import random
import torch
from torch.utils.data import Dataset, get_worker_info
from typing import Dict, List, Any

from oracle_fit import create_ground_truth_net, TwoLayerNet

from accelerate.logging import get_logger


logger = get_logger(__name__, log_level="INFO")


def pad_sequence_with_side(sequences: List[torch.Tensor], padding_value: Any, side: str) -> torch.Tensor:
    max_len = max(len(x) for x in sequences)
    padded = []
    for x in sequences:
        pad_len = max_len - len(x)
        if pad_len > 0:
            pad = torch.full((pad_len, *x.shape[1:]), padding_value, dtype=x.dtype, device=x.device)
            if side == 'left':
                x = torch.cat([pad, x])
            else:
                x = torch.cat([x, pad])
        padded.append(x)
    return torch.stack(padded)


def debug_extra_pad_tensors(
        tensors: List[torch.Tensor],
        padding_values: List[Any],
        pad_len: int,
        side: str,
    ) -> List[torch.Tensor]:
    assert len(tensors) == len(padding_values)
    if pad_len == -1:
        pad_len = random.randint(1, 15) # arbitrary
    padded_tensors = []
    for arg, padding_value in zip(tensors, padding_values):
        pad = torch.full((arg.shape[0], pad_len, *arg.shape[2:]), padding_value, device=arg.device, dtype=arg.dtype)
        if side == 'right':
            padded_tensor = torch.cat([arg, pad], dim=1)
        else:
            padded_tensor = torch.cat([pad, arg], dim=1)
        padded_tensors.append(padded_tensor)
    return padded_tensors


def get_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


########################################
# Training Dataset
########################################

class TrainDataset(Dataset):
    def __init__(
        self,
        total_steps: int,
        seed: int,
        pad_side: str,
        debug_random_pad: bool,
        debug_pad_len: int,
        process_index: int,
        min_num_pair: int,
        max_num_pair: int,
        num_workers: int,
        net_input_dim: int,
        net_hidden_dim: int,
        groundtruth_nets: List[TwoLayerNet],
    ):
        self.net_input_dim = net_input_dim
        self.net_hidden_dim = net_hidden_dim
        self._length = total_steps
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        self.groundtruth_nets = groundtruth_nets
        self.pad_side = pad_side
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len

        self.num_workers = num_workers
        self.process_index = process_index
        self.seed = seed

        # set rngs
        self.set_rngs(epoch=0)


    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # We'll do random sampling in the collate fn
        return 0

    def set_rngs(self, epoch: int):
        epoch_seed = epoch * 1000
        # seed and process_index
        if self.num_workers == 0:
            self.rngs = [get_torch_generator(self.seed + epoch_seed + self.process_index)]
        else:
            self.rngs = [get_torch_generator(self.seed + epoch_seed + i) for i in range(self.num_workers * self.process_index, self.num_workers * (self.process_index + 1))]
        # num pair must be the same across gpus
        if self.num_workers == 0:
            self.num_pair_rngs = [get_torch_generator(self.seed + epoch_seed)]
        else:
            self.num_pair_rngs = [get_torch_generator(self.seed + epoch_seed + i) for i in range(self.num_workers)]


def collate_fn_train(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]
    num_pair_rng = dataset.num_pair_rngs[int(worker_id)]

    # the restriction here is to enforce all list of pairs in batch are equal length
    out_list = []

    # must sample this number of pairs to avoid GPU synchronization issues
    required_num_pair = torch.randint(
        low=dataset.min_num_pair, high=dataset.max_num_pair + 1,
        size=(1,),
        generator=num_pair_rng,
    ).item()
    assert isinstance(required_num_pair, int)

    # sample random task from random dataset, if grid size >30 or does not have enough for required_num_pair, retry
    for _ in range(batch_size):
        # select or create generator
        net_idx = torch.tensor(-1)
        if len(dataset.groundtruth_nets) > 0:
            net_idx = torch.randint(low=0, high=len(dataset.groundtruth_nets), size=(1,), generator=rng)
            ground_truth_net = dataset.groundtruth_nets[net_idx]
        else:
            ground_truth_net = create_ground_truth_net(dataset.net_input_dim, dataset.net_hidden_dim, generator=rng)

        # generate data
        X = torch.randn(required_num_pair, dataset.net_input_dim, dtype=torch.float32, generator=rng)
        Y = ground_truth_net(X)

        # pad Y with 0
        Y_padded = torch.cat([Y, torch.zeros((required_num_pair, dataset.net_input_dim - 1), dtype=torch.float32)], dim=1)

        # interleave X and Y
        inputs_embeds = torch.zeros((2 * required_num_pair, dataset.net_input_dim), dtype=torch.float32)
        inputs_embeds[0::2] = X
        inputs_embeds[1::2] = Y_padded

        labels = torch.full((2 * required_num_pair,), -100, dtype=torch.float32)
        labels[1::2] = Y.squeeze(-1)

        out_list.append({
            "task_identifier": str(net_idx.item()),
            "inputs_embeds": torch.chunk(inputs_embeds, inputs_embeds.shape[0] // 2, dim=0),
            "labels": torch.chunk(labels, labels.shape[0] // 2, dim=0),
        })

    # parse into pair idx, then padded batch
    pair_idx_to_inputs_embeds = []
    pair_idx_to_attention_mask = []
    pair_idx_to_labels = []
    inputs_embeds_lens = []
    for pair_i in range(required_num_pair):
        # parse
        lens = [len(x['inputs_embeds'][pair_i]) for x in out_list]
        inputs_embeds = torch.stack([x['inputs_embeds'][pair_i] for x in out_list])
        labels = torch.stack([x['labels'][pair_i] for x in out_list])
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.int64)
        # assert
        assert all(l == 2 for l in lens)
        assert tuple(inputs_embeds.shape) == (batch_size, 2, dataset.net_input_dim)
        assert tuple(labels.shape) == (batch_size, 2)
        # aggregate
        inputs_embeds_lens.append(lens)
        pair_idx_to_inputs_embeds.append(inputs_embeds)
        pair_idx_to_attention_mask.append(attention_mask)
        pair_idx_to_labels.append(labels)

    task_identifiers = [x['task_identifier'] for x in out_list]

    extra_padded_inputs_embeds = []
    extra_padded_attention_mask = []
    extra_padded_labels = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for emb, mask, lab in zip(pair_idx_to_inputs_embeds, pair_idx_to_attention_mask, pair_idx_to_labels):
            emb, mask, lab = debug_extra_pad_tensors(
                [emb, mask, lab],
                padding_values=[0.1234, 0, -100],
                pad_len=dataset.debug_pad_len,
                side=dataset.pad_side,
            )
            extra_padded_inputs_embeds.append(emb)
            extra_padded_attention_mask.append(mask)
            extra_padded_labels.append(lab)
    else:
        extra_padded_inputs_embeds = pair_idx_to_inputs_embeds
        extra_padded_attention_mask = pair_idx_to_attention_mask
        extra_padded_labels = pair_idx_to_labels

    batch_dict = {
        "inputs_embeds": extra_padded_inputs_embeds,
        "attention_mask": extra_padded_attention_mask,
        "labels": extra_padded_labels,
        "inputs_embeds_lens": inputs_embeds_lens,
        "num_pairs": [required_num_pair] * batch_size,
        "is_same": False,
        "task_identifiers": task_identifiers,
    }
    return batch_dict


collate_fn_train_invar = collate_fn_train


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        seed: int,
        ntokens: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        debug_len: int,
        net_input_dim: int,
        net_hidden_dim: int,
        groundtruth_nets: List[TwoLayerNet],
        min_num_pair: int,
        max_num_pair: int,
    ):
        self.seed = seed
        self.ntokens = ntokens
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.debug_len = debug_len
        self.net_input_dim = net_input_dim
        self.net_hidden_dim = net_hidden_dim
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        assert len(groundtruth_nets) > 0

        # just generate data here
        rng = get_torch_generator(seed)
        required_num_pairs = torch.randint(
            low=min_num_pair, high=max_num_pair + 1,
            size=(len(groundtruth_nets),),
            generator=rng,
        ).tolist()

        self.data = []
        for required_num_pair, ground_truth_net in zip(required_num_pairs, groundtruth_nets):
            X = torch.randn(required_num_pair, net_input_dim, dtype=torch.float32, generator=rng)
            Y = ground_truth_net(X)
            self.data.append(self.format(X, Y))

    def format(self, X: torch.Tensor, Y: torch.Tensor) -> Dict:
        required_num_pair = X.shape[0]

        # pad Y with 0
        Y_padded = torch.cat([Y, torch.zeros((required_num_pair, self.net_input_dim - 1), dtype=torch.float32)], dim=1)

        # interleave X and Y
        inputs_embeds = torch.zeros((2 * required_num_pair, self.net_input_dim), dtype=torch.float32)
        inputs_embeds[0::2] = X
        inputs_embeds[1::2] = Y_padded

        attention_mask = torch.full((2 * required_num_pair,), 1, dtype=torch.int64, device='cpu')

        labels = torch.full((2 * required_num_pair,), -100, dtype=torch.float32)
        labels[1::2] = Y.squeeze(-1)

        # chunk and separate last pair into gen
        inputs_embeds = list(torch.chunk(inputs_embeds, inputs_embeds.shape[0] // 2, dim=0))
        attention_mask = list(torch.chunk(attention_mask, attention_mask.shape[0] // 2, dim=0))
        labels = list(torch.chunk(labels, labels.shape[0] // 2, dim=0))

        return {
            "inputs_embeds": inputs_embeds[:-1],
            "attention_mask": attention_mask[:-1],
            "labels": labels[:-1],
            "gen_inputs_embeds": inputs_embeds[-1],
            "gen_attention_mask": attention_mask[-1],
            "gen_labels": labels[-1],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)

    inputs_embeds = [x["inputs_embeds"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    gen_inputs_embeds = [x["gen_inputs_embeds"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    gen_labels = [x["gen_labels"] for x in batch]

    # save number of pairs before padding
    num_pairs = [len(i) for i in inputs_embeds]
    max_num_pairs = max(num_pairs)

    # pad all samples in batch with 0-tensor
    # also format then to [pair_idx, batch_size, seq_len]
    pair_idx_to_inputs_embeds = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    pair_idx_to_attention_mask = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    for batch_i, (embs, mask) in enumerate(zip(inputs_embeds, attention_mask)): # iterate over batch here
        embs += [embs[0]] * (max_num_pairs - len(embs)) # pad
        mask += [mask[0]] * (max_num_pairs - len(mask)) # pad
        for pair_i, (embs_, mask_) in enumerate(zip(embs, mask)):
            pair_idx_to_inputs_embeds[pair_i][batch_i] = embs_
            pair_idx_to_attention_mask[pair_i][batch_i] = mask_

    # get lengths of embs
    inputs_embeds_lens = []
    for pair_i in range(max_num_pairs):
        inputs_embeds_lens.append([len(embs) for embs in pair_idx_to_inputs_embeds[pair_i]])
    gen_inputs_embeds_lens = [len(embs) for embs in gen_inputs_embeds]

    # actual padding of sequences
    padded_inputs_embeds = []
    padded_attention_mask = []
    for inputs_embeds, attention_mask in zip(pair_idx_to_inputs_embeds, pair_idx_to_attention_mask):
        inputs_embeds = pad_sequence_with_side(inputs_embeds, padding_value=0.1234, side=dataset.pad_side)
        attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
        padded_inputs_embeds.append(inputs_embeds)
        padded_attention_mask.append(attention_mask)

    # debug extra padding
    extra_padded_inputs_embeds = []
    extra_padded_attention_mask = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for inputs_embeds, attention_mask in zip(padded_inputs_embeds, padded_attention_mask):
            inputs_embeds, attention_mask = debug_extra_pad_tensors(
                [inputs_embeds, attention_mask],
                padding_values=[0.1234, 0],
                pad_len=dataset.debug_pad_len,
                side=dataset.pad_side,
            )
            extra_padded_inputs_embeds.append(inputs_embeds)
            extra_padded_attention_mask.append(attention_mask)
    else:
        extra_padded_inputs_embeds = padded_inputs_embeds
        extra_padded_attention_mask = padded_attention_mask

    # pad the gen arguments (and debug padding again)
    gen_inputs_embeds = torch.stack(gen_inputs_embeds)
    gen_attention_mask = torch.stack(gen_attention_mask)
    gen_labels = torch.stack(gen_labels)
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        gen_inputs_embeds, gen_attention_mask, gen_labels = debug_extra_pad_tensors(
            [gen_inputs_embeds, gen_attention_mask, gen_labels],
            padding_values=[0.1234, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    batch_dict = {
        "inputs_embeds": extra_padded_inputs_embeds,
        "attention_mask": extra_padded_attention_mask,
        "gen_inputs_embeds": gen_inputs_embeds,
        "gen_attention_mask": gen_attention_mask,
        "gen_labels": gen_labels,
        "inputs_embeds_lens": inputs_embeds_lens,
        "gen_inputs_embeds_lens": gen_inputs_embeds_lens,
        "num_pairs": num_pairs,
    }
    return batch_dict


########################################
# Gradient Search Dataset
########################################
class GSDataset(Dataset):
    def __init__(
        self,
        data: Dict,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
    ):
        self.data = data
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side

        # format data (only use demonstration pairs)
        self.parsed_examples = self.format_all(data)

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format_all(self, data: Dict) -> List[Dict]:
        inputs_embeds = data['inputs_embeds']
        attention_mask = data['attention_mask']
        labels = data['labels']
        assert len(inputs_embeds) == len(attention_mask) == len(labels)
        assert all(x.shape[0] == 2 for x in inputs_embeds)
        assert all(x.shape[0] == y.shape[0] == z.shape[0] for x, y, z in zip(inputs_embeds, attention_mask, labels))

        return [{
            "inputs_embeds": emb,
            "attention_mask": mask,
            "labels": lab,
        } for emb, mask, lab in zip(inputs_embeds, attention_mask, labels)]


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    inputs_embeds = torch.stack([x["inputs_embeds"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    assert inputs_embeds.shape[:2] == attention_mask.shape == labels.shape
    assert inputs_embeds.shape[1] == 2

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        inputs_embeds, attention_mask = debug_extra_pad_tensors(
            [inputs_embeds, attention_mask],
            padding_values=[0.1234, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    batch_dict = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return batch_dict
