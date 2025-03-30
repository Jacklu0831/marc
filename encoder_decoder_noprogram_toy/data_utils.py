import random
import torch
from torch.utils.data import Dataset, get_worker_info
from typing import Dict, List, Any, Optional
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
        process_index: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        min_num_pair: int,
        max_num_pair: int,
        debug_len: int,
        num_workers: int,
        net_input_dim: int,
        net_hidden_dim: int,
        groundtruth_nets: List[TwoLayerNet],
    ):
        self.net_input_dim = net_input_dim
        self.net_hidden_dim = net_hidden_dim
        self._length = total_steps
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        self.debug_len = debug_len
        self.groundtruth_nets = groundtruth_nets

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

        attention_mask = torch.full((2 * required_num_pair,), 1, dtype=torch.int64, device='cpu')

        labels = torch.full((2 * required_num_pair,), -100, dtype=torch.float32)
        labels[1::2] = Y.squeeze(-1)

        out_list.append({
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    # collate does not require padding
    inputs_embeds_lens = [len(x['inputs_embeds']) for x in out_list] # backward compatible
    inputs_embeds = torch.stack([x['inputs_embeds'] for x in out_list])
    attention_mask = torch.stack([x['attention_mask'] for x in out_list])
    labels = torch.stack([x['labels'] for x in out_list])

    batch_dict = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
        "inputs_embeds_lens": inputs_embeds_lens,
    }
    return batch_dict


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        seed: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        min_num_pair: int,
        max_num_pair: int,
        debug_len: int,
        pad_side: str,
        net_input_dim: int,
        net_hidden_dim: int,
        groundtruth_nets: List[TwoLayerNet],
        ntokens: int,
    ):
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        self.debug_len = debug_len
        self.net_input_dim = net_input_dim
        self.net_hidden_dim = net_hidden_dim
        self.ntokens = ntokens
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
        labels[-1] = Y[-1].squeeze(-1)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            # for gs (no ntoken)
            "demon_inputs_embeds": inputs_embeds[:-2],
            "demon_attention_mask": attention_mask[:-2],
            "gen_inputs_embeds": inputs_embeds[-2:],
            "gen_attention_mask": attention_mask[-2:],
            "gen_labels": labels[-2:],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    inputs_embeds = [x["inputs_embeds"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    inputs_embeds_lens = [len(x) for x in inputs_embeds]
    inputs_embeds = pad_sequence_with_side(inputs_embeds, padding_value=0.1234, side=dataset.pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
    labels = pad_sequence_with_side(labels, padding_value=-100, side=dataset.pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        inputs_embeds, attention_mask, labels = debug_extra_pad_tensors(
            [inputs_embeds, attention_mask, labels],
            padding_values=[0.1234, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    # for gs (no ntoken)
    demon_inputs_embeds = [x["demon_inputs_embeds"] for x in batch]
    demon_attention_mask = [x["demon_attention_mask"] for x in batch]
    gen_inputs_embeds = [x["gen_inputs_embeds"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    gen_labels = [x["gen_labels"] for x in batch]

    demon_inputs_embeds = pad_sequence_with_side(demon_inputs_embeds, padding_value=0.1234, side=dataset.pad_side)
    demon_attention_mask = pad_sequence_with_side(demon_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_inputs_embeds = pad_sequence_with_side(gen_inputs_embeds, padding_value=0.1234, side=dataset.pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_labels = torch.stack(gen_labels) # no need to pad, always seq len of 2

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        demon_inputs_embeds, demon_attention_mask, gen_inputs_embeds, gen_attention_mask, gen_labels = debug_extra_pad_tensors(
            [demon_inputs_embeds, demon_attention_mask, gen_inputs_embeds, gen_attention_mask, gen_labels],
            padding_values=[0.1234, 0, 0.1234, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    batch_dict = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "labels": labels,
        "inputs_embeds_lens": inputs_embeds_lens,
        # for gs (no ntoken)
        "demon_inputs_embeds": demon_inputs_embeds,
        "demon_attention_mask": demon_attention_mask,
        "gen_inputs_embeds": gen_inputs_embeds,
        "gen_attention_mask": gen_attention_mask,
        "gen_labels": gen_labels,
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
        inputs_embeds = data['demon_inputs_embeds']
        assert inputs_embeds.shape[0] % 2 == 0 and inputs_embeds.dim() == 2

        return [
            {
                'inputs_embeds': emb,
                'attention_mask': torch.ones(emb.shape[:-1], dtype=torch.int64),
                'labels': emb[1, 0],
            }
            for emb in torch.chunk(inputs_embeds, inputs_embeds.shape[0] // 2, dim=0)
        ]


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    inputs_embeds = [x["inputs_embeds"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]

    inputs_embeds = torch.stack(inputs_embeds)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        # labels cant be padded, just single value no sequence length
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
