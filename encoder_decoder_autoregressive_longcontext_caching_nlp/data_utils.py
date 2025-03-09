import random
from collections import defaultdict
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import torch
from torch.utils.data import Dataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any, List, Optional
import numpy as np

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def pad_sequence_left(sequences: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    reversed_sequences = [seq.flip(0) for seq in sequences]
    padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
    return padded_reversed.flip(1)


def debug_extra_pad_tensors(
        tensors: List[torch.Tensor],
        padding_values: List[int],
        pad_len: int,
    ) -> List[torch.Tensor]:

    # left padding
    assert len(tensors) == len(padding_values)
    assert all(t.dim() == 2 for t in tensors)
    if pad_len == -1:
        pad_len = random.randint(0, 15) # arbitrary
    padded_tensors = []
    for arg, padding_value in zip(tensors, padding_values):
        pad = torch.full((arg.shape[0], pad_len), padding_value, device=arg.device, dtype=arg.dtype)
        padded_tensor = torch.cat([pad, arg], dim=-1)
        padded_tensors.append(padded_tensor)
    return padded_tensors


def verify_tokenizer_output(tokenizer_out: Any, tokenizer: PreTrainedTokenizerFast):
    # make sure the tokenizer doesnt fail me
    assert tokenizer_out['attention_mask'].numel() == tokenizer_out['attention_mask'].sum()
    assert tokenizer_out['input_ids'].shape == tokenizer_out['attention_mask'].shape
    assert tokenizer_out['input_ids'].dim() == 2 and tokenizer_out['input_ids'].shape[0] == 1
    assert tokenizer_out['input_ids'][0][0].item() == tokenizer.bos_token_id


########################################
# Training Dataset
########################################

class TrainDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        tokenizer: PreTrainedTokenizerFast,
        total_steps: int,
        seed: int,
        process_index: int,
        ntokens: int,
        num_pair: int,
        num_workers: int,
        max_seq_len: int,
        delimiter: str,
        debug_random_pad: bool,
        debug_len: int,
        debug_pad_len: int,
        debug_overfit: bool,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self._length = total_steps
        self.ntokens = ntokens
        self.num_pair = num_pair
        self.num_workers = num_workers if num_workers > 0 else 1
        self.max_seq_len = max_seq_len
        self.debug_random_pad = debug_random_pad
        self.debug_len = debug_len
        self.debug_pad_len = debug_pad_len
        self.debug_overfit = debug_overfit

        # get a customizable delimiter, use whitespace by default
        self.delimiter = delimiter
        delimiter_token_id = tokenizer(delimiter, return_tensors='pt')['input_ids'][0] # type: ignore
        assert len(delimiter_token_id) == 2 # bos, single token of interest
        self.delimiter_token_id = delimiter_token_id[1][None, ...] # (1,)

        # seed and process_index
        if num_workers == 0:
            self.rngs = [np.random.RandomState(seed + process_index)]
        else:
            self.rngs = [np.random.RandomState(seed + i) for i in range(num_workers * process_index, num_workers * (process_index + 1))]

        # task to indices
        self.task_to_indices = defaultdict(list)
        for idx, task in enumerate(data['task']):
            self.task_to_indices[task].append(idx)
        assert all(len(indices) >= self.num_pair for indices in self.task_to_indices.values())

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # We'll do random sampling in the collate fn
        return 0


def collate_fn_train(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]

    # each of (num_pair x batch_size, (pair_seq_len,))
    all_input_ids: List[Any] = [[] for _ in range(dataset.num_pair)]
    all_attention_mask: List[Any] = [[] for _ in range(dataset.num_pair)]
    all_label_ids: List[Any] = [[] for _ in range(dataset.num_pair)]

    while len(all_input_ids[0]) < batch_size:
        # choose task and indices
        task = rng.choice(list(dataset.task_to_indices)) # type: ignore
        indices = dataset.task_to_indices[task]

        if dataset.debug_overfit:
            chosen_indices = indices[:dataset.num_pair]
        else:
            chosen_indices = rng.choice(indices, size=dataset.num_pair, replace=False).tolist() # type: ignore

        # compute input_ids, attention_mask, label_ids for each pair
        input_ids_of_each_pair = []
        attention_mask_of_each_pair = []
        label_ids_of_each_pair = []

        examples = [dataset.data[i] for i in chosen_indices]
        for example_i, example in enumerate(examples):
            # tokenize
            tokenized_input = dataset.tokenizer(example['input'], return_tensors="pt")
            tokenized_output = dataset.tokenizer(example['output'], return_tensors="pt")
            verify_tokenizer_output(tokenized_input, dataset.tokenizer)
            verify_tokenizer_output(tokenized_output, dataset.tokenizer)
            input_input_ids = tokenized_input['input_ids'][0] # type: ignore
            output_input_ids = tokenized_output['input_ids'][0] # type: ignore
            dtype = input_input_ids.dtype

            # only keep bos for the first input
            if example_i > 0:
                input_input_ids = input_input_ids[1:]
            output_input_ids = output_input_ids[1:]

            # add whitespace at end of all inputs and outputs, except the last output
            input_input_ids = torch.cat([input_input_ids, torch.tensor([dataset.delimiter_token_id])])
            if example_i < len(examples) - 1:
                output_input_ids = torch.cat([output_input_ids, torch.tensor([dataset.delimiter_token_id])])

            # add eos at end of last output
            if example_i == len(examples) - 1:
                output_input_ids = torch.cat([output_input_ids, torch.tensor([dataset.tokenizer.eos_token_id])])

            # create input_ids, attention_mask, label_ids for pair
            input_ids = torch.cat([input_input_ids, output_input_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=dtype)

            label_ids = torch.cat([
                torch.full(input_input_ids.shape, -100, dtype=dtype),
                output_input_ids,
            ]) # do not predict input ids
            if example_i < len(examples) - 1:
                label_ids[-1] = -100 # do not predict delimiter

            assert input_ids.shape == attention_mask.shape == label_ids.shape

            input_ids_of_each_pair.append(input_ids)
            attention_mask_of_each_pair.append(attention_mask)
            label_ids_of_each_pair.append(label_ids)

        # check length
        if sum(len(x) for x in input_ids_of_each_pair) > dataset.max_seq_len:
            continue

        # add to batch
        for pair_i in range(dataset.num_pair):
            all_input_ids[pair_i].append(input_ids_of_each_pair[pair_i])
            all_attention_mask[pair_i].append(attention_mask_of_each_pair[pair_i])
            all_label_ids[pair_i].append(label_ids_of_each_pair[pair_i])

    assert len(all_input_ids) == dataset.num_pair
    assert set(len(x) for x in all_input_ids) == {batch_size}

    # get input ids lens (num_pair x batch_size)
    input_ids_lens = []
    for pair_input_ids in all_input_ids:
        input_ids_lens.append([len(ids) for ids in pair_input_ids])

    # collate
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    for pair_i in range(dataset.num_pair):
        all_input_ids[pair_i] = pad_sequence_left(all_input_ids[pair_i], padding_value=dataset.tokenizer.pad_token_id)
        all_attention_mask[pair_i] = pad_sequence_left(all_attention_mask[pair_i], padding_value=0)
        all_label_ids[pair_i] = pad_sequence_left(all_label_ids[pair_i], padding_value=-100)
        assert all_input_ids[pair_i].shape == all_attention_mask[pair_i].shape == all_label_ids[pair_i].shape

    # extra padding
    extra_padded_input_ids = []
    extra_padded_attention_mask = []
    extra_padded_label_ids = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for input_ids, attention_mask, label_ids in zip(all_input_ids, all_attention_mask, all_label_ids):
            input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
                [input_ids, attention_mask, label_ids],
                padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
                pad_len=dataset.debug_pad_len,
            )
            extra_padded_input_ids.append(input_ids)
            extra_padded_attention_mask.append(attention_mask)
            extra_padded_label_ids.append(label_ids)
    else:
        extra_padded_input_ids = all_input_ids
        extra_padded_attention_mask = all_attention_mask
        extra_padded_label_ids = all_label_ids

    assert len(extra_padded_input_ids) == len(extra_padded_attention_mask) == len(extra_padded_label_ids) == dataset.num_pair
    assert set(len(x) for x in extra_padded_input_ids) == {batch_size}
    assert set(len(x) for x in extra_padded_attention_mask) == {batch_size}
    assert set(len(x) for x in extra_padded_label_ids) == {batch_size}

    # dataset.tokenizer.decode(extra_padded_input_ids[-2][1], skip_special_tokens=False)
    # extra_padded_input_ids[-2][1]
    # extra_padded_label_ids[-2][1]

    return {
        'input_ids': extra_padded_input_ids,
        'attention_mask': extra_padded_attention_mask,
        'label_ids': extra_padded_label_ids,
        'input_ids_lens': input_ids_lens,
        'is_same': False,
    }


collate_fn_train_invar = collate_fn_train


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    sampled_indices = sorted(random.sample(range(1, dataset.debug_len), k=dataset.num_pair-1)) + [dataset.debug_len]
    pair_lens = []
    for i in range(len(sampled_indices)):
        if i == 0:
            pair_lens.append(sampled_indices[0])
        else:
            pair_lens.append(sampled_indices[i] - sampled_indices[i-1])
    assert len(pair_lens) == dataset.num_pair

    input_ids_of_each_pair = [torch.randint(0, 30, (batch_size, l), dtype=torch.int64, device='cpu') for l in pair_lens]
    attention_mask_of_each_pair = [torch.full((batch_size, l), 1, dtype=torch.int64, device='cpu') for l in pair_lens]
    input_ids_lens = [[l] * batch_size for l in pair_lens]
    assert sum(x.shape[1] for x in input_ids_of_each_pair) == dataset.debug_len

    return {
        "input_ids": input_ids_of_each_pair,
        "attention_mask": attention_mask_of_each_pair,
        "label_ids": input_ids_of_each_pair,
        "input_ids_lens": input_ids_lens,
        'is_same': False,
    }


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        data: Dataset,
        split_name: int,
        seed: int,
        tokenizer: PreTrainedTokenizerFast,
        ntokens: int,
        num_pair: int,
        max_seq_len: int,
        delimiter: str,
        eval_per_task: int,
        debug_len: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        debug_overfit: bool,
    ):
        self.data = data
        self.split_name = split_name
        self.seed = seed
        self.tokenizer = tokenizer
        self.ntokens = ntokens
        self.num_pair = num_pair
        self.max_seq_len = max_seq_len
        self.debug_len = debug_len
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.eval_per_task = eval_per_task

        # get a customizable delimiter, use whitespace by default
        self.delimiter = delimiter
        delimiter_token_id = tokenizer(delimiter, return_tensors='pt')['input_ids'][0] # type: ignore
        assert len(delimiter_token_id) == 2 # bos, single token of interest
        self.delimiter_token_id = delimiter_token_id[1][None, ...] # (1,)

        # task to indices
        self.task_to_indices = defaultdict(list)
        for idx, task in enumerate(data['task']):
            self.task_to_indices[task].append(idx)
        assert all(len(indices) >= self.num_pair for indices in self.task_to_indices.values())

        # pre-select tasks
        rng = np.random.RandomState(seed)
        chosen_indices_and_task_ids = []
        for task, indices in self.task_to_indices.items():
            for _ in range(self.eval_per_task):
                if debug_overfit:
                    chosen_indices = indices[:num_pair]
                else:
                    chosen_indices = rng.choice(indices, size=num_pair, replace=False).tolist() # type: ignore
                chosen_indices_and_task_ids.append((chosen_indices, task))

        # preset chosen task ids
        self.chosen_indices_and_task_ids = []
        for chosen_indices, task_id in chosen_indices_and_task_ids:
            formatted = self.format_and_filter(chosen_indices, task_id)
            if formatted is not None:
                self.chosen_indices_and_task_ids.append((chosen_indices, task_id))

        logger.info(f"evaluator filtered down to {len(self.chosen_indices_and_task_ids)}/{len(chosen_indices_and_task_ids)}")

    def format_and_filter(self, chosen_indices: List[int], task_id: str) -> Optional[Dict]:
        examples = [self.data[i] for i in chosen_indices]
        assert len(set(e['task'] for e in examples)) == 1

        # compute input_ids, attention_mask for each train pair and one for gen pair
        input_ids_of_each_pair = []
        gen_input_ids, gen_output_ids = None, None

        for example_i, example in enumerate(examples):
            # tokenize
            tokenized_input = self.tokenizer(example['input'], return_tensors="pt")
            tokenized_output = self.tokenizer(example['output'], return_tensors="pt")
            verify_tokenizer_output(tokenized_input, self.tokenizer)
            verify_tokenizer_output(tokenized_output, self.tokenizer)
            input_input_ids = tokenized_input['input_ids'][0] # type: ignore
            output_input_ids = tokenized_output['input_ids'][0] # type: ignore

            # only keep bos for the first input
            if example_i > 0:
                input_input_ids = input_input_ids[1:]
            output_input_ids = output_input_ids[1:]

            # add whitespace at end of all inputs and outputs, except the last output
            input_input_ids = torch.cat([input_input_ids, torch.tensor([self.delimiter_token_id])])
            if example_i < len(examples) - 1:
                output_input_ids = torch.cat([output_input_ids, torch.tensor([self.delimiter_token_id])])

            # add eos at end of last output
            if example_i == len(examples) - 1:
                output_input_ids = torch.cat([output_input_ids, torch.tensor([self.tokenizer.eos_token_id])])

            # create input_ids, attention_mask for pair
            if example_i == len(examples) - 1:
                gen_input_ids = input_input_ids
                gen_output_ids = output_input_ids
            else:
                input_ids = torch.cat([input_input_ids, output_input_ids])
                input_ids_of_each_pair.append(input_ids)

        # create full attention masks
        assert gen_input_ids is not None and gen_output_ids is not None
        attention_mask_of_each_pair = [torch.full(x.shape, 1, dtype=x.dtype) for x in input_ids_of_each_pair]
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=gen_input_ids.dtype)

        # deal with gen stuff
        assert gen_output_ids[-1] == self.tokenizer.eos_token_id
        out_token_length = len(gen_output_ids) - 1 # remove eos

        # filter by length
        if sum(len(x) for x in input_ids_of_each_pair) + len(gen_input_ids) + len(gen_output_ids) > self.max_seq_len:
            return None

        # self.tokenizer.decode(input_ids_of_each_pair[5], skip_special_tokens=False)
        # self.tokenizer.decode(gen_input_ids, skip_special_tokens=False)
        # self.tokenizer.decode(gen_output_ids, skip_special_tokens=False)

        return {
            "task_id": task_id,
            "input_ids": input_ids_of_each_pair,
            "attention_mask": attention_mask_of_each_pair,
            "gen_input_ids": gen_input_ids,
            "gen_output_ids": gen_output_ids,
            "gen_attention_mask": gen_attention_mask,
            "out_token_length": out_token_length,
        }

    def __len__(self):
        return len(self.chosen_indices_and_task_ids)

    def __getitem__(self, idx):
        # ignore idx, we sample instead
        formatted = self.format_and_filter(*self.chosen_indices_and_task_ids[idx])
        assert formatted is not None
        return formatted


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    task_ids = [x['task_id'] for x in batch]
    gen_input_ids = [x["gen_input_ids"] for x in batch]
    gen_output_ids = [x["gen_output_ids"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    out_token_length = [x["out_token_length"] for x in batch]
    batch_size = len(task_ids)

    # get input ids lens (num_pair-1 x batch_size)
    all_input_ids: List[Any] = [[] for _ in range(dataset.num_pair - 1)]
    all_attention_mask: List[Any] = [[] for _ in range(dataset.num_pair - 1)]
    for x in batch:
        input_ids_of_each_pair = x['input_ids']
        attention_mask_of_each_pair = x['attention_mask']
        assert len(input_ids_of_each_pair) == len(attention_mask_of_each_pair) == dataset.num_pair - 1
        for pair_i in range(dataset.num_pair - 1):
            all_input_ids[pair_i].append(input_ids_of_each_pair[pair_i])
            all_attention_mask[pair_i].append(attention_mask_of_each_pair[pair_i])

    assert set(len(x) for x in all_input_ids) == {batch_size}

    # get lengths of ids
    input_ids_lens = []
    for pair_input_ids in all_input_ids:
        input_ids_lens.append([len(ids) for ids in pair_input_ids])
    gen_input_ids_lens = [len(ids) for ids in gen_input_ids]

    # collate train
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    for pair_i in range(dataset.num_pair - 1):
        all_input_ids[pair_i] = pad_sequence_left(all_input_ids[pair_i], padding_value=dataset.tokenizer.pad_token_id)
        all_attention_mask[pair_i] = pad_sequence_left(all_attention_mask[pair_i], padding_value=0)

    # extra padding train
    extra_padded_input_ids = []
    extra_padded_attention_mask = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for input_ids, attention_mask in zip(all_input_ids, all_attention_mask):
            input_ids, attention_mask = debug_extra_pad_tensors(
                [input_ids, attention_mask],
                padding_values=[dataset.tokenizer.pad_token_id, 0],
                pad_len=dataset.debug_pad_len,
            )
            extra_padded_input_ids.append(input_ids)
            extra_padded_attention_mask.append(attention_mask)
    else:
        extra_padded_input_ids = all_input_ids
        extra_padded_attention_mask = all_attention_mask

    # collate gen
    # we do not pad gen_output_ids
    gen_input_ids = pad_sequence_left(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id)
    gen_attention_mask = pad_sequence_left(gen_attention_mask, padding_value=0)

    # extra padding gen
    gen_input_ids, gen_attention_mask = debug_extra_pad_tensors(
        [gen_input_ids, gen_attention_mask],
        padding_values=[dataset.tokenizer.pad_token_id, 0],
        pad_len=dataset.debug_pad_len,
    )

    batch_dict = {
        "task_ids": task_ids,
        "input_ids": extra_padded_input_ids,
        "attention_mask": extra_padded_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_output_ids": gen_output_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
    }
    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    sampled_indices = sorted(random.sample(range(1, dataset.debug_len), k=dataset.num_pair-1)) + [dataset.debug_len]
    pair_lens = []
    for i in range(len(sampled_indices)):
        if i == 0:
            pair_lens.append(sampled_indices[0])
        else:
            pair_lens.append(sampled_indices[i] - sampled_indices[i-1])
    assert len(pair_lens) == dataset.num_pair

    input_ids_of_each_pair = [torch.randint(0, 30, (batch_size, l), dtype=torch.int64, device='cpu') for l in pair_lens[:-1]]
    attention_mask_of_each_pair = [torch.full((batch_size, l), 1, dtype=torch.int64, device='cpu') for l in pair_lens[:-1]]

    gen_input_ids = torch.randint(0, 30, (batch_size, pair_lens[-1] // 2), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    gen_output_ids = torch.randint(0, 30, (batch_size, pair_lens[-1] // 2 + 1), dtype=torch.int64, device='cpu')
    out_token_length = gen_output_ids.shape[1]

    input_ids_lens = [[l] * batch_size for l in pair_lens[:-1]]
    gen_input_ids_lens = [pair_lens[-1] // 2] * batch_size

    return {
        "task_ids": ["task_0"] * batch_size,
        "input_ids": input_ids_of_each_pair,
        "attention_mask": attention_mask_of_each_pair,
        "gen_input_ids": gen_input_ids,
        "gen_output_ids": gen_output_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
    }
