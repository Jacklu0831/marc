from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any
from typing import List, Optional
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
        loss_type: str,
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
        self.loss_type = loss_type

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

    # each of (batch_size x pair_seq_len)
    all_input_ids = []
    all_attention_mask = []
    all_label_ids = []
    all_pair_start_idxs = [] # (batch_size, num_pair)

    while len(all_input_ids) < batch_size:
        # choose task and indices
        task = rng.choice(list(dataset.task_to_indices)) # type: ignore
        indices = dataset.task_to_indices[task]
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

            # set label ids to -100 here if not to be trained on
            if (example_i == 0 and dataset.loss_type == 'exclude_first') or (example_i < len(examples) - 1 and dataset.loss_type == 'only_last'):
                label_ids = torch.full(label_ids.shape, -100, dtype=dtype)

            assert input_ids.shape == attention_mask.shape == label_ids.shape

            input_ids_of_each_pair.append(input_ids)
            attention_mask_of_each_pair.append(attention_mask)
            label_ids_of_each_pair.append(label_ids)

        # start idxs computed from cumsum of lengths of pairs except last, set first start idx to 1 for bos
        pair_start_idxs = np.cumsum([0, *[len(x) for x in input_ids_of_each_pair[:-1]]]).tolist()
        pair_start_idxs[0] = 1

        # concat
        input_ids = torch.cat(input_ids_of_each_pair)
        attention_mask = torch.cat(attention_mask_of_each_pair)
        label_ids = torch.cat(label_ids_of_each_pair)

        # check length
        if len(input_ids) > dataset.max_seq_len:
            continue

        # add to batch
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_label_ids.append(label_ids)
        all_pair_start_idxs.append(pair_start_idxs)

    assert len(all_input_ids) == batch_size
    assert all(start_idxs[0] == 1 for start_idxs in all_pair_start_idxs)

    # get input ids lens (batch_size,)
    input_ids_lens = [len(x) for x in all_input_ids]

    # collate
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    all_input_ids = pad_sequence_left(all_input_ids, padding_value=dataset.tokenizer.pad_token_id)
    all_attention_mask = pad_sequence_left(all_attention_mask, padding_value=0)
    all_label_ids = pad_sequence_left(all_label_ids, padding_value=-100)
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    # extra padding
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        all_input_ids, all_attention_mask, all_label_ids = debug_extra_pad_tensors(
            [all_input_ids, all_attention_mask, all_label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
        )

    assert len(all_input_ids) == batch_size
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    # dataset.tokenizer.decode(all_input_ids[1], skip_special_tokens=False)
    # all_input_ids[1]
    # all_label_ids[1]

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'label_ids': all_label_ids,
        'input_ids_lens': input_ids_lens,
        'pair_start_idxs': all_pair_start_idxs,
    }


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.debug_len] * batch_size

    pair_start_idxs = [[1] + sorted(random.sample(range(2, dataset.debug_len), k=dataset.num_pair-1)) for _ in range(batch_size)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "input_ids_lens": input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
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
        eval_n_sample_per_task: int,
        debug_len: int,
        debug_random_pad: bool,
        debug_pad_len: int,
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
        self.eval_n_sample_per_task = eval_n_sample_per_task

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
            for sample_i in range(self.eval_n_sample_per_task):
                # task_id = f"{split_name}_{task}_{sample_i}"
                task_id = task
                chosen_indices = rng.choice(indices, size=num_pair, replace=False).tolist() # type: ignore
                chosen_indices_and_task_ids.append((chosen_indices, task_id))

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
        gen_input_ids, gen_output_ids = [], None

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
                gen_input_ids.append(input_input_ids)
                gen_output_ids = output_input_ids
            else:
                gen_input_ids.append(torch.cat([input_input_ids, output_input_ids]))

        # start idxs computed from cumsum of lengths of pairs except last, set first start idx to 1 for bos
        assert len(gen_input_ids) == self.num_pair
        pair_start_idxs = np.cumsum([0, *[len(x) for x in gen_input_ids[:-1]]]).tolist()
        pair_start_idxs[0] = 1

        # gen stuff
        gen_input_ids = torch.cat(gen_input_ids)
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=gen_input_ids.dtype)

        # deal with gen stuff
        assert gen_output_ids is not None
        assert gen_output_ids[-1] == self.tokenizer.eos_token_id
        out_token_length = len(gen_output_ids) - 1 # remove eos

        # filter by length
        if len(gen_input_ids) + len(gen_output_ids) > self.max_seq_len:
            return None

        # self.tokenizer.decode(gen_input_ids, skip_special_tokens=False)
        # self.tokenizer.decode(gen_output_ids, skip_special_tokens=False)

        return {
            "task_id": task_id,
            "gen_input_ids": gen_input_ids,
            "gen_output_ids": gen_output_ids,
            "gen_attention_mask": gen_attention_mask,
            "out_token_length": out_token_length,
            "pair_start_idxs": pair_start_idxs,
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
    pair_start_idxs = [x["pair_start_idxs"] for x in batch]
    assert all(start_idxs[0] == 1 for start_idxs in pair_start_idxs)

    # we do not pad gen_output_ids
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    gen_input_ids_lens = [len(x) for x in gen_input_ids]
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
        "gen_input_ids": gen_input_ids,
        "gen_output_ids": gen_output_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "gen_input_ids_lens": gen_input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
    }
    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len * 7 // 8 + 1), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full((batch_size, dataset.debug_len * 7 // 8 + 1), 1, dtype=torch.int64, device='cpu')
    gen_input_ids_lens = [dataset.debug_len * 7 // 8 + 1] * batch_size
    gen_output_ids = torch.randint(0, 30, (batch_size, dataset.debug_len // 8 + 1), dtype=torch.int64, device='cpu')

    pair_start_idxs = [[1] + sorted(random.sample(range(2, dataset.debug_len * 7 // 8 + 1), k=dataset.num_pair-1)) for _ in range(batch_size)]
    out_token_length = [dataset.debug_len // 8 + 1] * batch_size

    batch_dict = {
        "task_ids": ["task_0"] * batch_size,
        "gen_input_ids": gen_input_ids,
        "gen_output_ids": gen_output_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "gen_input_ids_lens": gen_input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
    }
    return batch_dict
