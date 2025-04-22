from datasets import load_dataset
import copy
import math
import os
import json
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, List, Optional, Union, Tuple
import numpy as np

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def pad_sequence_with_side(sequences: List[torch.Tensor], padding_value: int, side: str) -> torch.Tensor:
    if side == 'right':
        return pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    else:
        reversed_sequences = [seq.flip(0) for seq in sequences]
        padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
        return padded_reversed.flip(1)


def debug_extra_pad_tensors(
        tensors: List[torch.Tensor],
        padding_values: List[int],
        pad_len: int,
        side: str,
    ) -> List[torch.Tensor]:
    assert len(tensors) == len(padding_values)
    assert all(t.dim() == 2 for t in tensors)
    if pad_len == -1:
        pad_len = random.randint(1, 15) # arbitrary
    padded_tensors = []
    for arg, padding_value in zip(tensors, padding_values):
        pad = torch.full((arg.shape[0], pad_len), padding_value, device=arg.device, dtype=arg.dtype)
        if side == 'right':
            padded_tensor = torch.cat([arg, pad], dim=-1)
        else:
            padded_tensor = torch.cat([pad, arg], dim=-1)
        padded_tensors.append(padded_tensor)
    return padded_tensors


def tokenize(
        text: str,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast]
    ) -> torch.Tensor:

    tokenizer_out = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    assert tokenizer_out['input_ids'].shape == tokenizer_out['attention_mask'].shape # type: ignore
    assert tokenizer_out['input_ids'].dim() == 2 and tokenizer_out['input_ids'].shape[0] == 1 # type: ignore
    assert tokenizer_out['attention_mask'].numel() == tokenizer_out['attention_mask'].sum() # type: ignore

    input_ids = tokenizer_out['input_ids'][0] # type: ignore
    if not isinstance(tokenizer, GPT2TokenizerFast):
        assert input_ids[0].item() == tokenizer.bos_token_id # type: ignore
        input_ids = input_ids[1:]

    return input_ids


def parse_pairs(
    pairs: List[Dict],
    tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
    max_seq_len: int,
    delimiter_token_id: torch.Tensor,
    loss_type: str,
    is_train: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[List]]]:

    # compute input_ids, attention_mask, label_ids for each pair
    input_ids_of_each_pair = []
    label_ids_of_each_pair = []
    final_output_start_idx = -1

    # for gs (no ntoken)
    demon_input_ids = []
    demon_start_idxs = [0]

    for pair_i, pair in enumerate(pairs):
        # tokenize
        input_input_ids = tokenize(pair['input'], tokenizer)
        output_input_ids = tokenize(pair['output'], tokenizer)

        # append delimiter to input
        input_input_ids = torch.cat([input_input_ids, delimiter_token_id])

        # prepend \n\n for inputs that are not first
        if pair_i != 0:
            input_input_ids = torch.cat([delimiter_token_id, input_input_ids])

        # record the last output's start idx for optional truncation
        if pair_i == len(pairs) - 1:
            final_output_start_idx = sum(x.shape[0] for x in input_ids_of_each_pair) + input_input_ids.shape[0]

        # create input_ids and label_ids
        input_ids = torch.cat([input_input_ids, output_input_ids])
        if (pair_i == 0 and loss_type == 'exclude_first') or (pair_i < len(pairs) - 1 and loss_type == 'only_last'):
            label_ids = torch.full(input_ids.shape, -100, dtype=input_ids.dtype)
        else:
            label_ids = torch.cat([
                torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype),
                output_input_ids,
            ])

        input_ids_of_each_pair.append(input_ids)
        label_ids_of_each_pair.append(label_ids)

        # for gs (no token)
        demon_input_ids.append(input_input_ids)
        demon_input_ids.append(output_input_ids)

        # dont need genpair, dont need last demon pair
        if pair_i < len(pairs) - 2:
            demon_start_idxs.append(demon_start_idxs[-1] + len(input_ids))

    # concat
    input_ids = torch.cat(input_ids_of_each_pair)
    attention_mask = torch.full(input_ids.shape, 1, dtype=input_ids.dtype)
    label_ids = torch.cat(label_ids_of_each_pair)
    assert input_ids.shape == attention_mask.shape == label_ids.shape

    # optionally truncate, also messes up pair start idxs
    assert final_output_start_idx > -1
    if len(input_ids) > max_seq_len:
        return None

    # for gs (no ntoken)
    if not is_train:
        # gen input ids
        gen_input_ids = torch.cat(demon_input_ids[-2:])
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64)
        # gen label ids
        gen_label_ids = torch.full(demon_input_ids[-2].shape, -100, dtype=input_ids.dtype)
        gen_label_ids = torch.cat([gen_label_ids, demon_input_ids[-1]])
        # demon input ids
        demon_input_ids = torch.cat(demon_input_ids[:-2])
        demon_attention_mask = torch.full(demon_input_ids.shape, 1, dtype=torch.int64)
    else:
        gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask, demon_start_idxs = None, None, None, None, None, None

    return input_ids, attention_mask, label_ids, \
        gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask, demon_start_idxs


def collate_data(
    batch: List[Dict],
    tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
    debug_random_pad: bool,
    debug_pad_len: int,
    pad_side: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:

    batch_size = len(batch)
    all_input_ids = [x['input_ids'] for x in batch]
    all_attention_mask = [x['attention_mask'] for x in batch]
    all_label_ids = [x['label_ids'] for x in batch]

    # get input ids lens (batch_size,)
    input_ids_lens = [len(x) for x in all_input_ids]

    # collate
    assert isinstance(tokenizer.pad_token_id, int)
    all_input_ids = pad_sequence_with_side(all_input_ids, padding_value=tokenizer.pad_token_id, side=pad_side)
    all_attention_mask = pad_sequence_with_side(all_attention_mask, padding_value=0, side=pad_side)
    all_label_ids = pad_sequence_with_side(all_label_ids, padding_value=-100, side=pad_side)
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    # extra padding
    if debug_random_pad or debug_pad_len > -1:
        all_input_ids, all_attention_mask, all_label_ids = debug_extra_pad_tensors(
            [all_input_ids, all_attention_mask, all_label_ids],
            padding_values=[tokenizer.pad_token_id, 0, -100],
            pad_len=debug_pad_len,
            side=pad_side,
        )

    assert len(all_input_ids) == batch_size
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape
    return all_input_ids, all_attention_mask, all_label_ids, input_ids_lens


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        seed: int,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        debug_random_pad: bool,
        debug_pad_len: int,
        debug_max_len: bool,
        pad_side: str,
        max_seq_len: int,
        eval_test_per_task: int,
        eval_ratio: float,
        delimiter: str,
        num_demonstrations: int,
        eval_on_demonstrations: bool,
    ):
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        self.debug_max_len = debug_max_len
        self.seed = seed
        self.num_demonstrations = num_demonstrations

        # separate input and output by newline
        self.delimiter_token_id = tokenize(delimiter, tokenizer)

        # load mmlu
        dataset = load_dataset("cais/mmlu", 'all')['test'] # type: ignore
        self.tasks = sorted(set(dataset['subject'])) # type: ignore

        # preprocess a bit
        data = [dataset[i] for i in range(len(dataset))]
        for i, d in enumerate(data):
            data[i] = {
                'task': d['subject'],
                'input': d['question'],
                'output': d['choices'][d['answer']],
                'options': d['choices'],
            }

        # load data
        task_to_demonstrations = defaultdict(list)
        task_to_test_pairs = defaultdict(list)
        for task in self.tasks:
            task_data = [d for d in data if d['task'] == task]

            # get demonstrations
            assert num_demonstrations <= len(task_data)
            task_to_demonstrations[task] = task_data[:num_demonstrations]

            # get test
            if eval_on_demonstrations:
                task_to_test_pairs[task] = copy.deepcopy(task_to_demonstrations[task])
            else:
                num_chosen = math.ceil((len(task_data[num_demonstrations:])) * eval_ratio)
                task_to_test_pairs[task] = task_data[num_demonstrations: num_demonstrations + num_chosen]

        # debug: check eval on demonstrations is correct
        if eval_on_demonstrations:
            assert task_to_test_pairs == task_to_demonstrations

        assert set(task_to_demonstrations.keys()) == set(task_to_test_pairs.keys()) == set(self.tasks)

        total_unfiltered_tasks = len(self.tasks)

        # process and filter down
        logger.info(f'generating samples for mmlu')
        self.data = []
        unfiltered_total_test, filtered_total_test, unfiltered_total_sample = 0, 0, 0

        for task_i, (task, test_pairs) in enumerate(task_to_test_pairs.items()):
            logger.info(f'{task_i+1}/{len(task_to_test_pairs)}')

            test_added = 0 # we only add maximum of eval_test_per_task number of tests
            patience = 0 # some tasks just contains sequences that are too long
            for test_idx, test_pair in enumerate(test_pairs):
                assert len(test_pair['options']) > 1
                assert test_pair['output'] in test_pair['options']
                correct_option = test_pair['output']

                # get outputs for each option
                outs = []
                for option in test_pair['options']:
                    test_pair['output'] = option
                    outs.append(self.format_and_filter(task_to_demonstrations[task], test_pair, test_idx, correct_option))

                # add to data, accumulate filter and unfilter stats
                unfiltered_total_test += 1
                unfiltered_total_sample += len(test_pair['options'])
                if None not in outs: # all tests fit in sequence length
                    self.data += outs
                    filtered_total_test += 1
                    test_added += 1
                    patience = 0
                else:
                    patience += 1

                if test_added == eval_test_per_task or patience == 100:
                    break

        # some tasks may be completely filtered due to max sequence length
        self.tasks = sorted(set(data['task'] for data in self.data))
        self.task_to_demonstrations = {t: task_to_demonstrations[t] for t in self.tasks}
        assert set(self.tasks) == set(self.task_to_demonstrations.keys())
        logger.info(f'eval split filtered to {len(self.tasks)}/{total_unfiltered_tasks} tasks, {filtered_total_test}/{unfiltered_total_test} tests, {len(self.data)}/{unfiltered_total_sample} samples')

        # get maxseqlen
        new_max_seq_len = 0
        for d in self.data:
            new_max_seq_len = max(new_max_seq_len, len(d['demon_input_ids']) + len(d['gen_input_ids']))
        self.max_seq_len = new_max_seq_len
        logger.info(f'new max seq len {self.max_seq_len}')

        # debug: check demon pairs are same for each task
        task_to_demon_input_ids = {}
        task_to_demon_start_idxs = {}
        for d in self.data:
            assert d['demon_attention_mask'].numel() == d['demon_attention_mask'].sum()
            task, ids, idxs = d['task'], d['demon_input_ids'], d['demon_start_idxs']
            if task not in task_to_demon_input_ids:
                assert task not in task_to_demon_start_idxs
                task_to_demon_input_ids[task] = ids
                task_to_demon_start_idxs[task] = idxs
            assert torch.equal(ids, task_to_demon_input_ids[task])
            assert idxs == task_to_demon_start_idxs[task]
        del task_to_demon_input_ids
        del task_to_demon_start_idxs


    def format_and_filter(self, demonstrations: List[Dict], test_pair: Dict, test_idx: int, correct_option: str) -> Optional[Dict]:
        # make sure they are all the same task with the same non-empty options
        task = test_pair['task']
        assert all(e['task'] == task for e in demonstrations) # test and demonstration pair have same task (dont need same option)
        assert correct_option in test_pair['options']

        out = parse_pairs(
            pairs=demonstrations + [test_pair],
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            delimiter_token_id=self.delimiter_token_id,
            loss_type="only_last",
            is_train=False,
        )
        if out == None:
            return None
        input_ids, attention_mask, label_ids, gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask, demon_start_idxs = out

        return {
            "task": task,
            "test_idx": test_idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "option": test_pair['output'],
            "correct_option": correct_option,
            # for gs (no ntoken)
            "demon_input_ids": demon_input_ids,
            "demon_attention_mask": demon_attention_mask,
            "gen_input_ids": gen_input_ids,
            "gen_label_ids": gen_label_ids,
            "gen_attention_mask": gen_attention_mask,
            "demonstrations_pairs": demonstrations,
            "demon_start_idxs": demon_start_idxs,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)

    all_input_ids, all_attention_mask, all_label_ids, input_ids_lens = collate_data(
        batch=batch,
        tokenizer=dataset.tokenizer,
        debug_random_pad=dataset.debug_random_pad,
        debug_pad_len=dataset.debug_pad_len,
        pad_side=dataset.pad_side,
    )
    assert len(all_input_ids) == batch_size

    # eval only
    task = [x['task'] for x in batch]
    test_idx = [x['test_idx'] for x in batch]
    option = [x['option'] for x in batch]
    correct_option = [x['correct_option'] for x in batch]

    # for gs (no token)
    demon_input_ids = [x['demon_input_ids'] for x in batch]
    demon_attention_mask = [x['demon_attention_mask'] for x in batch]
    gen_input_ids = [x['gen_input_ids'] for x in batch]
    gen_attention_mask = [x['gen_attention_mask'] for x in batch]
    gen_label_ids = [x['gen_label_ids'] for x in batch]
    demon_start_idxs = [x['demon_start_idxs'] for x in batch]

    # collate
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    demon_input_ids = pad_sequence_with_side(demon_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    demon_attention_mask = pad_sequence_with_side(demon_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_input_ids = pad_sequence_with_side(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_label_ids = pad_sequence_with_side(gen_label_ids, padding_value=-100, side=dataset.pad_side)
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    # extra padding
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        demon_input_ids, demon_attention_mask = debug_extra_pad_tensors(
            [demon_input_ids, demon_attention_mask],
            padding_values=[dataset.tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )
        gen_input_ids, gen_attention_mask, gen_label_ids = debug_extra_pad_tensors(
            [gen_input_ids, gen_attention_mask, gen_label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    return {
        'task': task,
        'test_idx': test_idx,
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'label_ids': all_label_ids,
        'input_ids_lens': input_ids_lens,
        "option": option,
        "correct_option": correct_option,
        # for gs (no ntoken)
        "demon_input_ids": demon_input_ids,
        "demon_attention_mask": demon_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "gen_label_ids": gen_label_ids,
        "demon_start_idxs": demon_start_idxs,
    }


def collate_fn_eval_dummy(batch: List[int], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.max_seq_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.max_seq_len] * batch_size

    # for gs (no ntoken)
    demon_len = dataset.max_seq_len * 4 // 5
    demon_input_ids = torch.randint(0, 30, (batch_size, demon_len), dtype=torch.int64, device='cpu')
    demon_attention_mask = torch.full(demon_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len // 5), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    demon_start_idxs = [x * (demon_len // 16) for x in range(16)]

    return {
        "task": ["dummy"] * batch_size,
        "test_idx": list(range(batch_size)),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "input_ids_lens": input_ids_lens,
        "option": [''] * batch_size,
        "correct_option": [''] * batch_size,
        # for gs (no ntoken)
        "demon_input_ids": demon_input_ids,
        "demon_attention_mask": demon_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "gen_label_ids": gen_input_ids,
        "demon_start_idxs": demon_start_idxs,
    }

########################################
# Gradient Search Dataset
########################################
class GSDataset(Dataset):
    def __init__(
        self,
        demonstration_pairs: Dict[int, Dict],
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        past_kv_len: int,
        max_seq_len: int,
        delimiter: str,
        loss_on_input: bool,
    ):
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.past_kv_len = past_kv_len
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        self.loss_on_input = loss_on_input

        # separate input and output by newline
        self.delimiter_token_id = tokenize(delimiter, tokenizer)

        # format data (only use demonstration pairs)
        parsed_examples = [self.format(i, example) for i, example in demonstration_pairs.items()]
        self.parsed_examples = [e for e in parsed_examples if e is not None]

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format(self, example_idx: int, pair: Dict) -> Optional[Dict]:
        # tokenize
        input_input_ids = tokenize(pair['input'], self.tokenizer)
        output_input_ids = tokenize(pair['output'], self.tokenizer)

        # append delimiter to input
        input_input_ids = torch.cat([input_input_ids, self.delimiter_token_id])

        # prepend \n\n for inputs that are not first
        input_input_ids = torch.cat([self.delimiter_token_id, input_input_ids])

        # create input_ids and label_ids
        input_ids = torch.cat([input_input_ids, output_input_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)

        if self.loss_on_input:
            label_ids = input_ids
        else:
            label_ids = torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype)
            label_ids = torch.cat([label_ids, output_input_ids])

        # GS dataset can overflow too because it is taking from eval dataset before filtering
        overflow = len(input_ids) - (self.max_seq_len - self.past_kv_len)
        if overflow > 0:
            return None

        return {
            "example_idx": example_idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
        }


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]

    assert isinstance(dataset.tokenizer.pad_token_id, int)
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
            [input_ids, attention_mask, label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "example_idx": [x['example_idx'] for x in batch],
    }
    return batch_dict


def collate_fn_gs_dummy(batch: List[Dict], dataset: GSDataset) -> Dict:
    batch_size = len(batch)
    del batch

    input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len // 5), dtype=torch.int64, device='cpu')
    attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64, device='cpu')

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "example_idx": [0] * batch_size,
    }
    return batch_dict


########################################
# Test-Time-Training Dataset
########################################
class TTTDataset(Dataset):
    def __init__(
        self,
        demonstration_pairs: List[Dict],
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        max_seq_len: int,
        delimiter: str,
        permute_n: int,
        seed: int,
        loss_type: str,
    ):
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        self.permute_n = permute_n
        self.seed = seed
        self.loss_type = loss_type

        # separate input and output by newline
        self.delimiter_token_id = tokenize(delimiter, tokenizer)

        # generate data
        self.data = self.unique_permutations(
            permute_n=permute_n,
            seed=seed
        )

    def unique_permutations(self, permute_n: int, seed: int):
        rng = np.random.RandomState(seed)
        seen = set()
        all_data = []

        for _ in range(100000):
            perm = tuple(rng.permutation(len(self.demonstration_pairs)))
            if perm in seen:
                continue

            seen.add(perm)
            pairs = [self.demonstration_pairs[i] for i in perm]
            data = self.format_and_filter(
                demonstrations=pairs[:-1],
                test_pair=pairs[-1],
            )
            if data is not None:
                all_data.append(data)

            if len(all_data) >= permute_n:
                break

        return all_data

    def format_and_filter(self, demonstrations: List[Dict], test_pair: Dict) -> Optional[Dict]:
        # make sure they are all the same task with the same non-empty options
        task = test_pair['task']
        assert all(e['task'] == task for e in demonstrations) # test and demonstration pair have same task (dont need same option)

        out = parse_pairs(
            pairs=demonstrations + [test_pair],
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            delimiter_token_id=self.delimiter_token_id,
            loss_type=self.loss_type,
            is_train=True,
        )

        if out == None:
            return None
        input_ids, attention_mask, label_ids, _, _, _, _, _, _ = out

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def collate_fn_ttt(batch: List[Dict], dataset: TTTDataset) -> Dict:
    input_ids, attention_mask, label_ids, _ = collate_data(
        batch=batch,
        tokenizer=dataset.tokenizer,
        debug_random_pad=dataset.debug_random_pad,
        debug_pad_len=dataset.debug_pad_len,
        pad_side=dataset.pad_side,
    )

    # dataset.tokenizer.decode(input_ids[2], skip_special_tokens=True)
    # attention_mask[2]
    # dataset.tokenizer.decode(label_ids[2][label_ids[2] != -100])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label_ids': label_ids,
    }


def collate_fn_ttt_dummy(batch: List[int], dataset: TTTDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.max_seq_len), 1, dtype=torch.int64, device='cpu')

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
    }
