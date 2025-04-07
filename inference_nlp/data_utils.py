import math
import os
import json
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, get_worker_info
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
    max_pair_len: int,
    max_seq_len: int,
    delimiter_token_id: torch.Tensor,
    loss_type: str,
    allow_truncate: bool,
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

        # truncate each pair like in metaICL, not account for newlines tho
        if len(input_input_ids) > max_pair_len - len(output_input_ids):
            input_input_ids = input_input_ids[: max_pair_len - len(output_input_ids)]

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
        if allow_truncate:
            inp, out = input_ids[:final_output_start_idx], input_ids[final_output_start_idx:]
            input_ids = torch.cat([inp[:max_seq_len - out.shape[0]], out])
            inp, out = label_ids[:final_output_start_idx], label_ids[final_output_start_idx:]
            label_ids = torch.cat([inp[:max_seq_len - out.shape[0]], out])
            attention_mask = torch.full(input_ids.shape, 1, dtype=input_ids.dtype) # attention mask
            assert input_ids.shape == attention_mask.shape == label_ids.shape
            assert input_ids.shape[0] == max_seq_len
        else:
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
# Training Dataset
########################################

class TrainDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        config_file: str,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        total_steps: int,
        seed: int,
        process_index: int,
        debug_fixed_order: bool,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        loss_type: str,
        num_workers: int,
        max_seq_len: int,
        max_pair_len: int,
        allow_truncate: bool,
        delimiter: str,
    ):

        self.tokenizer = tokenizer
        self._length = total_steps
        self.debug_fixed_order = debug_fixed_order
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.loss_type = loss_type
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.max_pair_len = max_pair_len
        self.allow_truncate = allow_truncate
        self.delimiter = delimiter

        self.num_workers = num_workers
        self.process_index = process_index
        self.seed = seed

        # separate input and output by newline
        self.delimiter_token_id = tokenize(delimiter, tokenizer)

        # set rngs
        self.set_rngs(epoch=0)

        # load data
        self.tasks = json.load(open(config_file))['train']
        self.task_to_pairs = defaultdict(list)
        for task in self.tasks:
            task_data_dir = os.path.join(data_dir, task)
            assert os.path.exists(os.path.join(task_data_dir, f"{task}_16384_100_train.jsonl"))
            all_lines = open(os.path.join(task_data_dir, f"{task}_16384_100_train.jsonl"), 'r').readlines()
            for l in all_lines:
                example = json.loads(l)
                assert len(example['input']) > 0 and len(example['output']) > 0 # non-empty input output
                assert example['task'] == task # correct task in file
                assert len(example['options']) != 1 # either no option, or 2+ options
                if len(example['options']) > 1:
                    assert example['output'] in example['options'] # example in option
                del example['task'], example['options']
                assert set(example.keys()) == {'input', 'output'}
                self.task_to_pairs[task].append(example)

        logger.info(f'number of train tasks: {len(self.task_to_pairs)}')
        for task, pairs in self.task_to_pairs.items():
            logger.info(f'{task} with {len(pairs)} pairs')

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # We'll do random sampling in the collate fn
        return 0

    def set_rngs(self, epoch: int):
        epoch_seed = epoch * 1000
        # seed and process_index
        if self.num_workers == 0:
            self.rngs = [np.random.RandomState(self.seed + epoch_seed + self.process_index)]
        else:
            self.rngs = [np.random.RandomState(self.seed + epoch_seed + i) for i in range(self.num_workers * self.process_index, self.num_workers * (self.process_index + 1))]


def collate_fn_train(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]

    # must sample this number of pairs to avoid GPU synchronization issues
    required_num_pair = 17 # hardcode this

    out_batch = []
    while len(out_batch) < batch_size:
        task = rng.choice(list(dataset.task_to_pairs)) # type: ignore
        all_pairs = dataset.task_to_pairs[task]

        if dataset.debug_fixed_order:
            required_num_pair = len(all_pairs)
            chosen_pairs = all_pairs[:required_num_pair]
        else:
            chosen_pairs = rng.choice(all_pairs, size=required_num_pair, replace=False).tolist()

        out = parse_pairs(
            pairs=chosen_pairs,
            tokenizer=dataset.tokenizer,
            max_pair_len=dataset.max_pair_len,
            max_seq_len=dataset.max_seq_len,
            delimiter_token_id=dataset.delimiter_token_id,
            loss_type=dataset.loss_type,
            allow_truncate=dataset.allow_truncate,
            is_train=True,
        )
        if out == None:
            continue
        input_ids, attention_mask, label_ids, _, _, _, _, _, _ = out

        # add to batch
        out_batch.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids,
        })

    input_ids, attention_mask, label_ids, input_ids_lens = collate_data(
        batch=out_batch,
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
        'input_ids_lens': input_ids_lens,
    }


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.max_seq_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.max_seq_len] * batch_size

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "input_ids_lens": input_ids_lens,
    }


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        data_dir: str,
        config_file: str,
        seed: int,
        eval_seed: str,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        debug_random_pad: bool,
        debug_pad_len: int,
        debug_max_len: bool,
        pad_side: str,
        max_seq_len: int,
        max_pair_len: int,
        eval_test_per_task: int,
        eval_ratio: float,
        split: str,
        allow_truncate: bool,
        delimiter: str,
    ):
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.max_pair_len = max_pair_len
        self.allow_truncate = allow_truncate
        self.delimiter = delimiter
        self.debug_max_len = debug_max_len
        self.seed = seed

        # needed in evaluate
        self.split = split

        # separate input and output by newline
        self.delimiter_token_id = tokenize(delimiter, tokenizer)

        # rng
        rng = np.random.RandomState(seed)

        # get tasks
        self.tasks = json.load(open(config_file))[split]
        assert len(self.tasks) == len(set(self.tasks))

        # load data
        tasks_to_remove = set()
        task_to_demonstrations = defaultdict(list)
        task_to_test_pairs = defaultdict(list)
        for task in self.tasks:
            task_data_dir = os.path.join(data_dir, task)

            # train demonstration pairs
            train_file = os.path.join(task_data_dir, f"{task}_16_{eval_seed}_train.jsonl")
            for l in open(train_file, 'r').readlines():
                example = json.loads(l)
                task_to_demonstrations[task].append(example)
            assert len(task_to_demonstrations[task]) == 16

            # subset test pairs
            with_empty_options = False
            test_file = os.path.join(task_data_dir, f"{task}_16_{eval_seed}_test.jsonl")
            lines = open(test_file, 'r').readlines()
            rng.shuffle(lines)
            num_chosen = math.ceil(len(lines) * eval_ratio)
            for l in lines[:num_chosen]:
                example = json.loads(l)
                task_to_test_pairs[task].append(example)
                with_empty_options = with_empty_options or (len(example['options'])) == 0

            # keep track of tasks that have empty options, we filter them out
            if with_empty_options:
                tasks_to_remove.add(task)

        if split == 'train' and not config_file.endswith('toy.json'):
            # remove tasks with 10+ options
            for task, test_pairs in task_to_test_pairs.items():
                max_num_options = max(len(x['options']) for x in test_pairs)
                if max_num_options >= 10:
                    tasks_to_remove.add(task)
            # manually remove these
            tasks_to_remove.update({'piqa', 'cosmos_qa', 'art', 'paws', 'quail'})

        # filter out tasks with empty options
        if split == 'test':
            assert len(tasks_to_remove) == 0
        else:
            logger.info(f'evaluating on train split found {len(tasks_to_remove)}/{len(self.tasks)} tasks to remove')
            for task in tasks_to_remove:
                self.tasks.remove(task)
                del task_to_demonstrations[task]
                del task_to_test_pairs[task]
        assert set(task_to_demonstrations.keys()) == set(task_to_test_pairs.keys()) == set(self.tasks)
        total_unfiltered_tasks = len(self.tasks)

        # process and filter down
        logger.info(f'generating samples for split {split}-{eval_seed}...')
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
        logger.info(f'eval split {split}-{eval_seed} filtered to {len(self.tasks)}/{total_unfiltered_tasks} tasks, {filtered_total_test}/{unfiltered_total_test} tests, {len(self.data)}/{unfiltered_total_sample} samples')

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
            max_pair_len=self.max_pair_len,
            max_seq_len=self.max_seq_len,
            delimiter_token_id=self.delimiter_token_id,
            loss_type="only_last",
            allow_truncate=self.allow_truncate,
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
    demon_len = dataset.max_seq_len - dataset.max_pair_len
    demon_input_ids = torch.randint(0, 30, (batch_size, demon_len), dtype=torch.int64, device='cpu')
    demon_attention_mask = torch.full(demon_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.max_pair_len), dtype=torch.int64, device='cpu')
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
        demonstration_pairs: List[Dict],
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        past_kv_len: int,
        max_seq_len: int,
        max_pair_len: int,
        allow_truncate: bool,
        delimiter: str,
    ):
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.past_kv_len = past_kv_len
        self.max_seq_len = max_seq_len
        self.max_pair_len = max_pair_len
        self.allow_truncate = allow_truncate
        self.delimiter = delimiter

        # separate input and output by newline
        self.delimiter_token_id = tokenize(delimiter, tokenizer)

        # format data (only use demonstration pairs)
        parsed_examples = [self.format(example) for example in demonstration_pairs]
        self.parsed_examples = [e for e in parsed_examples if e is not None]

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format(self, pair: Dict) -> Optional[Dict]:
        # tokenize
        input_input_ids = tokenize(pair['input'], self.tokenizer)
        output_input_ids = tokenize(pair['output'], self.tokenizer)

        # truncate each pair like in metaICL, not account for newlines tho
        if len(input_input_ids) > self.max_pair_len - len(output_input_ids):
            input_input_ids = input_input_ids[: self.max_pair_len - len(output_input_ids)]

        # append delimiter to input
        input_input_ids = torch.cat([input_input_ids, self.delimiter_token_id])

        # prepend \n\n for inputs that are not first
        input_input_ids = torch.cat([self.delimiter_token_id, input_input_ids])

        # create input_ids and label_ids
        input_ids = torch.cat([input_input_ids, output_input_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
        label_ids = torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype)
        label_ids = torch.cat([label_ids, output_input_ids])

        # GS dataset can overflow too because it is taking from eval dataset before filtering
        overflow = len(input_ids) - (self.max_seq_len - self.past_kv_len)
        if overflow > 0:
            if self.allow_truncate and overflow < len(input_input_ids):
                input_ids = torch.cat([input_input_ids[:-overflow], output_input_ids])
                attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
                label_ids = torch.full(input_input_ids[:-overflow].shape, -100, dtype=input_ids.dtype)
                label_ids = torch.cat([label_ids, output_input_ids])
                assert input_ids.shape == attention_mask.shape == label_ids.shape
                assert input_ids.shape[0] == self.max_seq_len
            else:
                return None

        return {
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
    }
    return batch_dict


def collate_fn_gs_dummy(batch: List[Dict], dataset: GSDataset) -> Dict:
    batch_size = len(batch)
    del batch

    input_ids = torch.randint(0, 30, (batch_size, dataset.max_pair_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64, device='cpu')

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
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
        max_pair_len: int,
        allow_truncate: bool,
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
        self.max_pair_len = max_pair_len
        self.allow_truncate = allow_truncate
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

        for _ in range(1000):
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
            max_pair_len=self.max_pair_len,
            max_seq_len=self.max_seq_len,
            delimiter_token_id=self.delimiter_token_id,
            loss_type=self.loss_type,
            allow_truncate=self.allow_truncate,
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
