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
    newline_token_id: torch.Tensor,
    loss_type: str,
    allow_truncate: bool,
    is_train: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray,
                    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]:

    # compute input_ids, attention_mask, label_ids for each pair
    input_ids_of_each_pair = []
    label_ids_of_each_pair = []
    final_output_start_idx = -1

    # for gs (no ntoken)
    demon_input_ids = []

    for pair_i, pair in enumerate(pairs):
        # tokenize
        input_input_ids = tokenize(pair['input'], tokenizer)
        output_input_ids = tokenize(pair['output'], tokenizer)

        # truncate each pair like in metaICL, not account for newlines tho
        if len(input_input_ids) > max_pair_len - len(output_input_ids):
            input_input_ids = input_input_ids[: max_pair_len - len(output_input_ids)]

        # append delimiter to input
        input_input_ids = torch.cat([input_input_ids, newline_token_id])

        # prepend \n\n for inputs that are not first
        if pair_i != 0:
            input_input_ids = torch.cat([newline_token_id, newline_token_id, input_input_ids])

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

    # start idxs computed from cumsum of lengths of pairs except last
    pair_start_idxs = np.cumsum([0, *[len(x) for x in input_ids_of_each_pair[:-1]]]).tolist()

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
        gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask = None, None, None, None, None

    return input_ids, attention_mask, label_ids, pair_start_idxs, \
        gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask


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
        min_num_pair: int,
        max_num_pair: int,
        loss_type: str,
        debug_len: int,
        num_workers: int,
        max_seq_len: int,
        max_pair_len: int,
        allow_truncate: bool,
    ):

        self.tokenizer = tokenizer
        self._length = total_steps
        self.debug_fixed_order = debug_fixed_order
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        self.loss_type = loss_type
        self.debug_len = debug_len
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.max_pair_len = max_pair_len
        self.allow_truncate = allow_truncate

        self.num_workers = num_workers
        self.process_index = process_index
        self.seed = seed

        # separate input and output by newline
        self.newline_token_id = tokenize("\n", tokenizer)

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
        assert all(len(pairs) >= self.max_num_pair for pairs in self.task_to_pairs.values())

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
        # num pair must be the same across gpus
        if self.num_workers == 0:
            self.num_pair_rngs = [np.random.RandomState(self.seed + epoch_seed)]
        else:
            self.num_pair_rngs = [np.random.RandomState(self.seed + epoch_seed + i) for i in range(self.num_workers)]


def collate_fn_train(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]
    num_pair_rng = dataset.num_pair_rngs[int(worker_id)]

    # must sample this number of pairs to avoid GPU synchronization issues
    required_num_pair = num_pair_rng.choice(list(range(dataset.min_num_pair, dataset.max_num_pair + 1)))

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
            newline_token_id=dataset.newline_token_id,
            loss_type=dataset.loss_type,
            allow_truncate=dataset.allow_truncate,
            is_train=True,
        )
        if out == None:
            continue
        input_ids, attention_mask, label_ids, pair_start_idxs, _, _, _, _, _ = out

        # add to batch
        out_batch.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_ids': label_ids,
            'pair_start_idxs': pair_start_idxs,
        })

    input_ids, attention_mask, label_ids, input_ids_lens = collate_data(
        batch=out_batch,
        tokenizer=dataset.tokenizer,
        debug_random_pad=dataset.debug_random_pad,
        debug_pad_len=dataset.debug_pad_len,
        pad_side=dataset.pad_side,
    )
    pair_start_idxs = [x['pair_start_idxs'] for x in out_batch]
    assert all(start_idxs[0] == 0 for start_idxs in pair_start_idxs)

    # dataset.tokenizer.decode(input_ids[2], skip_special_tokens=True)
    # attention_mask[2]
    # dataset.tokenizer.decode(label_ids[2][label_ids[2] != -100])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label_ids': label_ids,
        'input_ids_lens': input_ids_lens,
        'pair_start_idxs': pair_start_idxs,
    }


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.debug_len] * batch_size
    pair_start_idxs = [[0] + sorted(random.sample(range(2, dataset.debug_len), k=dataset.max_num_pair-1)) for _ in range(batch_size)]

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
        data_dir: str,
        config_file: str,
        seed: int,
        eval_seed: int,
        tokenizer: Union[PreTrainedTokenizerFast, GPT2TokenizerFast],
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        debug_len: int,
        max_seq_len: int,
        max_pair_len: int,
        min_num_train_pair: int,
        max_num_train_pair: int,
        ntokens: int,
        eval_test_per_task: int,
        eval_ratio: float,
        split: str,
        allow_truncate: bool,
        debug_fixed_order: bool,
    ):
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.debug_len = debug_len
        self.max_seq_len = max_seq_len
        self.max_pair_len = max_pair_len
        self.allow_truncate = allow_truncate
        self.max_num_train_pair = max_num_train_pair # needed by collate
        self.debug_fixed_order = debug_fixed_order

        # needed in evaluate
        self.ntokens = ntokens # needed by eval
        self.split = split

        # separate input and output by newline
        self.newline_token_id = tokenize("\n", tokenizer)

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
            assert len(task_to_demonstrations[task]) == 16 >= max_num_train_pair

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
                # get random demonstration pairs
                num_demonstration = int(rng.choice(range(min_num_train_pair, max_num_train_pair + 1), size=1))
                if debug_fixed_order:
                    demonstrations = task_to_demonstrations[task][:num_demonstration]
                else:
                    demonstrations = rng.choice(task_to_demonstrations[task], size=num_demonstration).tolist()

                assert len(test_pair['options']) > 1
                assert test_pair['output'] in test_pair['options']
                correct_option = test_pair['output']

                # get outputs for each option
                outs = []
                for option in test_pair['options']:
                    test_pair['output'] = option
                    outs.append(self.format_and_filter(demonstrations, test_pair, test_idx, correct_option))

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
        logger.info(f'eval split {split}-{eval_seed} filtered to {len(self.tasks)}/{total_unfiltered_tasks} tasks, {filtered_total_test}/{unfiltered_total_test} tests, {len(self.data)}/{unfiltered_total_sample} samples')

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
            newline_token_id=self.newline_token_id,
            loss_type="only_last",
            allow_truncate=self.allow_truncate,
            is_train=False,
        )
        if out == None:
            return None
        input_ids, attention_mask, label_ids, pair_start_idxs, gen_input_ids, gen_attention_mask, gen_label_ids, demon_input_ids, demon_attention_mask = out

        return {
            "task": task,
            "test_idx": test_idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "pair_start_idxs": pair_start_idxs,
            "option": test_pair['output'],
            "correct_option": correct_option,
            # for gs (no ntoken)
            "demon_input_ids": demon_input_ids,
            "demon_attention_mask": demon_attention_mask,
            "gen_input_ids": gen_input_ids,
            "gen_label_ids": gen_label_ids,
            "gen_attention_mask": gen_attention_mask,
            "demonstrations_pairs": demonstrations,
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
    pair_start_idxs = [x['pair_start_idxs'] for x in batch]
    task = [x['task'] for x in batch]
    test_idx = [x['test_idx'] for x in batch]
    option = [x['option'] for x in batch]
    correct_option = [x['correct_option'] for x in batch]
    assert all(start_idxs[0] == 0 for start_idxs in pair_start_idxs)

    # for gs (no token)
    demon_input_ids = [x['demon_input_ids'] for x in batch]
    demon_attention_mask = [x['demon_attention_mask'] for x in batch]
    gen_input_ids = [x['gen_input_ids'] for x in batch]
    gen_attention_mask = [x['gen_attention_mask'] for x in batch]
    gen_label_ids = [x['gen_label_ids'] for x in batch]

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
        'pair_start_idxs': pair_start_idxs,
        "option": option,
        "correct_option": correct_option,
        # for gs (no ntoken)
        "demon_input_ids": demon_input_ids,
        "demon_attention_mask": demon_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "gen_label_ids": gen_label_ids,
    }


def collate_fn_eval_dummy(batch: List[int], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.debug_len] * batch_size
    pair_start_idxs = [[0] + sorted(random.sample(range(2, dataset.debug_len), k=dataset.max_num_train_pair)) for _ in range(batch_size)]

    # for gs (no ntoken)
    demon_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len * 8 // 10), dtype=torch.int64, device='cpu')
    demon_attention_mask = torch.full(demon_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len * 2 // 10), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64, device='cpu')

    return {
        "task": ["dummy"] * batch_size,
        "test_idx": list(range(batch_size)),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "input_ids_lens": input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
        "option": [''] * batch_size,
        "correct_option": [''] * batch_size,
        # for gs (no ntoken)
        "demon_input_ids": demon_input_ids,
        "demon_attention_mask": demon_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "gen_label_ids": gen_input_ids,
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
        train_pad_side: str,
        max_pair_len: int,
    ):
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.train_pad_side = train_pad_side
        self.max_pair_len = max_pair_len

        # separate input and output by newline
        self.newline_token_id = tokenize("\n", tokenizer)

        # format data (only use demonstration pairs)
        self.parsed_examples = [self.format(example) for example in demonstration_pairs]

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
        input_input_ids = torch.cat([input_input_ids, self.newline_token_id])

        # prepend \n\n for inputs that are not first
        input_input_ids = torch.cat([self.newline_token_id, self.newline_token_id, input_input_ids])

        # create input_ids and label_ids
        input_ids = torch.cat([input_input_ids, output_input_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
        label_ids = torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype)
        label_ids = torch.cat([label_ids, output_input_ids])

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
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.train_pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.train_pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.train_pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
            [input_ids, attention_mask, label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.train_pad_side,
        )

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
    }
    return batch_dict