import math
from tasks import TASKS
import json
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import GPT2TokenizerFast, Qwen2TokenizerFast
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, List, Optional, Union
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

    tokenizer_out = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)

    assert tokenizer_out['input_ids'].shape == tokenizer_out['attention_mask'].shape # type: ignore
    assert tokenizer_out['input_ids'].dim() == 2 and tokenizer_out['input_ids'].shape[0] == 1 # type: ignore
    assert tokenizer_out['attention_mask'].numel() == tokenizer_out['attention_mask'].sum() # type: ignore

    input_ids = tokenizer_out['input_ids'][0] # type: ignore
    if not isinstance(tokenizer, GPT2TokenizerFast) and not isinstance(tokenizer, Qwen2TokenizerFast):
        assert input_ids[0].item() == tokenizer.bos_token_id # type: ignore
        input_ids = input_ids[1:]

    return input_ids


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        data_dir: str,
        seed: int,
        tokenizer: PreTrainedTokenizerFast,
        debug_random_pad: bool,
        debug_pad_len: int,
        debug_max_len: bool,
        pad_side: str,
        max_seq_len: int,
        num_demonstrations: int,
        eval_ratio: float,
        eval_on_demonstrations: bool,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.debug_max_len = debug_max_len
        self.seed = seed
        self.num_demonstrations = num_demonstrations
        self.eval_ratio = eval_ratio
        self.eval_on_demonstrations = eval_on_demonstrations

        # data stuff
        self.task_to_demonstrations = {}
        task_to_test_pairs = {}

        for task_name in TASKS:
            # load data
            task_file = f"{data_dir}/{task_name}.json"
            with open(task_file, "r") as f:
                task_json = json.load(f)
            all_examples = task_json.get("examples", [])
            assert len(all_examples) > num_demonstrations

            # split to few-shot train and rest for eval
            random.seed(seed)
            random.shuffle(all_examples)
            self.task_to_demonstrations[task_name] = all_examples[:num_demonstrations]

            # subset test pairs
            test_pairs = all_examples[num_demonstrations:]
            test_pairs = test_pairs[:math.ceil(len(test_pairs) * eval_ratio)]
            if eval_on_demonstrations:
                task_to_test_pairs[task_name] = all_examples[:num_demonstrations]
            else:
                task_to_test_pairs[task_name] = test_pairs

        # parse and filter data
        self.data = []
        for task_i, (task, test_pairs) in enumerate(task_to_test_pairs.items()):
            logger.info(f'{task_i+1}/{len(task_to_test_pairs)}')
            for test_idx, test_pair in enumerate(test_pairs):
                out = self.format_and_filter(
                    task=task,
                    test_idx=test_idx,
                    demonstrations=self.task_to_demonstrations[task],
                    test_pair=test_pair,
                )
                if out is not None:
                    self.data.append(out)

        logger.info(f'filtered to {len(self.data)}/{sum(len(v) for v in task_to_test_pairs.values())} data')

        # info: max length
        max_seq_len = max(len(d['demon_input_ids']) + len(d['gen_input_ids'] + d['generation_length']) for d in self.data)
        assert max_seq_len <= self.max_seq_len
        self.max_seq_len = max_seq_len
        logger.info(f'max seq len {self.max_seq_len}')

        # # info: number of tests per task
        # for task in TASKS:
        #     sub = [d for d in self.data if d['task'] == task]
        #     print(f'Task {task} has {len(sub)} examples')

        # debug: no task filtered
        # assert set(d['task'] for d in self.data) == set(TASKS.keys())

        # debug: check demon pairs are same for each task
        task_to_demon_input_ids = {}
        task_to_demon_start_idxs = {}
        for d in self.data:
            task, ids, idxs = d['task'], d['demon_input_ids'], d['demon_start_idxs']
            if task not in task_to_demon_input_ids:
                assert task not in task_to_demon_start_idxs
                task_to_demon_input_ids[task] = ids
                task_to_demon_start_idxs[task] = idxs
            assert torch.equal(ids, task_to_demon_input_ids[task])
            assert idxs == task_to_demon_start_idxs[task]
        del task_to_demon_input_ids
        del task_to_demon_start_idxs


    def format_and_filter(self, task: str, test_idx: int, demonstrations: List[Dict], test_pair: Dict) -> Optional[Dict]:
        info = TASKS[task]
        instruction_ids = tokenize(f"{info['task_prompt']} {info['answer_format']}", self.tokenizer)

        # for gs (no ntoken)
        pair_input_ids = []
        demon_start_idxs = [len(instruction_ids)]

        for pair_i, pair in enumerate(demonstrations + [test_pair]):
            # tokenize
            input_input_ids = tokenize('\n\nQ: ' + pair['input'] + '\nA: ', self.tokenizer)
            output_input_ids = tokenize(pair['target'], self.tokenizer)

            pair_input_ids.append(input_input_ids)
            pair_input_ids.append(output_input_ids)

            # dont need last demon pair
            if pair_i < len(demonstrations) - 1:
                demon_start_idxs.append(demon_start_idxs[-1] + len(input_input_ids) + len(output_input_ids))

        # gen input ids
        gen_input_ids = pair_input_ids[-2]
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64)
        # demon input ids
        demon_input_ids = torch.cat([instruction_ids, *pair_input_ids[:-2]])

        if len(demon_input_ids) + len(gen_input_ids) + info["generation_length"] > self.max_seq_len:
            return None

        return {
            "task": task,
            "test_idx": test_idx,
            "label": test_pair['target'],
            "demon_input_ids": demon_input_ids,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "demon_start_idxs": demon_start_idxs,
            "generation_length": info["generation_length"],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    # eval only
    task = [x['task'] for x in batch]
    test_idx = [x['test_idx'] for x in batch]
    label = [x['label'] for x in batch]
    generation_length = [x['generation_length'] for x in batch]

    # for gs (no token)
    demon_input_ids = [x['demon_input_ids'] for x in batch]
    gen_input_ids = [x['gen_input_ids'] for x in batch]
    gen_attention_mask = [x['gen_attention_mask'] for x in batch]
    demon_start_idxs = [x['demon_start_idxs'] for x in batch]

    # collate
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    demon_input_ids = pad_sequence_with_side(demon_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    gen_input_ids = pad_sequence_with_side(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.pad_side)

    # extra padding
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        demon_input_ids = debug_extra_pad_tensors(
            [demon_input_ids],
            padding_values=[dataset.tokenizer.pad_token_id],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )
        gen_input_ids, gen_attention_mask = debug_extra_pad_tensors(
            [gen_input_ids, gen_attention_mask],
            padding_values=[dataset.tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    return {
        'task': task,
        'test_idx': test_idx,
        "label": label,
        "demon_input_ids": demon_input_ids,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "demon_start_idxs": demon_start_idxs,
        "generation_length": generation_length,
    }


def collate_fn_eval_dummy(batch: List[int], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    # for gs (no ntoken)
    demon_len = dataset.max_seq_len * 8 // 10
    demon_input_ids = torch.randint(0, 30, (batch_size, demon_len), dtype=torch.int64, device='cpu')
    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len // 10), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    demon_start_idxs = [x * (demon_len // 10) for x in range(10)]

    return {
        "task": ["dummy"] * batch_size,
        "test_idx": list(range(batch_size)),
        "label": ["dummy"] * batch_size,
        "demon_input_ids": demon_input_ids,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "demon_start_idxs": demon_start_idxs,
        "generation_length": [dataset.max_seq_len // 10] * batch_size,
    }


########################################
# Gradient Search Dataset
########################################
class GSDataset(Dataset):
    def __init__(
        self,
        demonstration_pairs: Dict[int, Dict],
        tokenizer: PreTrainedTokenizerFast,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        max_seq_len: int,
        loss_on_input: bool,
    ):
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.loss_on_input = loss_on_input

        # format data (only use demonstration pairs)
        self.parsed_examples = [self.format(i, example) for i, example in demonstration_pairs.items()]

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format(self, example_idx: int, pair: Dict) -> Dict:
        # tokenize
        input_input_ids = tokenize('\n\nQ: ' + pair['input'] + '\nA: ', self.tokenizer)
        output_input_ids = tokenize(pair['target'], self.tokenizer)

        # create input_ids and label_ids
        input_ids = torch.cat([input_input_ids, output_input_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)

        if self.loss_on_input:
            label_ids = input_ids
        else:
            label_ids = torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype)
            label_ids = torch.cat([label_ids, output_input_ids])

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

    input_ids = torch.randint(0, 30, (batch_size, dataset.max_seq_len * 2 // 10), dtype=torch.int64, device='cpu')
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
        task: str,
        demonstration_pairs: List[Dict],
        tokenizer: PreTrainedTokenizerFast,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        max_seq_len: int,
        permute_n: int,
        seed: int,
        loss_type: str,
    ):
        self.task = task
        self.demonstration_pairs = demonstration_pairs
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.permute_n = permute_n
        self.seed = seed
        self.loss_type = loss_type

        # generate data
        self.data = self.unique_permutations(
            permute_n=permute_n,
            seed=seed
        )

    def unique_permutations(self, permute_n: int, seed: int):
        rng = np.random.RandomState(seed)
        all_data = []

        for _ in range(1000):
            perm = tuple(rng.permutation(len(self.demonstration_pairs)))
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
        info = TASKS[self.task]
        instruction_ids = tokenize(f"{info['task_prompt']} {info['answer_format']}", self.tokenizer)

        # compute input_ids, attention_mask, label_ids for each pair
        input_ids_of_each_pair = []
        label_ids_of_each_pair = []

        for pair_i, pair in enumerate(demonstrations + [test_pair]):
            # tokenize
            input_input_ids = tokenize('\n\nQ: ' + pair['input'] + '\nA: ', self.tokenizer)
            output_input_ids = tokenize(pair['target'], self.tokenizer)

            # create input_ids and label_ids
            input_ids = torch.cat([input_input_ids, output_input_ids])
            if (pair_i == 0 and self.loss_type == 'exclude_first') or (pair_i < len(demonstrations) and self.loss_type == 'only_last'):
                label_ids = torch.full(input_ids.shape, -100, dtype=input_ids.dtype)
            else:
                label_ids = torch.cat([
                    torch.full(input_input_ids.shape, -100, dtype=input_ids.dtype),
                    output_input_ids,
                ])

            input_ids_of_each_pair.append(input_ids)
            label_ids_of_each_pair.append(label_ids)

        # concat
        input_ids = torch.cat([instruction_ids, *input_ids_of_each_pair])
        attention_mask = torch.full(input_ids.shape, 1, dtype=input_ids.dtype)
        label_ids = torch.cat([
            torch.full(instruction_ids.shape, -100, dtype=input_ids.dtype),
            *label_ids_of_each_pair,
        ])
        assert input_ids.shape == attention_mask.shape == label_ids.shape

        if len(input_ids) > self.max_seq_len:
            return None

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
    batch_size = len(batch)
    all_input_ids = [x['input_ids'] for x in batch]
    all_attention_mask = [x['attention_mask'] for x in batch]
    all_label_ids = [x['label_ids'] for x in batch]

    # collate
    assert isinstance(dataset.tokenizer.pad_token_id, int)
    all_input_ids = pad_sequence_with_side(all_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    all_attention_mask = pad_sequence_with_side(all_attention_mask, padding_value=0, side=dataset.pad_side)
    all_label_ids = pad_sequence_with_side(all_label_ids, padding_value=-100, side=dataset.pad_side)
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    # extra padding
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        all_input_ids, all_attention_mask, all_label_ids = debug_extra_pad_tensors(
            [all_input_ids, all_attention_mask, all_label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    assert len(all_input_ids) == batch_size
    assert all_input_ids.shape == all_attention_mask.shape == all_label_ids.shape

    # dataset.tokenizer.decode(input_ids[2], skip_special_tokens=True)
    # attention_mask[2]
    # dataset.tokenizer.decode(label_ids[2][label_ids[2] != -100])

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'label_ids': all_label_ids,
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
