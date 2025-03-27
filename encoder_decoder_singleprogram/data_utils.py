import hashlib
import ast
from collections import Counter
import itertools
import csv
import math
import copy
import os
import json
import random
import torch
from torch.utils.data import Dataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List, Tuple, Iterator, Union
from pathlib import Path
from typing import List, Optional
import numpy as np

from arclib.augmenters import (
    inverse,
    PermuteExamples,
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
    PermuteColors,
)
from accelerate.logging import get_logger

from arclib.arc import Task, Example

logger = get_logger(__name__, log_level="INFO")


class ARCTokenizer:
    def __init__(self, tokens: List[str], bos_token: str, eos_token: str):
        assert eos_token in tokens
        assert bos_token in tokens
        assert len(set(tokens)) == len(tokens)

        # mapping
        self.id_to_token = {i: token for i, token in enumerate(tokens)}
        self.token_to_id = {token: i for i, token in enumerate(tokens)}

        # special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bos_token_id = self.token_to_id[bos_token]
        self.eos_token_id = self.token_to_id[eos_token]
        self.special_token_ids = set([self.bos_token_id, self.eos_token_id])

        # set pad as eos
        self.pad_token = self.eos_token
        self.pad_token_id = self.eos_token_id

    def __call__(self, text: str) -> Dict[str, List[int]]:
        input_ids = [self.bos_token_id]
        while text:
            # special tokens
            if text.startswith(self.eos_token):
                input_ids.append(self.eos_token_id)
                text = text[len(self.eos_token):]
            elif text.startswith(self.bos_token):
                input_ids.append(self.bos_token_id)
                text = text[len(self.bos_token):]
            # input output
            elif text.startswith("input"):
                input_ids.append(self.token_to_id["input"])
                text = text[5:]
            elif text.startswith("output"):
                input_ids.append(self.token_to_id["output"])
                text = text[6:]
            elif text.startswith(":\n"):
                input_ids.append(self.token_to_id[":\n"])
                text = text[len(":\n"):]
            elif text.startswith("<CLS"):
                cls_token = text[:text.index(">") + 1]
                input_ids.append(self.token_to_id[cls_token])
                text = text[len(cls_token):]
            elif text[0] in self.token_to_id:
                input_ids.append(self.token_to_id[text[0]])
                text = text[1:]
            else:
                raise ValueError(f"cannot tokenize: {text}")

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def encode_to_tensor(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self(text)
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.int64)[None, ...],
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.int64)[None, ...],
        }

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False) -> str:
        tokens = []
        for token_id in token_ids:
            if not skip_special_tokens or (int(token_id) not in self.special_token_ids):
                tokens.append(self.id_to_token[int(token_id)])
        return "".join(tokens)

    def batch_decode(self, batch_token_ids: torch.Tensor, skip_special_tokens: bool) -> List[str]:
        assert batch_token_ids.dim() == 2
        texts = []
        for token_ids in batch_token_ids:
            text = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            texts.append(text)
        return texts


def get_extra_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply: List = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0), # type: ignore
            Concat((IdentityAugmenter(), Rotate(180)), axis=1), # type: ignore
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]), # type: ignore
            Chain([Rotate(270), IncreaseResolution(2)]), # type: ignore
            Chain([Rotate(180), IncreaseResolution(2)]), # type: ignore
            Chain([Flip(0), IncreaseResolution(2)]), # type: ignore
            Chain([Flip(1), IncreaseResolution(2)]), # type: ignore
            Chain([Transpose(), IncreaseResolution(2)]), # type: ignore
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )
    return augmenters_to_apply


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
        side: str
    ) -> List[torch.Tensor]:
    assert len(tensors) == len(padding_values)
    assert all(t.dim() == 2 for t in tensors)
    if pad_len == -1:
        pad_len = random.randint(0, 15) # arbitrary
    padded_tensors = []
    for arg, padding_value in zip(tensors, padding_values):
        pad = torch.full((arg.shape[0], pad_len), padding_value, device=arg.device, dtype=arg.dtype)
        if side == 'right':
            padded_tensor = torch.cat([arg, pad], dim=-1)
        else:
            padded_tensor = torch.cat([pad, arg], dim=-1)
        padded_tensors.append(padded_tensor)
    return padded_tensors


def load_re_arc_from_data_dir(data_dir: str) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Training data directory '{data_dir}' not found.")
    tasks_dict = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            task_id = filename.replace(".json", "")
            path = os.path.join(data_dir, filename)
            with open(path, "r") as f:
                data = json.load(f)
                tasks_dict[task_id] = data
    return tasks_dict


def grid_to_text(grid: List[List[int]], is_input: bool) -> str:
    label = "input:" if is_input else "output:"
    height = len(grid)
    assert height > 0, f"Grid height cannot be 0, got {grid}"
    width = len(grid[0])
    lines = [label, f"{height}", f"{width}"]
    for row in grid:
        assert len(row) == width, f"Inconsistent row width in grid: expected {width}, got {len(row)}."
        row_str = "".join(str(x) for x in row)
        lines.append(row_str)
    return "\n".join(lines)


def plot_histogram_with_frequencies(data: List[int], save_path: str, bins: int = 20) -> None:
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    # Create histogram
    counts, bins, bars = ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)  # type: ignore
    # Add frequency labels to each bar
    for bar, count in zip(bars, counts): # type: ignore
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(count)}',
                ha='center', va='bottom', fontsize=10)
    # Set x-ticks if bins are evenly spaced integers
    if np.allclose(np.diff(bins), bins[1] - bins[0]):  # Check if bins are evenly spaced  # type: ignore
        ax.set_xticks(bins[:-1] + np.diff(bins) / 2) # type: ignore
    ax.tick_params(axis='x', labelsize=5)
    plt.savefig(save_path, dpi=200)
    plt.close()


########################################
# Test-Time-Training Dataset
########################################
class TTTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_samples_per_task: int,
        permute_n: int,
        encoder_tokenizer: ARCTokenizer,
        decoder_tokenizer: ARCTokenizer,
        max_seq_len: int,
        seed: int,
        ntokens: int,
        encoder_pad_side: str,
        decoder_pad_side: str,
        encoder_loss_type: bool,
        debug_no_aug: bool,
    ):
        self.permute_n = permute_n
        # for ttt, only use mit augmenters for now
        self.augmenters: List[Augmenter] = get_extra_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)
        self.seed = seed
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_seq_len = max_seq_len
        self.ntokens = ntokens
        self.cls_tokens = [f"<CLS{token_i}>" for token_i in range(ntokens)]
        self.encoder_pad_side = encoder_pad_side
        self.decoder_pad_side = decoder_pad_side
        self.encoder_loss_type = encoder_loss_type
        self.debug_no_aug = debug_no_aug

        self.encoder_input_token_id = encoder_tokenizer("input")['input_ids'][1]
        self.encoder_output_token_id = encoder_tokenizer("output")['input_ids'][1]

        # load data
        self.task_id = Path(data_path).stem
        with open(data_path, "r") as f:
            task_data = json.load(f)

        # create task
        train_examples = [Example(input=np.array(x["input"]), output=np.array(x["output"])) for x in task_data['train']]
        self.task = Task(
            name=f'{self.task_id}',
            train_examples=train_examples,
            test_example=None, # type: ignore
        )

        # get data
        rng = np.random.RandomState(seed)
        self.ttt_tasks: List[Any] = self.task_to_ttt_filtered_data(max_gen=max_samples_per_task)
        rng.shuffle(self.ttt_tasks)

    def task_to_ttt_filtered_data(self, max_gen: int) -> List[Task]:
        # if leave 1 is enough, return it
        leave_1_train_tasks = self.task_to_ttt_filtered_data_leave_n(leave_n=1, max_gen=max_gen)
        if len(leave_1_train_tasks) >= max_gen:
            return leave_1_train_tasks
        # else generate leave 2 and append to leave 1
        max_gen_leave_2 = max_gen - len(leave_1_train_tasks)
        leave_1_train_tasks += self.task_to_ttt_filtered_data_leave_n(leave_n=2, max_gen=max_gen_leave_2)
        return leave_1_train_tasks

    def task_to_ttt_filtered_data_leave_n(self, leave_n: int, max_gen: int) -> List[Task]:
        rng = np.random.RandomState(self.seed)

        # get leave_n tasks
        initial_tasks = []
        n_train_examples = len(self.task.train_examples)
        for test_idx in range(n_train_examples):
            potential_train_idxs = set(range(n_train_examples)) - {test_idx}
            # we already remove i, so we need to remove n-1 more
            for leave_idxs in itertools.combinations(potential_train_idxs, leave_n - 1):
                train_idxs = potential_train_idxs - set(leave_idxs)
                examples = self.task.train_examples.copy()
                initial_tasks.append(
                    Task(name="", train_examples=[examples[i] for i in train_idxs], test_example=examples[test_idx])
                )

        if self.debug_no_aug:
            augmented_tasks = initial_tasks
        else:
            # get augmented tasks
            augmented_tasks = []
            for augmenter in self.augmenters:
                for task in initial_tasks:
                    task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
                    if task.max_height() <= 30 and task.max_width() <= 30:
                        augmented_tasks.append(task)
            augmented_tasks = list(dict.fromkeys(augmented_tasks + initial_tasks))

            # get permute-color then i/o-permuted tasks
            color_and_permute_augmented_tasks = []
            for _ in range(self.permute_n):
                for task in augmented_tasks:
                    new_task = task
                    if len(self.augmenters) > 0:
                        new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
                    new_task = PermuteExamples().apply_to_task(new_task, rng=rng, to_input=True, to_output=True)
                    color_and_permute_augmented_tasks.append(new_task)
            augmented_tasks = list(dict.fromkeys(color_and_permute_augmented_tasks + augmented_tasks))

        # format
        rng.shuffle(augmented_tasks)
        filtered_tasks = []
        for task in augmented_tasks:
            if len(filtered_tasks) >= max_gen:
                break
            if self.format_and_filter(task) is not None:
                filtered_tasks.append(task)
        return filtered_tasks

    def format_and_filter(self, task: Task) -> Optional[Dict]:
        # big grids are filtered out during augmentation already
        assert task.max_height() <= 30 and task.max_width() <= 30

        # Build encoder text
        prefix_texts = []
        for p in task.train_examples:
            prefix_texts.append(grid_to_text(p.input.tolist(), True))
            prefix_texts.append(grid_to_text(p.output.tolist(), False))
        encoder_text = "\n".join(prefix_texts) + "".join(self.cls_tokens)

        dec_in_text  = grid_to_text(task.test_example.input.tolist(), True)
        dec_out_text = grid_to_text(task.test_example.output.tolist(), False)

        # tiny optimization to include output:\n in input
        assert dec_out_text.startswith("output:\n")
        dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output:\n")]
        dec_out_text = dec_out_text[len("output:\n"):]
        dec_out_text += self.decoder_tokenizer.eos_token

        enc_tokens = self.encoder_tokenizer.encode_to_tensor(encoder_text)
        assert self.encoder_tokenizer.decode(enc_tokens["input_ids"][0][-self.ntokens:]) == "".join(self.cls_tokens)
        dec_in_tokens = self.decoder_tokenizer.encode_to_tensor(dec_in_text)
        dec_out_tokens = self.decoder_tokenizer.encode_to_tensor(dec_out_text)
        assert dec_out_tokens["input_ids"][0][-1].item() == self.decoder_tokenizer.eos_token_id

        # remove begin of sentence of dec_out_tokens
        assert dec_out_tokens['input_ids'][0][0] == self.decoder_tokenizer.bos_token_id
        dec_out_tokens['input_ids'] = dec_out_tokens['input_ids'][:, 1:]
        dec_out_tokens['attention_mask'] = dec_out_tokens['attention_mask'][:, 1:]

        # Build final decoder input + labels
        decoder_input_ids = torch.cat([
            dec_in_tokens["input_ids"].squeeze(0),
            dec_out_tokens["input_ids"].squeeze(0),
        ], dim=0)
        decoder_labels = torch.cat([
            torch.full(dec_in_tokens["input_ids"].shape[1:], -100),
            dec_out_tokens["input_ids"].squeeze(0),
        ], dim=0)
        decoder_attention_mask = torch.cat([
            dec_in_tokens["attention_mask"].squeeze(0),
            dec_out_tokens["attention_mask"].squeeze(0),
        ], dim=0)

        # Check length
        if enc_tokens["input_ids"].shape[1] - self.ntokens > self.max_seq_len or decoder_input_ids.shape[0] > self.max_seq_len // 2: # dec_len should be short
            return None

        # construct encoder label
        prefix_count = len(task.train_examples)
        encoder_input_ids = enc_tokens["input_ids"].squeeze(0)
        input_token_positions = [enc_i for enc_i, enc_token_id in enumerate(encoder_input_ids) if enc_token_id == self.encoder_input_token_id]
        output_token_positions = [enc_i for enc_i, enc_token_id in enumerate(encoder_input_ids) if enc_token_id == self.encoder_output_token_id]
        assert len(input_token_positions) == len(output_token_positions) == prefix_count
        assert all(p1 < p2 for p1, p2 in zip(input_token_positions, output_token_positions))

        encoder_labels = torch.full_like(encoder_input_ids, -100, dtype=encoder_input_ids.dtype)
        end_position = len(encoder_input_ids) - self.ntokens
        for pos, (p1, p2) in enumerate(zip(output_token_positions, input_token_positions[1:] + [end_position])):
            is_first = (pos == 0)
            is_last = (pos == prefix_count - 1)
            if self.encoder_loss_type == "last":
                if is_last:
                    p1 += 2 # remove output and :\n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
            elif self.encoder_loss_type == "rest":
                if not is_first:
                    p1 += 2 # remove output and :\n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
            else:
                p1 += 2 # remove output and :\n
                p2 -= not is_last # remove \n
                encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": enc_tokens["attention_mask"].squeeze(0),
            "encoder_labels": encoder_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_labels": decoder_labels,
        }

    def __len__(self):
        return len(self.ttt_tasks)

    def __getitem__(self, idx):
        return self.format_and_filter(self.ttt_tasks[idx])


def collate_fn_ttt(batch: List[Dict], dataset: TTTDataset) -> Dict:
    enc_ids = [x["encoder_input_ids"] for x in batch]
    enc_mask = [x["encoder_attention_mask"] for x in batch]
    enc_labs = [x["encoder_labels"] for x in batch]
    dec_ids = [x["decoder_input_ids"] for x in batch]
    dec_mask = [x["decoder_attention_mask"] for x in batch]
    dec_labs = [x["decoder_labels"] for x in batch]

    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]
    enc_ids = pad_sequence_with_side(enc_ids, padding_value=dataset.encoder_tokenizer.pad_token_id, side=dataset.encoder_pad_side)
    enc_mask = pad_sequence_with_side(enc_mask, padding_value=0, side=dataset.encoder_pad_side)
    enc_labs = pad_sequence_with_side(enc_labs, padding_value=-100, side=dataset.encoder_pad_side)
    dec_ids = pad_sequence_with_side(dec_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_pad_side)
    dec_mask = pad_sequence_with_side(dec_mask, padding_value=0, side=dataset.decoder_pad_side)
    dec_labs = pad_sequence_with_side(dec_labs, padding_value=-100, side=dataset.decoder_pad_side)

    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_labs,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_labs,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
    }


def collate_fn_ttt_dummy(batch: List[Dict], ntokens: int, debug_enc_len: int, debug_dec_len: int) -> Dict:
    batch_size = len(batch)
    assert batch_size > 0 and batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    del batch  # we don't use it directly

    enc_ids = torch.randint(1, 101, (batch_size, debug_enc_len + ntokens), dtype=torch.int64, device='cpu')
    enc_mask = torch.full((batch_size, debug_enc_len + ntokens), 1, dtype=torch.int64, device='cpu')
    dec_ids = torch.randint(1, 101, (batch_size, debug_dec_len), dtype=torch.int64, device='cpu')
    dec_mask = torch.full((batch_size, debug_dec_len), 1, dtype=torch.int64, device='cpu')
    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]

    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_ids,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_ids,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
    }


########################################
# Training Dataset
########################################
class TrainDataset(Dataset):
    def __init__(
        self,
        train_data_dir: str,
        eval_train_dir: str,
        verifier_file: str,
        encoder_tokenizer: ARCTokenizer,
        decoder_tokenizer: ARCTokenizer,
        total_steps: int,
        min_prefix: int,
        max_prefix: int,
        max_seq_len: int,
        augment_ratio: float,
        augment_single_grid: bool,
        seed: int,
        process_index: int,
        ntokens: int,
        debug_fixed_train_order: bool,
        debug_random_pad: bool,
        debug_pad_len: int,
        encoder_pad_side: str,
        decoder_pad_side: str,
        encoder_loss_type: bool,
        anti_invar_ratio: float,
        num_workers: int,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self._length = total_steps
        self.min_prefix = min_prefix
        self.max_prefix = max_prefix
        self.max_seq_len = max_seq_len
        self.augmenters: List[Augmenter] = get_extra_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)
        self.augment_ratio = augment_ratio
        self.augment_single_grid = augment_single_grid
        self.ntokens = ntokens
        self.cls_tokens = [f"<CLS{token_i}>" for token_i in range(ntokens)]
        self.debug_fixed_train_order = debug_fixed_train_order
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.encoder_pad_side = encoder_pad_side
        self.decoder_pad_side = decoder_pad_side
        self.encoder_loss_type = encoder_loss_type
        self.anti_invar_ratio = anti_invar_ratio

        # seed and process_index
        if num_workers == 0:
            self.rngs = [np.random.RandomState(seed + process_index)]
        else:
            self.rngs = [np.random.RandomState(seed + i) for i in range(num_workers * process_index, num_workers * (process_index + 1))]

        self.tasks_dict = load_re_arc_from_data_dir(train_data_dir)
        self.all_task_ids = list(self.tasks_dict.keys())

        self.encoder_input_token_id = encoder_tokenizer("input")['input_ids'][1]
        self.encoder_output_token_id = encoder_tokenizer("output")['input_ids'][1]

        # find unique tasks
        with open(verifier_file, "r") as f:
            file_content = f.read()
        tree = ast.parse(file_content)
        self.task_id_to_hash = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_body = ast.unparse(node.body) # type: ignore
                function_hash = hashlib.sha256(function_body.encode('utf-8')).hexdigest()
                assert node.name not in self.task_id_to_hash
                self.task_id_to_hash[node.name.split('_')[1]] = function_hash
        logger.info(f"found {len(self.task_id_to_hash)} functions from {verifier_file}")
        if "debug" not in eval_train_dir:
            assert set(self.task_id_to_hash.keys()) == set(self.tasks_dict.keys())

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # We'll do random sampling in the collate fn
        return 0


def collate_fn_train(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    assert batch_size > 0 and batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    del batch  # we don't use it directly

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]

    out_list = []

    while len(out_list) < batch_size:
        # whether to do opposite invar loss
        anti_invar = rng.rand() < dataset.anti_invar_ratio
        if anti_invar:
            # must be different tasks
            while True:
                task_ids = rng.choice(dataset.all_task_ids, size=2, replace=False)
                if dataset.task_id_to_hash[task_ids[0]] != dataset.task_id_to_hash[task_ids[1]]:
                    break
            pairs_for_tasks = [dataset.tasks_dict[task_ids[0]], dataset.tasks_dict[task_ids[1]]]
            # same or different augmentation
            augmenters = []
            io_augmentation_choices = []
            for _ in range(2):
                if rng.rand() < dataset.augment_ratio:
                    augmenters.append(rng.choice(dataset.augmenters)) # type: ignore
                else:
                    augmenters.append(None)
                io_augmentation_choice = rng.choice(["input_only", "output_only", "both"]) if dataset.augment_single_grid else "both"
                io_augmentation_choices.append(io_augmentation_choice)
        else:
            # must be same task
            task_id = rng.choice(dataset.all_task_ids)
            task_ids = [task_id, task_id]
            pairs_for_tasks = [dataset.tasks_dict[task_id], dataset.tasks_dict[task_id]]
            # must be same augmentation
            augmenters = [None, None]
            io_augmentation_choices = [rng.choice(["input_only", "output_only", "both"]) if dataset.augment_single_grid else "both"] * 2
            if rng.rand() < dataset.augment_ratio:
                augmenters = [rng.choice(dataset.augmenters)] * 2 # type: ignore

        # We'll attempt to produce exactly 2 examples from this single task
        two_task_list = []
        for task_i12 in range(2):
            pairs_for_task = pairs_for_tasks[task_i12]
            augmenter = augmenters[task_i12]
            io_augmentation_choice = io_augmentation_choices[task_i12]

            prefix_count = random.randint(dataset.min_prefix, dataset.max_prefix)
            required_count = prefix_count + 1
            if len(pairs_for_task) < required_count:
                break

            # sample task
            chosen_pairs = pairs_for_task[:required_count]
            if not dataset.debug_fixed_train_order:
                chosen_pairs = rng.choice(pairs_for_task, size=required_count, replace=False) # type: ignore
            chosen_pairs = copy.deepcopy(chosen_pairs)

            # apply augmentation
            if augmenter is not None:
                for pair in chosen_pairs:
                    if io_augmentation_choice in ['input_only', 'both']:
                        pair['input'] = augmenter.apply_to_grid(np.array(pair['input']), rng)
                    if io_augmentation_choice in ['output_only', 'both']:
                        pair['output'] = augmenter.apply_to_grid(np.array(pair['output']), rng)

            if any(len(pair['input']) > 30 or len(pair['output']) > 30 for pair in chosen_pairs):
                break
            if any(len(pair['input'][0]) > 30 or len(pair['output'][0]) > 30 for pair in chosen_pairs):
                break

            train_pairs = chosen_pairs[:prefix_count]
            test_pair = chosen_pairs[prefix_count]

            prefix_texts = []
            for pair in train_pairs:
                prefix_texts.append(grid_to_text(pair["input"], True))
                prefix_texts.append(grid_to_text(pair["output"], False))
            encoder_text = "\n".join(prefix_texts) + "".join(dataset.cls_tokens)

            dec_in_text  = grid_to_text(test_pair["input"], True)
            dec_out_text = grid_to_text(test_pair["output"], False)

            # tiny optimization to include output:\n in input
            assert dec_out_text.startswith("output:\n")
            dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output:\n")]
            dec_out_text = dec_out_text[len("output:\n"):]
            dec_out_text += dataset.decoder_tokenizer.eos_token

            enc_tokens = dataset.encoder_tokenizer.encode_to_tensor(encoder_text)
            assert dataset.encoder_tokenizer.decode(enc_tokens["input_ids"][0][-dataset.ntokens:]) == "".join(dataset.cls_tokens)
            dec_in_tokens  = dataset.decoder_tokenizer.encode_to_tensor(dec_in_text)
            dec_out_tokens = dataset.decoder_tokenizer.encode_to_tensor(dec_out_text)
            assert dec_out_tokens["input_ids"][0][-1].item() == dataset.decoder_tokenizer.eos_token_id

            # remove begin of sentence of dec_out_tokens
            assert dec_out_tokens['input_ids'][0][0] == dataset.decoder_tokenizer.bos_token_id
            dec_out_tokens['input_ids'] = dec_out_tokens['input_ids'][:, 1:]
            dec_out_tokens['attention_mask'] = dec_out_tokens['attention_mask'][:, 1:]

            # Build final decoder input + labels
            decoder_input_ids = torch.cat([
                dec_in_tokens["input_ids"].squeeze(0),
                dec_out_tokens["input_ids"].squeeze(0),
            ], dim=0)
            decoder_labels = torch.cat([
                torch.full(dec_in_tokens["input_ids"].shape[1:], -100),
                dec_out_tokens["input_ids"].squeeze(0),
            ], dim=0)
            decoder_attention_mask = torch.cat([
                dec_in_tokens["attention_mask"].squeeze(0),
                dec_out_tokens["attention_mask"].squeeze(0),
            ], dim=0)

            # Check length
            if enc_tokens["input_ids"].shape[1] - dataset.ntokens > dataset.max_seq_len or decoder_input_ids.shape[0] > dataset.max_seq_len // 2: # dec_len should be short
                break

            # construct encoder label
            encoder_input_ids = enc_tokens["input_ids"].squeeze(0)
            input_token_positions = [enc_i for enc_i, enc_token_id in enumerate(encoder_input_ids) if enc_token_id == dataset.encoder_input_token_id]
            output_token_positions = [enc_i for enc_i, enc_token_id in enumerate(encoder_input_ids) if enc_token_id == dataset.encoder_output_token_id]
            assert len(input_token_positions) == len(output_token_positions) == prefix_count
            assert all(p1 < p2 for p1, p2 in zip(input_token_positions, output_token_positions))

            encoder_labels = torch.full_like(encoder_input_ids, -100, dtype=encoder_input_ids.dtype)
            end_position = len(encoder_input_ids) - dataset.ntokens
            for pos, (p1, p2) in enumerate(zip(output_token_positions, input_token_positions[1:] + [end_position])):
                is_first = (pos == 0)
                is_last = (pos == prefix_count - 1)
                if dataset.encoder_loss_type == "last":
                    if is_last:
                        p1 += 2 # remove output and :\n
                        p2 -= not is_last # remove \n
                        encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
                elif dataset.encoder_loss_type == "rest":
                    if not is_first:
                        p1 += 2 # remove output and :\n
                        p2 -= not is_last # remove \n
                        encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
                else:
                    p1 += 2 # remove output and :\n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])

            two_task_list.append({
                "task_ids": task_ids[task_i12], # purely for debugging
                "prefix_counts": prefix_count, # purely for debugging
                "encoder_input_ids": encoder_input_ids,
                "encoder_attention_mask": enc_tokens["attention_mask"].squeeze(0),
                "encoder_labels": encoder_labels,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_labels": decoder_labels,
                "anti_invar": anti_invar,
                "input_token_positions": input_token_positions,
                "output_token_positions": output_token_positions,
            })
            # # show task data
            # print("////////////////////////////////////////////")
            # print(f"Encode text:{encoder_text}")
            # print(f"Task ID: {task_ids[task_i12]}")
            # print(f"Prefix Counts: {prefix_count}")
            # print("input_len",len(encoder_input_ids))

            # print(f"Encoder Input IDs: {encoder_input_ids}")
            # print(f"Encoder Attention Mask: {enc_tokens['attention_mask'].squeeze(0)}")
            # print(f"Encoder Labels: {encoder_labels}")

            # print(f"Decoder Input IDs: {decoder_input_ids}")
            # print(f"Decoder Attention Mask: {decoder_attention_mask}")
            # print(f"Decoder Labels: {decoder_labels}")

            # print(f"Anti Invariance: {anti_invar}")
            # print(f"Input Token Positions: {input_token_positions}")
            # print(f"Output Token Positions: {output_token_positions}")

            # print("////////////////////////////////////////////")

        # Check if we got 2 valid items from this task
        if len(two_task_list) == 2:
            if dataset.debug_fixed_train_order or not torch.equal(two_task_list[0]['encoder_input_ids'], two_task_list[1]['encoder_input_ids']):
                # Add them to out_list
                out_list.extend(two_task_list)

    # Now we must truncate out_list if we overshoot
    assert len(out_list) == batch_size, f"Should produce exactly {batch_size} items"

    enc_ids  = [x["encoder_input_ids"] for x in out_list]
    enc_mask = [x["encoder_attention_mask"] for x in out_list]
    enc_labs = [x["encoder_labels"] for x in out_list]
    dec_ids  = [x["decoder_input_ids"] for x in out_list]
    dec_mask = [x["decoder_attention_mask"] for x in out_list]
    dec_labs = [x["decoder_labels"] for x in out_list]
    anti_invars = [x["anti_invar"] for x in out_list]
    prefix_counts = [x["prefix_counts"] for x in out_list]
    input_token_positions = [x['input_token_positions'] for x in out_list]
    output_token_positions = [x['output_token_positions'] for x in out_list]

    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]
    enc_ids  = pad_sequence_with_side(enc_ids, padding_value=dataset.encoder_tokenizer.pad_token_id, side=dataset.encoder_pad_side)
    enc_mask = pad_sequence_with_side(enc_mask, padding_value=0, side=dataset.encoder_pad_side)
    enc_labs = pad_sequence_with_side(enc_labs, padding_value=-100, side=dataset.encoder_pad_side)
    dec_ids  = pad_sequence_with_side(dec_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_pad_side)
    dec_mask = pad_sequence_with_side(dec_mask, padding_value=0, side=dataset.decoder_pad_side)
    dec_labs = pad_sequence_with_side(dec_labs, padding_value=-100, side=dataset.decoder_pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        enc_ids, enc_mask, enc_labs = debug_extra_pad_tensors(
            [enc_ids, enc_mask, enc_labs],
            padding_values=[dataset.encoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.encoder_pad_side,
        )
        dec_ids, dec_mask, dec_labs = debug_extra_pad_tensors(
            [dec_ids, dec_mask, dec_labs],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_pad_side,
        )

    # task_ids = [x["task_ids"] for x in out_list] # purely for debugging
    # prefix_counts = [x["prefix_counts"] for x in out_list] # purely for debugging
    # import pickle
    # import glob
    # idx = max([int(Path(p).stem) for p in glob.glob(f"lots_debug_data/debug_old_train_dicts/*.pkl")], default=-1) + 1
    # pickle.dump({
    #     "task_ids": task_ids,
    #     "encoder_input_ids_lens": enc_ids_lens,
    #     "decoder_input_ids_lens": dec_ids_lens,
    #     "num_pairs": prefix_counts,
    #     "trainable_pair_indices": [c - 1 for c in prefix_counts],
    # }, open(f'lots_debug_data/debug_old_train_dicts/{idx}.pkl', 'wb'))
    # print(input_token_positions.shape)
    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_labs,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_labs,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
        "anti_invars": anti_invars,
        "prefix_counts": prefix_counts,
        "input_token_positions": input_token_positions,
        "output_token_positions": output_token_positions,
    }


def collate_fn_train_dummy(batch: List[int], ntokens: int, debug_enc_len: int, debug_dec_len: int) -> Dict:
    batch_size = len(batch)
    assert batch_size > 0 and batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    del batch  # we don't use it directly

    enc_ids = torch.randint(1, 101, (batch_size, debug_enc_len + ntokens), dtype=torch.int64, device='cpu')
    enc_mask = torch.full((batch_size, debug_enc_len + ntokens), 1, dtype=torch.int64, device='cpu')
    dec_ids = torch.randint(1, 101, (batch_size, debug_dec_len), dtype=torch.int64, device='cpu')
    dec_mask = torch.full((batch_size, debug_dec_len), 1, dtype=torch.int64, device='cpu')
    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]
    anti_invars = [False] * batch_size

    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_ids,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_ids,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
        "anti_invars": anti_invars,
    }


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    def __init__(
        self,
        eval_dir: str,
        select_tasks_path: Optional[str],
        leave_ns: List[int],
        leave_ns_inc: bool,
        permute_n: int,
        augment_n: int,
        permute_iters: int,
        seed: int,
        encoder_tokenizer: ARCTokenizer,
        decoder_tokenizer: ARCTokenizer,
        max_seq_len: int,
        ntokens: int,
        encoder_loss_type: str,
        debug_random_pad: bool,
        debug_pad_len: int,
        encoder_pad_side: str,
        decoder_pad_side: str,
        decoder_gen_pad_side: str,
    ):
        self.permute_n = permute_n
        self.permute_iters = permute_iters
        self.augment_n = augment_n
        self.augmenters = [Transpose(), Flip(0), Flip(1), Rotate(90), Rotate(180)]
        self.seed = seed
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_seq_len = max_seq_len
        self.ntokens = ntokens
        self.cls_tokens = [f"<CLS{token_i}>" for token_i in range(ntokens)]
        self.encoder_loss_type = encoder_loss_type
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.encoder_pad_side = encoder_pad_side
        self.decoder_pad_side = decoder_pad_side
        self.decoder_gen_pad_side = decoder_gen_pad_side

        self.encoder_input_token_id = encoder_tokenizer("input")['input_ids'][1]
        self.encoder_output_token_id = encoder_tokenizer("output")['input_ids'][1]

        # get file_paths
        if not os.path.isdir(eval_dir):
            raise FileNotFoundError(f"Eval directory '{eval_dir}' not found.")
        file_paths = []
        for filename in os.listdir(eval_dir):
            if filename.endswith(".json"):
                file_paths.append(os.path.join(eval_dir, filename))
        file_paths.sort()
        logger.info(f"found {len(file_paths)} files")

        # filter based on select tasks file
        if select_tasks_path is not None:
            with open(select_tasks_path, mode='r') as file:
                csv_reader = csv.reader(file)
                data_as_tuples = [tuple(row) for row in csv_reader]
                data_as_tuples = data_as_tuples[1:] # first row contains col names
                select_task_ids = [d[0] for d in data_as_tuples]
                assert len(select_task_ids) == len(set(select_task_ids))
                select_task_ids = set(select_task_ids)
            # filter
            file_paths = [p for p in file_paths if Path(p).stem in select_task_ids]
            assert len(file_paths) == len(select_task_ids), (len(file_paths), len(select_task_ids))
            logger.info(f"filtered to {len(file_paths)} files from {select_tasks_path}")

        # get actual data
        tasks = []
        for file_path in file_paths:
            # load data
            task_id = Path(file_path).stem
            with open(file_path, "r") as f:
                task_data = json.load(f)
            # create tasks
            train_examples = [Example(input=np.array(x["input"]), output=np.array(x["output"])) for x in task_data['train']]
            test_examples = [Example(input=np.array(x["input"]), output=np.array(x["output"])) for x in task_data['test']]
            for test_i, test_example in enumerate(test_examples):
                tasks.append(Task(
                    name=f'{task_id}-{test_i}',
                    train_examples=train_examples,
                    test_example=test_example,
                ))
        tasks.sort(key=lambda d: d.name)
        logger.info(f"found {len(tasks)} tasks")

        # print task num io
        task_num_ios = [len(task.train_examples) for task in tasks]
        logger.info(f"task num io range from {min(task_num_ios)} to {max(task_num_ios)}")

        # get task id to gt for competition evaluation
        self.task_id_to_gt = {task.name: task.test_example.output.tolist() for task in tasks}
        assert len(self.task_id_to_gt) == len(tasks)

        # augment data for voting
        # since this function has to do filtering, might as well parse data as well
        self.eval_tasks = []
        for task in tasks:
            new_tasks = self.get_task_augmentations_leave_ns_filtered(task, leave_ns=leave_ns)
            if len(new_tasks) == 0 and leave_ns_inc:
                new_tasks = self.get_task_augmentations_leave_ns_filtered(task, leave_ns=leave_ns + [leave_ns[-1] + 1])
            self.eval_tasks += new_tasks
        logger.info(f'augmented and filtered data from {len(tasks)} to {len(self.eval_tasks)}')

        # print details of parsed data
        parsed_data = [self[data_i] for data_i in range(len(self))]
        # task to number of augmented queries
        task_id_to_counts = Counter(d["task_id"] for d in parsed_data) # type: ignore
        task_id_to_counts = [(task_id, count) for task_id, count in task_id_to_counts.items()]
        for task_id, count in sorted(task_id_to_counts):
            logger.info(f"{task_id}: Number of Queries: {count}")
        # min and max sequence length
        enc_min_len, enc_max_len = 1e6, 0
        dec_min_len, dec_max_len = 1e6, 0
        for d in parsed_data:
            enc_min_len = min(enc_min_len, len(d['encoder_input_ids'])) # type: ignore
            enc_max_len = max(enc_max_len, len(d['encoder_input_ids'])) # type: ignore
            dec_min_len = min(dec_min_len, len(d['decoder_input_ids'])) # type: ignore
            dec_max_len = max(dec_max_len, len(d['decoder_input_ids'])) # type: ignore
        logger.info(f"encoder sequence length range from {enc_min_len} to {enc_max_len}]")
        logger.info(f"decoder sequence length range from {dec_min_len} to {dec_max_len}]")
        del parsed_data

        # print statistics
        # encoder_input_ids_lens = [x['encoder_input_ids'].shape[0] for x in self.parsed_data]
        # decoder_input_ids_lens = [x['decoder_input_ids'].shape[0] for x in self.parsed_data]
        # print([x for x in encoder_input_ids_lens if x > 8192])
        # print('encoderlen min-max seqlen:', min(encoder_input_ids_lens), max(encoder_input_ids_lens))
        # print('decoderlen min-max seqlen:', min(decoder_input_ids_lens), max(decoder_input_ids_lens))
        # plot_histogram_with_frequencies(encoder_input_ids_lens, "encoder.jpg")
        # plot_histogram_with_frequencies(decoder_input_ids_lens, "decoder.jpg")
        # breakpoint()

    def get_task_augmentations_leave_ns_filtered(
            self,
            task: Task,
            leave_ns: list[int],
        ) -> List[Task]:
        # get augmented queries
        augmented_tasks = []
        for leave_n in leave_ns:
            augmented_tasks += self.get_task_augmentations_leave_n(task, leave_n=leave_n)
        # filter augmented queries
        filtered_tasks = []
        for task in augmented_tasks:
            if self.format_and_filter(task) is not None:
                filtered_tasks.append(task)
        return filtered_tasks

    def get_task_augmentations_leave_n(
            self,
            task: Task,
            leave_n: int,
        ) -> List[Task]:
        rng = np.random.RandomState(self.seed)
        test_tasks = []

        # add leave n tasks
        indices = list(range(len(task.train_examples)))
        leave_n_indices = [set(indices) - set(comb) for comb in itertools.combinations(indices, leave_n)]
        train_examples = task.train_examples.copy()
        for comb in leave_n_indices:
            # add non-permute
            new_task = Task(
                name=task.name,
                train_examples=[train_examples[j] for j in comb],
                test_example=task.test_example,
            )
            test_tasks.append(new_task)
            # add permuted
            for _ in range(self.permute_n):
                permuted_task = PermuteExamples().apply_to_task(new_task, to_input=True, to_output=True, rng=rng)
                test_tasks.append(permuted_task)

        # remove duplicates
        # logger.info(f"{task.name} has {len(test_tasks)} after permute")
        test_tasks = list(dict.fromkeys(test_tasks))
        # logger.info(f"{task.name} has {len(test_tasks)} after permute set")

        # get augmented tasks
        augmented_tasks = []
        augmenters = rng.choice(self.augmenters, size=self.augment_n, replace=False)
        for augmenter in augmenters:
            # pick a random permutation and apply the augmenter to a random leave_n task
            new_task = rng.choice(test_tasks)
            augmented_task = augmenter.apply_to_task(new_task, to_input=True, to_output=True)
            if augmented_task in test_tasks:
                continue
            inverter = str(inverse(augmenter))
            augmented_task.inverter = inverter
            augmented_tasks.append(augmented_task)
        test_tasks += augmented_tasks

        # remove duplicates
        # logger.info(f"{task.name} has {len(test_tasks)} after permute set augment")
        test_tasks = list(dict.fromkeys(test_tasks))
        # logger.info(f"{task.name} has {len(test_tasks)} after permute set augment set")
        return test_tasks

    def format_and_filter(self, task: Task, permutation: Optional[List[int]] = None) -> Optional[Dict]:
        # this function only filters by token length, not by grid dimension
        # even the voting augmentation does not increase resolution
        assert task.max_height() <= 30 and task.max_width() <= 30

        # permute if given
        train_examples = task.train_examples
        if permutation is not None:
            assert set(permutation) == set(range(len(train_examples)))
            train_examples = [train_examples[permute_i] for permute_i in permutation]

        # Build encoder text
        prefix_texts = []
        for p in train_examples:
            prefix_texts.append(grid_to_text(p.input.tolist(), True))
            prefix_texts.append(grid_to_text(p.output.tolist(), False))
        encoder_text = "\n".join(prefix_texts) + "".join(self.cls_tokens)

        dec_in_text  = grid_to_text(task.test_example.input.tolist(), True)
        dec_out_text = grid_to_text(task.test_example.output.tolist(), False)

        # tiny optimization to include output:\n in input
        assert dec_out_text.startswith("output:\n")
        dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output:\n")]
        dec_out_text = dec_out_text[len("output:\n"):]
        dec_out_text += self.decoder_tokenizer.eos_token

        enc_tokens = self.encoder_tokenizer.encode_to_tensor(encoder_text)
        assert self.encoder_tokenizer.decode(enc_tokens["input_ids"][0][-self.ntokens:]) == "".join(self.cls_tokens)
        dec_in_tokens = self.decoder_tokenizer.encode_to_tensor(dec_in_text)
        dec_out_tokens = self.decoder_tokenizer.encode_to_tensor(dec_out_text)
        assert dec_out_tokens["input_ids"][0][-1].item() == self.decoder_tokenizer.eos_token_id

        # remove begin of sentence of dec_out_tokens
        assert dec_out_tokens['input_ids'][0][0] == self.decoder_tokenizer.bos_token_id
        dec_out_tokens['input_ids'] = dec_out_tokens['input_ids'][:, 1:]
        dec_out_tokens['attention_mask'] = dec_out_tokens['attention_mask'][:, 1:]

        # Build final decoder input + labels
        decoder_input_ids = torch.cat([
            dec_in_tokens["input_ids"].squeeze(0),
            dec_out_tokens["input_ids"].squeeze(0),
        ], dim=0)
        decoder_labels = torch.cat([
            torch.full(dec_in_tokens["input_ids"].shape[1:], -100),
            dec_out_tokens["input_ids"].squeeze(0),
        ], dim=0)
        decoder_attention_mask = torch.cat([
            dec_in_tokens["attention_mask"].squeeze(0),
            dec_out_tokens["attention_mask"].squeeze(0),
        ], dim=0)

        dec_label_texts = dec_out_text[:-len(self.decoder_tokenizer.eos_token)]

        # Check length
        if enc_tokens["input_ids"].shape[1] - self.ntokens > self.max_seq_len or decoder_input_ids.shape[0] > self.max_seq_len // 2: # dec_len should be short
            return None

        # construct encoder label
        prefix_count = len(train_examples)
        encoder_input_ids = enc_tokens["input_ids"].squeeze(0)
        input_token_positions = [enc_i for enc_i, enc_token_id in enumerate(encoder_input_ids) if enc_token_id == self.encoder_input_token_id]
        output_token_positions = [enc_i for enc_i, enc_token_id in enumerate(encoder_input_ids) if enc_token_id == self.encoder_output_token_id]
        assert len(input_token_positions) == len(output_token_positions) == prefix_count
        assert all(p1 < p2 for p1, p2 in zip(input_token_positions, output_token_positions))

        encoder_labels = torch.full_like(encoder_input_ids, -100, dtype=encoder_input_ids.dtype)
        end_position = len(encoder_input_ids) - self.ntokens
        for pos, (p1, p2) in enumerate(zip(output_token_positions, input_token_positions[1:] + [end_position])):
            is_first = (pos == 0)
            is_last = (pos == prefix_count - 1)
            if self.encoder_loss_type == "last":
                if is_last:
                    p1 += 2 # remove output and :\n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
            elif self.encoder_loss_type == "rest":
                if not is_first:
                    p1 += 2 # remove output and :\n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
            else:
                p1 += 2 # remove output and :\n
                p2 -= not is_last # remove \n
                encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])

        return {
            "task_id": task.name,
            "prefix_count": prefix_count, # purely for debugging
            "inverter": task.inverter if hasattr(task, "inverter") else "", # type: ignore
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": enc_tokens["attention_mask"].squeeze(0),
            "encoder_labels": encoder_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_labels": decoder_labels,
            "decoder_gen_input_ids": dec_in_tokens["input_ids"].squeeze(0),
            "decoder_gen_attention_mask": dec_in_tokens["attention_mask"].squeeze(0),
            "decoder_out_token_length": dec_out_tokens["input_ids"].shape[1] - 1, # remove eos token
            "decoder_label_texts": dec_label_texts,  # used for exact match
            "input_token_positions": input_token_positions,
            "output_token_positions": output_token_positions,
        }

    def __len__(self):
        return len(self.eval_tasks)

    def __getitem__(self, idx):
        return self.format_and_filter(self.eval_tasks[idx])

    def get_io_permuted_batches(self, batch_idxs: List[int]) -> Iterator[Tuple[List[Optional[Dict]], List[bool]]]:
        # TODO: optionally can leave1, but might mess with underlying program
        batch_size = len(batch_idxs)
        eval_tasks = [self.eval_tasks[idx] for idx in batch_idxs]

        rng = np.random.RandomState(self.seed)
        permutations_of_tasks = []
        for task in eval_tasks:
            num_train_examples = len(task.train_examples)
            # first permutation is always default
            permutations = list(itertools.permutations(range(num_train_examples)))
            permutations = [permutations[0]] + rng.permutation(permutations[1:]).tolist()[:self.permute_iters]
            permutations_of_tasks.append(permutations)

        # pad permutations
        # TODO: maybe this introduces issues
        # max_num_permutation = max(len(p) for p in permutations_of_tasks)
        # for idx in range(len(permutations_of_tasks)):
        #     l = permutations_of_tasks[idx]
        #     permutations_of_tasks[idx] = l * (max_num_permutation // len(l)) + l[:max_num_permutation % len(l)]
        # assert set(len(p) for p in permutations_of_tasks) == {max_num_permutation}

        max_num_permutation = max(len(p) for p in permutations_of_tasks)
        assert max_num_permutation > 0
        for permute_i in range(max_num_permutation):
            avail = [len(permutations) > permute_i for permutations in permutations_of_tasks]
            permutations = [permutations_of_tasks[idx][permute_i] for idx in range(batch_size) if avail[idx]]
            avail_eval_tasks = [eval_tasks[idx] for idx in range(batch_size) if avail[idx]]
            permuted_tasks = [self.format_and_filter(task, permutation) for task, permutation in zip(avail_eval_tasks, permutations)]
            yield (permuted_tasks, avail)


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    task_ids = [x['task_id'] for x in batch]
    inverters = [x['inverter'] for x in batch]
    enc_ids = [x["encoder_input_ids"] for x in batch]
    enc_mask = [x["encoder_attention_mask"] for x in batch]
    enc_labs = [x["encoder_labels"] for x in batch]
    dec_ids = [x["decoder_input_ids"] for x in batch]
    dec_mask = [x["decoder_attention_mask"] for x in batch]
    dec_gen_ids = [x["decoder_gen_input_ids"] for x in batch]
    dec_gen_mask = [x["decoder_gen_attention_mask"] for x in batch]
    dec_labs = [x["decoder_labels"] for x in batch]
    decoder_out_token_length = [x["decoder_out_token_length"] for x in batch]
    decoder_label_texts = [x["decoder_label_texts"] for x in batch]
    input_token_positions = [x["input_token_positions"] for x in batch]
    output_token_positions = [x["output_token_positions"] for x in batch]

    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]
    dec_gen_ids_lens = [len(x) for x in dec_gen_ids]
    enc_ids = pad_sequence_with_side(enc_ids, padding_value=dataset.encoder_tokenizer.pad_token_id, side=dataset.encoder_pad_side)
    enc_mask = pad_sequence_with_side(enc_mask, padding_value=0, side=dataset.encoder_pad_side)
    enc_labs = pad_sequence_with_side(enc_labs, padding_value=-100, side=dataset.encoder_pad_side)
    dec_ids = pad_sequence_with_side(dec_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_pad_side)
    dec_mask = pad_sequence_with_side(dec_mask, padding_value=0, side=dataset.decoder_pad_side)
    dec_labs = pad_sequence_with_side(dec_labs, padding_value=-100, side=dataset.decoder_pad_side)
    dec_gen_ids = pad_sequence_with_side(dec_gen_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_gen_pad_side)
    dec_gen_mask = pad_sequence_with_side(dec_gen_mask, padding_value=0, side=dataset.decoder_gen_pad_side)

    if dataset.debug_random_pad and dataset.debug_pad_len > -1:
        enc_ids, enc_mask, enc_labs = debug_extra_pad_tensors(
            [enc_ids, enc_mask, enc_labs],
            padding_values=[dataset.encoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.encoder_pad_side,
        )
        dec_ids, dec_mask, dec_labs = debug_extra_pad_tensors(
            [dec_ids, dec_mask, dec_labs],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_pad_side,
        )
        dec_gen_ids, dec_gen_mask = debug_extra_pad_tensors(
            [dec_gen_ids, dec_gen_mask],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_gen_pad_side,
        )

    # prefix_count = [x['prefix_count'] for x in batch] # purely for debugging
    # import pickle
    # import glob
    # idx = max([int(Path(p).stem) for p in glob.glob(f"lots_debug_data/debug_old_eval_train_dicts/*.pkl")], default=-1) + 1
    # pickle.dump({
    #     "task_id": task_ids,
    #     "encoder_input_ids_len": enc_ids_lens,
    #     "decoder_input_ids_len": dec_ids_lens,
    #     "decoder_gen_input_ids_lens": dec_gen_ids_lens,
    #     "decoder_out_token_length": decoder_out_token_length,
    #     "num_pairs": prefix_count,
    # }, open(f'lots_debug_data/debug_old_eval_train_dicts/{idx}.pkl', 'wb'))

    batch_dict = {
        "task_ids": task_ids,
        "inverters": inverters,
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_labs,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_labs,
        "decoder_gen_input_ids": dec_gen_ids,
        "decoder_gen_attention_mask": dec_gen_mask,
        "decoder_out_token_length": decoder_out_token_length,
        "decoder_label_texts": decoder_label_texts,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
        "decoder_gen_input_ids_lens": dec_gen_ids_lens,
        "input_token_positions": input_token_positions,
        "output_token_positions": output_token_positions,
    }

    

    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], ntokens: int, debug_enc_len: int, debug_dec_len: int) -> Dict:
    batch_size = len(batch)
    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    enc_ids = torch.randint(1, 101, (batch_size, debug_enc_len + ntokens), dtype=torch.int64, device='cpu')
    enc_mask = torch.full((batch_size, debug_enc_len + ntokens), 1, dtype=torch.int64, device='cpu')
    dec_ids = torch.randint(1, 101, (batch_size, debug_dec_len), dtype=torch.int64, device='cpu')
    dec_mask = torch.full((batch_size, debug_dec_len), 1, dtype=torch.int64, device='cpu')
    dec_gen_ids = torch.randint(1, 101, (batch_size, int(debug_dec_len * 0.4)), dtype=torch.int64, device='cpu')
    dec_gen_mask = torch.full((batch_size, int(debug_dec_len * 0.4)), 1, dtype=torch.int64, device='cpu')
    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]
    dec_gen_ids_lens = [len(x) for x in dec_gen_ids]

    batch_dict = {
        "task_ids": task_ids,
        "inverters": [""] * batch_size,
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_ids,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_ids,
        "decoder_gen_input_ids": dec_gen_ids,
        "decoder_gen_attention_mask": dec_gen_mask,
        "decoder_out_token_length": [math.ceil(debug_dec_len * 0.6)] * batch_size,
        "decoder_label_texts": ['1\n1\n1'] * batch_size,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
        "decoder_gen_input_ids_lens": dec_gen_ids_lens,
    }
    return batch_dict


########################################
# Gradient Search Dataset
########################################
class GSDataset(Dataset):
    def __init__(
        self,
        task: Task,
        encoder_tokenizer: ARCTokenizer,
        decoder_tokenizer: ARCTokenizer,
        max_seq_len: int,
        ntokens: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        decoder_pad_side: str,
    ):
        self.task = task
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_seq_len = max_seq_len
        self.ntokens = ntokens
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.decoder_pad_side = decoder_pad_side

        # format and filter data
        self.parsed_examples = []
        for example in task.train_examples:
            data = self.format_and_filter(example)
            if data is not None:
                self.parsed_examples.append(data)

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format_and_filter(self, example: Example) -> Optional[Dict]:
        # tasks are filtered by EvalDataset already, shouldn't have grids too big
        # but each example might still be too long (in practice this probably won't happen)
        assert max(example.input.shape) <= 30 and max(example.output.shape) <= 30

        dec_in_text  = grid_to_text(example.input.tolist(), True)
        dec_out_text = grid_to_text(example.output.tolist(), False)

        # tiny optimization to include output:\n in input
        assert dec_out_text.startswith("output:\n")
        dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output:\n")]
        dec_out_text = dec_out_text[len("output:\n"):]
        dec_out_text += self.decoder_tokenizer.eos_token

        dec_in_tokens = self.decoder_tokenizer.encode_to_tensor(dec_in_text)
        dec_out_tokens = self.decoder_tokenizer.encode_to_tensor(dec_out_text)
        assert dec_out_tokens["input_ids"][0][-1].item() == self.decoder_tokenizer.eos_token_id

        # remove begin of sentence of dec_out_tokens
        assert dec_out_tokens['input_ids'][0][0] == self.decoder_tokenizer.bos_token_id
        dec_out_tokens['input_ids'] = dec_out_tokens['input_ids'][:, 1:]
        dec_out_tokens['attention_mask'] = dec_out_tokens['attention_mask'][:, 1:]

        # Build final decoder input + labels
        decoder_input_ids = torch.cat([
            dec_in_tokens["input_ids"].squeeze(0),
            dec_out_tokens["input_ids"].squeeze(0),
        ], dim=0)
        decoder_labels = torch.cat([
            torch.full(dec_in_tokens["input_ids"].shape[1:], -100),
            dec_out_tokens["input_ids"].squeeze(0),
        ], dim=0)
        decoder_attention_mask = torch.cat([
            dec_in_tokens["attention_mask"].squeeze(0),
            dec_out_tokens["attention_mask"].squeeze(0),
        ], dim=0)

        # Check length
        # should never be evoked in practice
        if decoder_input_ids.shape[0] > self.max_seq_len // 2: # dec_len should be short
            return None

        return {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_labels": decoder_labels,
        }


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    dec_ids = [x["decoder_input_ids"] for x in batch]
    dec_mask = [x["decoder_attention_mask"] for x in batch]
    dec_labs = [x["decoder_labels"] for x in batch]

    dec_ids_lens = [len(x) for x in dec_ids]
    dec_ids = pad_sequence_with_side(dec_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_pad_side)
    dec_mask = pad_sequence_with_side(dec_mask, padding_value=0, side=dataset.decoder_pad_side)
    dec_labs = pad_sequence_with_side(dec_labs, padding_value=-100, side=dataset.decoder_pad_side)

    if dataset.debug_random_pad and dataset.debug_pad_len > -1:
        dec_ids, dec_mask, dec_labs = debug_extra_pad_tensors(
            [dec_ids, dec_mask, dec_labs],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_pad_side,
        )

    batch_dict = {
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_labs,
        "decoder_input_ids_lens": dec_ids_lens,
    }
    return batch_dict


# not used yet
def collate_fn_gs_dummy(batch: List[Dict], ntokens: int, debug_enc_len: int, debug_dec_len: int) -> Dict:
    batch_size = len(batch)
    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    enc_ids = torch.randint(1, 101, (batch_size, debug_enc_len + ntokens), dtype=torch.int64, device='cpu')
    enc_mask = torch.full((batch_size, debug_enc_len + ntokens), 1, dtype=torch.int64, device='cpu')
    dec_ids = torch.randint(1, 101, (batch_size, debug_dec_len), dtype=torch.int64, device='cpu')
    dec_mask = torch.full((batch_size, debug_dec_len), 1, dtype=torch.int64, device='cpu')
    dec_gen_ids = torch.randint(1, 101, (batch_size, int(debug_dec_len * 0.4)), dtype=torch.int64, device='cpu')
    dec_gen_mask = torch.full((batch_size, int(debug_dec_len * 0.4)), 1, dtype=torch.int64, device='cpu')
    enc_ids_lens = [len(x) for x in enc_ids]
    dec_ids_lens = [len(x) for x in dec_ids]
    dec_gen_ids_lens = [len(x) for x in dec_gen_ids]

    batch_dict = {
        "task_ids": task_ids,
        "inverters": [""] * batch_size,
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "encoder_labels": enc_ids,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_ids,
        "decoder_gen_input_ids": dec_gen_ids,
        "decoder_gen_attention_mask": dec_gen_mask,
        "decoder_out_token_length": [math.ceil(debug_dec_len * 0.6)] * batch_size,
        "decoder_label_texts": ['1\n1\n1'] * batch_size,
        "encoder_input_ids_lens": enc_ids_lens,
        "decoder_input_ids_lens": dec_ids_lens,
        "decoder_gen_input_ids_lens": dec_gen_ids_lens,
    }
    return batch_dict