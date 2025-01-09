# data_utils.py

import math
import copy
import os
import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Any, List
from pathlib import Path
from typing import List

import numpy as np

from arclib.augmenters import (
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
)
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def get_augmenters(
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

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
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


def pad_sequence_left(sequences, padding_value):
    reversed_sequences = [seq.flip(0) for seq in sequences]
    padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
    return padded_reversed.flip(1)


def load_tasks_from_data_dir(data_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scans 'data_dir' for *.json files, each containing a list of
    { "input": 2D_grid, "output": 2D_grid } items.

    Returns a dict: { task_id: [ {input:2D, output:2D}, ... ] }.
    """
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


def grid_to_text(
        grid: List[List[int]],
        is_input: bool,
    ) -> str:
    """
    Convert a 2D grid of ints into text lines:

      input:
      <height>
      <width>
      row_of_ints
      row_of_ints
      ...

    or:

      output:
      <height>
      <width>
      ...

    We keep the original approach: label + (height, width) + rows.
    """
    label = "input:" if is_input else "output:"
    height = len(grid)
    assert height > 0, f"Grid height cannot be 0, got {grid}"
    width = len(grid[0])
    lines = [label, f"{height}", f"{width}"]
    for row in grid:
        assert len(row) == width, f"Inconsistent row width in grid: expected {width}, got {len(row)}."
        row_str = " ".join(str(x) for x in row)
        lines.append(row_str)
    return "\n".join(lines)


########################################
# Training Dataset
########################################
class TrainDataset(Dataset):
    """
    A pseudo-infinite training dataset:
      - tasks_dict: {task_id: [ {input:2D, output:2D}, ... ] }
      - We define __len__ = total_steps,
        but each __getitem__ is just a placeholder (0).
      - We do random sampling in collate_fn_train.
    """
    def __init__(
        self,
        tasks_dict: Dict[str, List[Dict[str, Any]]],
        encoder_tokenizer,
        decoder_tokenizer,
        total_steps: int,
        min_prefix: int,
        max_prefix: int,
        max_seq_len: int,
        augment_ratio: float,
        seed: int,
        compact_grids: bool,
        num_virtual_tokens: int,
    ):
        self.tasks_dict = tasks_dict
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.all_task_ids = list(tasks_dict.keys())
        self._length = total_steps
        self.min_prefix = min_prefix
        self.max_prefix = max_prefix
        self.max_seq_len = max_seq_len
        self.augmenters = get_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)
        self.augment_ratio = augment_ratio
        self.rng = np.random.RandomState(seed)
        self.compact_grids = compact_grids
        self.num_virtual_tokens = num_virtual_tokens

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # We'll do random sampling in the collate fn
        return 0


def remove_val_from_1dtokenized(tokenized, val):
    mask = tokenized['input_ids'][0] != val
    tokenized['input_ids'] = tokenized['input_ids'][0][mask][None, ...]
    tokenized['attention_mask'] = tokenized['attention_mask'][0][mask][None, ...]


def collate_fn_train(batch, dataset: "TrainDataset", debug_fixed_train_order: bool):
    """
    We'll produce len(batch) examples. For each 2 items, we pick 1 random task
    and produce 2 examples from that task, or if it's too large => skip.

    Implementation details:
      - batch_size = len(batch). Must be even (assert).
      - We'll fill a list `out_list` with exactly batch_size items.
      - Each iteration, we pick 1 random task, try to produce 2 examples.
      - If EITHER example is invalid (too large, etc.), we skip BOTH and pick a new random task.
      - Once we've collected `batch_size` items, we do the final PADDING inside this function.
    """
    batch_size = len(batch)
    assert batch_size > 0 and batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    del batch  # we don't use it directly

    encoder_space_token_id = dataset.encoder_tokenizer(" ")['input_ids'][1]
    decoder_space_token_id = dataset.decoder_tokenizer(" ")['input_ids'][1]

    out_list = []

    while len(out_list) < batch_size:
        # Pick a random task
        task_id = dataset.rng.choice(dataset.all_task_ids)
        pairs_for_task = dataset.tasks_dict[task_id]

        # We'll attempt to produce exactly 2 examples from this single task
        task_out_list = []
        for _ in range(2):
            prefix_count = random.randint(dataset.min_prefix, dataset.max_prefix)
            required_count = prefix_count + 1
            if len(pairs_for_task) < required_count:
                break

            # sample task
            chosen_pairs = pairs_for_task[:required_count]
            if not debug_fixed_train_order:
                chosen_pairs = dataset.rng.choice(pairs_for_task, size=required_count, replace=False)
            chosen_pairs = copy.deepcopy(chosen_pairs)

            # apply augmentation
            if dataset.rng.rand() < dataset.augment_ratio:
                augmenter = dataset.rng.choice(dataset.augmenters)
                io_augmentation_choice = dataset.rng.choice(['input_only', 'output_only', 'both'])
                for pair in chosen_pairs:
                    if io_augmentation_choice in ['input_only', 'both']:
                        pair['input'] = augmenter.apply_to_grid(np.array(pair['input']), dataset.rng)
                    if io_augmentation_choice in ['output_only', 'both']:
                        pair['output'] = augmenter.apply_to_grid(np.array(pair['output']), dataset.rng)

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
            encoder_text = "\n".join(prefix_texts) + "[CLS]" * dataset.num_virtual_tokens

            dec_in_text  = grid_to_text(test_pair["input"], True)
            dec_out_text = grid_to_text(test_pair["output"], False)

            # tiny optimization to include output:\n in input
            assert dec_out_text.startswith("output:\n")
            dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output:\n")]
            dec_out_text = dec_out_text[len("output:\n"):]
            dec_out_text += "<|eot_id|>"

            enc_tokens = dataset.encoder_tokenizer(encoder_text, return_tensors="pt", truncation=False)
            assert enc_tokens["input_ids"][0][-dataset.num_virtual_tokens:].tolist() == [dataset.encoder_tokenizer.cls_token_id] * dataset.num_virtual_tokens
            dec_in_tokens  = dataset.decoder_tokenizer(dec_in_text, return_tensors="pt", truncation=False)
            dec_out_tokens = dataset.decoder_tokenizer(dec_out_text, return_tensors="pt", truncation=False)
            assert dec_out_tokens["input_ids"][0][-1].item() == dataset.decoder_tokenizer.eos_token_id
            # remove begin of sentence of dec_out_tokens
            dec_out_tokens['input_ids'] = dec_out_tokens['input_ids'][:, 1:]
            dec_out_tokens['attention_mask'] = dec_out_tokens['attention_mask'][:, 1:]

            # compact grids (optional)
            if dataset.compact_grids:
                remove_val_from_1dtokenized(enc_tokens, encoder_space_token_id)
                remove_val_from_1dtokenized(dec_in_tokens, decoder_space_token_id)
                remove_val_from_1dtokenized(dec_out_tokens, decoder_space_token_id)

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
            if enc_tokens["input_ids"].shape[1] > dataset.max_seq_len or decoder_input_ids.shape[0] > dataset.max_seq_len // 2: # dec_len should be short
                break

            example_dict = {
                "encoder_input_ids": enc_tokens["input_ids"].squeeze(0),
                "encoder_attention_mask": enc_tokens["attention_mask"].squeeze(0),
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "decoder_labels": decoder_labels,
            }
            task_out_list.append(example_dict)

        # Check if we got 2 valid items from this task
        if len(task_out_list) == 2:
            if debug_fixed_train_order or not torch.equal(task_out_list[0]['encoder_input_ids'], task_out_list[1]['encoder_input_ids']):
                # Add them to out_list
                out_list.extend(task_out_list)

    # Now we must truncate out_list if we overshoot
    assert len(out_list) == batch_size, f"Should produce exactly {batch_size} items"

    enc_ids  = [x["encoder_input_ids"] for x in out_list]
    enc_mask = [x["encoder_attention_mask"] for x in out_list]
    dec_ids  = [x["decoder_input_ids"] for x in out_list]
    dec_mask = [x["decoder_attention_mask"] for x in out_list]
    dec_labs = [x["decoder_labels"] for x in out_list]

    enc_ids  = pad_sequence_left(enc_ids, padding_value=dataset.encoder_tokenizer.pad_token_id)
    enc_mask = pad_sequence_left(enc_mask, padding_value=0)
    dec_ids  = pad_sequence_left(dec_ids, padding_value=dataset.decoder_tokenizer.pad_token_id)
    dec_mask = pad_sequence_left(dec_mask, padding_value=0)
    dec_labs = pad_sequence_left(dec_labs, padding_value=-100)

    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_labs,
    }


def collate_fn_train_dummy(batch, dummy_seq_enc_len: int, dummy_seq_dec_len: int):
    """
    We'll produce len(batch) examples. For each 2 items, we pick 1 random task
    and produce 2 examples from that task, or if it's too large => skip.

    Implementation details:
      - batch_size = len(batch). Must be even (assert).
      - We'll fill a list `out_list` with exactly batch_size items.
      - Each iteration, we pick 1 random task, try to produce 2 examples.
      - If EITHER example is invalid (too large, etc.), we skip BOTH and pick a new random task.
      - Once we've collected `batch_size` items, we do the final PADDING inside this function.
    """
    batch_size = len(batch)
    assert batch_size > 0 and batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    del batch  # we don't use it directly

    enc_ids = torch.randint(1, 101, (batch_size, dummy_seq_enc_len), dtype=torch.int64, device='cpu')
    enc_mask = torch.full((batch_size, dummy_seq_enc_len), 1, dtype=torch.int64, device='cpu')
    dec_ids = torch.randint(1, 101, (batch_size, dummy_seq_dec_len), dtype=torch.int64, device='cpu')
    dec_mask = torch.full((batch_size, dummy_seq_dec_len), 1, dtype=torch.int64, device='cpu')

    return {
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_labels": dec_ids,
    }


########################################
# Evaluation Dataset
########################################
class EvalDataset:
    """
    Each .json in the directory => 1 sample in dataset.
    The JSON format:
      {
        "train": [ {input:2D, output:2D}, ... ],
        "test": [ {input:2D, output:2D} ]  # single item in this list
      }
    We produce exactly 1 example per file.
    """
    def __init__(
        self,
        eval_dir: str,
        encoder_tokenizer,
        decoder_tokenizer,
        max_seq_len: int,
        keep_ratio: float,
        compact_grids: bool,
        num_virtual_tokens: int,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_seq_len = max_seq_len
        self.keep_ratio = keep_ratio
        self.compact_grids = compact_grids
        self.num_virtual_tokens = num_virtual_tokens

        self.encoder_space_token_id = encoder_tokenizer(" ")['input_ids'][1]
        self.decoder_space_token_id = decoder_tokenizer(" ")['input_ids'][1]

        # get filepaths
        if not os.path.isdir(eval_dir):
            raise FileNotFoundError(f"Eval directory '{eval_dir}' not found.")
        filepaths = []
        for filename in os.listdir(eval_dir):
            if filename.endswith(".json"):
                filepaths.append(os.path.join(eval_dir, filename))
        filepaths.sort()
        filepaths = filepaths[:int(keep_ratio * len(filepaths))]

        # get actual data
        self.data = []
        for filepath in filepaths:
            task_id = Path(filepath).stem
            with open(filepath, "r") as f:
                data = json.load(f)
            for test_i, test_pair in enumerate(data["test"]):
                self.data.append({
                    'task_id': f'{task_id}_{test_i}',
                    'train': data['train'],
                    'test': test_pair,
                })

        # parse data
        self.parsed_data = [self.parse_data(idx) for idx in range(len(self.data))]
        self.parsed_data = [data for data in self.parsed_data if data is not None]
        logger.info(f'filtered data from {len(self.data)} to {len(self.parsed_data)}')

        # print statistics
        # encoder_input_ids_lens = [x['encoder_input_ids'].shape[0] for x in self.parsed_data]
        # decoder_input_ids_lens = [x['decoder_input_ids'].shape[0] for x in self.parsed_data]
        # for x in encoder_input_ids_lens:
        #     if x > 8192:
        #         print(x)
        # import matplotlib.pyplot as plt
        # plt.hist(encoder_input_ids_lens)
        # plt.savefig('encoder.jpg')
        # plt.close()
        # plt.hist(decoder_input_ids_lens)
        # plt.savefig('decoder.jpg')
        # plt.close()

    def __len__(self):
        return len(self.parsed_data)

    def __getitem__(self, idx):
        return self.parsed_data[idx]

    def parse_data(self, idx):
        task_id = self.data[idx]['task_id']
        train_pairs = self.data[idx]['train']
        test_pair = self.data[idx]['test']

        # Build encoder text
        prefix_texts = []
        for p in train_pairs:
            prefix_texts.append(grid_to_text(p["input"], True))
            prefix_texts.append(grid_to_text(p["output"], False))
        encoder_text = "\n".join(prefix_texts) + "[CLS]" * self.num_virtual_tokens

        dec_in_text  = grid_to_text(test_pair["input"], True)
        dec_out_text = grid_to_text(test_pair["output"], False)

        # tiny optimization to include output:\n in input
        assert dec_out_text.startswith("output:\n")
        dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output:\n")]
        dec_out_text = dec_out_text[len("output:\n"):]
        dec_out_text += "<|eot_id|>"

        enc_tokens = self.encoder_tokenizer(encoder_text, return_tensors="pt", truncation=False)
        assert enc_tokens["input_ids"][0][-self.num_virtual_tokens:].tolist() == [self.encoder_tokenizer.cls_token_id] * self.num_virtual_tokens
        dec_in_tokens  = self.decoder_tokenizer(dec_in_text,  return_tensors="pt", truncation=False)
        dec_out_tokens = self.decoder_tokenizer(dec_out_text, return_tensors="pt", truncation=False)
        assert all(t[-1].item() == self.decoder_tokenizer.eos_token_id for t in dec_out_tokens["input_ids"])
        assert dec_out_tokens["input_ids"][0][-1].item() == self.decoder_tokenizer.eos_token_id

        # remove begin of sentence of dec_out_tokens
        dec_out_tokens['input_ids'] = dec_out_tokens['input_ids'][:, 1:]
        dec_out_tokens['attention_mask'] = dec_out_tokens['attention_mask'][:, 1:]

        # compact grids (optional)
        if self.compact_grids:
            remove_val_from_1dtokenized(enc_tokens, self.encoder_space_token_id)
            remove_val_from_1dtokenized(dec_in_tokens, self.decoder_space_token_id)
            remove_val_from_1dtokenized(dec_out_tokens, self.decoder_space_token_id)

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

        dec_label_texts = dec_out_text[:-len("<|eot_id|>")]
        if self.compact_grids:
            dec_label_texts = dec_label_texts.replace(' ', '')

        # Check length
        if enc_tokens["input_ids"].shape[1] > self.max_seq_len or decoder_input_ids.shape[0] > self.max_seq_len // 2: # dec_len should be short
            return None

        return {
            "task_id": task_id,
            "encoder_input_ids": enc_tokens["input_ids"].squeeze(0),
            "encoder_attention_mask": enc_tokens["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_labels": decoder_labels,
            "decoder_gen_input_ids": dec_in_tokens["input_ids"].squeeze(0),
            "decoder_gen_attention_mask": dec_in_tokens["attention_mask"].squeeze(0),
            "decoder_out_token_length": dec_out_tokens["input_ids"].shape[1] - 1, # remove eos token
            "decoder_label_texts": dec_label_texts,  # used for exact match
        }


def collate_fn_eval(batch, encoder_tokenizer, decoder_tokenizer):
    """
    Filter out None, then pad each field. Return the final dict.
    Also store how many valid items we have, for logging purposes.
    """
    task_ids = [x['task_id'] for x in batch]
    enc_ids = [x["encoder_input_ids"] for x in batch]
    enc_mask = [x["encoder_attention_mask"] for x in batch]
    dec_ids = [x["decoder_input_ids"] for x in batch]
    dec_mask = [x["decoder_attention_mask"] for x in batch]
    dec_gen_ids = [x["decoder_gen_input_ids"] for x in batch]
    dec_gen_mask = [x["decoder_gen_attention_mask"] for x in batch]
    dec_labs = [x["decoder_labels"] for x in batch]
    decoder_out_token_length = [x["decoder_out_token_length"] for x in batch]
    decoder_label_texts = [x["decoder_label_texts"] for x in batch]

    enc_ids = pad_sequence_left(enc_ids, padding_value=encoder_tokenizer.pad_token_id)
    enc_mask = pad_sequence_left(enc_mask, padding_value=0)
    dec_ids = pad_sequence_left(dec_ids, padding_value=decoder_tokenizer.pad_token_id)
    dec_mask = pad_sequence_left(dec_mask, padding_value=0)
    dec_gen_ids = pad_sequence_left(dec_gen_ids, padding_value=decoder_tokenizer.pad_token_id)
    dec_gen_mask = pad_sequence_left(dec_gen_mask, padding_value=0)
    dec_labs = pad_sequence_left(dec_labs, padding_value=-100)

    batch_dict = {
        "task_ids": task_ids,
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_gen_input_ids": dec_gen_ids,
        "decoder_gen_attention_mask": dec_gen_mask,
        "decoder_labels": dec_labs,
        "decoder_out_token_length": decoder_out_token_length,
        "decoder_label_texts": decoder_label_texts,
    }
    return batch_dict


def collate_fn_eval_dummy(batch, dummy_seq_enc_len: int, dummy_seq_dec_len: int):
    """
    Filter out None, then pad each field. Return the final dict.
    Also store how many valid items we have, for logging purposes.
    """
    batch_size = len(batch)
    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    enc_ids = torch.randint(1, 101, (batch_size, dummy_seq_enc_len), dtype=torch.int64, device='cpu')
    enc_mask = torch.full((batch_size, dummy_seq_enc_len), 1, dtype=torch.int64, device='cpu')
    dec_ids = torch.randint(1, 101, (batch_size, dummy_seq_dec_len), dtype=torch.int64, device='cpu')
    dec_mask = torch.full((batch_size, dummy_seq_dec_len), 1, dtype=torch.int64, device='cpu')
    dec_gen_ids = torch.randint(1, 101, (batch_size, int(dummy_seq_dec_len * 0.4)), dtype=torch.int64, device='cpu')
    dec_gen_mask = torch.full((batch_size, int(dummy_seq_dec_len * 0.4)), 1, dtype=torch.int64, device='cpu')

    batch_dict = {
        "task_ids": task_ids,
        "encoder_input_ids": enc_ids,
        "encoder_attention_mask": enc_mask,
        "decoder_input_ids": dec_ids,
        "decoder_attention_mask": dec_mask,
        "decoder_gen_input_ids": dec_gen_ids,
        "decoder_gen_attention_mask": dec_gen_mask,
        "decoder_labels": dec_ids,
        "decoder_label_texts": ['helloworld'] * batch_size,
        "decoder_out_token_length": [math.ceil(dummy_seq_dec_len * 0.6)] * batch_size,
    }
    return batch_dict
