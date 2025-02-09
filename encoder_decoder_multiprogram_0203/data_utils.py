from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ast
import hashlib
import glob
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
    def __init__(self, tokens: List[str], bos_token: str, eos_token: str, pad_token: str):
        assert eos_token in tokens
        assert bos_token in tokens
        assert len(set(tokens)) == len(tokens)

        # mapping
        self.id_to_token = {i: token for i, token in enumerate(tokens)}
        self.token_to_id = {token: i for i, token in enumerate(tokens)}

        # special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.bos_token_id = self.token_to_id[bos_token]
        self.eos_token_id = self.token_to_id[eos_token]
        self.pad_token_id = self.token_to_id[pad_token]
        self.special_token_ids = set([self.bos_token_id, self.eos_token_id, self.pad_token_id])

    # def __call__(self, text: str, add_bos: bool, add_eos: bool) -> List[int]:
    #     input_ids = [self.bos_token_id] if add_bos else []

    #     while text:
    #         # special tokens
    #         if text.startswith(self.eos_token):
    #             input_ids.append(self.eos_token_id)
    #             text = text[len(self.eos_token):]
    #         elif text.startswith(self.bos_token):
    #             input_ids.append(self.bos_token_id)
    #             text = text[len(self.bos_token):]
    #         elif text.startswith(self.pad_token):
    #             input_ids.append(self.pad_token_id)
    #             text = text[len(self.pad_token):]
    #         # input output
    #         elif text.startswith("input"):
    #             input_ids.append(self.token_to_id["input"])
    #             text = text[5:]
    #         elif text.startswith("output"):
    #             input_ids.append(self.token_to_id["output"])
    #             text = text[6:]
    #         # double digit
    #         elif text[:2] in self.token_to_id:
    #             input_ids.append(self.token_to_id[text[:2]])
    #             text = text[2:]
    #         # single digit
    #         elif text[0] in self.token_to_id:
    #             input_ids.append(self.token_to_id[text[0]])
    #             text = text[1:]
    #         else:
    #             raise ValueError(f"cannot tokenize: {text}")

    #     if add_eos:
    #         input_ids.append(self.eos_token_id)

    #     return input_ids

    # def encode_to_tensor(self, text: str, add_bos: bool, add_eos: bool) -> torch.Tensor:
    #     input_ids = self(text, add_bos, add_eos)
    #     return torch.tensor(input_ids, dtype=torch.int64)

    def encode_dimensions_to_tensor(self, height: int, width: int) -> torch.Tensor:
        return torch.tensor(
            [self.token_to_id[str(height)],
             self.token_to_id[str(width)],
             self.token_to_id["\n"]],
            dtype=torch.int64)

    def encode_grid_to_tensor(self, grid: np.ndarray) -> torch.Tensor:
        assert grid.ndim == 2
        token_ids = []
        for row in grid:
            for x in row:
                token_ids.append(self.token_to_id[str(x)])
            token_ids.append(self.token_to_id["\n"])
        token_ids = token_ids[:-1] # no \n at end
        return torch.tensor(token_ids, dtype=torch.int64)

    def convert_token_to_id(self, token: str) -> torch.Tensor:
        return torch.tensor([self.token_to_id[token]], dtype=torch.int64)

    def get_input_and_output_grid_ids(self, example: Example, add_bos: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dim_ids = self.encode_dimensions_to_tensor(len(example.input), len(example.input[0]))
        output_dim_ids = self.encode_dimensions_to_tensor(len(example.output), len(example.output[0]))
        input_grid_ids = self.encode_grid_to_tensor(example.input)
        output_grid_ids = self.encode_grid_to_tensor(example.output)

        # input_grid_ids should contain everything except the output_grid_ids
        input_grid_ids = torch.cat([
            self.convert_token_to_id("input"),
            input_dim_ids,
            input_grid_ids,
            self.convert_token_to_id("output"),
        ])
        if add_bos:
            input_grid_ids = torch.cat([self.convert_token_to_id(self.bos_token), input_grid_ids])

        # output_grid_ids should contain a eos
        output_grid_ids = torch.cat([
            output_dim_ids,
            output_grid_ids,
            self.convert_token_to_id(self.eos_token),
        ])
        return input_grid_ids, output_grid_ids

    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor],
            skip_special_tokens: bool = False,
            extra_id_to_token: Dict[int, str] = {}
        ) -> str:
        tokens = []
        for token_id in token_ids:
            if not skip_special_tokens or (int(token_id) not in self.special_token_ids):
                token_id = int(token_id)
                if token_id not in self.id_to_token:
                    tokens.append(extra_id_to_token[token_id])
                else:
                    tokens.append(self.id_to_token[token_id])
        return "".join(tokens)

    def get_grid_dimensions(self, token_ids: Union[List[int], torch.Tensor]) -> List[Tuple[int, int]]:
        dimensions = []
        for i, token_id in enumerate(token_ids):
            if token_id in [self.token_to_id["input"], self.token_to_id["output"]]:
                height = self.id_to_token[int(token_ids[i+1])]
                width = self.id_to_token[int(token_ids[i+2])]
                dimensions.append((height, width))
        return dimensions

    def batch_decode(self, batch_token_ids: torch.Tensor, skip_special_tokens: bool) -> List[str]:
        assert batch_token_ids.dim() == 2
        texts = []
        for token_ids in batch_token_ids:
            text = self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            texts.append(text)
        return texts


def get_d8_augmenters() -> List[Augmenter]:
    return [
        Rotate(0),
        Rotate(90),
        Rotate(180),
        Rotate(270),
        Flip(0),
        Flip(1),
        Chain([Flip(0), Rotate(90)]), # type: ignore
        Chain([Flip(1), Rotate(90)]), # type: ignore
    ]


def get_mit_augmenters(
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


def parse_input_output_grids(s: str, dimensions: List[Tuple[int, int]]) -> List[Dict[str, List[List[int]]]]:
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    result = []
    i = 0
    n = len(lines)
    dim_i = 0

    while i < n:
        # get input grid
        assert lines[i].startswith("input")
        lines[i] = lines[i][len("input"):]
        input_height_width = lines[i]; i += 1
        input_height, input_width = dimensions[dim_i]
        dim_i += 1
        assert f"{input_height}{input_width}" == input_height_width, (f"{input_height}{input_width}", input_height_width)
        input_height, input_width = int(input_height), int(input_width)
        # read input grid rows
        input_grid = []
        for row_i in range(input_height):
            if row_i < input_height - 1:
                row = [int(x) for x in lines[i]]
                i += 1
            else:
                row = [int(x) for x in lines[i][:lines[i].find("output")]]
                lines[i] = lines[i][lines[i].find("output"):]
            input_grid.append(row)
        # verify input dimensions
        assert len(input_grid) == input_height
        assert all(len(row) == input_width for row in input_grid)

        # get output grid
        assert lines[i].startswith("output")
        lines[i] = lines[i][len("output"):]
        output_height_width = lines[i]; i += 1
        output_height, output_width = dimensions[dim_i]
        dim_i += 1
        assert f"{output_height}{output_width}" == output_height_width, (f"{output_height}{output_width}", output_height_width)
        output_height, output_width = int(output_height), int(output_width)
        # read output grid rows
        output_grid = []
        for row_i in range(output_height):
            if row_i < output_height - 1 or "input" not in lines[i]:
                row = [int(x) for x in lines[i]]
                i += 1
            else:
                row = [int(x) for x in lines[i][:lines[i].find("input")]]
                lines[i] = lines[i][lines[i].find("input"):]
            output_grid.append(row)
        # verify output dimensions
        assert len(output_grid) == output_height
        assert all(len(row) == output_width for row in output_grid)

        # add the pair to the result
        result.append({"input": input_grid, "output": output_grid})

    assert dim_i == len(dimensions)
    return result


def pad_grid_to_30x30(grid: List[List[int]], padding_value: int = 10) -> List[List[int]]:
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    assert rows <= 30 and cols <= 30
    # Calculate padding needed on each side
    pad_top = (30 - rows) // 2
    pad_bottom = 30 - rows - pad_top
    pad_left = (30 - cols) // 2
    pad_right = 30 - cols - pad_left
    # Pad the grid with the specified value
    padded_grid = np.pad(
        grid_array,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=padding_value,
    )
    return padded_grid.tolist()


def visualize_task(
        task: Union[Task, List[Dict[str, List]]],
        name: str = "",
        out_path: str = "temp.jpg",
    ) -> None:
    # do some parsing
    grids_row1 = None
    grids_row2 = None
    if isinstance(task, Task):
        assert task.name == name
        grids_row1 = []
        for e in task.train_examples:
            grids_row1.append(e.input)
            grids_row1.append(e.output)
        grids_row2 = [task.test_example.input, task.test_example.output]
    elif isinstance(task, list):
        grids_row1 = []
        for t in task[:-1]:
            grids_row1.append(t["input"])
            grids_row1.append(t["output"])
        grids_row2 = [task[-1]["input"], task[-1]["output"]]
    else:
        raise ValueError(f"unrecognized task type")

    grids_row1 = [pad_grid_to_30x30(grid) for grid in grids_row1]
    grids_row2 = [pad_grid_to_30x30(grid) for grid in grids_row2]

    color_map_list = [
        "#000000", # black
        "#0074D9", # blue
        "#FF4136", # red
        "#2ECC40", # green
        "#FFDC00", # yellow
        "#AAAAAA", # grey
        "#F012BE", # fuschia
        "#FF851B", # orange
        "#7FDBFF", # teal
        "#870C25", # brown
        "#ffffff", # white (background)
    ]
    cmap = ListedColormap(color_map_list)
    n_cols = max(len(grids_row1), len(grids_row2))
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 8))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    # Plot each grid in the first row
    for i, grid in enumerate(grids_row1):
        ax = axes[0, i]
        ax.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=10)
        ax.axis("off")
    if len(grids_row1) < n_cols:
        for j in range(len(grids_row1), n_cols):
            axes[0, j].axis("off")
    # Plot each grid in the second row
    for i, grid in enumerate(grids_row2):
        ax = axes[1, i]
        ax.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=10)
        ax.axis("off")
    if len(grids_row2) < n_cols:
        for j in range(len(grids_row2), n_cols):
            axes[1, j].axis("off")
    # format
    fig.suptitle(name, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def load_concept_arc() -> Dict[str, List[Dict[str, List[List[int]]]]]:
    files = glob.glob(f'/scratch/zy3101/ConceptARC/corpus/*/*.json')
    files += glob.glob(f'/scratch/zy3101/ConceptARC/MinimalTasks/*.json')
    tasks_dict = {}
    for data_i, file in enumerate(files):
        data = json.load(open(file, 'r'))
        tasks_dict[f"concept{data_i}"] = data["train"] + data["test"]
    return tasks_dict


def load_train_original_data_from_dir(data_dir: str) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    tasks_dict = {}
    for file in glob.glob(f"{data_dir}/*.json"):
        data = json.load(open(file, 'r'))
        tasks_dict[Path(file).stem] = data["train"] + data["test"]
    return tasks_dict


# def grid_to_text(grid: np.ndarray, is_input: bool) -> str:
#     assert grid.ndim == 2
#     height = len(grid)
#     width = len(grid[0])
#     lines = [f"{height}", f"{width}"]
#     for row in grid:
#         assert len(row) == width
#         row_str = "".join(str(x) for x in row)
#         lines.append(row_str)
#     assert len(lines) == height
#     # final parsed
#     header = "input" if is_input else ""
#     footer = "output" if is_input else ""
#     parsed = header + "\n".join(lines) + footer
#     return parsed


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
        self.augmenters: List[Augmenter] = get_mit_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)
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

        # tiny optimization to include output\n in input
        assert dec_out_text.startswith("output\n")
        dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output\n")]
        dec_out_text = dec_out_text[len("output\n"):]
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
                    p1 += 2 # remove output and \n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
            elif self.encoder_loss_type == "rest":
                if not is_first:
                    p1 += 2 # remove output and \n
                    p2 -= not is_last # remove \n
                    encoder_labels[p1:p2] = copy.deepcopy(encoder_input_ids[p1:p2])
            else:
                p1 += 2 # remove output and \n
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
        re_arc_ratio: float,
        concept_arc_ratio: float,
        arc_heavy_ratio: float,
        encoder_tokenizer: ARCTokenizer,
        decoder_tokenizer: ARCTokenizer,
        total_steps: int,
        max_seq_len: int,
        extra_augment_ratio: float,
        extra_augment_single_grid: bool,
        seed: int,
        process_index: int,
        ntokens: int,
        debug_fixed_order: bool,
        debug_random_pad: bool,
        debug_pad_len: int,
        encoder_pad_side: str,
        decoder_pad_side: str,
        encoder_loss_type: bool,
        anti_invar_ratio: float,
        debug_train_data: bool,
        no_color_permute: bool,
        no_pair_permute: bool,
        no_d8: bool,
        max_num_sample_program: int,
        min_num_pair_for_program: int,
        no_train_original: bool,
        only_train_original: bool,
        debug_enc_len: int,
        debug_dec_len: int,
        max_num_train_program: int,
        num_workers: int,
    ):
        self.re_arc_ratio = re_arc_ratio
        self.concept_arc_ratio = concept_arc_ratio
        self.arc_heavy_ratio = arc_heavy_ratio

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self._length = total_steps
        self.max_seq_len = max_seq_len
        self.extra_augment_ratio = extra_augment_ratio
        self.extra_augment_single_grid = extra_augment_single_grid
        self.ntokens = ntokens
        self.debug_fixed_order = debug_fixed_order
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.encoder_pad_side = encoder_pad_side
        self.decoder_pad_side = decoder_pad_side
        self.encoder_loss_type = encoder_loss_type
        self.anti_invar_ratio = anti_invar_ratio
        self.debug_train_data = debug_train_data
        self.no_color_permute = no_color_permute
        self.no_pair_permute = no_pair_permute
        self.no_d8 = no_d8
        self.max_num_sample_program = max_num_sample_program
        self.min_num_pair_for_program = min_num_pair_for_program
        self.debug_enc_len = debug_enc_len
        self.debug_dec_len = debug_dec_len
        self.max_num_train_program = max_num_train_program

        # setup args
        self.normalized_ratio = np.array([self.re_arc_ratio, self.concept_arc_ratio, self.arc_heavy_ratio])
        self.normalized_ratio /= np.sum(self.normalized_ratio)
        self.d8_augmenters = get_d8_augmenters()
        self.extra_augmenters = get_mit_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)

        # seed and process_index
        if num_workers == 0:
            self.rngs = [np.random.RandomState(seed + process_index)]
        else:
            self.rngs = [np.random.RandomState(seed + i) for i in range(num_workers * process_index, num_workers * (process_index + 1))]

        # load re-arc data + train original data
        if only_train_original:
            re_arc_task_id_to_pairs = load_train_original_data_from_dir(eval_train_dir)
        elif no_train_original:
            re_arc_task_id_to_pairs = load_re_arc_from_data_dir(train_data_dir)
        else:
            re_arc_task_id_to_pairs = load_re_arc_from_data_dir(train_data_dir)
            train_original_task_id_to_pairs = load_train_original_data_from_dir(eval_train_dir)
            assert set(re_arc_task_id_to_pairs.keys()) == set(train_original_task_id_to_pairs.keys())
            for task_id in re_arc_task_id_to_pairs:
                re_arc_task_id_to_pairs[task_id] += train_original_task_id_to_pairs[task_id]
        self.arc_train_id_to_pairs = re_arc_task_id_to_pairs
        logger.info(f'loaded {len(self.arc_train_id_to_pairs)} re-arc/train-original tasks, total \
                    {sum(len(x) for x in self.arc_train_id_to_pairs.values())} pairs')

        # load concept-arc data
        self.concept_arc_task_id_to_pairs = {}
        if concept_arc_ratio > 0.0:
            self.concept_arc_task_id_to_pairs = load_concept_arc()
            logger.info(f'loaded {len(self.concept_arc_task_id_to_pairs)} concept-arc tasks')

        # load heavy-arc data
        self.heavy_arc_data = {}
        if arc_heavy_ratio > 0.0:
            self.heavy_arc_data = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")["train"] # type: ignore
            logger.info(f'loaded {len(self.heavy_arc_data)} arc-heavy tasks')

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
            assert set(self.task_id_to_hash.keys()) == set(self.arc_train_id_to_pairs.keys())

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

    def augment_and_count_length(
            pair: Dict[str, List[List[int]]],
            d8_augmenter: Optional[Augmenter],
            extra_augmenter: Optional[Augmenter],
            io_augmentation_choice: str,
        ) -> Tuple[int, int, Dict[str, np.ndarray]]:

        assert set(pair.keys()) == {"input", "output"}
        np_pair = {
            "input": np.array(copy.deepcopy(pair["input"])).astype(int),
            "output": np.array(copy.deepcopy(pair["output"])).astype(int),
        }
        # apply d8 augmentation
        if d8_augmenter is not None:
            np_pair['input'] = d8_augmenter.apply_to_grid(np_pair['input'], rng)
            np_pair['output'] = d8_augmenter.apply_to_grid(np_pair['output'], rng)
        # apply extra augmentation
        if extra_augmenter is not None:
            if io_augmentation_choice in ['input_only', 'both']:
                np_pair['input'] = extra_augmenter.apply_to_grid(np_pair['input'], rng)
            if io_augmentation_choice in ['output_only', 'both']:
                np_pair['output'] = extra_augmenter.apply_to_grid(np_pair['output'], rng)
        else:
            assert io_augmentation_choice is None

        input_h, input_w, output_h, output_w = len(np_pair['input']), len(np_pair['input'][0]), len(np_pair['output']), len(np_pair['output'][0])
        max_dim = max(input_h, input_w, output_h, output_w)

        pair_token_len = input_h * (input_w + 1) + output_h * (output_w + 1) - 2 # cells and \n
        pair_token_len += 3 # input, output, eos
        pair_token_len += 6 # hw\n for both input and output

        return max_dim, pair_token_len, np_pair

    out_list = []
    while len(out_list) < batch_size:
        two_task_list = []
        dataset_name = rng.choice(["re-arc", "concept-arc", "arc-heavy"], p=dataset.normalized_ratio)

        # some tasks are just not suitable? skip them
        while len(two_task_list) < 2:

            # STEP 1: get antiinvar, 2 all pairs and 2 task ids
            if dataset_name == "re-arc":
                anti_invar = rng.rand() < dataset.anti_invar_ratio
                anti_invar = int(anti_invar)

                if anti_invar:
                    # for anti-invariance, tasks need to be different
                    while True:
                        task_id_2 = rng.choice(list(dataset.arc_train_id_to_pairs.keys()), size=2, replace=False)
                        if dataset.task_id_to_hash[task_id_2[0]] != dataset.task_id_to_hash[task_id_2[1]]:
                            break
                    all_pairs_2 = [
                        dataset.arc_train_id_to_pairs[task_id_2[0]],
                        dataset.arc_train_id_to_pairs[task_id_2[1]],
                    ]
                else:
                    # for invariance, task need to be same
                    task_id = rng.choice(list(dataset.arc_train_id_to_pairs.keys()))
                    task_id_2 = [task_id, task_id]
                    all_pairs_2 = [
                        dataset.arc_train_id_to_pairs[task_id],
                        dataset.arc_train_id_to_pairs[task_id],
                    ]

            elif dataset_name == "concept-arc":
                anti_invar = 2
                # just sample two random tasks (same or different)
                task_id_2 = rng.choice(list(dataset.concept_arc_task_id_to_pairs.keys()), size=2, replace=True)
                all_pairs_2 = [
                    dataset.concept_arc_task_id_to_pairs[task_id_2[0]],
                    dataset.concept_arc_task_id_to_pairs[task_id_2[1]],
                ]

            else:
                anti_invar = 2
                # just sample two random tasks (same or different)
                task_id_2 = []
                all_pairs_2 = []
                for idx in rng.choice(list(range(len(dataset.heavy_arc_data))), size=2, replace=True).tolist():
                    pairs = dataset.heavy_arc_data[idx]["examples"]
                    assert all(len(pair) == 2 for pair in pairs)
                    pairs = [{"input": pair[0], "output": pair[1]} for pair in pairs]
                    task_id_2.append(f"heavy{idx}")
                    all_pairs_2.append(pairs)

            assert anti_invar in [0, 1, 2]
            assert len(task_id_2) == 2
            assert len(all_pairs_2) == 2


            # STEP 2: decide on extra augmentations and io augmentation choice for pairs
            d8_augmenter_2 = [] # apply early because it affects input_id length
            extra_augmenter_2 = []
            io_augmentation_choice_2 = [] # only for extra augmentation

            if anti_invar == 0:
                # same augmenter for both tasks
                basic_augmenter = rng.choice(dataset.d8_augmenters) # type: ignore
                d8_augmenter_2 = [basic_augmenter, basic_augmenter]
                if rng.rand() < dataset.extra_augment_ratio:
                    augmenter = rng.choice(dataset.extra_augmenters) # type: ignore
                    io_augmentation_choice = rng.choice(["input_only", "output_only", "both"]) if dataset.extra_augment_single_grid else "both"
                    extra_augmenter_2 = [augmenter, augmenter]
                    io_augmentation_choice_2 = [io_augmentation_choice, io_augmentation_choice]
                else:
                    extra_augmenter_2 = [None, None]
                    io_augmentation_choice_2 = [None, None]
            else:
                # can be a different augmenter for each task
                d8_augmenter_2 = rng.choice(dataset.d8_augmenters, size=2, replace=True) # type: ignore
                for _ in range(2):
                    if rng.rand() < dataset.extra_augment_ratio:
                        augmenter = rng.choice(dataset.extra_augmenters) # type: ignore
                        extra_augmenter_2.append(augmenter)
                        io_augmentation_choice = rng.choice(["input_only", "output_only", "both"]) if dataset.extra_augment_single_grid else "both"
                        io_augmentation_choice_2.append(io_augmentation_choice)
                    else:
                        extra_augmenter_2.append(None)
                        io_augmentation_choice_2.append(None)

            if dataset.no_d8:
                d8_augmenter_2 = [None, None]

            assert len(d8_augmenter_2) == 2
            assert len(extra_augmenter_2) == 2
            assert len(io_augmentation_choice_2) == 2


            # STEP 3: choose pairs and parse until maxing out sequence length
            decoder_pair_2 = []
            encoder_pairs_2 = []
            total_encoder_npair_to_len_2 = []
            total_decoder_len_2 = []

            for all_pairs, d8_augmenter, extra_augmenter, io_augmentation_choice in zip(all_pairs_2, d8_augmenter_2, extra_augmenter_2, io_augmentation_choice_2):
                # print('length all pairs', len(all_pairs))

                # indices to sample from, make sure we have at least enough pairs for an encoder program and a decoder pair
                available_idxs = list(range(len(all_pairs)))
                if not dataset.debug_fixed_order:
                    rng.shuffle(available_idxs)
                if len(available_idxs) < dataset.min_num_pair_for_program + 1:
                    break

                # STEP 3.1: get decoder input, quit if too big
                # print('selected pair', available_idxs[-1], 'for decoder')
                total_decoder_len = 1 # bos
                decoder_max_dim, decoder_pair_token_len, decoder_pair = augment_and_count_length(
                    pair=all_pairs[available_idxs[-1]],
                    d8_augmenter=d8_augmenter,
                    extra_augmenter=extra_augmenter,
                    io_augmentation_choice=io_augmentation_choice,
                )
                total_decoder_len += decoder_pair_token_len
                if decoder_max_dim > 30 or total_decoder_len > dataset.max_seq_len // 2:
                    break
                available_idxs = available_idxs[:-1]

                # STEP 3.2: get encoder input, sample up to either max num program, exhaust all pairs, grid too big, or sequence too long
                # note grid too big results in completely resampling from the chosen dataset
                encoder_pairs = [] # list of input ids for each grid, no bos
                big_grid_failure = False
                total_encoder_npair_to_len = [1] # just bos when npair == 0

                for idx in available_idxs[:dataset.max_num_sample_program + dataset.min_num_pair_for_program - 1]:
                    # print('selected pair', idx, 'for encoder')
                    encoder_max_dim, encoder_pair_token_len, encoder_pair = augment_and_count_length(
                        d8_augmenter=d8_augmenter,
                        pair=all_pairs[idx],
                        extra_augmenter=extra_augmenter,
                        io_augmentation_choice=io_augmentation_choice,
                    )
                    # if grid too big, quit
                    if encoder_max_dim > 30:
                        big_grid_failure = True
                        break
                    # if we can add it, add it, otherwise stop
                    if total_encoder_npair_to_len[-1] + encoder_pair_token_len > dataset.max_seq_len:
                        break
                    # update
                    total_encoder_npair_to_len.append(total_encoder_npair_to_len[-1] + encoder_pair_token_len)
                    encoder_pairs.append(encoder_pair)

                # encoder should have at least enough pairs for a program
                if big_grid_failure or len(encoder_pairs) < dataset.min_num_pair_for_program:
                    break

                # update
                decoder_pair_2.append(decoder_pair)
                encoder_pairs_2.append(encoder_pairs)
                total_encoder_npair_to_len_2.append(total_encoder_npair_to_len)
                total_decoder_len_2.append(total_decoder_len)

            if len(decoder_pair_2) < 2:
                two_task_list = []
                continue

            # limit nprogram of first to be same as second
            # n_pairs = min(len(encoder_pairs) for encoder_pairs in encoder_pairs_2)
            # encoder_pairs_2 = [pairs[:n_pairs] for pairs in encoder_pairs_2]
            # total_encoder_len_2 = [pair_to_len[n_pairs] for pair_to_len in total_encoder_npair_to_len_2]

            # get program_idxs_2 (sorted and same length) and limit num pairs
            program_idxs_2 = []
            program_idx_choices_2 = []
            for i, encoder_pairs in enumerate(encoder_pairs_2):
                program_idx_choices = list(range(dataset.min_num_pair_for_program - 1, len(encoder_pairs)))
                program_idx_choices_2.append(program_idx_choices)
            num_trainable_program = min([len(program_idx_choices) for program_idx_choices in program_idx_choices_2])
            for i, program_idx_choices in enumerate(program_idx_choices_2):
                program_idxs = rng.choice(program_idx_choices, size=min(dataset.max_num_train_program, num_trainable_program), replace=False).tolist()
                program_idxs.sort()
                program_idxs_2.append(program_idxs)
            assert len(program_idxs_2[0]) == len(program_idxs_2[1]) <= dataset.max_num_train_program
            assert all(i + 1 >= dataset.min_num_pair_for_program for i in program_idxs_2[0] + program_idxs_2[1])

            # update encoder lengths for debugging
            total_encoder_len_2 = []
            for i, (program_idxs, encoder_pairs, pair_to_len) in enumerate(zip(program_idxs_2, encoder_pairs_2, total_encoder_npair_to_len_2)):
                encoder_pairs_2[i] = encoder_pairs[:program_idxs[-1] + 1] # shorten num encoder pairs to not waste encoder context
                total_encoder_len_2.append(pair_to_len[len(encoder_pairs_2[i])])


            # STEP 4: convert to tasks, apply basic augmentations
            task_2 = [Task(
                name="dummy",
                train_examples=[
                    Example(input=pair["input"], output=pair["output"])
                    for pair in encoder_pairs
                ],
                test_example=Example(input=decoder_pair["input"], output=decoder_pair["output"]),
            ) for encoder_pairs, decoder_pair in zip(encoder_pairs_2, decoder_pair_2)]

            # permute examples and permute colors require full tasks, so put here, they do not affect input ids length
            if not dataset.no_pair_permute:
                # always apply pair permutation
                task_2 = [PermuteExamples().apply_to_task(task, to_input=True, to_output=True, rng=rng) for task in task_2]

            if not dataset.no_color_permute:
                # always apply color permutation (same color map if invariance)
                if anti_invar == 0:
                    augmenter = PermuteColors()
                    task_2[0] = augmenter.apply_to_task(task_2[0], to_input=True, to_output=True, rng=rng)
                    color_mapper = augmenter.color_mapper
                    task_2[1] = augmenter.apply_to_task(task_2[1], to_input=True, to_output=True, rng=rng, color_mapper=color_mapper)
                else:
                    task_2 = [PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng) for task in task_2]

            # STEP 5: tokenize and format to output
            for task_id, task, program_idxs, extra_augmenter, total_encoder_len, total_decoder_len in zip(task_id_2, task_2, program_idxs_2, extra_augmenter_2, total_encoder_len_2, total_decoder_len_2):

                # get encoder_input_ids, encoder_label_ids, encoder_program_starts, encoder_attention_mask
                # use dummy ids for program tokens
                encoder_input_ids = []
                encoder_label_ids = []
                all_encoder_program_starts = []
                trainable_encoder_program_starts = []

                program_input_ids = torch.full((dataset.ntokens,), dataset.encoder_tokenizer.pad_token_id, dtype=torch.int64)
                for example_i, example in enumerate(task.train_examples):
                    input_grid_ids, output_grid_ids = dataset.encoder_tokenizer.get_input_and_output_grid_ids(
                        example=example,
                        add_bos=(example_i == 0),
                    )

                    # update encoder_input_ids
                    encoder_input_ids.append(input_grid_ids)
                    encoder_input_ids.append(output_grid_ids)

                    # update encoder_label_ids (mask input, optinally unmask output)
                    encoder_label_ids.append(torch.full(input_grid_ids.shape, -100, dtype=torch.int64))
                    if (dataset.encoder_loss_type == "all") or \
                        (dataset.encoder_loss_type == "rest" and example_i > 0) or \
                        (dataset.encoder_loss_type == "last" and example_i == len(task.train_examples) - 1):
                        encoder_label_ids.append(output_grid_ids)
                    else:
                        encoder_label_ids.append(torch.full(output_grid_ids.shape, -100, dtype=torch.int64))

                    # program tokens are based on min_num_pair_for_program
                    if example_i + 1 >= dataset.min_num_pair_for_program:
                        program_start_idx = sum(len(x) for x in encoder_input_ids)
                        all_encoder_program_starts.append(program_start_idx)
                        encoder_input_ids.append(program_input_ids)
                        encoder_label_ids.append(torch.full(program_input_ids.shape, -100, dtype=torch.int64))
                        if example_i in program_idxs:
                            trainable_encoder_program_starts.append(program_start_idx)

                encoder_input_ids = torch.cat(encoder_input_ids)
                encoder_label_ids = torch.cat(encoder_label_ids)
                assert encoder_input_ids.shape == encoder_label_ids.shape
                encoder_attention_mask = torch.full(encoder_input_ids.shape, 1, dtype=torch.int64)

                # get decoder_input_ids, decoder_label_ids, decoder_attention_mask
                input_grid_ids, output_grid_ids = dataset.decoder_tokenizer.get_input_and_output_grid_ids(
                    example=task.test_example,
                    add_bos=True,
                )
                decoder_input_ids = torch.cat([input_grid_ids, output_grid_ids])
                decoder_label_ids = torch.cat([torch.full(input_grid_ids.shape, -100, dtype=torch.int64),
                                               output_grid_ids])
                assert decoder_input_ids.shape == decoder_label_ids.shape
                decoder_attention_mask = torch.full(decoder_input_ids.shape, 1, dtype=torch.int64)

                # assert lengths are not over limit
                # not too many trainable or all encoder programs
                # pre-computed lengths are correct
                assert len(encoder_input_ids) <= (dataset.max_seq_len + dataset.max_num_sample_program * dataset.ntokens)
                assert len(decoder_input_ids) <= (dataset.max_seq_len // 2)
                assert len(all_encoder_program_starts) <= dataset.max_num_sample_program
                assert len(trainable_encoder_program_starts) <= dataset.max_num_train_program
                if extra_augmenter is None:
                    encoder_total_program_len = (all_encoder_program_starts.index(max(trainable_encoder_program_starts)) + 1) * dataset.ntokens
                    temp = total_encoder_len + encoder_total_program_len
                    assert encoder_input_ids.shape[0] == temp, (encoder_input_ids.shape[0], temp)
                    assert decoder_input_ids.shape[0] == total_decoder_len, (decoder_input_ids.shape[0], total_decoder_len)

                assert encoder_attention_mask.sum() == encoder_attention_mask.numel()
                assert decoder_attention_mask.sum() == decoder_attention_mask.numel()

                # print('num train example', len(task.train_examples))
                # print('num trainable program start', len(trainable_encoder_program_starts))
                # print('num all program start', len(all_encoder_program_starts))
                # print('selected program starts', trainable_encoder_program_starts)
                # print('all program starts', all_encoder_program_starts)
                # print('program_idxs', program_idxs)
                # print('encoder input ids len', encoder_input_ids.shape[0])

                # if task_id_2[0] == "7b7f7511":
                #     print("encoder_input_ids\n", encoder_input_ids)
                #     print("encoder_label_ids\n", encoder_label_ids)
                #     print("decoder_input_ids\n", decoder_input_ids)
                #     print("decoder_label_ids\n", decoder_label_ids)
                #     print("all_encoder_program_starts\n", all_encoder_program_starts)
                #     print("trainable_encoder_program_starts\n", trainable_encoder_program_starts)
                #     print()
                #     breakpoint()

                two_task_list.append({
                    "task_ids": task_id, # purely for debugging
                    "program_idxs": program_idxs, # purely for debugging
                    "encoder_input_ids": encoder_input_ids,
                    "encoder_attention_mask": encoder_attention_mask,
                    "encoder_label_ids": encoder_label_ids,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                    "decoder_label_ids": decoder_label_ids,
                    "anti_invar": anti_invar,
                    "all_encoder_program_starts": all_encoder_program_starts,
                    "trainable_encoder_program_starts": trainable_encoder_program_starts,
                })

            if len(two_task_list) < 2:
                two_task_list = []

        # Check if we got 2 valid items from this task
        assert len(two_task_list) == 2
        out_list.extend(two_task_list)

    # Now we must truncate out_list if we overshoot
    assert len(out_list) == batch_size, f"Should produce exactly {batch_size} items"

    task_ids = [x["task_ids"] for x in out_list] # purely for debugging
    program_idxs = [x["program_idxs"] for x in out_list] # purely for debugging
    encoder_input_ids  = [x["encoder_input_ids"] for x in out_list]
    encoder_attention_mask = [x["encoder_attention_mask"] for x in out_list]
    encoder_label_ids = [x["encoder_label_ids"] for x in out_list]
    decoder_input_ids  = [x["decoder_input_ids"] for x in out_list]
    decoder_attention_mask = [x["decoder_attention_mask"] for x in out_list]
    decoder_label_ids = [x["decoder_label_ids"] for x in out_list]
    all_encoder_program_starts = [x["all_encoder_program_starts"] for x in out_list]
    trainable_encoder_program_starts = [x["trainable_encoder_program_starts"] for x in out_list]
    anti_invars = [x["anti_invar"] for x in out_list]

    # visualizing some training data
    if dataset.debug_train_data:
        for i in range(0, len(anti_invars), 2):
            assert anti_invars[i] == anti_invars[i+1]
        img_idx = max([int(Path(p).stem.split('_')[0]) for p in glob.glob(f"debug_train_data/*.jpg")], default=-1) + 1
        for batch_i in range(len(out_list)):
            # remove encoder programs
            encoder_text = dataset.encoder_tokenizer.decode(encoder_input_ids[batch_i], skip_special_tokens=True, extra_id_to_token={-100: ""})
            decoder_text = dataset.decoder_tokenizer.decode(decoder_input_ids[batch_i], skip_special_tokens=True)
            encoder_dimensions = dataset.encoder_tokenizer.get_grid_dimensions(encoder_input_ids[batch_i])
            decoder_dimensions = dataset.decoder_tokenizer.get_grid_dimensions(decoder_input_ids[batch_i])
            encoder_grids = parse_input_output_grids(encoder_text, encoder_dimensions)
            decoder_grids = parse_input_output_grids(decoder_text, decoder_dimensions)
            visualize_task(
                task=encoder_grids + decoder_grids,
                name=f"{dataset_name}_{task_id_2[batch_i]}_antiinvar{anti_invars[batch_i]}", # type: ignore
                out_path=f"debug_train_data/{img_idx}_{batch_i}.jpg",
            )

    # uncomment to debug
    # print(dataset.encoder_tokenizer.decode(encoder_input_ids[0]))
    # print(dataset.encoder_tokenizer.decode(encoder_label_ids[0], extra_id_to_token={-100: "_"}))
    # print(dataset.decoder_tokenizer.decode(decoder_input_ids[0]))
    # print(dataset.decoder_tokenizer.decode(decoder_label_ids[0], extra_id_to_token={-100: "_"}))
    # breakpoint()

    encoder_input_ids_lens = [len(x) for x in encoder_input_ids]
    decoder_input_ids_lens = [len(x) for x in decoder_input_ids]
    encoder_input_ids  = pad_sequence_with_side(encoder_input_ids, padding_value=dataset.encoder_tokenizer.pad_token_id, side=dataset.encoder_pad_side)
    encoder_attention_mask = pad_sequence_with_side(encoder_attention_mask, padding_value=0, side=dataset.encoder_pad_side)
    encoder_label_ids = pad_sequence_with_side(encoder_label_ids, padding_value=-100, side=dataset.encoder_pad_side)
    decoder_input_ids  = pad_sequence_with_side(decoder_input_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_pad_side)
    decoder_attention_mask = pad_sequence_with_side(decoder_attention_mask, padding_value=0, side=dataset.decoder_pad_side)
    decoder_label_ids = pad_sequence_with_side(decoder_label_ids, padding_value=-100, side=dataset.decoder_pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        encoder_input_ids, encoder_attention_mask, encoder_label_ids = debug_extra_pad_tensors(
            [encoder_input_ids, encoder_attention_mask, encoder_label_ids],
            padding_values=[dataset.encoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.encoder_pad_side,
        )
        decoder_input_ids, decoder_attention_mask, decoder_label_ids = debug_extra_pad_tensors(
            [decoder_input_ids, decoder_attention_mask, decoder_label_ids],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_pad_side,
        )

    # import pickle
    # idx = max([int(Path(p).stem) for p in glob.glob(f"lots_debug_data/debug_new_train_dicts/*.pkl")], default=-1) + 1
    # pickle.dump({
    #     "task_ids": task_ids,
    #     "encoder_input_ids_lens": encoder_input_ids_lens,
    #     "decoder_input_ids_lens": decoder_input_ids_lens,
    #     "num_pairs": [len(x) + 1 for x in all_encoder_program_starts],
    #     "trainable_pair_indices": program_idxs,
    # }, open(f'lots_debug_data/debug_new_train_dicts/{idx}.pkl', 'wb'))

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "encoder_label_ids": encoder_label_ids,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_label_ids": decoder_label_ids,
        "encoder_input_ids_lens": encoder_input_ids_lens,
        "decoder_input_ids_lens": decoder_input_ids_lens,
        "all_encoder_program_starts": all_encoder_program_starts,
        "trainable_encoder_program_starts": trainable_encoder_program_starts,
        "anti_invars": anti_invars,
    }


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    assert batch_size > 0 and batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    del batch  # we don't use it directly

    # for multi-program mode, we do not have program for the first max_num_sample_program - 1 pairs
    ntoken_len = dataset.ntokens * dataset.max_num_sample_program
    encoder_attention_mask = torch.full((batch_size, dataset.debug_enc_len + ntoken_len), 1, dtype=torch.int64, device='cpu')
    decoder_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_dec_len), dtype=torch.int64, device='cpu')
    decoder_attention_mask = torch.full((batch_size, dataset.debug_dec_len), 1, dtype=torch.int64, device='cpu')
    decoder_input_ids_lens = [len(x) for x in decoder_input_ids]

    # simulate program tokens for encoder_input_ids and encoder_program_starts
    all_encoder_program_starts = []
    old_encoder_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_enc_len), dtype=torch.int64, device='cpu')
    old_encoder_input_ids_chunk = torch.chunk(old_encoder_input_ids, dataset.max_num_sample_program, dim=1)
    program = torch.tensor(
        [dataset.encoder_tokenizer.pad_token_id] * dataset.ntokens,
        dtype=torch.int64, device='cpu')[None, ...].repeat(batch_size, 1)
    encoder_input_ids = []

    for i in range(dataset.max_num_sample_program):
        encoder_input_ids.append(old_encoder_input_ids_chunk[i])
        all_encoder_program_starts.append(sum(x.shape[1] for x in encoder_input_ids))
        encoder_input_ids.append(program)
    encoder_input_ids = torch.cat(encoder_input_ids, dim=1)
    assert len(encoder_input_ids) == len(encoder_attention_mask)

    trainable_encoder_program_starts = all_encoder_program_starts[-dataset.max_num_train_program:]
    trainable_encoder_program_starts = [trainable_encoder_program_starts] * batch_size
    all_encoder_program_starts = [all_encoder_program_starts] * batch_size
    encoder_input_ids_lens = [len(x) for x in encoder_input_ids]

    anti_invars = random.choice([0, 1, 2])
    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "encoder_label_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_label_ids": decoder_input_ids,
        "encoder_input_ids_lens": encoder_input_ids_lens,
        "decoder_input_ids_lens": decoder_input_ids_lens,
        "all_encoder_program_starts": all_encoder_program_starts,
        "trainable_encoder_program_starts": trainable_encoder_program_starts,
        "anti_invars": [anti_invars] * batch_size,
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
        min_num_pair_for_program: int,
        max_num_sample_program: int,
        debug_enc_len: int,
        debug_dec_len: int,
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
        self.encoder_loss_type = encoder_loss_type
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.encoder_pad_side = encoder_pad_side
        self.decoder_pad_side = decoder_pad_side
        self.decoder_gen_pad_side = decoder_gen_pad_side
        self.min_num_pair_for_program = min_num_pair_for_program
        self.max_num_sample_program = max_num_sample_program
        self.debug_enc_len = debug_enc_len
        self.debug_dec_len = debug_dec_len

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
        # task num pairs
        n_pairs = [len(task.train_examples) for task in self.eval_tasks]
        logger.info(f"encoder npairs range from {min(n_pairs)} to {max(n_pairs)}")
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
        # do not add any randomness to this function!
        # this function only filters by token length, not by grid dimension
        # even the voting augmentation does not increase resolution
        assert task.max_height() <= 30 and task.max_width() <= 30

        # permute if given
        train_examples = task.train_examples
        if permutation is not None:
            assert set(permutation) == set(range(len(train_examples)))
            task.train_examples = [train_examples[permute_i] for permute_i in permutation]

        # Build encoder text
        task = copy.deepcopy(task)

        # get encoder_input_ids, encoder_label_ids, encoder_program_start, encoder_attention_mask
        # use dummy ids for program tokens
        encoder_input_ids = []
        encoder_label_ids = []
        all_encoder_program_starts = []
        trainable_encoder_program_starts = [] # for now, program start is only on the last program

        program_input_ids = torch.full((self.ntokens,), self.encoder_tokenizer.pad_token_id, dtype=torch.int64)
        for example_i, example in enumerate(task.train_examples):
            input_grid_ids, output_grid_ids = self.encoder_tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=(example_i == 0),
            )

            # update encoder_input_ids
            encoder_input_ids.append(input_grid_ids)
            encoder_input_ids.append(output_grid_ids)

            # update encoder_label_ids
            # input grid should always be masked
            encoder_label_ids.append(torch.full(input_grid_ids.shape, -100, dtype=torch.int64))
            # output grid is nuanced based on encoder_loss_type
            if (self.encoder_loss_type == "all") or \
                (self.encoder_loss_type == "rest" and example_i > 0) or \
                (self.encoder_loss_type == "last" and example_i == len(task.train_examples) - 1):
                encoder_label_ids.append(output_grid_ids)
            else:
                encoder_label_ids.append(torch.full(output_grid_ids.shape, -100, dtype=torch.int64))

            # program tokens are based on min_num_pair_for_program
            if example_i + 1 >= self.min_num_pair_for_program:
                program_start_idx = sum(len(x) for x in encoder_input_ids)
                all_encoder_program_starts.append(program_start_idx)
                encoder_input_ids.append(program_input_ids)
                encoder_label_ids.append(torch.full(program_input_ids.shape, -100, dtype=torch.int64))
                if example_i == len(task.train_examples) - 1:
                    trainable_encoder_program_starts.append(program_start_idx)

        encoder_input_ids = torch.cat(encoder_input_ids)
        encoder_label_ids = torch.cat(encoder_label_ids)
        assert encoder_input_ids.shape == encoder_label_ids.shape
        encoder_attention_mask = torch.full(encoder_input_ids.shape, 1, dtype=torch.int64)

        if len(encoder_input_ids) > self.max_seq_len + self.max_num_sample_program * self.ntokens:
            return None

        # get decoder_input_ids, decoder_label_ids, decoder_attention_mask
        input_grid_ids, output_grid_ids = self.decoder_tokenizer.get_input_and_output_grid_ids(
            example=task.test_example,
            add_bos=True,
        )

        decoder_input_ids = torch.cat([input_grid_ids, output_grid_ids])
        decoder_label_ids = torch.cat([torch.full(input_grid_ids.shape, -100, dtype=torch.int64),
                                        output_grid_ids])
        assert decoder_input_ids.shape == decoder_label_ids.shape
        decoder_attention_mask = torch.full(decoder_input_ids.shape, 1, dtype=torch.int64)

        if len(decoder_input_ids) > self.max_seq_len // 2:
            return None

        # eval only stuff
        decoder_gen_attention_mask = torch.full(input_grid_ids.shape, 1, dtype=torch.int64)
        decoder_label_texts = self.decoder_tokenizer.decode(output_grid_ids)[:-len(self.decoder_tokenizer.eos_token)]
        decoder_out_token_length = len(output_grid_ids) - 1 # remove eos token

        assert encoder_attention_mask.sum() == encoder_attention_mask.numel()
        assert decoder_attention_mask.sum() == decoder_attention_mask.numel()
        assert decoder_gen_attention_mask.sum() == decoder_gen_attention_mask.numel()
        assert len(trainable_encoder_program_starts) == 1

        # print(task.name)
        # if "7b7f7511" in task.name:
        #     print("encoder_input_ids\n", encoder_input_ids)
        #     print("encoder_label_ids\n", encoder_label_ids)
        #     print("decoder_input_ids\n", decoder_input_ids)
        #     print("decoder_label_ids\n", decoder_label_ids)
        #     print("decoder_gen_input_ids\n", input_grid_ids)
        #     print("all_encoder_program_starts\n", all_encoder_program_starts)
        #     print("trainable_encoder_program_starts\n", trainable_encoder_program_starts)
        #     print("decoder_out_token_length\n", decoder_out_token_length)
        #     breakpoint()
        # decoder_out_token_length = 1

        return {
            "task_id": task.name,
            "inverter": task.inverter if hasattr(task, "inverter") else "", # type: ignore
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_label_ids": encoder_label_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_label_ids": decoder_label_ids,
            "decoder_gen_input_ids": input_grid_ids,
            "decoder_gen_attention_mask": decoder_gen_attention_mask,
            "decoder_out_token_length": decoder_out_token_length,
            "decoder_label_texts": decoder_label_texts,  # used for exact match
            "all_encoder_program_starts": all_encoder_program_starts,
            "trainable_encoder_program_starts": trainable_encoder_program_starts,
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
    encoder_input_ids = [x["encoder_input_ids"] for x in batch]
    encoder_attention_mask = [x["encoder_attention_mask"] for x in batch]
    encoder_label_ids = [x["encoder_label_ids"] for x in batch]
    decoder_input_ids = [x["decoder_input_ids"] for x in batch]
    decoder_attention_mask = [x["decoder_attention_mask"] for x in batch]
    decoder_label_ids = [x["decoder_label_ids"] for x in batch]
    decoder_gen_input_ids = [x["decoder_gen_input_ids"] for x in batch]
    decoder_gen_attention_mask = [x["decoder_gen_attention_mask"] for x in batch]
    decoder_out_token_length = [x["decoder_out_token_length"] for x in batch]
    decoder_label_texts = [x["decoder_label_texts"] for x in batch]
    all_encoder_program_starts = [x["all_encoder_program_starts"] for x in batch]
    trainable_encoder_program_starts = [x["trainable_encoder_program_starts"] for x in batch]

    encoder_input_ids_lens = [len(x) for x in encoder_input_ids]
    decoder_input_ids_lens = [len(x) for x in decoder_input_ids]
    decoder_gen_input_ids_lens = [len(x) for x in decoder_gen_input_ids]
    encoder_input_ids = pad_sequence_with_side(encoder_input_ids, padding_value=dataset.encoder_tokenizer.pad_token_id, side=dataset.encoder_pad_side)
    encoder_attention_mask = pad_sequence_with_side(encoder_attention_mask, padding_value=0, side=dataset.encoder_pad_side)
    encoder_label_ids = pad_sequence_with_side(encoder_label_ids, padding_value=-100, side=dataset.encoder_pad_side)
    decoder_input_ids = pad_sequence_with_side(decoder_input_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_pad_side)
    decoder_attention_mask = pad_sequence_with_side(decoder_attention_mask, padding_value=0, side=dataset.decoder_pad_side)
    decoder_label_ids = pad_sequence_with_side(decoder_label_ids, padding_value=-100, side=dataset.decoder_pad_side)
    decoder_gen_input_ids = pad_sequence_with_side(decoder_gen_input_ids, padding_value=dataset.decoder_tokenizer.pad_token_id, side=dataset.decoder_gen_pad_side)
    decoder_gen_attention_mask = pad_sequence_with_side(decoder_gen_attention_mask, padding_value=0, side=dataset.decoder_gen_pad_side)

    if dataset.debug_random_pad and dataset.debug_pad_len > -1:
        encoder_input_ids, encoder_attention_mask, encoder_label_ids = debug_extra_pad_tensors(
            [encoder_input_ids, encoder_attention_mask, encoder_label_ids],
            padding_values=[dataset.encoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.encoder_pad_side,
        )
        decoder_input_ids, decoder_attention_mask, decoder_label_ids = debug_extra_pad_tensors(
            [decoder_input_ids, decoder_attention_mask, decoder_label_ids],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_pad_side,
        )
        decoder_gen_input_ids, decoder_gen_attention_mask = debug_extra_pad_tensors(
            [decoder_gen_input_ids, decoder_gen_attention_mask],
            padding_values=[dataset.decoder_tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.decoder_gen_pad_side,
        )

    # import pickle
    # import glob
    # idx = max([int(Path(p).stem) for p in glob.glob(f"lots_debug_data/debug_new_eval_eval_dicts/*.pkl")], default=-1) + 1
    # pickle.dump({
    #     "task_id": task_ids,
    #     "encoder_input_ids_len": encoder_input_ids_lens,
    #     "decoder_input_ids_len": decoder_input_ids_lens,
    #     "decoder_gen_input_ids_lens": decoder_gen_input_ids_lens,
    #     "decoder_out_token_length": decoder_out_token_length,
    #     "num_pairs": [len(p) + 1 for p in all_encoder_program_starts],
    # }, open(f'lots_debug_data/debug_new_eval_eval_dicts/{idx}.pkl', 'wb'))

    batch_dict = {
        "task_ids": task_ids,
        "inverters": inverters,
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "encoder_label_ids": encoder_label_ids,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_label_ids": decoder_label_ids,
        "decoder_gen_input_ids": decoder_gen_input_ids,
        "decoder_gen_attention_mask": decoder_gen_attention_mask,
        "decoder_out_token_length": decoder_out_token_length,
        "decoder_label_texts": decoder_label_texts,
        "all_encoder_program_starts": all_encoder_program_starts,
        "trainable_encoder_program_starts": trainable_encoder_program_starts,
        "encoder_input_ids_lens": encoder_input_ids_lens,
        "decoder_input_ids_lens": decoder_input_ids_lens,
        "decoder_gen_input_ids_lens": decoder_gen_input_ids_lens,
    }
    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    ntoken_len = dataset.ntokens * dataset.max_num_sample_program
    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    encoder_input_ids = torch.randint(1, 30, (batch_size, dataset.debug_enc_len + ntoken_len), dtype=torch.int64, device='cpu')
    encoder_attention_mask = torch.full((batch_size, dataset.debug_enc_len + ntoken_len), 1, dtype=torch.int64, device='cpu')
    decoder_input_ids = torch.randint(1, 30, (batch_size, dataset.debug_dec_len), dtype=torch.int64, device='cpu')
    decoder_attention_mask = torch.full((batch_size, dataset.debug_dec_len), 1, dtype=torch.int64, device='cpu')
    decoder_gen_input_ids = torch.randint(1, 30, (batch_size, int(dataset.debug_dec_len * 0.4)), dtype=torch.int64, device='cpu')
    decoder_gen_attention_mask = torch.full((batch_size, int(dataset.debug_dec_len * 0.4)), 1, dtype=torch.int64, device='cpu')
    decoder_input_ids_lens = [len(x) for x in decoder_input_ids]
    decoder_gen_input_ids_lens = [len(x) for x in decoder_gen_input_ids]

    # simulate program tokens for encoder_input_ids and encoder_program_start
    all_encoder_program_starts = []
    old_encoder_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_enc_len), dtype=torch.int64, device='cpu')
    old_encoder_input_ids_chunk = torch.chunk(old_encoder_input_ids, dataset.max_num_sample_program, dim=1)
    program = torch.tensor(
        [dataset.encoder_tokenizer.pad_token_id] * dataset.ntokens,
        dtype=torch.int64, device='cpu')[None, ...].repeat(batch_size, 1)
    encoder_input_ids = []

    for i in range(dataset.max_num_sample_program):
        encoder_input_ids.append(old_encoder_input_ids_chunk[i])
        all_encoder_program_starts.append(sum(x.shape[1] for x in encoder_input_ids))
        encoder_input_ids.append(program)

    encoder_input_ids = torch.cat(encoder_input_ids, dim=1)
    assert len(encoder_input_ids) == len(encoder_attention_mask)

    trainable_encoder_program_starts = all_encoder_program_starts[-1:]
    trainable_encoder_program_starts = [trainable_encoder_program_starts] * batch_size
    all_encoder_program_starts = [all_encoder_program_starts] * batch_size
    encoder_input_ids_lens = [len(x) for x in encoder_input_ids]

    batch_dict = {
        "task_ids": task_ids,
        "inverters": [""] * batch_size,
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "encoder_label_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_label_ids": decoder_input_ids,
        "decoder_gen_input_ids": decoder_gen_input_ids,
        "decoder_gen_attention_mask": decoder_gen_attention_mask,
        "decoder_out_token_length": [math.ceil(dataset.debug_dec_len * 0.6)] * batch_size,
        "decoder_label_texts": ['1\n1\n1'] * batch_size,
        "all_encoder_program_starts": all_encoder_program_starts,
        "trainable_encoder_program_starts": trainable_encoder_program_starts,
        "encoder_input_ids_lens": encoder_input_ids_lens,
        "decoder_input_ids_lens": decoder_input_ids_lens,
        "decoder_gen_input_ids_lens": decoder_gen_input_ids_lens,
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

        # tiny optimization to include output\n in input
        assert dec_out_text.startswith("output\n")
        dec_in_text = dec_in_text + "\n" + dec_out_text[:len("output\n")]
        dec_out_text = dec_out_text[len("output\n"):]
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