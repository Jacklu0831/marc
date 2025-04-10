import hashlib
import ast
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from collections import Counter
import itertools
import csv
import copy
import os
import json
import random
import torch
from torch.utils.data import Dataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Union
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

    def encode_dimensions_to_tensor(self, height: int, width: int) -> torch.Tensor:
        return torch.tensor(
            [self.token_to_id[str(height)],
             self.token_to_id[str(width)],
             self.token_to_id["\n"]],
            dtype=torch.int64)

    def encode_grid_to_tensor(self, grid: np.ndarray, no_separate_color_tokens: bool) -> torch.Tensor:
        assert grid.ndim == 2
        token_ids = []
        for row in grid:
            for x in row:
                if no_separate_color_tokens:
                    token_ids.append(self.token_to_id[str(x)])
                else:
                    token_ids.append(self.token_to_id[f"c{str(x)}"])
            token_ids.append(self.token_to_id["\n"])
        token_ids = token_ids[:-1] # no \n at end
        return torch.tensor(token_ids, dtype=torch.int64)

    def convert_token_to_id(self, token: str) -> torch.Tensor:
        return torch.tensor([self.token_to_id[token]], dtype=torch.int64)

    def get_input_and_output_grid_ids(self, example: Example, add_bos: bool, no_dim: bool, no_separate_color_tokens: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        input_grid_ids = self.encode_grid_to_tensor(example.input, no_separate_color_tokens)
        output_grid_ids = self.encode_grid_to_tensor(example.output, no_separate_color_tokens)

        # input_grid_ids should contain everything except the output_grid_ids
        if no_dim:
            input_grid_ids = torch.cat([
                self.convert_token_to_id("input"),
                input_grid_ids,
                self.convert_token_to_id("output"),
            ])
        else:
            input_grid_ids = torch.cat([
                self.convert_token_to_id("input"),
                self.encode_dimensions_to_tensor(len(example.input), len(example.input[0])),
                input_grid_ids,
                self.convert_token_to_id("output"),
            ])
        if add_bos:
            input_grid_ids = torch.cat([self.convert_token_to_id(self.bos_token), input_grid_ids])

        # output_grid_ids should contain a eos
        if no_dim:
            output_grid_ids = torch.cat([
                output_grid_ids,
                self.convert_token_to_id(self.eos_token),
            ])
        else:
            output_grid_ids = torch.cat([
                self.encode_dimensions_to_tensor(len(example.output), len(example.output[0])),
                output_grid_ids,
                self.convert_token_to_id(self.eos_token),
            ])
        return input_grid_ids, output_grid_ids

    def decode(
            self,
            token_ids: Union[List[int], torch.Tensor],
            no_separate_color_tokens: bool,
            skip_special_tokens: bool = False,
            extra_id_to_token: Dict[int, str] = {},
        ) -> str:
        tokens = []
        for token_id in token_ids:
            if not skip_special_tokens or (int(token_id) not in self.special_token_ids):
                token_id = int(token_id)
                if token_id not in self.id_to_token:
                    tokens.append(extra_id_to_token[token_id])
                else:
                    token = self.id_to_token[token_id]
                    if token.startswith('c') and (not no_separate_color_tokens):
                        token = token[1:]
                    tokens.append(token)
        return "".join(tokens)

    def get_grid_dimensions(self, token_ids: Union[List[int], torch.Tensor]) -> List[Tuple[int, int]]:
        dimensions = []
        for i, token_id in enumerate(token_ids):
            if token_id in [self.token_to_id["input"], self.token_to_id["output"]]:
                height = self.id_to_token[int(token_ids[i+1])]
                width = self.id_to_token[int(token_ids[i+2])]
                dimensions.append((height, width))
        return dimensions

    def batch_decode(self, batch_token_ids: torch.Tensor, skip_special_tokens: bool, no_separate_color_tokens: bool) -> List[str]:
        assert batch_token_ids.dim() == 2
        texts = []
        for token_ids in batch_token_ids:
            text = self.decode(
                token_ids=token_ids,
                no_separate_color_tokens=no_separate_color_tokens,
                skip_special_tokens=skip_special_tokens
            )
            texts.append(text)
        return texts


def get_d8_augmenters(include_identity: bool) -> List[Augmenter]:
    augmenters = [
        Rotate(90),
        Rotate(180),
        Rotate(270),
        Flip(0),
        Flip(1),
        Chain([Flip(0), Rotate(90)]), # type: ignore
        Chain([Flip(1), Rotate(90)]), # type: ignore
    ]
    if include_identity:
        augmenters = [Rotate(0)] + augmenters
    return augmenters # type: ignore


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

    grids_row1 = [pad_grid_to_30x30(grid) for grid in grids_row1] # type: ignore
    grids_row2 = [pad_grid_to_30x30(grid) for grid in grids_row2] # type: ignore

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
    files = glob.glob(f'./data/ConceptARC/corpus/*/*.json')
    files += glob.glob(f'./data/ConceptARC/MinimalTasks/*.json')
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
        tokenizer: ARCTokenizer,
        total_steps: int,
        extra_augment_ratio: float,
        extra_augment_single_grid: bool,
        seed: int,
        process_index: int,
        ntokens: int,
        debug_fixed_order: bool,
        debug_random_pad: bool,
        debug_pad_len: int,
        train_pad_side: str,
        debug_train_data: bool,
        no_color_permute: bool,
        no_pair_permute: bool,
        no_d8: bool,
        min_num_pair: int,
        max_num_pair: int,
        no_train_original: bool,
        only_train_original: bool,
        debug_len: int,
        num_workers: int,
        no_dim: bool,
        no_separate_color_tokens: bool,
        max_seq_len: int,
        no_bos: bool,
        only_first_bos: bool,
        same_task_identifier_across_gpus: bool,
    ):
        self.re_arc_ratio = re_arc_ratio
        self.concept_arc_ratio = concept_arc_ratio
        self.arc_heavy_ratio = arc_heavy_ratio

        self.tokenizer = tokenizer
        self._length = total_steps
        self.extra_augment_ratio = extra_augment_ratio
        self.extra_augment_single_grid = extra_augment_single_grid
        self.ntokens = ntokens
        self.debug_fixed_order = debug_fixed_order
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.train_pad_side = train_pad_side
        self.debug_train_data = debug_train_data
        self.no_color_permute = no_color_permute
        self.no_pair_permute = no_pair_permute
        self.no_d8 = no_d8
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        self.debug_len = debug_len
        self.no_dim = no_dim
        self.no_separate_color_tokens = no_separate_color_tokens
        self.max_seq_len = max_seq_len
        self.no_bos = no_bos
        self.only_first_bos = only_first_bos
        self.same_task_identifier_across_gpus = same_task_identifier_across_gpus

        self.num_workers = num_workers
        self.process_index = process_index
        self.seed = seed

        # setup args
        self.normalized_ratio = np.array([self.re_arc_ratio, self.concept_arc_ratio, self.arc_heavy_ratio])
        self.normalized_ratio /= np.sum(self.normalized_ratio)
        self.d8_augmenters = get_d8_augmenters(include_identity=True)
        self.extra_augmenters = get_mit_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)

        # set rngs
        self.set_rngs(epoch=0)

        # load re-arc data + train original data
        if only_train_original:
            re_arc_task_id_to_pairs = load_train_original_data_from_dir(eval_train_dir)
        elif no_train_original:
            re_arc_task_id_to_pairs = load_re_arc_from_data_dir(train_data_dir)
        else:
            re_arc_task_id_to_pairs = load_re_arc_from_data_dir(train_data_dir)
            train_original_task_id_to_pairs = load_train_original_data_from_dir(eval_train_dir)
            if set(re_arc_task_id_to_pairs.keys()) != set(train_original_task_id_to_pairs.keys()):
                assert set(re_arc_task_id_to_pairs.keys()).issubset(set(train_original_task_id_to_pairs.keys()))
                logger.info(f'[WARNING] loaded {len(set(re_arc_task_id_to_pairs.keys()))} re-arc tasks, less than 400')
                train_original_task_id_to_pairs = {task_id: pairs for task_id, pairs in train_original_task_id_to_pairs.items() if task_id in re_arc_task_id_to_pairs}
            for task_id in re_arc_task_id_to_pairs:
                re_arc_task_id_to_pairs[task_id] += train_original_task_id_to_pairs[task_id]
        self.arc_train_id_to_pairs = re_arc_task_id_to_pairs
        logger.info(f'loaded {len(self.arc_train_id_to_pairs)} re-arc/train-original tasks, total \
                    {sum(len(x) for x in self.arc_train_id_to_pairs.values())} pairs')

        # load concept-arc data
        self.concept_arc_id_to_pairs = {}
        if concept_arc_ratio > 0.0:
            self.concept_arc_id_to_pairs = load_concept_arc()
            logger.info(f'loaded {len(self.concept_arc_id_to_pairs)} concept-arc tasks')

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
            self.gpu_consistent_rngs = [np.random.RandomState(self.seed + epoch_seed)]
        else:
            self.gpu_consistent_rngs = [np.random.RandomState(self.seed + epoch_seed + i) for i in range(self.num_workers)]


def collate_fn_train(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]
    gpu_consistent_rng = dataset.gpu_consistent_rngs[int(worker_id)]
    task_identifier_rng = gpu_consistent_rng if dataset.same_task_identifier_across_gpus else rng

    # the restriction here is to enforce all list of pairs in batch are equal length
    all_task_ids = []
    token_lens = []
    all_np_chosen_pairs = []

    # must sample this number of pairs to avoid GPU synchronization issues
    required_num_pair = gpu_consistent_rng.choice(list(range(dataset.min_num_pair, dataset.max_num_pair + 1)))
    task_identifiers = []

    # sample random task from random dataset, if grid size >30 or does not have enough for required_num_pair, retry
    while len(all_task_ids) < batch_size:
        dataset_name = task_identifier_rng.choice(["re-arc", "concept-arc", "arc-heavy"], p=dataset.normalized_ratio)

        # STEP 1: get task id and pairs, sample task id until reaching num chosen pair
        if dataset_name == "re-arc":
            task_id = task_identifier_rng.choice(list(dataset.arc_train_id_to_pairs.keys()))
            all_pairs = dataset.arc_train_id_to_pairs[task_id]
        elif dataset_name == "concept-arc":
            task_id = task_identifier_rng.choice(list(dataset.concept_arc_id_to_pairs.keys()))
            all_pairs = dataset.concept_arc_id_to_pairs[task_id]
        else:
            idx = task_identifier_rng.choice(list(range(len(dataset.heavy_arc_data))))
            task_id = f"heavy{idx}"
            all_pairs = dataset.heavy_arc_data[int(idx)]["examples"]
            all_pairs = [{"input": pair[0], "output": pair[1]} for pair in all_pairs]
            assert all(len(pair) == 2 for pair in all_pairs)

        if dataset.debug_fixed_order:
            required_num_pair = len(all_pairs)

        # need at least required_num_pair to proceed
        if len(all_pairs) < required_num_pair:
            continue

        # STEP 2: decide on extra augmentations and io augmentation choice for pairs
        if not dataset.no_d8:
            d8_augmenter = task_identifier_rng.choice(dataset.d8_augmenters) # type: ignore
        else:
            d8_augmenter = None

        if task_identifier_rng.rand() < dataset.extra_augment_ratio:
            extra_augmenter = task_identifier_rng.choice(dataset.extra_augmenters) # type: ignore
            io_augmentation_choice = task_identifier_rng.choice(["input_only", "output_only", "both"]) if dataset.extra_augment_single_grid else "both"
        else:
            extra_augmenter = None
            io_augmentation_choice = None

        # STEP 3: choose pairs and apply augmentations
        if dataset.debug_fixed_order:
            chosen_pairs = all_pairs[:required_num_pair]
        else:
            chosen_pairs = rng.choice(all_pairs, size=required_num_pair, replace=False) # type: ignore

        np_chosen_pairs = []
        for pair in chosen_pairs:
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
            np_chosen_pairs.append(np_pair)

        if any(max(*pair["input"].shape, *pair["output"].shape) > 30 for pair in np_chosen_pairs):
            continue

        # HACK: just hardcode some calculation here to limit maxseqlen
        token_len = 0
        for pair_i, pair in enumerate(np_chosen_pairs):
            h1, w1 = len(pair['input']), len(pair['input'][0])
            h2, w2 = len(pair['output']), len(pair['output'][0])
            pair_token_len = h1 * (w1 + 1) + h2 * (w2 + 1) - 2 # cells and \n
            pair_token_len += 3 # input, output, eos
            pair_token_len += 6 # hw\n for both input and output
            if not dataset.no_bos:
                if dataset.only_first_bos:
                    pair_token_len += int(pair_i == 0)
                else:
                    pair_token_len += 1
            token_len += pair_token_len

        if token_len > dataset.max_seq_len:
            continue

        # STEP 4: found a valid task!
        all_task_ids.append(task_id) # type: ignore
        token_lens.append(token_len)
        all_np_chosen_pairs.append(np_chosen_pairs)

        # used for program caching
        task_identifiers.append(
            '-'.join([dataset_name, task_id, str(d8_augmenter), str(extra_augmenter), str(io_augmentation_choice)])
        )

    # apply color and pair permutation
    tasks = [Task(
        name=task_id,
        train_examples=[
            Example(input=pair["input"], output=pair["output"])
            for pair in pairs[:-1]
        ],
        test_example=Example(input=pairs[-1]["input"], output=pairs[-1]["output"]),
    ) for task_id, pairs in zip(all_task_ids, all_np_chosen_pairs)]

    if not dataset.no_pair_permute:
        tasks = [PermuteExamples().apply_to_task(task, to_input=True, to_output=True, rng=rng) for task in tasks]

    if not dataset.no_color_permute:
        temp = []
        for task_i, task in enumerate(tasks):
            augmenter = PermuteColors()
            temp.append(augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng))
            task_identifiers[task_i] += f'-{str(augmenter._color_map)}'
        tasks = temp

    # we do a lil parsing
    pair_idx_to_input_ids = []
    pair_idx_to_attention_mask = []
    pair_idx_to_label_ids = []

    for pair_i in range(required_num_pair):
        # get inputids, attention, labelids for batch of pairs at pair_i
        batch_input_ids = []
        batch_attention_mask = []
        batch_label_ids = []
        for task in tasks:
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = dataset.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=False if dataset.no_bos else (pair_i == 0 if dataset.only_first_bos else True),
                no_dim=dataset.no_dim,
                no_separate_color_tokens=dataset.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_grid_ids])
            # append
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_label_ids.append(label_ids)
        # aggregate
        pair_idx_to_input_ids.append(batch_input_ids)
        pair_idx_to_attention_mask.append(batch_attention_mask)
        pair_idx_to_label_ids.append(batch_label_ids)
    assert [sum(x[batch_i].shape[0] for x in pair_idx_to_input_ids) for batch_i in range(batch_size)] == token_lens

    # visualize some training data
    if dataset.debug_train_data:
        img_idx = max([int(Path(p).stem.split('_')[0]) for p in glob.glob(f"debug_train_data/*.jpg")], default=-1) + 1
        for batch_i in range(batch_size):
            input_ids = [pair_idx_to_input_ids[pair_i][batch_i] for pair_i in range(required_num_pair)]
            texts = [
                dataset.tokenizer.decode(ids, skip_special_tokens=True, no_separate_color_tokens=dataset.no_separate_color_tokens)
                for ids in input_ids
            ]
            dimensions = [dataset.tokenizer.get_grid_dimensions(pair_idx_to_input_ids[pair_i][batch_i]) for pair_i in range(required_num_pair)]
            assert all(len(d) == 2 for d in dimensions)
            grids = [parse_input_output_grids(t, d) for t, d in zip(texts, dimensions)]
            grids = [item for sublist in grids for item in sublist]
            visualize_task(
                task=grids,
                name=f"{dataset_name}_{all_task_ids[batch_i]}", # type: ignore
                out_path=f"debug_train_data/{img_idx}_{batch_i}.jpg",
            )

    # get input ids lens
    input_ids_lens = []
    for pair_i in range(required_num_pair):
        input_ids_lens.append([len(ids) for ids in pair_idx_to_input_ids[pair_i]])

    # pad
    padded_input_ids = []
    padded_attention_mask = []
    padded_label_ids = []
    for input_ids, attention_mask, label_ids in zip(pair_idx_to_input_ids, pair_idx_to_attention_mask, pair_idx_to_label_ids):
        input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.train_pad_side)
        attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.train_pad_side)
        label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.train_pad_side)
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_label_ids.append(label_ids)

    extra_padded_input_ids = []
    extra_padded_attention_mask = []
    extra_padded_label_ids = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for input_ids, attention_mask, label_ids in zip(padded_input_ids, padded_attention_mask, padded_label_ids):
            input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
                [input_ids, attention_mask, label_ids],
                padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
                pad_len=dataset.debug_pad_len,
                side=dataset.train_pad_side,
            )
            extra_padded_input_ids.append(input_ids)
            extra_padded_attention_mask.append(attention_mask)
            extra_padded_label_ids.append(label_ids)
    else:
        extra_padded_input_ids = padded_input_ids
        extra_padded_attention_mask = padded_attention_mask
        extra_padded_label_ids = padded_label_ids

    # for i in range(len(extra_padded_input_ids)): print(dataset.tokenizer.decode(extra_padded_input_ids[i][1], no_separate_color_tokens=dataset.no_separate_color_tokens), '\n')
    batch_dict = {
        "input_ids": extra_padded_input_ids,
        "attention_mask": extra_padded_attention_mask,
        "label_ids": extra_padded_label_ids,
        "input_ids_lens": input_ids_lens,
        "num_pairs": [required_num_pair] * batch_size,
        "is_same": False,
        "task_identifiers": task_identifiers,
    }
    return batch_dict


def collate_fn_train_invar(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    assert batch_size == 2
    del batch  # we don't use it directly

    def get_all_pairs(required_num_pair: int, exclude_id: Optional[str] = None):
        while True:
            dataset_name = rng.choice(["re-arc", "concept-arc", "arc-heavy"], p=dataset.normalized_ratio)

            if dataset_name == "re-arc":
                task_id = rng.choice(list(dataset.arc_train_id_to_pairs.keys()))
                all_pairs = dataset.arc_train_id_to_pairs[task_id]
            elif dataset_name == "concept-arc":
                task_id = rng.choice(list(dataset.concept_arc_id_to_pairs.keys()))
                all_pairs = dataset.concept_arc_id_to_pairs[task_id]
            else:
                idx = rng.choice(list(range(len(dataset.heavy_arc_data))))
                task_id = f"heavy{idx}"
                all_pairs = dataset.heavy_arc_data[int(idx)]["examples"]
                all_pairs = [{"input": pair[0], "output": pair[1]} for pair in all_pairs]
                assert all(len(pair) == 2 for pair in all_pairs)

            if len(all_pairs) < required_num_pair:
                continue

            # for re-arc, some tasks replicate
            if dataset_name == "re-arc" and (exclude_id is not None) and dataset.task_id_to_hash[task_id] == dataset.task_id_to_hash[exclude_id]:
                continue

            return task_id, all_pairs

    def get_augmentations():
        # d8
        if not dataset.no_d8:
            d8_augmenter = rng.choice(dataset.d8_augmenters) # type: ignore
        else:
            d8_augmenter = None
        # extra
        if rng.rand() < dataset.extra_augment_ratio:
            extra_augmenter = rng.choice(dataset.extra_augmenters) # type: ignore
            io_augmentation_choice = rng.choice(["input_only", "output_only", "both"]) if dataset.extra_augment_single_grid else "both"
        else:
            extra_augmenter = None
            io_augmentation_choice = None
        return d8_augmenter, extra_augmenter, io_augmentation_choice

    def get_np_chosen_pairs(chosen_pairs, d8_augmenter, extra_augmenter, io_augmentation_choice):
        np_chosen_pairs = []
        for pair in chosen_pairs:
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
            np_chosen_pairs.append(np_pair)
        return np_chosen_pairs

    def get_token_len(np_chosen_pairs):
        # HACK: just hardcode some calculation here to limit maxseqlen
        token_len = 0
        for pair_i, pair in enumerate(np_chosen_pairs):
            h1, w1 = len(pair['input']), len(pair['input'][0])
            h2, w2 = len(pair['output']), len(pair['output'][0])
            pair_token_len = h1 * (w1 + 1) + h2 * (w2 + 1) - 2 # cells and \n
            pair_token_len += 3 # input, output, eos
            pair_token_len += 6 # hw\n for both input and output
            if not dataset.no_bos:
                if dataset.only_first_bos:
                    pair_token_len += int(pair_i == 0)
                else:
                    pair_token_len += 1
            token_len += pair_token_len
        return token_len

    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0 # could be single-thread
    rng = dataset.rngs[int(worker_id)]
    num_pair_rng = dataset.gpu_consistent_rngs[int(worker_id)]

    # must sample this number of pairs to avoid GPU synchronization issues
    required_num_pair = num_pair_rng.choice(list(range(dataset.min_num_pair, dataset.max_num_pair + 1)))

    # is same
    is_same = rng.rand() < 0.5

    # get tasks
    while True:
        # task_ids, np_chosen_pairs, and token_lens
        if is_same:
            # sample same task and augmentation scheme
            task_id, all_pairs = get_all_pairs(required_num_pair=required_num_pair)
            d8_augmenter, extra_augmenter, io_augmentation_choice = get_augmentations()
            # choose
            chosen_pairs1 = rng.choice(all_pairs, size=required_num_pair, replace=False) # type: ignore
            chosen_pairs2 = rng.choice(all_pairs, size=required_num_pair, replace=False) # type: ignore
            np_chosen_pairs1 = get_np_chosen_pairs(chosen_pairs1, d8_augmenter, extra_augmenter, io_augmentation_choice)
            np_chosen_pairs2 = get_np_chosen_pairs(chosen_pairs2, d8_augmenter, extra_augmenter, io_augmentation_choice)
        else:
            # sample different task and different augmentation scheme
            task_id1, all_pairs1 = get_all_pairs(required_num_pair=required_num_pair)
            task_id2, all_pairs2 = get_all_pairs(required_num_pair=required_num_pair, exclude_id=task_id1)
            d8_augmenter1, extra_augmenter1, io_augmentation_choice1 = get_augmentations()
            d8_augmenter2, extra_augmenter2, io_augmentation_choice2 = get_augmentations()
            # choose
            chosen_pairs1 = rng.choice(all_pairs1, size=required_num_pair, replace=False) # type: ignore
            chosen_pairs2 = rng.choice(all_pairs2, size=required_num_pair, replace=False) # type: ignore
            np_chosen_pairs1 = get_np_chosen_pairs(chosen_pairs1, d8_augmenter1, extra_augmenter1, io_augmentation_choice1)
            np_chosen_pairs2 = get_np_chosen_pairs(chosen_pairs2, d8_augmenter2, extra_augmenter2, io_augmentation_choice2)

        if any(max(*pair["input"].shape, *pair["output"].shape) > 30 for pair in np_chosen_pairs1):
            continue
        if any(max(*pair["input"].shape, *pair["output"].shape) > 30 for pair in np_chosen_pairs2):
            continue

        token_len1 = get_token_len(np_chosen_pairs1)
        token_len2 = get_token_len(np_chosen_pairs2)
        if max(token_len1, token_len2) > dataset.max_seq_len:
            continue

        all_task_ids = [task_id, task_id] if is_same else [task_id1, task_id2] # type: ignore
        token_lens = [token_len1, token_len2]
        all_np_chosen_pairs = [np_chosen_pairs1, np_chosen_pairs2]

        # create tasks
        tasks = [Task(
            name=task_id,
            train_examples=[
                Example(input=pair["input"], output=pair["output"])
                for pair in pairs[:-1]
            ],
            test_example=Example(input=pairs[-1]["input"], output=pairs[-1]["output"]),
        ) for task_id, pairs in zip(all_task_ids, all_np_chosen_pairs)]

        # permute examples
        if not dataset.no_pair_permute:
            tasks = [PermuteExamples().apply_to_task(task, to_input=True, to_output=True, rng=rng) for task in tasks]

        # color permute depends on is_same
        if not dataset.no_color_permute:
            if is_same:
                augmenter = PermuteColors()
                tasks[0] = augmenter.apply_to_task(tasks[0], to_input=True, to_output=True, rng=rng)
                color_mapper = augmenter.color_mapper
                tasks[1] = augmenter.apply_to_task(tasks[1], to_input=True, to_output=True, rng=rng, color_mapper=color_mapper)
            else:
                tasks = [PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng) for task in tasks]

        break

    # we do a lil parsing
    pair_idx_to_input_ids = []
    pair_idx_to_attention_mask = []
    pair_idx_to_label_ids = []

    for pair_i in range(required_num_pair):
        # get inputids, attention, labelids for batch of pairs at pair_i
        batch_input_ids = []
        batch_attention_mask = []
        batch_label_ids = []
        for task in tasks:
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = dataset.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=False if dataset.no_bos else (pair_i == 0 if dataset.only_first_bos else True),
                no_dim=dataset.no_dim,
                no_separate_color_tokens=dataset.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_grid_ids])
            # append
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_label_ids.append(label_ids)
        # aggregate
        pair_idx_to_input_ids.append(batch_input_ids)
        pair_idx_to_attention_mask.append(batch_attention_mask)
        pair_idx_to_label_ids.append(batch_label_ids)
    assert [sum(x[batch_i].shape[0] for x in pair_idx_to_input_ids) for batch_i in range(batch_size)] == token_lens

    # visualize some training data
    if dataset.debug_train_data:
        img_idx = max([int(Path(p).stem.split('_')[0]) for p in glob.glob(f"debug_train_data/*.jpg")], default=-1) + 1
        for batch_i in range(batch_size):
            input_ids = [pair_idx_to_input_ids[pair_i][batch_i] for pair_i in range(required_num_pair)]
            texts = [
                dataset.tokenizer.decode(ids, skip_special_tokens=True, no_separate_color_tokens=dataset.no_separate_color_tokens)
                for ids in input_ids
            ]
            dimensions = [dataset.tokenizer.get_grid_dimensions(pair_idx_to_input_ids[pair_i][batch_i]) for pair_i in range(required_num_pair)]
            assert all(len(d) == 2 for d in dimensions)
            grids = [parse_input_output_grids(t, d) for t, d in zip(texts, dimensions)]
            grids = [item for sublist in grids for item in sublist]
            visualize_task(
                task=grids,
                name=f"{all_task_ids[batch_i]}_{is_same}", # type: ignore
                out_path=f"debug_train_data/{img_idx}_{batch_i}.jpg",
            )

    # get input ids lens
    input_ids_lens = []
    for pair_i in range(required_num_pair):
        input_ids_lens.append([len(ids) for ids in pair_idx_to_input_ids[pair_i]])

    # pad
    padded_input_ids = []
    padded_attention_mask = []
    padded_label_ids = []
    for input_ids, attention_mask, label_ids in zip(pair_idx_to_input_ids, pair_idx_to_attention_mask, pair_idx_to_label_ids):
        input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.train_pad_side)
        attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.train_pad_side)
        label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.train_pad_side)
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_label_ids.append(label_ids)

    extra_padded_input_ids = []
    extra_padded_attention_mask = []
    extra_padded_label_ids = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for input_ids, attention_mask, label_ids in zip(padded_input_ids, padded_attention_mask, padded_label_ids):
            input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
                [input_ids, attention_mask, label_ids],
                padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
                pad_len=dataset.debug_pad_len,
                side=dataset.train_pad_side,
            )
            extra_padded_input_ids.append(input_ids)
            extra_padded_attention_mask.append(attention_mask)
            extra_padded_label_ids.append(label_ids)
    else:
        extra_padded_input_ids = padded_input_ids
        extra_padded_attention_mask = padded_attention_mask
        extra_padded_label_ids = padded_label_ids

    batch_dict = {
        "input_ids": extra_padded_input_ids,
        "attention_mask": extra_padded_attention_mask,
        "label_ids": extra_padded_label_ids,
        "input_ids_lens": input_ids_lens,
        "num_pairs": [required_num_pair] * batch_size,
        "is_same": is_same,
        "task_identifiers": ["-"] * batch_size,
    }
    return batch_dict


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    sampled_indices = sorted(random.sample(range(1, dataset.debug_len), k=dataset.max_num_pair-1)) + [dataset.debug_len]
    pair_lens = []
    for i in range(len(sampled_indices)):
        if i == 0:
            pair_lens.append(sampled_indices[0])
        else:
            pair_lens.append(sampled_indices[i] - sampled_indices[i-1])
    assert len(pair_lens) == dataset.max_num_pair

    input_ids_of_each_pair = [torch.randint(0, 30, (batch_size, l), dtype=torch.int64, device='cpu') for l in pair_lens]
    attention_mask_of_each_pair = [torch.full((batch_size, l), 1, dtype=torch.int64, device='cpu') for l in pair_lens]
    input_ids_lens = [[l] * batch_size for l in pair_lens]
    assert sum(x.shape[1] for x in input_ids_of_each_pair) == dataset.debug_len

    return {
        "input_ids": input_ids_of_each_pair,
        "attention_mask": attention_mask_of_each_pair,
        "label_ids": input_ids_of_each_pair,
        "input_ids_lens": input_ids_lens,
        "num_pairs": [dataset.max_num_pair] * batch_size,
        "is_same": False,
        "task_identifiers": ["-"] * batch_size,
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
        tokenizer: ARCTokenizer,
        ntokens: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        train_pad_side: str,
        gen_pad_side: str,
        debug_len: int,
        no_dim: bool,
        no_separate_color_tokens: bool,
        extra_inference_pairs: int,
        limit_inference_pairs: bool,
        limit_inference_pairs_strict: bool,
        max_num_train_pair: int,
        max_seq_len: int,
        no_bos: bool,
        only_first_bos: bool,
    ):
        self.permute_n = permute_n
        self.augment_n = augment_n
        self.permute_iters = permute_iters
        self.seed = seed
        self.tokenizer = tokenizer
        self.ntokens = ntokens
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.train_pad_side = train_pad_side
        self.gen_pad_side = gen_pad_side
        self.debug_len = debug_len
        self.no_dim = no_dim
        self.no_separate_color_tokens = no_separate_color_tokens
        self.limit_inference_pairs = limit_inference_pairs
        self.limit_inference_pairs_strict = limit_inference_pairs_strict
        self.max_num_train_pair = max_num_train_pair # max num pair in training, used for limiting inference
        self.max_seq_len = max_seq_len
        self.no_bos = no_bos
        self.only_first_bos = only_first_bos

        self.extra_inference_pairs = extra_inference_pairs
        self.inference_pair_rng = np.random.RandomState(seed)

        self.augmenters = [Transpose(), Flip(0), Flip(1), Rotate(90), Rotate(180)]

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

        # NOTE: we are not filtering here, just pray it works anyway
        # augment data for voting
        # since this function has to do filtering, might as well parse data as well
        self.eval_tasks = []
        for task in tasks:
            new_tasks = self.get_task_augmentations_leave_ns_filtered(task, leave_ns=leave_ns)
            if len(new_tasks) == 0 and leave_ns_inc:
                new_tasks = self.get_task_augmentations_leave_ns_filtered(task, leave_ns=leave_ns + [leave_ns[-1] + 1])
            self.eval_tasks += new_tasks
        logger.info(f'augmented data from {len(tasks)} to {len(self.eval_tasks)}')

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
        min_len, max_len = 1e6, 0
        for d in parsed_data:
            min_len = min(min_len, sum(len(i) for i in d['input_ids']) + len(d['gen_input_ids']) + d['out_token_length'] + 1) # type: ignore
            max_len = max(max_len, sum(len(i) for i in d['input_ids']) + len(d['gen_input_ids']) + d['out_token_length'] + 1) # type: ignore
        logger.info(f"encoder sequence length range from {min_len} to {max_len}]")
        del parsed_data

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

    def format_and_filter(self, task: Task, permutation: Optional[List[int]] = None, extra_inference_pairs: int = 0) -> Optional[Dict]:
        # do not add any randomness to this function!
        # this function only filters by token length, not by grid dimension
        # even the voting augmentation does not increase resolution
        assert task.max_height() <= 30 and task.max_width() <= 30

        # Build encoder text
        task = copy.deepcopy(task)

        # permute if given
        if permutation is not None:
            assert set(permutation) == set(range(len(task.train_examples)))
            task.train_examples = [task.train_examples[permute_i] for permute_i in permutation]

        # extra inference pairs for inference scaling
        if extra_inference_pairs > 0:
            # determine number of pairs to sample
            num_new_pairs = extra_inference_pairs
            if self.limit_inference_pairs:
                num_new_pairs = min(extra_inference_pairs, self.max_num_train_pair - len(task.train_examples))
            # sample new pairs
            if num_new_pairs > 0:
                task.train_examples += self.inference_pair_rng.choice(
                    task.train_examples, # type: ignore
                    size=num_new_pairs,
                    replace=True,
                ).tolist() # type: ignore

        # if strict, eval samples are always at most max_num_train_pair
        if self.limit_inference_pairs_strict:
            task.train_examples = task.train_examples[:self.max_num_train_pair]

        # parse task
        pair_idx_to_input_ids = []
        pair_idx_to_attention_mask = []
        gen_input_ids = None # collect the last input grid
        gen_output_ids = None # collect the last output grid
        out_token_length = -1

        # demonstration pairs
        for pair_i, example in enumerate(task.train_examples):
            input_grid_ids, output_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=False if self.no_bos else (pair_i == 0 if self.only_first_bos else True),
                no_dim=self.no_dim,
                no_separate_color_tokens=self.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            pair_idx_to_input_ids.append(input_ids)
            pair_idx_to_attention_mask.append(attention_mask)

        # test pair
        gen_input_ids, gen_output_ids = self.tokenizer.get_input_and_output_grid_ids(
            example=task.test_example,
            add_bos=False if self.no_bos else (not self.only_first_bos),
            no_dim=self.no_dim,
            no_separate_color_tokens=self.no_separate_color_tokens,
        )
        assert isinstance(gen_input_ids, torch.Tensor) and isinstance(gen_output_ids, torch.Tensor)
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64)
        out_token_length = len(gen_output_ids) - 1 # remove eos token

        # filter just in case (test with eval script later to see if this limits performance)
        total_seq_len = sum(x.shape[0] for x in pair_idx_to_input_ids) + gen_input_ids.shape[0] + gen_output_ids.shape[0]
        if total_seq_len > self.max_seq_len:
            return None

        # decoder label text has to be non-color-mapped
        label_texts = self.tokenizer.decode(
            gen_output_ids,
            no_separate_color_tokens=self.no_separate_color_tokens
        )[:-len(self.tokenizer.eos_token)]

        # for i in range(len(pair_idx_to_input_ids)): print(self.tokenizer.decode(pair_idx_to_input_ids[i], no_separate_color_tokens=self.no_separate_color_tokens), '\n')

        return {
            "task_id": task.name,
            "inverter": task.inverter if hasattr(task, "inverter") else "", # type: ignore
            "input_ids": pair_idx_to_input_ids,
            "attention_mask": pair_idx_to_attention_mask,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "out_token_length": out_token_length,
            "label_texts": label_texts,  # used for exact match
        }

    def __len__(self):
        return len(self.eval_tasks)

    def __getitem__(self, idx):
        return self.format_and_filter(self.eval_tasks[idx])


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    task_ids = [x['task_id'] for x in batch]
    inverters = [x['inverter'] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    gen_input_ids = [x["gen_input_ids"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    out_token_length = [x["out_token_length"] for x in batch]
    label_texts = [x["label_texts"] for x in batch]

    batch_size = len(task_ids)

    # save number of pairs before padding
    num_pairs = [len(i) for i in input_ids]
    max_num_pairs = max(num_pairs)

    # pad all samples in batch with 0-tensor
    # also format then to [pair_idx, batch_size, seq_len]
    pair_idx_to_input_ids = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    pair_idx_to_attention_mask = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    for batch_i, (ids, mask) in enumerate(zip(input_ids, attention_mask)): # iterate over batch here
        pad_ids = torch.tensor([0], dtype=ids[0].dtype)
        pad_mask = torch.tensor([0], dtype=mask[0].dtype)
        ids += [pad_ids] * (max_num_pairs - len(ids))
        mask += [pad_mask] * (max_num_pairs - len(mask))
        for pair_i, (ids_, mask_) in enumerate(zip(ids, mask)):
            pair_idx_to_input_ids[pair_i][batch_i] = ids_
            pair_idx_to_attention_mask[pair_i][batch_i] = mask_

    # get lengths of ids
    input_ids_lens = []
    for pair_i in range(max_num_pairs):
        input_ids_lens.append([len(ids) for ids in pair_idx_to_input_ids[pair_i]])
    gen_input_ids_lens = [len(ids) for ids in gen_input_ids]

    # actual padding of sequences
    padded_input_ids = []
    padded_attention_mask = []
    for input_ids, attention_mask in zip(pair_idx_to_input_ids, pair_idx_to_attention_mask):
        input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.train_pad_side)
        attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.train_pad_side)
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)

    # debug extra padding
    extra_padded_input_ids = []
    extra_padded_attention_mask = []
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        for input_ids, attention_mask in zip(padded_input_ids, padded_attention_mask):
            input_ids, attention_mask = debug_extra_pad_tensors(
                [input_ids, attention_mask],
                padding_values=[dataset.tokenizer.pad_token_id, 0],
                pad_len=dataset.debug_pad_len,
                side=dataset.train_pad_side,
            )
            extra_padded_input_ids.append(input_ids)
            extra_padded_attention_mask.append(attention_mask)
    else:
        extra_padded_input_ids = padded_input_ids
        extra_padded_attention_mask = padded_attention_mask

    # pad the gen arguments (and debug padding again)
    gen_input_ids = pad_sequence_with_side(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.gen_pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.gen_pad_side)
    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        gen_input_ids, gen_attention_mask = debug_extra_pad_tensors(
            [gen_input_ids, gen_attention_mask],
            padding_values=[dataset.tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.gen_pad_side,
        )

    batch_dict = {
        "task_ids": task_ids,
        "inverters": inverters,
        "input_ids": extra_padded_input_ids,
        "attention_mask": extra_padded_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "label_texts": label_texts,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
        "num_pairs": num_pairs,
    }
    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    sampled_indices = sorted(random.sample(range(1, dataset.debug_len), k=dataset.max_num_train_pair)) + [dataset.debug_len]
    pair_lens = []
    for i in range(len(sampled_indices)):
        if i == 0:
            pair_lens.append(sampled_indices[0])
        else:
            pair_lens.append(sampled_indices[i] - sampled_indices[i-1])
    assert len(pair_lens) == dataset.max_num_train_pair + 1

    input_ids_of_each_pair = [torch.randint(0, 30, (batch_size, l), dtype=torch.int64, device='cpu') for l in pair_lens[:-1]]
    attention_mask_of_each_pair = [torch.full((batch_size, l), 1, dtype=torch.int64, device='cpu') for l in pair_lens[:-1]]

    gen_input_ids = torch.randint(0, 30, (batch_size, pair_lens[-1] // 2), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64, device='cpu')
    gen_output_ids = torch.randint(0, 30, (batch_size, pair_lens[-1] // 2 + 1), dtype=torch.int64, device='cpu')
    out_token_length = [gen_output_ids.shape[1]] * batch_size

    input_ids_lens = [[l] * batch_size for l in pair_lens[:-1]]
    gen_input_ids_lens = [pair_lens[-1] // 2] * batch_size

    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    label_texts = ['1\n1\n1'] * batch_size
    num_pairs = [dataset.max_num_train_pair] * batch_size

    batch_dict = {
        "task_ids": task_ids,
        "inverters": [""] * batch_size,
        "input_ids": input_ids_of_each_pair,
        "attention_mask": attention_mask_of_each_pair,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "label_texts": label_texts,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
        "num_pairs": num_pairs,
    }
    return batch_dict


########################################
# Gradient Search Dataset
########################################
class GSDataset(Dataset):
    def __init__(
        self,
        task: Task,
        tokenizer: ARCTokenizer,
        debug_random_pad: bool,
        debug_pad_len: int,
        train_pad_side: str,
        no_dim: bool,
        no_separate_color_tokens: bool,
        no_bos: bool,
        only_first_bos: bool,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.train_pad_side = train_pad_side
        self.no_dim = no_dim
        self.no_separate_color_tokens = no_separate_color_tokens
        self.no_bos = no_bos
        self.only_first_bos = only_first_bos

        # format data (only use demonstration pairs)
        self.parsed_examples = [self.format(example) for example in task.train_examples]

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format(self, example: Example) -> Optional[Dict]:
        # tasks are filtered by EvalDataset already, shouldn't have grids too big
        assert max(example.input.shape) <= 30 and max(example.output.shape) <= 30

        input_grid_ids, output_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
            example=example,
            add_bos=False if self.no_bos else (not self.only_first_bos),
            no_dim=self.no_dim,
            no_separate_color_tokens=self.no_separate_color_tokens,
        )
        input_ids = torch.cat([input_grid_ids, output_grid_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
        label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
        label_ids = torch.cat([label_ids, output_grid_ids])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
        }


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]

    input_ids_lens = [len(x) for x in input_ids]
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
        "input_ids_lens": input_ids_lens,
    }
    return batch_dict


########################################
# Test-Time-Training Dataset
########################################
class TTTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_samples_per_task: int,
        permute_n: int,
        tokenizer: ARCTokenizer,
        seed: int,
        pad_side: str,
        debug_no_aug: bool,
        aug_type: str,
        no_dim: bool,
        no_separate_color_tokens: bool,
        max_seq_len: int,
        no_bos: bool,
        only_first_bos: bool,
    ):
        self.permute_n = permute_n
        self.tokenizer = tokenizer
        self.seed = seed
        self.pad_side = pad_side
        self.debug_no_aug = debug_no_aug
        self.no_dim = no_dim
        self.no_separate_color_tokens = no_separate_color_tokens
        self.max_seq_len = max_seq_len
        self.no_bos = no_bos
        self.only_first_bos = only_first_bos

        # get all augmenters
        d8_augmenters = get_d8_augmenters(include_identity=False)
        extra_augmenters = get_mit_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)
        if aug_type == "none":
            augmenters = []
        elif aug_type == "both":
            augmenters = d8_augmenters + extra_augmenters
        elif aug_type == "d8":
            augmenters = d8_augmenters
        else:
            augmenters = extra_augmenters

        # keep unique augmenters
        self.augmenters = []
        for aug in augmenters:
            if str(aug) not in [str(x) for x in self.augmenters]:
                self.augmenters.append(aug)

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
        self.ttt_tasks = self.task_to_ttt_filtered_data(max_gen=max_samples_per_task)
        rng.shuffle(self.ttt_tasks) # type: ignore

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
        task = copy.deepcopy(task)
        num_pair = len(task.train_examples) + 1

        # parse task
        pair_idx_to_input_ids = []
        pair_idx_to_attention_mask = []
        pair_idx_to_label_ids = []

        for pair_i in range(num_pair):
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=False if self.no_bos else (pair_i == 0 if self.only_first_bos else True),
                no_dim=self.no_dim,
                no_separate_color_tokens=self.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_grid_ids])
            # append
            pair_idx_to_input_ids.append(input_ids)
            pair_idx_to_attention_mask.append(attention_mask)
            pair_idx_to_label_ids.append(label_ids)

        return {
            "input_ids": pair_idx_to_input_ids,
            "attention_mask": pair_idx_to_attention_mask,
            "label_ids": pair_idx_to_label_ids,
        }

    def __len__(self):
        return len(self.ttt_tasks)

    def __getitem__(self, idx):
        return self.format_and_filter(self.ttt_tasks[idx])


def collate_fn_ttt(batch: List[Dict], dataset: TTTDataset) -> Dict:
    batch_size = len(batch)

    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]

    # save number of pairs before padding
    num_pairs = [len(i) for i in input_ids]
    max_num_pairs = max(num_pairs)

    # for now, we just pad so that all samples in batch have the same number of pairs]
    # also format then to [pair_idx, batch_size, seq_len]
    pair_idx_to_input_ids = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    pair_idx_to_attention_mask = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    pair_idx_to_label_ids = [[torch.tensor(0.0) for _ in range(batch_size)] for _ in range(max_num_pairs)]
    for batch_i, (ids, mask, label) in enumerate(zip(input_ids, attention_mask, label_ids)): # iterate over batch here
        min_idx, pad_ids = min(enumerate(ids), key=lambda x: len(x[1]))
        pad_mask = mask[min_idx]
        pad_label = label[min_idx]
        ids += [pad_ids] * (max_num_pairs - len(ids))
        mask += [pad_mask] * (max_num_pairs - len(mask))
        label += [pad_label] * (max_num_pairs - len(label))
        for pair_i, (ids_, mask_, label_) in enumerate(zip(ids, mask, label)):
            pair_idx_to_input_ids[pair_i][batch_i] = ids_
            pair_idx_to_attention_mask[pair_i][batch_i] = mask_
            pair_idx_to_label_ids[pair_i][batch_i] = label_

    # get lengths of ids
    input_ids_lens = []
    for pair_i in range(max_num_pairs):
        input_ids_lens.append([len(ids) for ids in pair_idx_to_input_ids[pair_i]])

    # actual padding of sequences
    padded_input_ids = []
    padded_attention_mask = []
    padded_label_ids = []
    for input_ids, attention_mask, label_ids in zip(pair_idx_to_input_ids, pair_idx_to_attention_mask, pair_idx_to_label_ids):
        input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
        attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
        label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.pad_side)
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_label_ids.append(label_ids)

    batch_dict = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "label_ids": padded_label_ids,
        "input_ids_lens": input_ids_lens,
        "num_pairs": num_pairs,
    }
    return batch_dict
