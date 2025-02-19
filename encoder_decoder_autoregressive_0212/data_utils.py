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
from typing import Dict, List, Tuple, Iterator, Union
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

    def encode_grid_to_tensor(self, grid: np.ndarray, separate_color_tokens: bool) -> torch.Tensor:
        assert grid.ndim == 2
        token_ids = []
        for row in grid:
            for x in row:
                if separate_color_tokens:
                    token_ids.append(self.token_to_id[f"c{str(x)}"])
                else:
                    token_ids.append(self.token_to_id[str(x)])
            token_ids.append(self.token_to_id["\n"])
        token_ids = token_ids[:-1] # no \n at end
        return torch.tensor(token_ids, dtype=torch.int64)

    def convert_token_to_id(self, token: str) -> torch.Tensor:
        return torch.tensor([self.token_to_id[token]], dtype=torch.int64)

    def get_input_and_output_grid_ids(self, example: Example, add_bos: bool, no_dim: bool, separate_color_tokens: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        input_grid_ids = self.encode_grid_to_tensor(example.input, separate_color_tokens)
        output_grid_ids = self.encode_grid_to_tensor(example.output, separate_color_tokens)

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
            separate_color_tokens: bool,
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
                    if token.startswith('c') and separate_color_tokens:
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

    def batch_decode(self, batch_token_ids: torch.Tensor, skip_special_tokens: bool, separate_color_tokens: bool) -> List[str]:
        assert batch_token_ids.dim() == 2
        texts = []
        for token_ids in batch_token_ids:
            text = self.decode(
                token_ids=token_ids,
                separate_color_tokens=separate_color_tokens,
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
    files = glob.glob(f'/scratch/yl11330/ConceptARC/corpus/*/*.json')
    files += glob.glob(f'/scratch/yl11330/ConceptARC/MinimalTasks/*.json')
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


def pairs_to_color_equiv_mapping(task: Task) -> Tuple[Task, np.ndarray]:
    # turn to dict 2dlist format
    chosen_pairs = task.train_examples + [task.test_example]
    chosen_pairs = [{'input': p.input.tolist(), 'output': p.output.tolist()} for p in chosen_pairs]
    # get color mapping
    flattened = [[p['input'], p['output']] for p in chosen_pairs]
    flattened = list(itertools.chain.from_iterable(itertools.chain.from_iterable(itertools.chain.from_iterable(flattened))))
    color_mapping = {val: idx for idx, val in enumerate(dict.fromkeys(flattened))}
    # some color may not be in task, assume they are in order
    for c in range(10):
        if c not in color_mapping:
            color_mapping[c] = len(color_mapping)
    assert set(color_mapping.keys()) == set(color_mapping.values()) == set(range(10))
    color_mapping = np.array([color_mapping[i] for i in range(10)], dtype=np.int64)
    # apply mapping
    for pair in chosen_pairs:
        pair['input'] = color_mapping[np.array(pair['input'])].tolist()
        pair['output'] = color_mapping[np.array(pair['output'])].tolist()
    # inverse mapping
    inverse_mapping = np.zeros(10, dtype=np.int64)
    for key, value in enumerate(color_mapping):
        inverse_mapping[value] = key
    # turn to task
    task = Task(
        name=task.name,
        train_examples=[
            Example(input=np.array(p['input']), output=np.array(p['output'])) for p in chosen_pairs[:-1]
        ],
        test_example=Example(input=np.array(chosen_pairs[-1]['input']), output=np.array(chosen_pairs[-1]['output']))
    )
    return task, inverse_mapping


########################################
# Training Dataset
########################################

class TrainDataset(Dataset):
    def __init__(
        self,
        train_data_dir: str,
        eval_train_dir: str,
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
        color_equiv: bool,
        curriculum_iters: int,
        global_batch_size: int,
        no_dim: bool,
        separate_color_tokens: bool,
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
        self.color_equiv = color_equiv
        self.curriculum_iters = curriculum_iters
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers if num_workers > 0 else 1
        self.no_dim = no_dim
        self.separate_color_tokens = separate_color_tokens

        # setup args
        self.normalized_ratio = np.array([self.re_arc_ratio, self.concept_arc_ratio, self.arc_heavy_ratio])
        self.normalized_ratio /= np.sum(self.normalized_ratio)
        self.d8_augmenters = get_d8_augmenters(include_identity=True)
        self.extra_augmenters = get_mit_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)

        # seed and process_index
        if num_workers == 0:
            self.rngs = [np.random.RandomState(seed + process_index)]
        else:
            self.rngs = [np.random.RandomState(seed + i) for i in range(num_workers * process_index, num_workers * (process_index + 1))]

        # keep track of how many samples for this worker
        if num_workers == 0:
            self.workers_to_num_sample = {0: 0}
        else:
            self.workers_to_num_sample = {i: 0 for i in range(num_workers)}

        # num pair must be the same across gpus
        if num_workers == 0:
            self.num_pair_rngs = [np.random.RandomState(seed)]
        else:
            self.num_pair_rngs = [np.random.RandomState(seed + i) for i in range(num_workers)]

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
        self.concept_arc_id_to_pairs = {}
        if concept_arc_ratio > 0.0:
            self.concept_arc_id_to_pairs = load_concept_arc()
            logger.info(f'loaded {len(self.concept_arc_id_to_pairs)} concept-arc tasks')

        # load heavy-arc data
        self.heavy_arc_data = {}
        if arc_heavy_ratio > 0.0:
            self.heavy_arc_data = load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems")["train"] # type: ignore
            logger.info(f'loaded {len(self.heavy_arc_data)} arc-heavy tasks')

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
    num_pair_rng = dataset.num_pair_rngs[int(worker_id)]

    # update curriculum
    dataset.workers_to_num_sample[int(worker_id)] += batch_size

    # the restriction here is to enforce all list of pairs in batch are equal length
    all_task_ids = []
    all_np_chosen_pairs = []

    # must sample this number of pairs to avoid GPU synchronization issues
    if dataset.curriculum_iters > 0:
        required_num_pair = dataset.min_num_pair + \
            (dataset.workers_to_num_sample[int(worker_id)] * dataset.num_workers) // (dataset.global_batch_size * dataset.curriculum_iters)
        required_num_pair = min(required_num_pair, dataset.max_num_pair)
    else:
        required_num_pair = num_pair_rng.choice(list(range(dataset.min_num_pair, dataset.max_num_pair + 1)))

    # sample random task from random dataset, if grid size >30 or does not have enough for required_num_pair, retry
    while len(all_task_ids) < batch_size:
        dataset_name = rng.choice(["re-arc", "concept-arc", "arc-heavy"], p=dataset.normalized_ratio)

        # STEP 1: get task id and pairs, sample task id until reaching num chosen pair
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

        # need at least required_num_pair to proceed
        if len(all_pairs) < required_num_pair:
            continue

        # STEP 2: decide on extra augmentations and io augmentation choice for pairs
        if not dataset.no_d8:
            d8_augmenter = rng.choice(dataset.d8_augmenters) # type: ignore
        else:
            d8_augmenter = None

        if rng.rand() < dataset.extra_augment_ratio:
            extra_augmenter = rng.choice(dataset.extra_augmenters) # type: ignore
            io_augmentation_choice = rng.choice(["input_only", "output_only", "both"]) if dataset.extra_augment_single_grid else "both"
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

        # STEP 4: found a valid task!
        all_task_ids.append(task_id) # type: ignore
        all_np_chosen_pairs.append(np_chosen_pairs)

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
        tasks = [PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng) for task in tasks]

    # color equivariance AFTER all augmentations
    if dataset.color_equiv:
        tasks = [pairs_to_color_equiv_mapping(task)[0] for task in tasks]

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
                add_bos=True,
                no_dim=dataset.no_dim,
                separate_color_tokens=dataset.separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_grid_ids])
            # append
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_label_ids.append(label_ids)

        pair_idx_to_input_ids.append(batch_input_ids)
        pair_idx_to_attention_mask.append(batch_attention_mask)
        pair_idx_to_label_ids.append(batch_label_ids)

    # visualize some training data
    if dataset.debug_train_data:
        img_idx = max([int(Path(p).stem.split('_')[0]) for p in glob.glob(f"debug_train_data/*.jpg")], default=-1) + 1
        for batch_i in range(batch_size):
            input_ids = [pair_idx_to_input_ids[pair_i][batch_i] for pair_i in range(required_num_pair)]
            texts = [
                dataset.tokenizer.decode(ids, skip_special_tokens=True, separate_color_tokens=dataset.separate_color_tokens)
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

    batch_dict = {
        "input_ids": extra_padded_input_ids,
        "attention_mask": extra_padded_attention_mask,
        "label_ids": extra_padded_label_ids,
        "input_ids_lens": input_ids_lens,
        "num_pairs": [required_num_pair] * batch_size,
    }
    return batch_dict


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = [torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu') for _ in range(dataset.max_num_pair)]
    attention_mask = [torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu') for _ in range(dataset.max_num_pair)]
    input_ids_lens = [[dataset.debug_len] * batch_size for _ in range(dataset.max_num_pair)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "input_ids_lens": input_ids_lens,
        "num_pairs": [dataset.max_num_pair] * batch_size,
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
        color_equiv: bool,
        no_dim: bool,
        separate_color_tokens: bool,
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
        self.color_equiv = color_equiv
        self.no_dim = no_dim
        self.separate_color_tokens = separate_color_tokens

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
            new_tasks = self.get_task_augmentations_leave_ns(task, leave_ns=leave_ns)
            if len(new_tasks) == 0 and leave_ns_inc:
                new_tasks = self.get_task_augmentations_leave_ns(task, leave_ns=leave_ns + [leave_ns[-1] + 1])
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
            min_len = min(min_len, sum(len(i) for i in d['input_ids'])) # type: ignore
            max_len = max(max_len, sum(len(i) for i in d['input_ids'])) # type: ignore
        logger.info(f"encoder sequence length range from {min_len} to {max_len}]")
        del parsed_data

    def get_task_augmentations_leave_ns(
            self,
            task: Task,
            leave_ns: list[int],
        ) -> List[Task]:
        # get augmented queries
        augmented_tasks = []
        for leave_n in leave_ns:
            augmented_tasks += self.get_task_augmentations_leave_n(task, leave_n=leave_n)
        return augmented_tasks

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

    def format(self, task: Task, permutation: Optional[List[int]] = None) -> Dict:
        # do not add any randomness to this function!
        # this function only filters by token length, not by grid dimension
        # even the voting augmentation does not increase resolution
        assert task.max_height() <= 30 and task.max_width() <= 30

        # color equivariance
        inverse_color_map = np.array(range(10), dtype=np.int64)
        original_test_example = copy.deepcopy(task.test_example)
        if self.color_equiv:
            task, inverse_color_map = pairs_to_color_equiv_mapping(task)

        # permute if given
        train_examples = task.train_examples
        if permutation is not None:
            assert set(permutation) == set(range(len(train_examples)))
            task.train_examples = [train_examples[permute_i] for permute_i in permutation]

        # Build encoder text
        task = copy.deepcopy(task)
        num_pair = len(task.train_examples) + 1

        # parse task
        pair_idx_to_input_ids = []
        pair_idx_to_attention_mask = []
        pair_idx_to_label_ids = []
        gen_input_ids = None # collect the last input grid
        gen_output_ids = None # collect the last output grid
        out_token_length = -1

        for pair_i in range(num_pair):
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=True,
                no_dim=self.no_dim,
                separate_color_tokens=self.separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_grid_ids])
            pair_idx_to_input_ids.append(input_ids)
            pair_idx_to_attention_mask.append(attention_mask)
            pair_idx_to_label_ids.append(label_ids)

            if pair_i == num_pair - 1:
                gen_input_ids = input_grid_ids
                gen_output_ids = output_grid_ids
                out_token_length = len(output_grid_ids) - 1 # remove eos token

        assert isinstance(gen_input_ids, torch.Tensor) and isinstance(gen_output_ids, torch.Tensor)
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64)

        # decoder label text has to be non-color-mapped
        if self.color_equiv:
            _, original_test_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
                example=original_test_example,
                add_bos=True,
                no_dim=self.no_dim,
                separate_color_tokens=self.separate_color_tokens,
            )
            label_texts = self.tokenizer.decode(
                original_test_grid_ids,
                separate_color_tokens=self.separate_color_tokens
            )[:-len(self.tokenizer.eos_token)]
        else:
            label_texts = self.tokenizer.decode(
                gen_output_ids,
                separate_color_tokens=self.separate_color_tokens
            )[:-len(self.tokenizer.eos_token)]

        return {
            "task_id": task.name,
            "inverter": task.inverter if hasattr(task, "inverter") else "", # type: ignore
            "input_ids": pair_idx_to_input_ids,
            "attention_mask": pair_idx_to_attention_mask,
            "label_ids": pair_idx_to_label_ids,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "out_token_length": out_token_length,
            "label_texts": label_texts,  # used for exact match
            "inverse_color_map": inverse_color_map,
        }

    def __len__(self):
        return len(self.eval_tasks)

    def __getitem__(self, idx):
        return self.format(self.eval_tasks[idx])

    def get_io_permuted_batches(self, batch_idxs: List[int]) -> Iterator[Tuple[List[Dict], List[bool]]]:
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
            permuted_tasks = [self.format(task, permutation) for task, permutation in zip(avail_eval_tasks, permutations)]
            yield (permuted_tasks, avail)


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    task_ids = [x['task_id'] for x in batch]
    inverters = [x['inverter'] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]
    gen_input_ids = [x["gen_input_ids"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    out_token_length = [x["out_token_length"] for x in batch]
    label_texts = [x["label_texts"] for x in batch]
    inverse_color_maps = [x["inverse_color_map"] for x in batch]

    batch_size = len(task_ids)

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
    gen_input_ids_lens = [len(ids) for ids in gen_input_ids]

    # actual padding of sequences
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

    # debug extra padding
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
        "label_ids": extra_padded_label_ids,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "label_texts": label_texts,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
        "num_pairs": num_pairs,
        "inverse_color_maps": inverse_color_maps,
    }
    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly
    max_num_pair = 10

    input_ids = [torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu') for _ in range(max_num_pair)]
    attention_mask = [torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu') for _ in range(max_num_pair)]
    input_ids_lens = [[dataset.debug_len] * batch_size for _ in range(max_num_pair)]

    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len // 2 + 1), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full((batch_size, dataset.debug_len // 2 + 1), 1, dtype=torch.int64, device='cpu')
    gen_input_ids_lens = [dataset.debug_len // 2 + 1] * batch_size

    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    out_token_length = [dataset.debug_len // 2 + 1] * batch_size
    label_texts = ['1\n1\n1'] * batch_size
    num_pairs = [max_num_pair] * batch_size
    inverse_color_maps = [np.array(range(10), dtype=np.int64)] * batch_size

    batch_dict = {
        "task_ids": task_ids,
        "inverters": [""] * batch_size,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "label_texts": label_texts,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
        "num_pairs": num_pairs,
        "inverse_color_maps": inverse_color_maps,
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
        ntokens: int,
        debug_random_pad: bool,
        debug_pad_len: int,
        gen_pad_side: str,
        no_dim: bool,
        separate_color_tokens: bool,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.ntokens = ntokens
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.gen_pad_side = gen_pad_side
        self.no_dim = no_dim
        self.separate_color_tokens = separate_color_tokens

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
            add_bos=True,
            no_dim=self.no_dim,
            separate_color_tokens=self.separate_color_tokens,
        )
        input_ids = torch.cat([input_grid_ids, output_grid_ids])
        attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
        label_ids = torch.cat([
            torch.full(input_grid_ids.shape, -100, dtype=torch.int64),
            output_grid_ids,
        ])

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
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.gen_pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.gen_pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.gen_pad_side)

    if dataset.debug_random_pad and dataset.debug_pad_len > -1:
        input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
            [input_ids, attention_mask, label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.gen_pad_side,
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
        separate_color_tokens: bool,
    ):
        self.permute_n = permute_n
        self.tokenizer = tokenizer
        self.seed = seed
        self.pad_side = pad_side
        self.debug_no_aug = debug_no_aug
        self.no_dim = no_dim
        self.separate_color_tokens = separate_color_tokens

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
        self.ttt_tasks = self.task_to_ttt_data(max_gen=max_samples_per_task)
        rng.shuffle(self.ttt_tasks) # type: ignore

    def task_to_ttt_data(self, max_gen: int) -> List[Task]:
        # if leave 1 is enough, return it
        leave_1_train_tasks = self.task_to_ttt_data_leave_n(leave_n=1, max_gen=max_gen)
        if len(leave_1_train_tasks) >= max_gen:
            return leave_1_train_tasks
        # else generate leave 2 and append to leave 1
        max_gen_leave_2 = max_gen - len(leave_1_train_tasks)
        leave_1_train_tasks += self.task_to_ttt_data_leave_n(leave_n=2, max_gen=max_gen_leave_2)
        return leave_1_train_tasks

    def task_to_ttt_data_leave_n(self, leave_n: int, max_gen: int) -> List[Task]:
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
        return augmented_tasks[:max_gen]

    def format(self, task: Task) -> Dict:
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
                add_bos=True,
                no_dim=self.no_dim,
                separate_color_tokens=self.separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            # label id for all except first pair
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            if pair_i == 0:
                label_ids = torch.cat([label_ids, torch.full(output_grid_ids.shape, -100, dtype=torch.int64)])
            else:
                label_ids = torch.cat([label_ids, output_grid_ids])
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
        return self.format(self.ttt_tasks[idx])


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
