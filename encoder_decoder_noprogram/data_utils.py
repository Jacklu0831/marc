from datasets import load_dataset
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

    def get_input_and_output_grid_ids(self, example: Example, add_bos: bool, no_separate_color_tokens: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dim_ids = self.encode_dimensions_to_tensor(len(example.input), len(example.input[0]))
        output_dim_ids = self.encode_dimensions_to_tensor(len(example.output), len(example.output[0]))
        input_grid_ids = self.encode_grid_to_tensor(example.input, no_separate_color_tokens)
        output_grid_ids = self.encode_grid_to_tensor(example.output, no_separate_color_tokens)

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
            no_separate_color_tokens: bool,
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
        re_arc_ratio: float,
        concept_arc_ratio: float,
        arc_heavy_ratio: float,
        tokenizer: ARCTokenizer,
        total_steps: int,
        extra_augment_ratio: float,
        extra_augment_single_grid: bool,
        seed: int,
        process_index: int,
        debug_fixed_order: bool,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        no_color_permute: bool,
        no_pair_permute: bool,
        no_d8: bool,
        min_num_pair: int,
        max_num_pair: int,
        no_train_original: bool,
        only_train_original: bool,
        debug_len: int,
        num_workers: int,
        no_separate_color_tokens: bool,
        max_seq_len: int,
        no_bos: bool,
    ):
        self.re_arc_ratio = re_arc_ratio
        self.concept_arc_ratio = concept_arc_ratio
        self.arc_heavy_ratio = arc_heavy_ratio

        self.tokenizer = tokenizer
        self._length = total_steps
        self.extra_augment_ratio = extra_augment_ratio
        self.extra_augment_single_grid = extra_augment_single_grid
        self.debug_fixed_order = debug_fixed_order
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.no_color_permute = no_color_permute
        self.no_pair_permute = no_pair_permute
        self.no_d8 = no_d8
        self.min_num_pair = min_num_pair
        self.max_num_pair = max_num_pair
        self.debug_len = debug_len
        self.no_separate_color_tokens = no_separate_color_tokens
        self.max_seq_len = max_seq_len
        self.no_bos = no_bos

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

    # the restriction here is to enforce all list of pairs in batch are equal length
    out_list = []

    # must sample this number of pairs to avoid GPU synchronization issues
    required_num_pair = num_pair_rng.choice(list(range(dataset.min_num_pair, dataset.max_num_pair + 1)))

    # sample random task from random dataset, if grid size >30 or does not have enough for required_num_pair, retry
    while len(out_list) < batch_size:
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

        if dataset.debug_fixed_order:
            required_num_pair = len(all_pairs)

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

        # apply color and pair permutation
        task = Task(
            name=task_id,
            train_examples=[
                Example(input=pair["input"], output=pair["output"])
                for pair in np_chosen_pairs[:-1]
            ],
            test_example=Example(input=np_chosen_pairs[-1]["input"], output=np_chosen_pairs[-1]["output"]),
        )

        if not dataset.no_pair_permute:
            task = PermuteExamples().apply_to_task(task, to_input=True, to_output=True, rng=rng)

        if not dataset.no_color_permute:
            task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)

        # we do a lil parsing
        pair_idx_to_input_ids = []
        pair_idx_to_attention_mask = []
        pair_idx_to_label_ids = []

        for pair_i in range(required_num_pair):
            # get inputids, attention, labelids for batch of pairs at pair_i
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = dataset.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=(pair_i == 0) if not dataset.no_bos else False,
                no_separate_color_tokens=dataset.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            # label id for all except first pair
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            if pair_i == 0:
                label_ids = torch.cat([label_ids, torch.full(output_grid_ids.shape, -100, dtype=torch.int64)])
            else:
                label_ids = torch.cat([label_ids, output_grid_ids])
            # aggregate
            pair_idx_to_input_ids.append(input_ids)
            pair_idx_to_attention_mask.append(attention_mask)
            pair_idx_to_label_ids.append(label_ids)

        # start idxs computed from cumsum of lengths of pairs except last, set first start idx to 1 for bos
        pair_start_idxs = np.cumsum([0, *[len(x) for x in pair_idx_to_input_ids[:-1]]]).tolist()
        pair_start_idxs[0] = 0 if dataset.no_bos else 1

        input_ids = torch.cat(pair_idx_to_input_ids)
        attention_mask = torch.cat(pair_idx_to_attention_mask)
        label_ids = torch.cat(pair_idx_to_label_ids)
        assert input_ids.shape == attention_mask.shape == label_ids.shape

        if len(input_ids) > dataset.max_seq_len:
            continue

        out_list.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "pair_start_idxs": pair_start_idxs,
        })

    input_ids = [x['input_ids'] for x in out_list]
    attention_mask = [x['attention_mask'] for x in out_list]
    label_ids = [x['label_ids'] for x in out_list]

    pair_start_idxs = [x['pair_start_idxs'] for x in out_list]
    if not dataset.no_bos:
        assert all(start_idxs[0] == 1 for start_idxs in pair_start_idxs)
    else:
        assert all(start_idxs[0] == 0 for start_idxs in pair_start_idxs)
    input_ids_lens = [len(i) for i in input_ids]

    # pad
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
        "input_ids_lens": input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
    }
    return batch_dict


def collate_fn_train_dummy(batch: List[int], dataset: TrainDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.debug_len] * batch_size

    if not dataset.no_bos:
        pair_start_idxs = [list(range(1, dataset.debug_len * 9 // 10, dataset.debug_len // 15))[:10] for _ in range(batch_size)]
    else:
        pair_start_idxs = [list(range(0, dataset.debug_len * 9 // 10, dataset.debug_len // 15))[:10] for _ in range(batch_size)]

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
        eval_dir: str,
        select_tasks_path: Optional[str],
        leave_ns: List[int],
        leave_ns_inc: bool,
        permute_n: int,
        augment_n: int,
        permute_iters: int,
        seed: int,
        tokenizer: ARCTokenizer,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        debug_len: int,
        no_separate_color_tokens: bool,
        max_seq_len: int,
        no_bos: bool,
    ):
        self.permute_n = permute_n
        self.augment_n = augment_n
        self.permute_iters = permute_iters
        self.seed = seed
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.debug_len = debug_len
        self.max_seq_len = max_seq_len
        self.no_separate_color_tokens = no_separate_color_tokens
        self.no_bos = no_bos

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
            min_len = min(min_len, len(d['input_ids'])) # type: ignore
            max_len = max(max_len, len(d['input_ids'])) # type: ignore
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

    def format_and_filter(self, task: Task) -> Optional[Dict]:
        # do not add any randomness to this function!
        # this function only filters by token length, not by grid dimension
        # even the voting augmentation does not increase resolution
        assert task.max_height() <= 30 and task.max_width() <= 30

        # Build encoder text
        task = copy.deepcopy(task)
        num_pair = len(task.train_examples) + 1

        # parse task
        pair_idx_to_input_ids = []
        pair_idx_to_attention_mask = []
        pair_idx_to_label_ids = []
        gen_input_ids = []
        gen_output_ids = None

        for pair_i in range(num_pair):
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=(pair_i == 0) if not self.no_bos else False,
                no_separate_color_tokens=self.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            # gen ids
            if pair_i < num_pair - 1:
                gen_input_ids.append(torch.cat([input_grid_ids, output_grid_ids]))
            else:
                gen_input_ids.append(input_grid_ids)
                gen_output_ids = output_grid_ids
            # label id for all except first pair
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            if pair_i == 0:
                label_ids = torch.cat([label_ids, torch.full(output_grid_ids.shape, -100, dtype=torch.int64)])
            else:
                label_ids = torch.cat([label_ids, output_grid_ids])
            pair_idx_to_input_ids.append(input_ids)
            pair_idx_to_attention_mask.append(attention_mask)
            pair_idx_to_label_ids.append(label_ids)

        # start idxs computed from cumsum of lengths of pairs except last, plus one for bos
        pair_start_idxs = np.cumsum([0, *[len(x) for x in pair_idx_to_input_ids[:-1]]]).tolist()
        pair_start_idxs[0] = 0 if self.no_bos else 1

        input_ids = torch.cat(pair_idx_to_input_ids)
        attention_mask = torch.cat(pair_idx_to_attention_mask)
        label_ids = torch.cat(pair_idx_to_label_ids)
        assert input_ids.shape == attention_mask.shape == label_ids.shape

        if len(input_ids) > self.max_seq_len:
            return None

        gen_input_ids = torch.cat(gen_input_ids)
        assert isinstance(gen_output_ids, torch.Tensor)
        out_token_length = len(gen_output_ids) - 1 # remove eos token
        gen_attention_mask = torch.full(gen_input_ids.shape, 1, dtype=torch.int64)
        label_texts = self.tokenizer.decode(
            gen_output_ids,
            no_separate_color_tokens=self.no_separate_color_tokens
        )[:-len(self.tokenizer.eos_token)]

        return {
            "task_id": task.name,
            "inverter": task.inverter if hasattr(task, "inverter") else "", # type: ignore
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "out_token_length": out_token_length,
            "label_texts": label_texts,  # used for exact match
            "pair_start_idxs": pair_start_idxs,
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
    label_ids = [x["label_ids"] for x in batch]
    gen_input_ids = [x["gen_input_ids"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    out_token_length = [x["out_token_length"] for x in batch]
    label_texts = [x["label_texts"] for x in batch]
    pair_start_idxs = [x["pair_start_idxs"] for x in batch]
    assert all(start_idxs[0] == 1 - dataset.no_bos for start_idxs in pair_start_idxs)

    input_ids_lens = [len(x) for x in input_ids]
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.pad_side)

    gen_input_ids_lens = [len(x) for x in gen_input_ids]
    gen_input_ids = pad_sequence_with_side(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
            [input_ids, attention_mask, label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )
        gen_input_ids, gen_attention_mask = debug_extra_pad_tensors(
            [gen_input_ids, gen_attention_mask],
            padding_values=[dataset.tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    batch_dict = {
        "task_ids": task_ids,
        "inverters": inverters,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "out_token_length": out_token_length,
        "label_texts": label_texts,
        "input_ids_lens": input_ids_lens,
        "gen_input_ids_lens": gen_input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
    }
    return batch_dict


def collate_fn_eval_dummy(batch: List[Dict], dataset: EvalDataset) -> Dict:
    batch_size = len(batch)
    del batch  # we don't use it directly

    input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.debug_len), 1, dtype=torch.int64, device='cpu')
    input_ids_lens = [dataset.debug_len] * batch_size

    gen_input_ids = torch.randint(0, 30, (batch_size, dataset.debug_len * 9 // 10 + 1), dtype=torch.int64, device='cpu')
    gen_attention_mask = torch.full((batch_size, dataset.debug_len * 9 // 10 + 1), 1, dtype=torch.int64, device='cpu')
    gen_input_ids_lens = [dataset.debug_len * 9 // 10 + 1] * batch_size

    if not dataset.no_bos:
        pair_start_idxs = [list(range(1, dataset.debug_len * 9 // 10, dataset.debug_len // 15))[:10] for _ in range(batch_size)]
    else:
        pair_start_idxs = [list(range(0, dataset.debug_len * 9 // 10, dataset.debug_len // 15))[:10] for _ in range(batch_size)]

    task_ids = [str(x) for x in range(100000, 100000 + batch_size)]
    out_token_length = [dataset.debug_len // 10 + 1] * batch_size
    label_texts = ['1\n1\n1'] * batch_size

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
        "pair_start_idxs": pair_start_idxs,
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
        max_seq_len: int,
        seed: int,
        pad_side: str,
        debug_no_aug: bool,
        aug_type: str,
        no_separate_color_tokens: bool,
        no_bos: bool,
    ):
        self.permute_n = permute_n
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.pad_side = pad_side
        self.debug_no_aug = debug_no_aug
        self.no_separate_color_tokens = no_separate_color_tokens
        self.no_bos = no_bos

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

        # we do a lil parsing
        pair_idx_to_input_ids = []
        pair_idx_to_attention_mask = []
        pair_idx_to_label_ids = []

        for pair_i in range(num_pair):
            # get inputids, attention, labelids for batch of pairs at pair_i
            example = (task.train_examples + [task.test_example])[pair_i]
            input_grid_ids, output_grid_ids = self.tokenizer.get_input_and_output_grid_ids(
                example=example,
                add_bos=(pair_i == 0) if not self.no_bos else False,
                no_separate_color_tokens=self.no_separate_color_tokens,
            )
            input_ids = torch.cat([input_grid_ids, output_grid_ids])
            attention_mask = torch.full(input_ids.shape, 1, dtype=torch.int64)
            # label id for all except first pair
            # NOTE: loss on first is applied here
            label_ids = torch.full(input_grid_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_grid_ids])
            # aggregate
            pair_idx_to_input_ids.append(input_ids)
            pair_idx_to_attention_mask.append(attention_mask)
            pair_idx_to_label_ids.append(label_ids)

        # start idxs computed from cumsum of lengths of pairs except last, plus one for bos
        pair_start_idxs = np.cumsum([0, *[len(x) for x in pair_idx_to_input_ids[:-1]]]).tolist()
        pair_start_idxs[0] = 0 if self.no_bos else 1

        input_ids = torch.cat(pair_idx_to_input_ids)
        attention_mask = torch.cat(pair_idx_to_attention_mask)
        label_ids = torch.cat(pair_idx_to_label_ids)
        assert input_ids.shape == attention_mask.shape == label_ids.shape

        if len(input_ids) > self.max_seq_len:
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "pair_start_idxs": pair_start_idxs,
        }

    def __len__(self):
        return len(self.ttt_tasks)

    def __getitem__(self, idx):
        return self.format_and_filter(self.ttt_tasks[idx])


def collate_fn_ttt(batch: List[Dict], dataset: TTTDataset) -> Dict:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]
    pair_start_idxs = [x["pair_start_idxs"] for x in batch]
    assert all(start_idxs[0] == 1 for start_idxs in pair_start_idxs)

    input_ids_lens = [len(x) for x in input_ids]
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.pad_side)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "input_ids_lens": input_ids_lens,
        "pair_start_idxs": pair_start_idxs,
    }
