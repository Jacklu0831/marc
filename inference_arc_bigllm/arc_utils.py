"""Utility functions extracted from inference_arc/train.py for use in bigllm evaluation."""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

import arclib.augmenters

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def set_up_main_process_logger(accelerator, logger) -> None:
    """Configure logging for the main process."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        import datasets
        import transformers
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        import datasets
        import transformers
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def best_match_count(s1: str, s2: str) -> int:
    """Sliding-window character match count between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    L, S = len(s1), len(s2)
    max_matches = 0
    for shift in range(-S + 1, L):
        matches = 0
        for i in range(S):
            j = i + shift
            if 0 <= j < L and s2[i] == s1[j]:
                matches += 1
        max_matches = max(max_matches, matches)
    return max_matches


def list2d_to_tuple(l: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Convert 2D list to nested tuple for hashing."""
    return tuple(tuple(row) for row in l)


def grid_2d_to_text(grid: List[List[int]]) -> str:
    """Convert 2D grid to text format: 'HW\\nrow1\\nrow2\\n...'

    First line is height+width concatenated, then each row is digits concatenated.
    """
    height, width = len(grid), len(grid[0])
    lines = [f"{str(height)}{str(width)}"]
    for row in grid:
        lines.append("".join([str(x) for x in row]))
    return "\n".join(lines)


def text_to_2d_grid(text: str) -> Tuple[Optional[List[List[int]]], bool]:
    """Parse grid text back to 2D list.

    Expected format: 'HW\\nrow1\\nrow2\\n...' where HW is dimensions (skipped)
    and each row is digits concatenated.

    Returns:
        (grid, is_valid): grid is None if parsing fails.
    """
    try:
        text = text.strip()
        grid_lines = text.split('\n')
        grid = []
        row_lens = []
        grid_lines = grid_lines[1:]  # skip dimensions
        for l in grid_lines:
            row = [int(x) for x in l]
            grid.append(row)
            row_lens.append(len(row))
            assert all(0 <= x and x < 10 for x in row)
        assert len(set(row_lens)) == 1  # so the grid is not empty
        assert row_lens[0] > 0
        return grid, True
    except:
        return None, False


def row_base_majority_voting(
        grids: List[Tuple[Tuple[int, ...], ...]],
        transpose: bool = False,
    ) -> Tuple[Tuple[int, ...], ...]:
    """Row-based majority voting across multiple grid predictions."""
    if transpose:
        grids = [list2d_to_tuple((np.array(grid).T).tolist()) for grid in grids]
    shapes = [np.array(grid).shape for grid in grids]
    most_common_n_row, most_common_n_col = max(set(shapes), key=shapes.count)
    grid_rows = []
    for row_i in range(most_common_n_row):
        all_rows = [
            grid[row_i]
            for grid in grids
            if len(grid) > row_i and len(grid[row_i]) == most_common_n_col
        ]
        most_common_row = max(set(all_rows), key=all_rows.count)
        grid_rows.append(most_common_row)
    grid = np.array(grid_rows).T if transpose else np.array(grid_rows)
    return list2d_to_tuple(grid.tolist())


def get_three_votes(grids: List[Tuple[Tuple[int, ...], ...]]) -> List[Tuple[Tuple[int, ...], ...]]:
    """Get top-3 grid candidates by frequency with tie-breaking."""
    unique_grids = list(set(grids))
    counts = [grids.count(grid) for grid in unique_grids]
    common1 = unique_grids[np.argmax(counts)]
    common2 = common1
    common3 = common1
    if len(unique_grids) > 2:
        common2 = unique_grids[np.argsort(counts)[-2]]
        common3 = unique_grids[np.argsort(counts)[-3]]
    elif len(unique_grids) > 1:
        common2 = unique_grids[np.argsort(counts)[-2]]
    row_based_majority = row_base_majority_voting(grids, transpose=False)
    col_based_majority = row_base_majority_voting(grids, transpose=True)
    if common2 == common1:
        common2 = (
            row_based_majority
            if row_based_majority != common1
            else col_based_majority
        )
    if common3 in [common1, common2]:
        common3 = (
            row_based_majority
            if row_based_majority not in (common1, common2)
            else col_based_majority
        )
    return [common1, common2, common3]


def invert_and_vote(
        inverters_and_grids: List[Tuple[str, Tuple[Tuple[int, ...], ...]]]
    ) -> Tuple[Any, Any, List]:
    """Two-round voting with augmentation inversion."""
    category_to_grids: Dict[str, list] = defaultdict(list)
    for inverter, grid in inverters_and_grids:
        inverter_fn = lambda x: x
        if inverter != "":
            inverter_fn = eval("arclib.augmenters." + inverter)
        grid = list2d_to_tuple(inverter_fn(np.array(grid)).tolist())
        category_to_grids[inverter].append(grid)
    grids_all: list = []
    for key in category_to_grids:
        grids_all += category_to_grids[key]
    category_to_grids["all"] = grids_all
    candidates: list = []
    for grids in category_to_grids.values():
        candidates += get_three_votes(grids)
    c1, c2, c3 = get_three_votes(candidates)
    if candidates.count(c2) == candidates.count(c3):
        if "identity" in category_to_grids:
            if category_to_grids["identity"].count(c2) < category_to_grids["identity"].count(c3):
                c2 = c3
    return c1, c2, grids_all


def print_trainable_parameters(model: nn.Module) -> None:
    """Log trainable vs total parameter counts."""
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
