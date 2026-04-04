"""Data utilities for ARC evaluation with off-the-shelf LLMs.

Key difference from inference_arc/data_utils.py: uses the model's native BPE tokenizer
instead of a custom 45-token ARCTokenizer. Grids are encoded as plain text:

    input
    3 4
    1234
    5678
    9012
    output
    2 3
    123
    456

The model sees this as natural text tokens. Grid parsing for evaluation uses the same
text_to_2d_grid() format as the original (HW\\nrow1\\nrow2...).
"""

import csv
import copy
import os
import json
import random
import itertools
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np

from arclib.arc import Task, Example
from arclib.augmenters import (
    PermuteExamples,
    Augmenter,
    Chain,
    Flip,
    Rotate,
    Transpose,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import Qwen2TokenizerFast

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


# ########################################
# Instruction
# ########################################

ARC_INSTRUCTION = (
    "Each task consists of input-output grid pairs. "
    "Grids use digits 0-9. "
    "Given the demonstrated pairs, predict the output grid for the test input. "
    "Output only the grid dimensions and rows, nothing else."
)


# ########################################
# Grid Text Encoding
# ########################################

def format_grid(grid: np.ndarray) -> str:
    """Convert a 2D numpy grid to text.

    Format:
        H W
        row1_digits_concatenated
        row2_digits_concatenated
        ...

    Example for a 3x4 grid:
        3 4
        1234
        5678
        9012
    """
    h, w = grid.shape
    lines = [f"{h} {w}"]
    for row in grid:
        lines.append("".join(str(x) for x in row))
    return "\n".join(lines)


def format_label_grid(grid: np.ndarray) -> str:
    """Format grid for label text (compatible with text_to_2d_grid parser).

    Format: 'HW\\nrow1\\nrow2...' where HW is digits concatenated.
    This matches the format produced by grid_2d_to_text() in arc_utils.py.
    """
    h, w = grid.shape
    lines = [f"{h}{w}"]
    for row in grid:
        lines.append("".join(str(x) for x in row))
    return "\n".join(lines)


def example_to_text(example: Example) -> Tuple[str, str]:
    """Convert an ARC example to (input_text, output_text) strings.

    Args:
        example: ARC Example with .input and .output numpy arrays.

    Returns:
        input_text: 'input\\nH W\\nrow1\\n...\\noutput\\n'
        output_text: 'H W\\nrow1\\n...\\n'
    """
    input_text = "input\n" + format_grid(example.input) + "\noutput\n"
    output_text = format_grid(example.output) + "\n"
    return input_text, output_text


def tokenize_text(
        text: str,
        tokenizer: PreTrainedTokenizerFast,
        add_special_tokens: bool = False,
    ) -> torch.Tensor:
    """Tokenize text and return 1D tensor of token IDs.

    Strips BOS token for Llama/Mistral tokenizers (they always prepend BOS).
    Qwen tokenizers don't prepend BOS, so nothing is stripped.
    """
    out = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = out["input_ids"][0]  # type: ignore  # [seq_len]
    # strip BOS if the tokenizer auto-prepends it (Llama, Mistral)
    if not isinstance(tokenizer, Qwen2TokenizerFast):
        if len(input_ids) > 0 and input_ids[0].item() == tokenizer.bos_token_id:
            input_ids = input_ids[1:]
    return input_ids


# ########################################
# Padding Utilities
# ########################################

def pad_sequence_with_side(sequences: List[torch.Tensor], padding_value: int, side: str) -> torch.Tensor:
    """Pad a list of 1D tensors to equal length, supporting left or right padding."""
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
    """Add extra padding for debugging purposes."""
    assert len(tensors) == len(padding_values)
    assert all(t.dim() == 2 for t in tensors)
    if pad_len == -1:
        pad_len = random.randint(0, 15)
    padded_tensors = []
    for arg, padding_value in zip(tensors, padding_values):
        pad = torch.full((arg.shape[0], pad_len), padding_value, device=arg.device, dtype=arg.dtype)
        if side == 'right':
            padded_tensor = torch.cat([arg, pad], dim=-1)
        else:
            padded_tensor = torch.cat([pad, arg], dim=-1)
        padded_tensors.append(padded_tensor)
    return padded_tensors


# ########################################
# Evaluation Dataset
# ########################################

class EvalDataset:
    """Loads ARC evaluation tasks and encodes them as text for LLM inference.

    Each task has train_examples (demonstrations) and a test_example.
    The full sequence for a task is:
        [BOS] demo1_input demo1_output demo2_input demo2_output ... test_input
    where each part is encoded as plain text using the model's tokenizer.
    """

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
        tokenizer: PreTrainedTokenizerFast,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        debug_max_len: bool,
        max_seq_len: int,
        no_bos: bool,
        eval_on_demonstrations: bool,
        chat_prefix_ids: Optional[torch.Tensor] = None,
        chat_suffix_ids: Optional[torch.Tensor] = None,
    ):
        self.permute_n = permute_n
        self.augment_n = augment_n
        self.permute_iters = permute_iters
        self.seed = seed
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.debug_max_len = debug_max_len
        self.max_seq_len = max_seq_len
        self.no_bos = no_bos
        self.chat_prefix_ids = chat_prefix_ids  # e.g. "<|im_start|>user\n" for Qwen3
        self.chat_suffix_ids = chat_suffix_ids  # e.g. "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.eval_on_demonstrations = eval_on_demonstrations

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
                data_as_tuples = data_as_tuples[1:]  # first row contains col names
                select_task_ids = [d[0] for d in data_as_tuples]
                assert len(select_task_ids) == len(set(select_task_ids))
                select_task_ids = set(select_task_ids)
            file_paths = [p for p in file_paths if Path(p).stem in select_task_ids]
            assert len(file_paths) == len(select_task_ids), (len(file_paths), len(select_task_ids))
            logger.info(f"filtered to {len(file_paths)} files from {select_tasks_path}")

        # get actual data
        tasks = []
        for file_path in file_paths:
            task_id = Path(file_path).stem
            with open(file_path, "r") as f:
                task_data = json.load(f)
            train_examples = [Example(input=np.array(x["input"]), output=np.array(x["output"])) for x in task_data['train']]
            if eval_on_demonstrations:
                for test_i, train_example in enumerate(train_examples):
                    tasks.append(Task(
                        name=f'{task_id}-{test_i}',
                        train_examples=train_examples,
                        test_example=train_example,
                    ))
            else:
                test_examples = [Example(input=np.array(x["input"]), output=np.array(x["output"])) for x in task_data['test']]
                for test_i, test_example in enumerate(test_examples):
                    tasks.append(Task(
                        name=f'{task_id}-{test_i}',
                        train_examples=train_examples,
                        test_example=test_example,
                    ))
        tasks.sort(key=lambda d: d.name)
        logger.info(f"found {len(tasks)} tasks")

        # pad pairs for gs to be max batch size (debug)
        if debug_max_len:
            max_num_pair = max(len(t.train_examples) for t in tasks)
            for t in tasks:
                n_example = len(t.train_examples)
                if n_example > 0:
                    t.train_examples += np.random.choice(t.train_examples, size=max_num_pair - n_example).tolist()

        task_num_ios = [len(task.train_examples) for task in tasks]
        logger.info(f"task num io range from {min(task_num_ios)} to {max(task_num_ios)}")

        # get task id to gt for competition evaluation
        self.task_id_to_gt = {task.name: task.test_example.output.tolist() for task in tasks}
        assert len(self.task_id_to_gt) == len(tasks)

        # augment data for voting
        self.eval_tasks = []
        for task in tasks:
            new_tasks = self.get_task_augmentations_leave_ns_filtered(task, leave_ns=leave_ns)
            if len(new_tasks) == 0 and leave_ns_inc:
                new_tasks = self.get_task_augmentations_leave_ns_filtered(task, leave_ns=leave_ns + [leave_ns[-1] + 1])
            self.eval_tasks += new_tasks
        logger.info(f'augmented data from {len(tasks)} to {len(self.eval_tasks)}')

        # print details of parsed data
        parsed_data = [self[data_i] for data_i in range(len(self))]
        from collections import Counter
        task_id_to_counts = Counter(d["task_id"] for d in parsed_data)  # type: ignore
        task_id_to_counts_list = [(task_id, count) for task_id, count in task_id_to_counts.items()]
        for task_id, count in sorted(task_id_to_counts_list):
            logger.info(f"{task_id}: Number of Queries: {count}")
        n_pairs = [len(task.train_examples) for task in self.eval_tasks]
        logger.info(f"encoder npairs range from {min(n_pairs)} to {max(n_pairs)}")
        min_len, max_len = 1e6, 0
        for d in parsed_data:
            min_len = min(min_len, len(d['input_ids']))  # type: ignore
            max_len = max(max_len, len(d['input_ids']))  # type: ignore
        logger.info(f"encoder sequence length range from {min_len} to {max_len}]")
        del parsed_data

    def get_task_augmentations_leave_ns_filtered(
            self,
            task: Task,
            leave_ns: list[int],
        ) -> List[Task]:
        """Get augmented tasks filtered by sequence length."""
        augmented_tasks = []
        for leave_n in leave_ns:
            augmented_tasks += self.get_task_augmentations_leave_n(task, leave_n=leave_n)
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
        """Generate leave-n-out + permutation + augmentation variants of a task."""
        rng = np.random.RandomState(self.seed)
        test_tasks = []

        indices = list(range(len(task.train_examples)))
        leave_n_indices = [set(indices) - set(comb) for comb in itertools.combinations(indices, leave_n)]
        train_examples = task.train_examples.copy()
        for comb in leave_n_indices:
            new_task = Task(
                name=task.name,
                train_examples=[train_examples[j] for j in comb],
                test_example=task.test_example,
            )
            test_tasks.append(new_task)
            for _ in range(self.permute_n):
                permuted_task = PermuteExamples().apply_to_task(new_task, to_input=True, to_output=True, rng=rng)
                test_tasks.append(permuted_task)

        test_tasks = list(dict.fromkeys(test_tasks))

        augmented_tasks = []
        augmenters = rng.choice(self.augmenters, size=self.augment_n, replace=False)
        for augmenter in augmenters:
            new_task = rng.choice(test_tasks)
            augmented_task = augmenter.apply_to_task(new_task, to_input=True, to_output=True)
            if augmented_task in test_tasks:
                continue
            from arclib.augmenters import inverse
            inverter = str(inverse(augmenter))
            augmented_task.inverter = inverter
            augmented_tasks.append(augmented_task)
        test_tasks += augmented_tasks
        test_tasks = list(dict.fromkeys(test_tasks))
        return test_tasks

    def format_and_filter(self, task: Task) -> Optional[Dict]:
        """Format a task as token IDs and filter by sequence length.

        Encodes all demonstrations + test query as text, tokenizes with the model's
        tokenizer, and returns tensors needed for KV init, GS, and generation.
        """
        assert task.max_height() <= 30 and task.max_width() <= 30
        task = copy.deepcopy(task)
        num_pair = len(task.train_examples) + 1

        # Tokenize instruction (prepended before all demos)
        instruction_text = ARC_INSTRUCTION + "\n\n"
        instruction_ids = tokenize_text(text=instruction_text, tokenizer=self.tokenizer, add_special_tokens=False)
        # Add BOS before instruction if requested (Llama etc.)
        if not self.no_bos and self.tokenizer.bos_token_id is not None:
            instruction_ids = torch.cat([
                torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int64),
                instruction_ids,
            ])
        # Prepend chat template prefix before instruction (Qwen3 etc.)
        if self.chat_prefix_ids is not None:
            instruction_ids = torch.cat([self.chat_prefix_ids, instruction_ids])
        instruction_label_ids = torch.full(instruction_ids.shape, -100, dtype=torch.int64)

        # Build token sequences for each pair
        pair_idx_to_input_ids = [instruction_ids]
        pair_idx_to_label_ids = [instruction_label_ids]
        all_input_ids = [instruction_ids.clone()]  # instruction + all demo pairs + last input grid
        gen_output_text = None
        demon_start_idxs = [0]

        for pair_i in range(num_pair):
            example = (task.train_examples + [task.test_example])[pair_i]
            input_text, output_text = example_to_text(example)

            # Tokenize input part (no special tokens — BOS handled in instruction)
            input_ids = tokenize_text(text=input_text, tokenizer=self.tokenizer, add_special_tokens=False)
            # Tokenize output part + EOS
            output_ids = tokenize_text(
                text=output_text + self.tokenizer.eos_token,
                tokenizer=self.tokenizer,
                add_special_tokens=False,
            )

            full_ids = torch.cat([input_ids, output_ids])

            # Track gen ids vs demon ids
            if pair_i < num_pair - 1:
                # demonstration pair: include full input+output
                all_input_ids.append(full_ids.clone())
            else:
                # test pair: only input (no output for generation)
                # Append chat suffix after test input (Qwen3: <|im_end|>\n<|im_start|>assistant\n<think>...</think>\n)
                test_input_ids = input_ids.clone()
                if self.chat_suffix_ids is not None:
                    test_input_ids = torch.cat([test_input_ids, self.chat_suffix_ids])
                all_input_ids.append(test_input_ids)
                gen_output_text = output_text  # raw text for label

            # Label IDs: -100 for input tokens, actual IDs for output tokens
            label_ids = torch.full(input_ids.shape, -100, dtype=torch.int64)
            if pair_i == 0:
                # first pair: mask everything (input + output)
                label_ids = torch.cat([label_ids, torch.full(output_ids.shape, -100, dtype=torch.int64)])
            else:
                # subsequent pairs: loss on output tokens
                label_ids = torch.cat([label_ids, output_ids])
            pair_idx_to_input_ids.append(full_ids)
            pair_idx_to_label_ids.append(label_ids)

            # track demon boundaries (don't need genpair, don't need last demon pair)
            if pair_i == 0:
                # first demon starts after instruction
                demon_start_idxs[0] = 0  # relative to demon_input_ids (instruction is part of it)
                if num_pair > 2:
                    demon_start_idxs.append(len(instruction_ids) + len(full_ids))
            elif pair_i < num_pair - 2:
                demon_start_idxs.append(demon_start_idxs[-1] + len(full_ids))

        input_ids = torch.cat(pair_idx_to_input_ids)
        label_ids = torch.cat(pair_idx_to_label_ids)
        attention_mask = torch.ones_like(input_ids)
        assert input_ids.shape == attention_mask.shape == label_ids.shape

        if len(input_ids) > self.max_seq_len:
            return None

        # for gs (no ntoken)
        demon_input_ids = torch.cat(all_input_ids[:-1])
        demon_attention_mask = torch.ones_like(demon_input_ids)
        gen_input_ids = all_input_ids[-1]
        gen_attention_mask = torch.ones_like(gen_input_ids)

        all_input_ids_cat = torch.cat(all_input_ids)
        all_attention_mask = torch.ones_like(all_input_ids_cat)

        # Compute label_texts in the format text_to_2d_grid() expects
        assert gen_output_text is not None
        # gen_output_text is "H W\nrow1\nrow2\n...\n"
        # label_texts needs format "HW\nrow1\nrow2..." for text_to_2d_grid()
        label_texts = format_label_grid(task.test_example.output)

        # out_token_length: number of tokens for the output grid (excluding EOS)
        output_only_ids = tokenize_text(
            text=gen_output_text,
            tokenizer=self.tokenizer,
            add_special_tokens=False,
        )
        out_token_length = len(output_only_ids)

        return {
            "task_id": task.name,
            "inverter": task.inverter if hasattr(task, "inverter") else "",  # type: ignore
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "all_input_ids": all_input_ids_cat,
            "all_attention_mask": all_attention_mask,
            "out_token_length": out_token_length,
            "label_texts": label_texts,
            # for gs (no ntoken)
            "demon_input_ids": demon_input_ids,
            "demon_attention_mask": demon_attention_mask,
            "gen_input_ids": gen_input_ids,
            "gen_attention_mask": gen_attention_mask,
            "demon_start_idxs": demon_start_idxs,
        }

    def __len__(self):
        return len(self.eval_tasks)

    def __getitem__(self, idx):
        return self.format_and_filter(self.eval_tasks[idx])


def collate_fn_eval(batch: List[Dict], dataset: EvalDataset) -> Dict:
    """Collate function for EvalDataset batches."""
    task_ids = [x['task_id'] for x in batch]
    inverters = [x['inverter'] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]
    all_input_ids = [x["all_input_ids"] for x in batch]
    all_attention_mask = [x["all_attention_mask"] for x in batch]
    out_token_length = [x["out_token_length"] for x in batch]
    label_texts = [x["label_texts"] for x in batch]

    input_ids_lens = [len(x) for x in input_ids]
    input_ids = pad_sequence_with_side(input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    attention_mask = pad_sequence_with_side(attention_mask, padding_value=0, side=dataset.pad_side)
    label_ids = pad_sequence_with_side(label_ids, padding_value=-100, side=dataset.pad_side)

    all_input_ids_lens = [len(x) for x in all_input_ids]
    all_input_ids = pad_sequence_with_side(all_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    all_attention_mask = pad_sequence_with_side(all_attention_mask, padding_value=0, side=dataset.pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        input_ids, attention_mask, label_ids = debug_extra_pad_tensors(
            [input_ids, attention_mask, label_ids],
            padding_values=[dataset.tokenizer.pad_token_id, 0, -100],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )
        all_input_ids, all_attention_mask = debug_extra_pad_tensors(
            [all_input_ids, all_attention_mask],
            padding_values=[dataset.tokenizer.pad_token_id, 0],
            pad_len=dataset.debug_pad_len,
            side=dataset.pad_side,
        )

    # for gs (no ntoken)
    demon_input_ids = [x["demon_input_ids"] for x in batch]
    demon_attention_mask = [x["demon_attention_mask"] for x in batch]
    gen_input_ids = [x["gen_input_ids"] for x in batch]
    gen_attention_mask = [x["gen_attention_mask"] for x in batch]
    demon_start_idxs = [x['demon_start_idxs'] for x in batch]

    demon_input_ids = pad_sequence_with_side(demon_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    demon_attention_mask = pad_sequence_with_side(demon_attention_mask, padding_value=0, side=dataset.pad_side)
    gen_input_ids = pad_sequence_with_side(gen_input_ids, padding_value=dataset.tokenizer.pad_token_id, side=dataset.pad_side)
    gen_attention_mask = pad_sequence_with_side(gen_attention_mask, padding_value=0, side=dataset.pad_side)

    if dataset.debug_random_pad or dataset.debug_pad_len > -1:
        demon_input_ids, demon_attention_mask = debug_extra_pad_tensors(
            [demon_input_ids, demon_attention_mask],
            padding_values=[dataset.tokenizer.pad_token_id, 0],
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
        "task_ids": task_ids,
        "inverters": inverters,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "all_input_ids": all_input_ids,
        "all_attention_mask": all_attention_mask,
        "out_token_length": out_token_length,
        "label_texts": label_texts,
        "input_ids_lens": input_ids_lens,
        "all_input_ids_lens": all_input_ids_lens,
        # for gs (no ntoken)
        "demon_input_ids": demon_input_ids,
        "demon_attention_mask": demon_attention_mask,
        "gen_input_ids": gen_input_ids,
        "gen_attention_mask": gen_attention_mask,
        "demon_start_idxs": demon_start_idxs,
    }


# ########################################
# Gradient Search Dataset
# ########################################

class GSDataset(Dataset):
    """Dataset for CT-KV gradient search — one entry per demonstration pair.

    Each entry tokenizes a single demo (input grid + output grid) for computing
    the leave-one-out loss during KV cache optimization.
    """

    def __init__(
        self,
        task: Task,
        tokenizer: PreTrainedTokenizerFast,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        max_seq_len: int,
        loss_on_input: bool,
        no_bos: bool,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.loss_on_input = loss_on_input
        self.no_bos = no_bos

        self.parsed_examples = [self.format(i, example) for i, example in enumerate(task.train_examples)]

    def __len__(self):
        return len(self.parsed_examples)

    def __getitem__(self, idx):
        return self.parsed_examples[idx]

    def format(self, example_idx: int, example: Example) -> Optional[Dict]:
        """Tokenize a single demonstration pair for GS loss computation."""
        assert max(example.input.shape) <= 30 and max(example.output.shape) <= 30

        input_text, output_text = example_to_text(example)

        # Tokenize without BOS (each demo is processed independently for GS)
        input_ids = tokenize_text(text=input_text, tokenizer=self.tokenizer, add_special_tokens=False)
        output_ids = tokenize_text(
            text=output_text + self.tokenizer.eos_token,
            tokenizer=self.tokenizer,
            add_special_tokens=False,
        )

        full_ids = torch.cat([input_ids, output_ids])
        attention_mask = torch.ones_like(full_ids)

        if self.loss_on_input:
            label_ids = full_ids.clone()
        else:
            label_ids = torch.full(input_ids.shape, -100, dtype=torch.int64)
            label_ids = torch.cat([label_ids, output_ids])

        return {
            "example_idx": example_idx,
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
        }


def collate_fn_gs(batch: List[Dict], dataset: GSDataset) -> Dict:
    """Collate function for GSDataset batches."""
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]

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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "example_idx": [x['example_idx'] for x in batch],
    }


def collate_fn_gs_dummy(batch: List[Dict], dataset: GSDataset) -> Dict:
    """Dummy collate function for GS (used in distributed padding)."""
    batch_size = len(batch)
    del batch
    input_ids = torch.randint(0, 1000, (batch_size, dataset.max_seq_len // 8), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.max_seq_len // 8), 1, dtype=torch.int64, device='cpu')
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "example_idx": [0] * batch_size,
    }


# ########################################
# Test-Time-Training Dataset
# ########################################

class TTTDataset(Dataset):
    """Dataset for test-time LoRA training — random permutations of demos.

    Each entry is a permutation of the task's demonstrations, with the last
    one serving as the leave-one-out test example.
    """

    def __init__(
        self,
        task: Task,
        tokenizer: PreTrainedTokenizerFast,
        debug_random_pad: bool,
        debug_pad_len: int,
        pad_side: str,
        max_seq_len: int,
        permute_n: int,
        seed: int,
        loss_type: str,
        no_bos: bool,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.debug_random_pad = debug_random_pad
        self.debug_pad_len = debug_pad_len
        self.pad_side = pad_side
        self.max_seq_len = max_seq_len
        self.permute_n = permute_n
        self.seed = seed
        self.loss_type = loss_type
        self.no_bos = no_bos

        self.data = self.unique_permutations(permute_n=permute_n, seed=seed)

    def unique_permutations(self, permute_n: int, seed: int) -> List[Dict]:
        """Generate unique random permutations of demonstrations."""
        rng = np.random.RandomState(seed)
        seen: set = set()
        all_data: List[Dict] = []

        for _ in range(1000):
            perm = tuple(rng.permutation(len(self.task.train_examples)))
            if perm in seen:
                continue
            seen.add(perm)
            permuted_examples = [self.task.train_examples[i] for i in perm]
            data = self.format_and_filter(
                task=Task(
                    name=self.task.name,
                    train_examples=permuted_examples[:-1],
                    test_example=permuted_examples[-1],
                )
            )
            if data is not None:
                all_data.append(data)
            if len(all_data) >= permute_n:
                break
        return all_data

    def format_and_filter(self, task: Task) -> Optional[Dict]:
        """Format a permuted task as token IDs for TTT training."""
        assert task.max_height() <= 30 and task.max_width() <= 30
        task = copy.deepcopy(task)
        num_pair = len(task.train_examples) + 1

        pair_idx_to_input_ids = []
        pair_idx_to_label_ids = []

        for pair_i in range(num_pair):
            example = (task.train_examples + [task.test_example])[pair_i]
            input_text, output_text = example_to_text(example)

            input_ids = tokenize_text(text=input_text, tokenizer=self.tokenizer, add_special_tokens=False)
            output_ids = tokenize_text(
                text=output_text + self.tokenizer.eos_token,
                tokenizer=self.tokenizer,
                add_special_tokens=False,
            )

            if pair_i == 0 and not self.no_bos:
                bos_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int64)
                input_ids = torch.cat([bos_tensor, input_ids])

            full_ids = torch.cat([input_ids, output_ids])

            # label masking based on loss_type
            if (pair_i == 0 and self.loss_type == 'exclude_first') or (pair_i < num_pair and self.loss_type == 'only_last'):
                label_ids = torch.full(full_ids.shape, -100, dtype=torch.int64)
            else:
                label_ids = torch.cat([
                    torch.full(input_ids.shape, -100, dtype=torch.int64),
                    output_ids,
                ])
            pair_idx_to_input_ids.append(full_ids)
            pair_idx_to_label_ids.append(label_ids)

        input_ids = torch.cat(pair_idx_to_input_ids)
        label_ids = torch.cat(pair_idx_to_label_ids)
        attention_mask = torch.ones_like(input_ids)
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
    """Collate function for TTTDataset batches."""
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]

    input_ids_lens = [len(x) for x in input_ids]
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
        "input_ids_lens": input_ids_lens,
    }


def collate_fn_ttt_dummy(batch: List[Dict], dataset: TTTDataset) -> Dict:
    """Dummy collate function for TTT (used in distributed padding)."""
    batch_size = len(batch)
    del batch
    input_ids = torch.randint(0, 1000, (batch_size, dataset.max_seq_len), dtype=torch.int64, device='cpu')
    attention_mask = torch.full((batch_size, dataset.max_seq_len), 1, dtype=torch.int64, device='cpu')
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": input_ids,
        "input_ids_lens": [dataset.max_seq_len] * batch_size,
    }
