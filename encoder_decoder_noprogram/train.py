from custom_llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import arclib # required
import numpy as np
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Optional, Iterator, Dict, Set
import pprint
import math
import json
from tqdm import tqdm
from functools import partial
import argparse
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training # type: ignore

import logging
import datasets
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

from data_utils import (
    TrainDataset,
    EvalDataset,
    collate_fn_train,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
    ARCTokenizer,
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # weird tokenizer issue
os.environ["NCCL_TIMEOUT"] = "28800" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "28800000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

logger = get_logger(__name__, log_level="INFO")


MODEL_NAME_TO_PATH = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3b_uncensored": "chuanli11/Llama-3.2-3B-Instruct-uncensored",
    "nemo8b": "nvidia/Mistral-NeMo-Minitron-8B-Base",
}
NBIT_TO_DTYPE = {
    16: torch.bfloat16,
    32: torch.float32,
}

class ProgramEmbeddings(nn.Module):
    def __init__(self, embedding: torch.Tensor):
        super(ProgramEmbeddings, self).__init__()
        self.embedding = nn.Parameter(embedding)

    def forward(self, program_i: int) -> torch.Tensor:
        del program_i
        return self.embedding

class ProgramEmbeddings(nn.Module):
    def __init__(self, embedding: torch.Tensor):
        super(ProgramEmbeddings, self).__init__()
        self.embedding = nn.Parameter(embedding)

    def forward(self, program_i: int) -> torch.Tensor:
        del program_i
        return self.embedding


def three_commas(x):
    x = str(x)
    b, a = divmod(len(x), 3)
    return ",".join(([x[:a]] if a else []) + \
                    [x[a + 3*i: a + 3*i + 3] for i in range(b)])


def set_up_main_process_logger(accelerator, logger):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def chunks(lst: List[int], n: int) -> Iterator[List[int]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunks_uniform_batch(task_ids: List[str], data_idxs: List[int], n: int) -> Iterator[List[int]]:
    assert len(task_ids) == len(data_idxs)
    # group by first item in tuple (task_id)
    task_id_to_data_idx = defaultdict(list)
    for task_id, data_idx in zip(task_ids, data_idxs):
        task_id_to_data_idx[task_id].append(data_idx)
    # for each task_id, yield chunks of data idxs
    for task_id, data_idxs in task_id_to_data_idx.items():
        yield from chunks(data_idxs, n)

def insert_based_on_sides(
        data: torch.Tensor,
        to_insert: torch.Tensor,
        insert_side: str,
        pad_side: str,
        pad_id: Union[int, torch.Tensor],
        lens: List[int]=None,
    ) -> torch.Tensor:

    if pad_side == "right":
        if insert_side == "left":
            return torch.cat([to_insert, data], dim=1)
        else:
            data_new = []
            for x, m, l in zip(data, to_insert, lens):
                if isinstance(pad_id, int):
                    assert torch.equal(x[l:], torch.full(x[l:].shape, pad_id, device=x[l:].device)), x[l:]
                else:
                    assert torch.equal(x[l:], pad_id.unsqueeze(0).expand(x[l:].shape[0], -1)), x[l:]
                x = torch.cat([x[:l], m, x[l:]])
                data_new.append(x)
            return torch.stack(data_new)
    else:
        if insert_side == "left":
            data_new = []
            for x, m, l in zip(data, to_insert, lens):
                if isinstance(pad_id, int):
                    assert torch.equal(x[:-l], torch.full(x[:-l].shape, pad_id, device=x[:-l].device)), x[:-l]
                else:
                    assert torch.equal(x[:-l], pad_id.unsqueeze(0).expand(x[:-l].shape[0], -1)), x[:-l]
                x = torch.cat([x[:-l], m, x[-l:]])
                data_new.append(x)
            return torch.stack(data_new)
        else:
            return torch.cat([data, to_insert], dim=1)

def get_memory_footprint(module: nn.Module):
    return sum(p.nelement() * p.element_size() for p in module.parameters()) + \
        sum(p.nelement() * p.element_size() for p in module.buffers())

def best_match_count(s1, s2):
    # Ensure s1 is the longer string
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    L, S = len(s1), len(s2)
    max_matches = 0

    # Slide s2 over s1.
    # Range of shifts: from -S+1 (s2 shifted so its end aligns with the start of s1)
    # to L-1 (s2 shifted so its start aligns with the end of s1)
    for shift in range(-S + 1, L):
        matches = 0
        # Loop over each index of the shorter string
        for i in range(S):
            j = i + shift  # corresponding index in s1
            # Only count if within bounds of s1
            if 0 <= j < L and s2[i] == s1[j]:
                matches += 1
        max_matches = max(max_matches, matches)

    return max_matches


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    task_to_ttt_path: Optional[Dict[str, str]],
    ttt_param_names: Optional[Set[str]],
    model: Union[nn.Module, DistributedDataParallel],
    program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    tokenizer: ARCTokenizer,
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    dry_eval_run: bool,
    output_dir: str,
<<<<<<< HEAD
    thinking_embeddings: Union[ProgramEmbeddings, DistributedDataParallel]=None,
=======
    ntokens: int,
    attention_cutoff: bool,
    attend_prev_programs: bool,
>>>>>>> origin/main
):
    model.eval()

    # get modules in case of DDP
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # if ttt provided, same model weights for the missing ttt task weights
    cached_weights_path = None
    curr_ttt_task_name = None
    if task_to_ttt_path is not None: # run on both processes
        # save model for default when ttt is missing
        cached_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_cache.pt")
        assert isinstance(ttt_param_names, set)
        names = set(name for name, _ in module.named_parameters())
        assert all(name in names for name in ttt_param_names), f"process{accelerator.process_index}\n\n{ttt_param_names}\n\n{names}"
        cache_weights = {name: param for name, param in module.named_parameters() if name in ttt_param_names}
        torch.save(cache_weights, cached_weights_path)
        logger.info(f"ttt provided, cached {len(cache_weights)} weights to {cached_weights_path}")
        # save default to model paths and set current ttt weights to default
        task_to_ttt_path["default"] = cached_weights_path
        curr_ttt_task_name = "default"

    # setup terminators and suppress warning
    module.generation_config.pad_token_id = dataset.tokenizer.pad_token_id

    distributed_state = PartialState()
    task_id_and_text_list = []
    task_id_and_inverter_grids = []
    exact_acc_list = []
    valid_grid_list = []
    correct_grid_dim_list = []
    token_acc_list = []
    relaxed_token_acc_list = []
    ttt_provided_list = []

    data_idxs = list(range(len(dataset)))
    assert len(data_idxs) >= accelerator.num_processes # avoid padding issue

    with distributed_state.split_between_processes(data_idxs) as process_data_idxs:
        assert isinstance(process_data_idxs, list)
        n_batches = math.ceil(len(process_data_idxs) / batch_size)

        # if ttt provided, make sure all batches are of the same task name
        if task_to_ttt_path is not None:
            # tackle tasks in orderly fashion
            task_names = [dataset.eval_tasks[idx].name for idx in process_data_idxs] # type: ignore
            task_ids = [task_name.split('-')[0] for task_name in task_names]
            n_batches = len(list(chunks_uniform_batch(task_ids, process_data_idxs, batch_size)))
            data_idx_iterator = tqdm(chunks_uniform_batch(task_ids, process_data_idxs, batch_size), total=n_batches) # type: ignore
        else:
            data_idx_iterator = tqdm(chunks(process_data_idxs, batch_size), total=n_batches)  # type: ignore

        for batch_idxs in data_idx_iterator:
            bs = len(batch_idxs)

            # optionally load ttt lora
            ttt_provided = [0] * bs
            if task_to_ttt_path is not None:
                # make sure task name is unique and set to default is missing
                task_names = [dataset.eval_tasks[idx].name.split('-')[0] for idx in batch_idxs]
                assert len(set(task_names)) == 1 # have to be the same task
                task_name = task_names[0]
                if task_name not in task_to_ttt_path:
                    task_name = "default"
                ttt_provided = [int(task_name != "default")] * bs
                # load ttt if necessary
                if task_name != curr_ttt_task_name:
                    # load model
                    ttt_path = task_to_ttt_path[task_name]
                    ttt_state_dict = torch.load(
                        ttt_path,
                        weights_only=True,
                        map_location=accelerator.device
                    )
                    assert set(ttt_state_dict.keys()) == ttt_param_names
                    module.load_state_dict(ttt_state_dict, strict=False)
                    del ttt_state_dict
                    curr_ttt_task_name = task_name # set current task name
                    model.eval() # another eval after loading weight just in case
            ttt_provided_list += ttt_provided

            batch_data = [dataset[i] for i in batch_idxs]
            batch = collate_fn(batch_data)

            if dry_eval_run:
                continue

            # get tensors
            task_ids = batch["task_ids"]
            inverters = batch["inverters"]
            gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_input_ids_lens = batch["gen_input_ids_lens"]
            out_token_length = batch["out_token_length"]
            label_texts = batch["label_texts"]
<<<<<<< HEAD

            # compute ce loss
            if thinking_embeddings != None:
                pad_side = 'right'
                ntokens = len(thinking_embeddings[0])
                batch_size = len(input_ids[0])
                module = model.module if isinstance(model, DistributedDataParallel) else model
                embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens
                dtype, device = input_ids[0].dtype, input_ids[0].device
                thinking_inputs_embeds = thinking_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
                extra_program_attention_mask = torch.full((batch_size, ntokens), 1, device=device, dtype=dtype)
                extra_program_label_ids = torch.full((batch_size, ntokens), -100, device=device, dtype=dtype)
                pad_embeds = embed_tokens(torch.tensor(dataset.tokenizer.pad_token_id, device=device))
                inputs_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in input_ids]
                with_thiking_embeds = insert_based_on_sides(
                    data=inputs_embeds,
                    to_insert=thinking_inputs_embeds,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=pad_embeds,
                )
                with_thinking_attention_mask = insert_based_on_sides(
                    data=attention_mask,
                    to_insert=extra_program_attention_mask,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=pad_embeds,
                )
=======
            pair_start_idxs = batch["pair_start_idxs"]

            # compute accuracy
            with accelerator.autocast():
                arbitrary_increase = 5
                if program_embeddings is None:
                    # generate with no program
                    gen_tokens = module.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max(out_token_length) + arbitrary_increase, # arbitrary increase
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                    )
                    gen_texts = dataset.tokenizer.batch_decode(
                        gen_tokens[:, gen_input_ids.shape[1]:],
                        skip_special_tokens=True,
                        no_separate_color_tokens=dataset.no_separate_color_tokens,
                    )
                else:
                    # generate with no program
                    device, dtype = gen_input_ids.device, gen_input_ids.dtype
                    max_num_pairs = max(len(start_idxs) for start_idxs in pair_start_idxs)
                    pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device)) # (hidden_dim,)
                    gen_inputs_embeds = embed_tokens(gen_input_ids)

                    # insert program embeddings
                    inputs_embeds_with_programs = []
                    attention_mask_with_programs = []
                    program_intervals = []

                    for task_input_ids, task_inputs_embeds, task_attention_mask, input_ids_len, start_idxs in zip(gen_input_ids, gen_inputs_embeds, gen_attention_mask, gen_input_ids_lens, pair_start_idxs):
                        assert start_idxs[0] == 1 # first pair starts after bos
                        # start_idxs are offset if padding side is left
                        if dataset.gen_pad_side == "left":
                            start_idxs = [s + task_input_ids.shape[0] - input_ids_len for s in start_idxs]
                        assert len(set(task_input_ids[start_idxs].tolist())) == 1 # all should be the input token (we remove bos)

                        # insert program token before every pair
                        task_inputs_embeds_with_programs = [task_inputs_embeds[:start_idxs[0]]]
                        task_attention_mask_with_programs = [task_attention_mask[:start_idxs[0]]]
                        task_program_intervals = []

                        for i, start_idx in enumerate(start_idxs):
                            end_idx = start_idxs[i+1] if i < len(start_idxs) - 1 else len(task_inputs_embeds)
                            # program start
                            program_start = sum(len(x) for x in task_inputs_embeds_with_programs)
                            task_program_intervals.append((program_start, program_start + ntokens))
                            # insert program embedding into inputs_embeds
                            task_inputs_embeds_with_programs.append(program_embeddings("dummy"))
                            task_inputs_embeds_with_programs.append(task_inputs_embeds[start_idx: end_idx])
                            # insert full attention for programs
                            task_attention_mask_with_programs.append(torch.full((ntokens,), 1, device=device, dtype=dtype))
                            task_attention_mask_with_programs.append(task_attention_mask[start_idx: end_idx])

                        # some programs in batch have fewer pairs, pad manually, so hacky
                        pad_length = (max_num_pairs - len(start_idxs)) * ntokens
                        task_pad_inputs_embeds = pad_embeds[None, ...].expand(pad_length, -1)
                        task_pad_attention_mask = torch.full((pad_length,), 0, device=device, dtype=dtype)
                        if dataset.gen_pad_side == 'left':
                            task_inputs_embeds_with_programs.insert(0, task_pad_inputs_embeds)
                            task_attention_mask_with_programs.insert(0, task_pad_attention_mask)
                            task_program_intervals = [(x[0] + pad_length, x[1] + pad_length) for x in task_program_intervals]
                        else:
                            task_inputs_embeds_with_programs.append(task_pad_inputs_embeds)
                            task_attention_mask_with_programs.append(task_pad_attention_mask)

                        # concat all
                        inputs_embeds_with_programs.append(torch.cat(task_inputs_embeds_with_programs))
                        attention_mask_with_programs.append(torch.cat(task_attention_mask_with_programs))
                        program_intervals.append(task_program_intervals)

                    # stack and check, now we have the three inputs with programs
                    inputs_embeds_with_programs = torch.stack(inputs_embeds_with_programs)
                    attention_mask_with_programs = torch.stack(attention_mask_with_programs)
                    assert inputs_embeds_with_programs.shape[1] == gen_input_ids.shape[1] + max_num_pairs * ntokens
                    assert inputs_embeds_with_programs.shape[:2] == attention_mask_with_programs.shape[:2]

                    # generate
                    if not attention_cutoff:
                        gen_tokens = module.generate(
                            inputs_embeds=inputs_embeds_with_programs,
                            attention_mask=attention_mask_with_programs,
                            max_new_tokens=max(out_token_length) + arbitrary_increase, # arbitrary increase
                            num_return_sequences=1,
                            temperature=1.0,
                            top_p=1.0,
                            do_sample=False,
                            eos_token_id=[dataset.tokenizer.eos_token_id],
                        )
                    else:
                        gen_tokens = module.generate(
                            inputs_embeds=inputs_embeds_with_programs,
                            attention_mask=attention_mask_with_programs,
                            max_new_tokens=max(out_token_length) + arbitrary_increase, # arbitrary increase
                            num_return_sequences=1,
                            temperature=1.0,
                            top_p=1.0,
                            do_sample=False,
                            eos_token_id=[dataset.tokenizer.eos_token_id],
                            program_intervals=program_intervals,
                            attend_prev_programs=attend_prev_programs,
                        )
                    gen_texts = dataset.tokenizer.batch_decode(
                        gen_tokens,
                        skip_special_tokens=True,
                        no_separate_color_tokens=dataset.no_separate_color_tokens,
                    )
>>>>>>> origin/main

                with_thinking_labels = insert_based_on_sides(
                    data=label_ids,
                    to_insert=extra_program_attention_mask,
                    lens=extra_program_label_ids,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=pad_embeds,
                )
                with accelerator.autocast():
                    ce_loss = model(
                        input_embeds=with_thiking_embeds,
                        attention_mask= with_thinking_attention_mask,
                        labels=with_thinking_labels,
                    ).loss
                    ce_loss_list += [ce_loss.item()] * bs


                gen_input_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in gen_input_ids]
                with_thiking_gen_embeds = insert_based_on_sides(
                    data=gen_input_embeds,
                    to_insert=thinking_inputs_embeds,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=pad_embeds,
                )
                with_thinking_gen_attention_mask = insert_based_on_sides(
                    data=gen_attention_mask,
                    to_insert=extra_program_attention_mask,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=pad_embeds,
                )

           
                # compute accuracy
                with accelerator.autocast():
                    # generate
                    gen_tokens = module.generate(
                        input_embeds=with_thiking_gen_embeds,
                        attention_mask=with_thinking_gen_attention_mask,
                        max_new_tokens=max(out_token_length), # arbitrary increase
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                    )
                    gen_texts = dataset.tokenizer.batch_decode(gen_tokens[:, with_thiking_gen_embeds.shape[1]:], skip_special_tokens=True)


            else:
                with accelerator.autocast():
                    ce_loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=label_ids,
                    ).loss
                    ce_loss_list += [ce_loss.item()] * bs

                    # compute accuracy
                with accelerator.autocast():
                    # generate
                    gen_tokens = module.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max(out_token_length), # arbitrary increase
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                    )
                    gen_texts = dataset.tokenizer.batch_decode(gen_tokens[:, gen_input_ids.shape[1]:], skip_special_tokens=True)


            
            # Compare each gen_text with label_texts
            assert len(task_ids) == len(inverters) == bs, (len(task_ids), len(inverters), bs)
            assert len(gen_texts) == len(label_texts) == bs, (len(gen_texts), len(label_texts), bs)
            for task_id, inverter, gen_text, label_text in zip(task_ids, inverters, gen_texts, label_texts):
                relaxed_token_acc_list.append(best_match_count(gen_text, label_text) / len(label_text))
                # is valid grid
                gen_grid, gen_is_grid = text_to_2d_grid(text=gen_text)
                label_grid, label_is_grid = text_to_2d_grid(text=label_text)
                assert label_is_grid
                valid_grid_list.append(int(gen_is_grid))
                if not gen_is_grid:
                    task_id_and_text_list.append((task_id, gen_text, label_text))
                    exact_acc_list.append(0)
                    correct_grid_dim_list.append(0)
                    token_acc_list.append(0)
                    continue
                assert isinstance(gen_grid, list)
                assert isinstance(label_grid, list)
                # now we know it's a valid grid
                gen_text = grid_2d_to_text(gen_grid)
                task_id_and_text_list.append((task_id, gen_text, label_text))
                gen_grid, label_grid = list2d_to_tuple(gen_grid), list2d_to_tuple(label_grid)
                # exact acc
                exact_acc_list.append(int(gen_grid == label_grid))
                # save gen and gt grid
                task_id_and_inverter_grids.append((task_id, inverter, gen_grid, label_grid))
                # correct grid dim
                is_correct_grid_dim = (len(gen_grid) == len(label_grid) and len(gen_grid[0]) == len(label_grid[0]))
                correct_grid_dim_list.append(int(is_correct_grid_dim))
                if not is_correct_grid_dim:
                    token_acc_list.append(0)
                    continue
                # token acc
                grid_size = len(label_grid) * len(label_grid[0])
                num_token_correct = 0
                for gen_row, label_row in zip(gen_grid, label_grid):
                    for gen_x, label_x in zip(gen_row, label_row):
                        num_token_correct += int(gen_x == label_x)
                token_acc_list.append(num_token_correct / grid_size)

    distributed_state.wait_for_everyone()
    # results
    task_id_and_text_list = gather_object(task_id_and_text_list)
    task_id_and_inverter_grids = gather_object(task_id_and_inverter_grids) # likely diff len from dataset
    # accuracies
    exact_acc_list = gather_object(exact_acc_list)
    valid_grid_list = gather_object(valid_grid_list)
    correct_grid_dim_list = gather_object(correct_grid_dim_list)
    token_acc_list = gather_object(token_acc_list)
    relaxed_token_acc_list = gather_object(relaxed_token_acc_list)
    # ttt
    ttt_provided_list = gather_object(ttt_provided_list)

    assert len(task_id_and_text_list) == len(dataset), (len(task_id_and_text_list), len(dataset))
    assert len(exact_acc_list) == len(dataset), (len(exact_acc_list), len(dataset))
    assert len(valid_grid_list) == len(dataset), (len(valid_grid_list), len(dataset))
    assert len(correct_grid_dim_list) == len(dataset), (len(correct_grid_dim_list), len(dataset))
    assert len(token_acc_list) == len(dataset), (len(token_acc_list), len(dataset))
    assert len(relaxed_token_acc_list) == len(dataset), (len(relaxed_token_acc_list), len(dataset))
    assert len(ttt_provided_list) == len(dataset), (len(ttt_provided_list), len(dataset))

    # average metrics
    # note these are all computed without accounting for skipped eval grids
    exact_acc = sum(exact_acc_list) / len(dataset)
    valid_grid = sum(valid_grid_list) / len(dataset)
    correct_grid_dim = sum(correct_grid_dim_list) / len(dataset)
    token_acc = sum(token_acc_list) / len(dataset)
    relaxed_token_acc = sum(relaxed_token_acc_list) / len(dataset)
    ttt_provided = sum(ttt_provided_list) / len(dataset)

    # grab all results
    task_id_to_texts = defaultdict(list)
    for task_id, gen_text, label_text in task_id_and_text_list:
        task_id_to_texts[task_id].append((gen_text, label_text))

    # voting
    votes = {}
    for task_id in dataset.task_id_to_gt:
        # get 2 vote results
        inverters_and_gen_grids = [(x[1], list2d_to_tuple(x[2])) for x in task_id_and_inverter_grids if x[0] == task_id]
        votes[task_id] = [[[0]], [[0]]]
        if len(inverters_and_gen_grids) > 0:
            attempt1, attempt2, _ = invert_and_vote(inverters_and_gen_grids)
            votes[task_id] = [attempt1, attempt2]
        # assert all label grids are the same after invert augmentation
        inverters_and_label_grids = [(x[1], list2d_to_tuple(x[3])) for x in task_id_and_inverter_grids if x[0] == task_id]
        if len(inverters_and_label_grids) > 0:
            _, _, inverted_labels = invert_and_vote(inverters_and_label_grids)
            assert len(set(inverted_labels)) == 1

    # competition evaluation
    task_name_to_corrects = defaultdict(list)
    for task_id, gt in dataset.task_id_to_gt.items():
        correct = list2d_to_tuple(gt) in votes[task_id]
        task_name = task_id.split('-')[0]
        task_name_to_corrects[task_name].append(correct)

    competition_sub_correct = sum(sum(corrects) for corrects in task_name_to_corrects.values())
    competition_all_correct = sum(all(corrects) for corrects in task_name_to_corrects.values())
    competition_sub_acc = competition_sub_correct / sum(len(corrects) for corrects in task_name_to_corrects.values())
    competition_all_acc = competition_all_correct / len(task_name_to_corrects)

    if cached_weights_path is not None:
        os.remove(cached_weights_path)

    return exact_acc, valid_grid, correct_grid_dim, token_acc, relaxed_token_acc, task_id_to_texts, \
        votes, competition_sub_acc, competition_all_acc, ttt_provided


def list2d_to_tuple(l: List[List[int]]) -> Tuple[Tuple[int]]:
    return tuple(tuple(row) for row in l) # type: ignore


def row_base_majority_voting(
        grids: List[Tuple[Tuple[int]]],
        transpose: bool = False,
    ) -> Tuple[Tuple[int]]:
    # transpose if needed
    if transpose:
        grids = [list2d_to_tuple((np.array(grid).T).tolist()) for grid in grids] # type: ignore
    # get most common shape
    shapes = [np.array(grid).shape for grid in grids]
    most_common_n_row, most_common_n_col = max(set(shapes), key=shapes.count)
    # for each row, find all grids with same number of column that also contain this row
    grid_rows = []
    for row_i in range(most_common_n_row):
        all_rows = [
            grid[row_i]
            for grid in grids
            if len(grid) > row_i and len(grid[row_i]) == most_common_n_col
        ]
        most_common_row = max(set(all_rows), key=all_rows.count)
        grid_rows.append(most_common_row)
    # transpose back if needed
    grid = np.array(grid_rows).T if transpose else np.array(grid_rows)
    return list2d_to_tuple(grid.tolist())


def get_three_votes(grids: List[Tuple[Tuple[int]]]) -> List[Tuple[Tuple[int]]]:
    unique_grids = list(set(grids))
    counts = [grids.count(grid) for grid in unique_grids]
    common1 = unique_grids[np.argmax(counts)]
    common2 = common1
    common3 = common1
    # assign common2 and common3
    if len(unique_grids) > 2:
        common2 = unique_grids[np.argsort(counts)[-2]]
        common3 = unique_grids[np.argsort(counts)[-3]]
    elif len(unique_grids) > 1:
        common2 = unique_grids[np.argsort(counts)[-2]]
    # break tie for common2 and common3
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


def invert_and_vote(inverters_and_grids: List[Tuple[str, Tuple[Tuple[int]]]]):
    # collect inverted grids by augmentation
    category_to_grids = defaultdict(list)
    for inverter, grid in inverters_and_grids:
        inverter_fn = lambda x: x
        if inverter != "":
            inverter_fn = eval("arclib.augmenters." + inverter)
        grid = list2d_to_tuple(inverter_fn(np.array(grid)).tolist())
        category_to_grids[inverter].append(grid)
    # add all grids as a category
    grids_all = []
    for key in category_to_grids:
        grids_all += category_to_grids[key]
    category_to_grids["all"] = grids_all
    # first voting round
    candidates = []
    for grids in category_to_grids.values():
        candidates += get_three_votes(grids)
    # second voting round
    c1, c2, c3 = get_three_votes(candidates)
    # break tie between c2 and c3
    if candidates.count(c2) == candidates.count(c3):
        if "identity" in category_to_grids:
            if category_to_grids["identity"].count(c2) < category_to_grids["identity"].count(c3):
                c2 = c3
    return c1, c2, grids_all


def grid_2d_to_text(grid: list[List[int]]):
    height, width = len(grid), len(grid[0])
    lines = [f"{str(height)}{str(width)}"]
    for row in grid:
        lines.append("".join([str(x) for x in row]))
    return "\n".join(lines)


def text_to_2d_grid(text: str) -> Tuple[Optional[List[List[int]]], bool]:
    try:
        text = text.strip() # label is appended by \n
        grid_lines = text.split('\n')
        grid = []
        row_lens = []
        grid_lines = grid_lines[1:]
        for l in grid_lines: # skip dimensions
            row = [int(x) for x in l]
            grid.append(row)
            row_lens.append(len(row))
            assert all(0 <= x and x < 10 for x in row)
        assert len(set(row_lens)) == 1 # so the grid is not empty
        assert row_lens[0] > 0
        return grid, True
    except:
        return None, False


@torch.no_grad()
def save_train_model(
        model: Union[nn.Module, DistributedDataParallel],
        output_dir: str,
        epoch: int,
    ) -> str:
    save_enc_path = os.path.join(output_dir, f"lora_epoch_{epoch+1}")
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.save_pretrained(save_enc_path, save_embedding_layers=True)
    logger.info(f"Saved model to {save_enc_path}")
    return save_enc_path


def print_trainable_parameters(model):
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


def initialize_program_embeddings(
        embeddings: torch.Tensor,
        accelerator: Accelerator,
        ntokens: int,
<<<<<<< HEAD
=======
        cov_scale: float,
>>>>>>> origin/main
    ) -> torch.Tensor:

    dtype = embeddings.dtype
    device = embeddings.device
    n_embeds = embeddings.shape[0]
    embeddings = embeddings.to(torch.float32).to(device=accelerator.device)
    mean_embeddings = torch.mean(embeddings, axis=0) # type: ignore
    centered_embeddings = embeddings - mean_embeddings
    covariance = centered_embeddings.T @ centered_embeddings / n_embeds
    eigenvalues = torch.linalg.eigvals(covariance)
    assert not ((covariance == covariance.T).all() and not torch.is_complex(eigenvalues) and (eigenvalues > 0).all())
<<<<<<< HEAD
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean_embeddings, covariance_matrix=1e-9 * covariance)
    return distribution.sample(sample_shape=(ntokens,)).to(device).to(dtype) # type: ignore

=======
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean_embeddings, covariance_matrix=cov_scale * covariance)
    return distribution.sample(sample_shape=(ntokens,)).to(device).to(dtype) # type: ignore


def model_loss(
    model: Union[nn.Module, DistributedDataParallel],
    tokenizer: ARCTokenizer,
    program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    # data
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,
    input_ids_lens: List[int],
    pair_start_idxs: List[List[int]],
    ntokens: int,
    # others
    pad_side: str,
    attention_cutoff: bool,
    attend_prev_programs: bool,
) -> torch.Tensor:

    # original baseline
    if program_embeddings is None:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
        ).loss

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens
    device, dtype = input_ids.device, input_ids.dtype
    max_num_pairs = max(len(start_idxs) for start_idxs in pair_start_idxs)

    # get embeddings
    pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device)) # (hidden_dim,)
    inputs_embeds = embed_tokens(input_ids)

    # insert program embeddings
    inputs_embeds_with_programs = []
    attention_mask_with_programs = []
    label_ids_with_programs = []
    program_intervals = []

    for task_input_ids, task_inputs_embeds, task_attention_mask, task_label_ids, input_ids_len, start_idxs in zip(input_ids, inputs_embeds, attention_mask, label_ids, input_ids_lens, pair_start_idxs):
        assert start_idxs[0] == 1 # first pair starts after bos
        # start_idxs are offset if padding side is left
        if pad_side == "left":
            start_idxs = [s + task_input_ids.shape[0] - input_ids_len for s in start_idxs]
        assert len(set(task_input_ids[start_idxs].tolist())) == 1 # all should be the input token (we remove bos)

        # insert program token before every pair
        task_inputs_embeds_with_programs = [task_inputs_embeds[:start_idxs[0]]]
        task_attention_mask_with_programs = [task_attention_mask[:start_idxs[0]]]
        task_label_ids_with_programs = [task_label_ids[:start_idxs[0]]]
        task_program_intervals = []

        for i, start_idx in enumerate(start_idxs):
            end_idx = start_idxs[i+1] if i < len(start_idxs) - 1 else len(task_inputs_embeds)
            # program intervals
            program_start = sum(len(x) for x in task_inputs_embeds_with_programs)
            task_program_intervals.append((program_start, program_start + ntokens))
            # insert program embedding into inputs_embeds
            task_inputs_embeds_with_programs.append(program_embeddings("dummy"))
            task_inputs_embeds_with_programs.append(task_inputs_embeds[start_idx: end_idx])
            # insert full attention for programs
            task_attention_mask_with_programs.append(torch.full((ntokens,), 1, device=device, dtype=dtype))
            task_attention_mask_with_programs.append(task_attention_mask[start_idx: end_idx])
            # insert no label supervision for programs
            task_label_ids_with_programs.append(torch.full((ntokens,), -100, device=device, dtype=dtype))
            task_label_ids_with_programs.append(task_label_ids[start_idx: end_idx])

        # some programs in batch have fewer pairs, pad manually, so hacky
        pad_length = (max_num_pairs - len(start_idxs)) * ntokens
        task_pad_inputs_embeds = pad_embeds[None, ...].expand(pad_length, -1)
        task_pad_attention_mask = torch.full((pad_length,), 0, device=device, dtype=dtype)
        task_pad_label_ids = torch.full((pad_length,), -100, device=device, dtype=dtype)
        if pad_side == 'left':
            task_inputs_embeds_with_programs.insert(0, task_pad_inputs_embeds)
            task_attention_mask_with_programs.insert(0, task_pad_attention_mask)
            task_label_ids_with_programs.insert(0, task_pad_label_ids)
            task_program_intervals = [(x[0] + pad_length, x[1] + pad_length) for x in task_program_intervals]
        else:
            task_inputs_embeds_with_programs.append(task_pad_inputs_embeds)
            task_attention_mask_with_programs.append(task_pad_attention_mask)
            task_label_ids_with_programs.append(task_pad_label_ids)

        # concat all
        inputs_embeds_with_programs.append(torch.cat(task_inputs_embeds_with_programs))
        attention_mask_with_programs.append(torch.cat(task_attention_mask_with_programs))
        label_ids_with_programs.append(torch.cat(task_label_ids_with_programs))
        program_intervals.append(task_program_intervals)

    # stack and check, now we have the three inputs with programs
    inputs_embeds_with_programs = torch.stack(inputs_embeds_with_programs)
    attention_mask_with_programs = torch.stack(attention_mask_with_programs)
    label_ids_with_programs = torch.stack(label_ids_with_programs)
    assert inputs_embeds_with_programs.shape[1] == input_ids.shape[1] + max_num_pairs * ntokens
    assert inputs_embeds_with_programs.shape[:2] == attention_mask_with_programs.shape[:2] == label_ids_with_programs.shape[:2]

    if not attention_cutoff:
        return model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            labels=label_ids_with_programs,
        ).loss
    else:
        return model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            labels=label_ids_with_programs,
            program_intervals=program_intervals,
            attend_prev_programs=attend_prev_programs,
        ).loss

>>>>>>> origin/main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--tracker_project_name", type=str, default="arc")
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_len", type=int, default=-1) # two grid -> 1867
    parser.add_argument("--debug_fixed_order", action="store_true")
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)
    parser.add_argument("--no_separate_color_tokens", action='store_true')
    parser.add_argument("--no_color_permute", action="store_true")
    parser.add_argument("--no_pair_permute", action="store_true")
    parser.add_argument("--no_d8", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--thinking_tokens", action="store_true")

    # Gist/thinking tokens
    # single means one program, all means pay attention to all programs
    parser.add_argument("--ntokens", type=int, default=-1)
    parser.add_argument("--attention_cutoff", action="store_true")
    parser.add_argument("--attend_prev_programs", action="store_true")

    # Training
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_program", type=float, default=1e-4)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--lr_thinking", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")
    parser.add_argument("--dry_train_run", action="store_true")
    parser.add_argument("--dry_eval_run", action="store_true")
    parser.add_argument("--warmup_epoch", type=int, default=1)

    # both data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_num_pair", type=int, default=3) # includes test pair
    parser.add_argument("--max_num_pair", type=int, default=8) # includes test pair
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--train_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--gen_pad_side", type=str, choices=["left", "right"], default="left")

    # re-arc train data
    parser.add_argument("--train_data_dir", type=str, default="./data/re-arc/train_data/tasks")
    parser.add_argument("--no_train_original", action="store_true")
    parser.add_argument("--only_train_original", action="store_true")

    # extra train data
    parser.add_argument("--re_arc_ratio", type=float, default=1.0)
    parser.add_argument("--concept_arc_ratio", type=float, default=0.0)
    parser.add_argument("--arc_heavy_ratio", type=float, default=0.0)

    # augmentation
    parser.add_argument("--extra_augment_ratio", type=float, default=0.0)
    parser.add_argument("--extra_augment_single_grid", action="store_true")

    # eval train data
    parser.add_argument("--eval_train_dir", type=str, default="./data/re-arc/arc_original/training")
    parser.add_argument("--eval_train_select_tasks_path", type=str, default=None)
    parser.add_argument("--eval_train_leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--eval_train_leave_ns_inc", action="store_true")
    parser.add_argument("--eval_train_permute_n", type=int, default=0)
    parser.add_argument("--eval_train_augment_n", type=int, default=0)
    parser.add_argument("--eval_train_permute_iters", type=int, default=0)

    # eval eval data (mirror eval train data)
    parser.add_argument("--eval_eval_dir", type=str, default="./data/re-arc/arc_original/evaluation")
    parser.add_argument("--eval_eval_select_tasks_path", type=str, default=None)
    parser.add_argument("--eval_eval_leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--eval_eval_leave_ns_inc", action="store_true")
    parser.add_argument("--eval_eval_permute_n", type=int, default=0)
    parser.add_argument("--eval_eval_augment_n", type=int, default=0)
    parser.add_argument("--eval_eval_permute_iters", type=int, default=0)

    # Lora
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=float, default=24.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument('--lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head'
    ])
    parser.add_argument("--no_rslora", action='store_true')

    parser.add_argument("--ntokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # override to debug stuff
    if args.debug:
        args.tag = 'test'
        args.wandb = False
        args.samples_per_epoch = 32
        args.log_every = 1

    # check args
    if args.model_name == "nemo8b":
        assert args.train_pad_side == args.gen_pad_side == "left"
    assert not (args.no_train_original and args.only_train_original)
    assert args.min_num_pair >= 3

    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable

    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    os.environ["WANDB_API_KEY"]="8f75801720d1540f03dd3a73e52f11d8ec74f395"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        project_config=project_config,
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_up_main_process_logger(accelerator, logger)
    set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(
            args.tracker_project_name,
            tracker_config,
            init_kwargs={"wandb": {"name": args.tag}}
        )
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("Accelerator and seed set up.")

    # log args
    logger.info("#### BEGIN ALL ARGUMENTS ####")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("#### END ALL ARGUMENTS ####\n")

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.model_name], cache_dir='./encoder_decoder_cache')
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.pad_token is None
    assert isinstance(tokenizer.bos_token, str)
    logger.info("Tokenizers loaded and pad tokens handled.")

    # Build base models
    from_pretrained_kwargs = {
        "cache_dir": "./encoder_decoder_cache",
        "low_cpu_mem_usage": True,
    }
    if args.flash_attn:
        from_pretrained_kwargs["attn_implementation"] = "flash_attention_2"
    if args.untrainable_nbit in NBIT_TO_DTYPE:
        from_pretrained_kwargs["torch_dtype"] = NBIT_TO_DTYPE[args.untrainable_nbit]
    elif args.untrainable_nbit == 4:
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=NBIT_TO_DTYPE[args.trainable_nbit],
        )
    elif args.untrainable_nbit == 3.6:
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=NBIT_TO_DTYPE[args.trainable_nbit],
        )
    elif args.untrainable_nbit == 8:
        # wtf why this more memory
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"unrecognized untrainable_nbit {args.untrainable_nbit}")

    base_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        **from_pretrained_kwargs,
    )

    if args.untrainable_nbit in [4, 8]:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logger.info("Base models loaded.")
    # for thinking tokens
    if args.thinking_tokens:
        # initialize program embeddings
        thinking_embeddings = ProgramEmbeddings(
            embedding=initialize_program_embeddings(
                base_model.model.embed_tokens.weight.data.detach().clone(),
                accelerator,
                ntokens=args.ntokens,
            ),
        )
  
        for param in thinking_embeddings.parameters():
            param.requires_grad = True
     
        for name, param in thinking_embeddings.named_parameters():
            assert param.requires_grad
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
        
        thinking_embeddings_n_params = sum(p.numel() for p in thinking_embeddings.parameters())
        logger.info(f'thking embeddings size {round(get_memory_footprint(thinking_embeddings) / 1024 ** 3, 2)}GB')

    # initialize program embeddings
    program_embeddings = None
    if args.ntokens > 0:
        program_embeddings = ProgramEmbeddings(
            embedding=initialize_program_embeddings(
                base_model.model.embed_tokens.weight.data.detach().clone(),
                accelerator,
                ntokens=args.ntokens,
                cov_scale=1e-9,
            ),
        )
        logger.info("Program embeddings initialized.")

    # only keep these tokens, resize model embedding (eos == pad)
    # we do not include program tokens here, those are added later during training and inference
    keep_tokens = [str(i) for i in range(31)] # dim
    keep_tokens += [tokenizer.bos_token, tokenizer.eos_token, "\n", "input", "output", "pad"]
    assert len(set(keep_tokens)) == len(keep_tokens)

    keep_token_ids = []
    for token in keep_tokens:
        token_id = tokenizer(token)["input_ids"] # type: ignore
        assert isinstance(token_id, list) and len(token_id) == 2 # with start token
        keep_token_ids.append(token_id[1])
    assert len(set(keep_token_ids)) == len(keep_token_ids)

    color_embeddings = None
    if not args.no_separate_color_tokens:
        color_embeddings = initialize_program_embeddings(
            base_model.model.embed_tokens.weight.data.detach().clone(),
            accelerator,
            ntokens=10,
            cov_scale=1.0,
        )

    # this breaks embedding tying, but whatever
    with torch.no_grad():
        # subset embeddings and lmheads
        assert base_model.model.embed_tokens.weight.shape == base_model.lm_head.weight.shape
        base_model.model.embed_tokens.weight = nn.Parameter(base_model.model.embed_tokens.weight[keep_token_ids])
        base_model.model.embed_tokens.num_embeddings = len(keep_token_ids) # type: ignore
        assert base_model.lm_head.bias is None
        base_model.lm_head.weight = nn.Parameter(base_model.lm_head.weight[keep_token_ids])
        base_model.lm_head.out_features = len(keep_token_ids)
        base_model.config.tie_word_embeddings = False

        if not args.no_separate_color_tokens:
            assert isinstance(color_embeddings, torch.Tensor)
            base_model.model.embed_tokens.weight = nn.Parameter(torch.cat([color_embeddings, base_model.model.embed_tokens.weight]))
            base_model.model.embed_tokens.num_embeddings += 10 # type: ignore
            base_model.lm_head.weight = nn.Parameter(torch.cat([color_embeddings, base_model.lm_head.weight]))
            base_model.lm_head.out_features += 10

    if not args.no_separate_color_tokens:
        keep_tokens = [f"c{c}" for c in range(10)] + keep_tokens

    # update configs
    assert base_model.config.vocab_size and base_model.config.bos_token_id and base_model.config.eos_token_id
    base_model.config.vocab_size = len(keep_token_ids) + (0 if args.no_separate_color_tokens else 10)
    base_model.config.bos_token_id = keep_tokens.index(tokenizer.bos_token) # type: ignore
    base_model.config.eos_token_id = keep_tokens.index(tokenizer.eos_token) # type: ignore

    # create custom tokenizer
    arc_tokenizer = ARCTokenizer(
        tokens=keep_tokens, # type: ignore
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token, # type: ignore
        pad_token="pad",
    )
    del tokenizer
    tokenizer = arc_tokenizer

    # lora
    model = None
    if args.no_lora:
        model = base_model
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            use_rslora=not args.no_rslora,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, peft_config)
    logger.info("LoRA-wrapped models initialized (optional)")

    # ensure require grad
    if program_embeddings is not None:
        for param in program_embeddings.parameters():
            param.requires_grad = True

    # convert model weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if program_embeddings is not None:
        for name, param in program_embeddings.named_parameters():
            assert param.requires_grad
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # number of parameters and size
    print_trainable_parameters(model)
    logger.info(f'model size {round(model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    if program_embeddings is not None:
        program_embeddings_n_params = sum(p.numel() for p in program_embeddings.parameters())
        logger.info(f'program embedding params {three_commas(program_embeddings_n_params)}')

    # Build training dataset
    train_dataset = TrainDataset(
        train_data_dir=args.train_data_dir,
        eval_train_dir=args.eval_train_dir,
        re_arc_ratio=args.re_arc_ratio,
        concept_arc_ratio=args.concept_arc_ratio,
        arc_heavy_ratio=args.arc_heavy_ratio,
        tokenizer=tokenizer,
        total_steps=args.samples_per_epoch,
        extra_augment_ratio=args.extra_augment_ratio,
        extra_augment_single_grid=args.extra_augment_single_grid,
        seed=args.seed,
        process_index=accelerator.process_index,
        debug_fixed_order=args.debug_fixed_order,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        train_pad_side=args.train_pad_side,
        no_color_permute=args.no_color_permute,
        no_pair_permute=args.no_pair_permute,
        no_d8=args.no_d8,
        min_num_pair=args.min_num_pair,
        max_num_pair=args.max_num_pair,
        no_train_original=args.no_train_original,
        only_train_original=args.only_train_original,
        debug_len=args.debug_len,
        num_workers=args.num_workers,
        no_separate_color_tokens=args.no_separate_color_tokens,
        max_seq_len=args.max_seq_len,
        ntokens=args.ntokens,
    )
    # special token id
    thinking_token_id = 666
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset,ntokens = args.ntokens, special_token_id = thinking_token_id)
    if args.debug_len > 0:
        train_collate_fn = partial(collate_fn_train_dummy, dataset=train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False, # this doesn't matter, collate does all the work
        collate_fn=train_collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )
    logger.info(f"len(train_dataset) = {len(train_dataset)}")
    logger.info(f"len(train_loader) = {len(train_loader)}")

    # Param groups for LoRA
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "embed" in name or "lm_head" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
<<<<<<< HEAD
    if args.thinking_tokens:
        thinking_params = [param for param in thinking_embeddings.parameters()]
        optimizer_grouped_params = [
            {"params": embedding_params, "lr": args.lr_embedding},
            {"params": other_params, "lr": args.lr_other},
            {"params": thinking_params, "lr": args.lr_thinking},
        ]
        all_params = embedding_params + other_params+thinking_params
    else:
        optimizer_grouped_params = [
            {"params": embedding_params, "lr": args.lr_embedding},
            {"params": other_params, "lr": args.lr_other},
        ]
        all_params = embedding_params + other_params
    
=======
    program_params = [param for param in program_embeddings.parameters()] if program_embeddings is not None else []

    optimizer_grouped_params = [
        {"params": embedding_params, "lr": args.lr_embedding},
        {"params": program_params, "lr": args.lr_program},
        {"params": other_params, "lr": args.lr_other},
    ]

    all_params = embedding_params + other_params + program_params
>>>>>>> origin/main
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay) # type: ignore
    elif args.optimizer == 'adamw8bit':
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_params, weight_decay=args.weight_decay)
        # optimizer = bnb.optim.PagedAdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_params) # type: ignore
    logger.info(f"Optimizer with {len(embedding_params)} embed-params lr={args.lr_embedding}")
    logger.info(f"Optimizer with {len(other_params)} other-params lr={args.lr_other}")

    # LR schedule
    steps_per_epoch = args.samples_per_epoch // (args.train_batch_size * args.grad_accum_steps * accelerator.num_processes)
    num_training_steps = steps_per_epoch * args.num_epochs
    if args.lr_scheduler == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps * args.warmup_epoch,
            num_training_steps=num_training_steps * args.grad_accum_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps * args.warmup_epoch,
        )
    logger.info(f'lr scheduler with {steps_per_epoch} warmup steps')

    # Prepare with accelerator
    (
        model,
        program_embeddings,
        optimizer,
        train_loader,
    ) = accelerator.prepare(
        model,
        program_embeddings,
        optimizer,
        train_loader,
    )
    assert isinstance(model, (nn.Module, DistributedDataParallel))
    if program_embeddings is not None:
        assert isinstance(program_embeddings, (ProgramEmbeddings, DistributedDataParallel))

    if args.dry_train_run:
        for _ in tqdm(train_loader, total=len(train_loader)):
            pass
        exit()

    # Build evaluation datasets
    eval_train_dataset = EvalDataset(
        args.eval_train_dir,
        select_tasks_path=args.eval_train_select_tasks_path,
        leave_ns=args.eval_train_leave_ns,
        leave_ns_inc=args.eval_train_leave_ns_inc,
        permute_n=args.eval_train_permute_n,
        augment_n=args.eval_train_augment_n,
        permute_iters=args.eval_train_permute_iters,
        seed=args.seed,
        tokenizer=tokenizer,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        train_pad_side=args.train_pad_side,
        gen_pad_side=args.gen_pad_side,
        debug_len=args.debug_len,
        no_separate_color_tokens=args.no_separate_color_tokens,
        max_seq_len=args.max_seq_len,
        ntokens=args.ntokens,
    )
    eval_eval_dataset = EvalDataset(
        eval_dir=args.eval_eval_dir,
        select_tasks_path=args.eval_eval_select_tasks_path,
        leave_ns=args.eval_eval_leave_ns,
        leave_ns_inc=args.eval_eval_leave_ns_inc,
        permute_n=args.eval_eval_permute_n,
        augment_n=args.eval_eval_augment_n,
        permute_iters=args.eval_eval_permute_iters,
        seed=args.seed,
        tokenizer=tokenizer,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        train_pad_side=args.train_pad_side,
        gen_pad_side=args.gen_pad_side,
        debug_len=args.debug_len,
        no_separate_color_tokens=args.no_separate_color_tokens,
        max_seq_len=args.max_seq_len,
        ntokens=args.ntokens,
    )
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_train_dataset)
    if args.debug_len > 0:
        eval_collate_fn = partial(collate_fn_eval_dummy, dataset=eval_train_dataset)

    logger.info(f'======= TRAINING INFO START =======')
    logger.info(f'num_epochs={args.num_epochs}')
    logger.info(f'train_batch_size={args.train_batch_size}')
    logger.info(f'grad_accum_steps={args.grad_accum_steps}')
    logger.info(f'accelerator.num_processes={accelerator.num_processes}')
    logger.info(f'steps_per_epoch={steps_per_epoch}')
    logger.info(f'{three_commas(sum(p.numel() for p in all_params))} trainable params')
    logger.info(f'======= TRAINING INFO END =======\n')

    global_step = 0
    progress_bar = tqdm(
        range(num_training_steps),
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Total Train Steps")

    # model saving
    last_save_model_path = None
    epoch_to_eval_exact_acc = {}

    # train!
    for epoch in range(args.num_epochs):
        model.train()

        ce_loss_accum = 0.0
        grad_norm_accum = 0.0
        pad_side = 'right'

        for batch_data in train_loader:
            input_ids = batch_data["input_ids"].to(accelerator.device)
            attention_mask = batch_data["attention_mask"].to(accelerator.device)
            label_ids = batch_data["label_ids"].to(accelerator.device)
            input_ids_lens = batch_data["input_ids_lens"]
            pair_start_idxs = batch_data["pair_start_idxs"]

            if args.thinking_tokens:
                ntokens = args.ntokens
                batch_size = len(input_ids[0])
                module = model.module if isinstance(model, DistributedDataParallel) else model
                embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens
                dtype, device = input_ids[0].dtype, input_ids[0].device
                # thinking_inputs_embeds = thinking_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
                mask = (x == thinking_token_id).unsqueeze(-1)
                inputs_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in input_ids]
                input_embeds = torch.where(mask, thinking_embeddings, inputs_embeds)

                # extra_program_attention_mask = torch.full((batch_size, ntokens), 1, device=device, dtype=dtype)
                # extra_program_label_ids = torch.full((batch_size, ntokens), -100, device=device, dtype=dtype)
                # pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device))
                # inputs_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in input_ids]
                # with_thiking_embeds = insert_based_on_sides(
                #     data=inputs_embeds,
                #     to_insert=thinking_inputs_embeds,
                #     insert_side="right",
                #     pad_side=pad_side,
                #     pad_id=pad_embeds,
                # )
                # with_thinking_attention_mask = insert_based_on_sides(
                #     data=attention_mask,
                #     to_insert=extra_program_attention_mask,
                #     insert_side="right",
                #     pad_side=pad_side,
                #     pad_id=pad_embeds,
                # )

                # with_thinking_labels = insert_based_on_sides(
                #     data=label_ids,
                #     to_insert=extra_program_attention_mask,
                #     lens=extra_program_label_ids,
                #     insert_side="right",
                #     pad_side=pad_side,
                #     pad_id=pad_embeds,
                # )
                
                

            with accelerator.accumulate(model):
                with accelerator.autocast():
<<<<<<< HEAD
                    if args.thinking_tokens:
                        ce_loss = model(
                            input_embeds=with_thiking_embeds,
                            attention_mask=with_thinking_attention_mask,
                            labels=with_thinking_labels,
                        ).loss
                    else:
                        ce_loss = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=label_ids,
                        ).loss
=======
                    ce_loss = model_loss(
                        model=model,
                        tokenizer=tokenizer,
                        program_embeddings=program_embeddings,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        label_ids=label_ids,
                        input_ids_lens=input_ids_lens,
                        pair_start_idxs=pair_start_idxs,
                        ntokens=args.ntokens,
                        pad_side=args.train_pad_side,
                        attention_cutoff=args.attention_cutoff,
                        attend_prev_programs=args.attend_prev_programs,
                    )
>>>>>>> origin/main

                avg_ce_loss = accelerator.gather(ce_loss.repeat(args.train_batch_size)).mean() # type: ignore
                ce_loss_accum += avg_ce_loss.item() / args.grad_accum_steps

                accelerator.backward(ce_loss)
                if accelerator.sync_gradients:
                    grad_norm_accum += accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item() / args.grad_accum_steps # type: ignore
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    progress_bar.update(args.log_every)
                    try:
                        accelerator.log({
                            "train/ce_loss": ce_loss_accum,
                            "train/grad_norm": grad_norm_accum,
                            "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                            "train/lr_program": lr_scheduler.get_last_lr()[1],
                            "train/lr_other": lr_scheduler.get_last_lr()[2],
                        }, step=global_step)
                    except:
                        logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                ce_loss_accum = 0.0
                grad_norm_accum = 0.0

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
            train_exact_acc, train_valid_grid, train_correct_grid_dim, train_token_acc, train_relaxed_token_acc, train_texts, \
                train_votes, train_competition_sub_acc, train_competition_all_acc, _ = evaluate(
                task_to_ttt_path=None,
                ttt_param_names=None,
                model=model,
                program_embeddings=program_embeddings,
                tokenizer=tokenizer,
                dataset=eval_train_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                dry_eval_run=args.dry_eval_run,
                output_dir=args.output_dir,
                ntokens=args.ntokens,
                attention_cutoff=args.attention_cutoff,
                attend_prev_programs=args.attend_prev_programs,
            )
            eval_exact_acc, eval_valid_grid, eval_correct_grid_dim, eval_token_acc, eval_relaxed_token_acc, eval_texts, \
                eval_votes, eval_competition_sub_acc, eval_competition_all_acc, _ = evaluate(
                task_to_ttt_path=None,
                ttt_param_names=None,
                model=model,
                program_embeddings=program_embeddings,
                tokenizer=tokenizer,
                dataset=eval_eval_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                dry_eval_run=args.dry_eval_run,
                output_dir=args.output_dir,
                ntokens=args.ntokens,
                attention_cutoff=args.attention_cutoff,
                attend_prev_programs=args.attend_prev_programs,
            )

            if accelerator.is_main_process:
                eval_metric_dict = {
                    "eval/train_exact_acc": train_exact_acc,
                    "eval/train_valid_grid": train_valid_grid,
                    "eval/train_correct_grid_dim": train_correct_grid_dim,
                    "eval/train_token_acc": train_token_acc,
                    "eval/train_relaxed_token_acc": train_relaxed_token_acc,
                    "eval/train_competition_sub_acc": train_competition_sub_acc,
                    "eval/train_competition_all_acc": train_competition_all_acc,
                    "eval/eval_exact_acc": eval_exact_acc,
                    "eval/eval_valid_grid": eval_valid_grid,
                    "eval/eval_correct_grid_dim": eval_correct_grid_dim,
                    "eval/eval_token_acc": eval_token_acc,
                    "eval/eval_relaxed_token_acc": eval_relaxed_token_acc,
                    "eval/eval_competition_sub_acc": eval_competition_sub_acc,
                    "eval/eval_competition_all_acc": eval_competition_all_acc,
                }
                logger.info(f'Evaluation results:\n{pprint.pformat(eval_metric_dict, indent=4)}')
                try:
                    accelerator.log(eval_metric_dict, step=global_step)
                except:
                    logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                # Save outputs
                save_eval_train_pred_gt_path = os.path.join(args.output_dir, f"eval_train_{epoch+1}_pred_gt.json")
                save_eval_eval_pred_gt_path = os.path.join(args.output_dir, f"eval_eval_{epoch+1}_pred_gt.json")
                with open(save_eval_train_pred_gt_path, 'w') as f:
                    json.dump(train_texts, f)
                with open(save_eval_eval_pred_gt_path, 'w') as f:
                    json.dump(eval_texts, f)
                logger.info(f"Saved eval train pred gt to {save_eval_train_pred_gt_path}")
                logger.info(f"Saved eval eval pred gt to {save_eval_eval_pred_gt_path}")

                # save votes
                save_eval_train_vote_path = os.path.join(args.output_dir, f"eval_train_{epoch+1}_vote.json")
                save_eval_eval_vote_path = os.path.join(args.output_dir, f"eval_eval_{epoch+1}_vote.json")
                with open(save_eval_train_vote_path, 'w') as f:
                    json.dump(train_votes, f)
                with open(save_eval_eval_vote_path, 'w') as f:
                    json.dump(eval_votes, f)
                logger.info(f"Saved eval train vote to {save_eval_train_vote_path}")
                logger.info(f"Saved eval eval vote to {save_eval_eval_vote_path}")

                # Save model
                do_save_model = args.save_all_models
                if not args.save_all_models:
                    if (not epoch_to_eval_exact_acc) or eval_exact_acc >= max(epoch_to_eval_exact_acc.values()):
                        do_save_model = True

                if do_save_model:
                    if (not args.save_all_models) and (last_save_model_path is not None):
                        save_enc_path = last_save_model_path
                        rm_cmd = f"rm -rf {save_enc_path}"
                        os.system(rm_cmd)
                    last_save_model_path = save_train_model(
                        model=model,
                        output_dir=args.output_dir,
                        epoch=epoch,
                    )
                epoch_to_eval_exact_acc[epoch] = eval_exact_acc

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
