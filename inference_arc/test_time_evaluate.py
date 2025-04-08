import math
import gc
import time
from tqdm import tqdm
from collections import defaultdict
from typing import Union, Tuple, List
from datetime import timedelta
import pprint
import json
from functools import partial
import argparse
from arclib.arc import Task

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from accelerate.logging import get_logger

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training # type: ignore

from data_utils import (
    ARCTokenizer,
    EvalDataset,
    GSDataset,
    TTTDataset,
    collate_fn_gs,
    collate_fn_gs_dummy,
    collate_fn_ttt,
    collate_fn_ttt_dummy,
)
from train import (
    set_up_main_process_logger,
    initialize_program_embeddings,
    best_match_count,
    text_to_2d_grid,
    list2d_to_tuple,
    invert_and_vote,
    grid_2d_to_text,
    chunks,
)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "28800" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "28800000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

logger = get_logger(__name__, log_level="INFO")


MODEL_NAME_TO_PATH = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
}
NBIT_TO_DTYPE = {
    16: torch.bfloat16,
    32: torch.float32,
}


def generate_unique_permute_masks(tensor, start_idxs, M):
    # ensure first perm is identity
    boundaries = start_idxs + [tensor.size(0)]
    chunk_indices = [torch.arange(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    num_chunks = len(chunk_indices)

    unique_masks = []
    seen_orders = set()

    M = min(M, math.factorial(num_chunks))

    while len(unique_masks) < M:
        # Generate a random permutation order for the chunks.
        if len(unique_masks) == 0:
            order = list(range(num_chunks))
        else:
            order = torch.randperm(num_chunks).tolist()
        order_tuple = tuple(order)

        # Check if this permutation has already been generated.
        if order_tuple in seen_orders:
            continue

        seen_orders.add(order_tuple)

        # Build the mask by concatenating the indices in the new order.
        mask = torch.cat([chunk_indices[i] for i in order])
        unique_masks.append(mask)

    return unique_masks


@torch.no_grad()
def initialize_kv(
    model: nn.Module,
    demon_input_ids: torch.Tensor,
    demon_start_idxs: List[int],
    accelerator: Accelerator,
    gs_iters: int,
    gs_random_kv: bool,
    gs_num_permute: int,
    gs_permute_batch_size: int,
    gs_permute_back: bool,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:

    # initialize kv
    if gs_iters > 0 and gs_random_kv:
        # random kv initialization
        past_key_values = tuple(
            (
                torch.randn((1, model.config.num_key_value_heads, demon_input_ids.shape[1], model.config.head_dim), device=accelerator.device, dtype=torch.float32),
                torch.randn((1, model.config.num_key_value_heads, demon_input_ids.shape[1], model.config.head_dim), device=accelerator.device, dtype=torch.float32),
            ) for _ in range(model.config.num_hidden_layers)
        )

    elif gs_iters == 0 or gs_num_permute == 1:
        # only one kv is needed
        with accelerator.autocast():
            past_key_values = model(input_ids=demon_input_ids, output_hidden_states=True).past_key_values

    else:
        # generate batches of permutations of them and average all
        permute_masks = generate_unique_permute_masks(demon_input_ids[0], demon_start_idxs, gs_num_permute)

        past_key_values = tuple(
            (
                torch.zeros((1, model.config.num_key_value_heads, demon_input_ids.shape[1], model.config.head_dim), device=accelerator.device, dtype=torch.float32),
                torch.zeros((1, model.config.num_key_value_heads, demon_input_ids.shape[1], model.config.head_dim), device=accelerator.device, dtype=torch.float32),
            ) for _ in range(model.config.num_hidden_layers)
        )
        for batch_permute_masks in chunks(permute_masks, gs_permute_batch_size):
            # get batch of permuted demon input ids
            batch_demon_input_ids = []
            for permute_mask in batch_permute_masks:
                batch_demon_input_ids.append(demon_input_ids.squeeze(0)[permute_mask])
            batch_demon_input_ids = torch.stack(batch_demon_input_ids)

            # get kv of each
            with accelerator.autocast():
                batch_past_key_values = model(input_ids=batch_demon_input_ids, output_hidden_states=True).past_key_values
            assert len(batch_permute_masks) == batch_past_key_values[0][0].shape[0]

            # optionally permute kv back
            inverse_mask = None
            if gs_permute_back:
                for batch_i, permute_mask in enumerate(batch_permute_masks):
                    inverse_mask = torch.empty_like(permute_mask)
                    inverse_mask[permute_mask] = torch.arange(len(permute_mask))
                    for layer_i in range(len(batch_past_key_values)):
                        for kv_i in range(2):
                            batch_past_key_values[layer_i][kv_i][batch_i] = batch_past_key_values[layer_i][kv_i][batch_i, :, inverse_mask, :]
                    # debug: make sure inverse mask is correct
                    assert torch.equal(batch_demon_input_ids[batch_i, inverse_mask], demon_input_ids.squeeze(0))

            # add batch sum to a total sum of past_key_values
            assert batch_past_key_values[0][0].shape[0] == len(batch_permute_masks)
            for layer_i in range(len(batch_past_key_values)):
                for kv_i in range(2):
                    past_key_values[layer_i][kv_i].add_(batch_past_key_values[layer_i][kv_i].sum(dim=0, keepdim=True))

            del batch_demon_input_ids, batch_past_key_values, inverse_mask

        # average the past key values
        for layer_i in range(len(past_key_values)):
            for kv_i in range(2):
                past_key_values[layer_i][kv_i].div_(len(permute_masks))

    return past_key_values # type: ignore


@torch.no_grad()
def test_time_evaluate(
    model: Union[nn.Module, DistributedDataParallel],
    dataset: EvalDataset,
    accelerator: Accelerator,
    trainable_nbit: int,
    no_flash_attn: bool,
    gs_iters: int,
    gs_batch_size: int,
    gs_grad_accum_steps: int,
    gs_lr: float,
    gs_beta1: float,
    gs_beta2: float,
    gs_weight_decay: float,
    gs_optimizer: str,
    gs_max_grad_norm: float,
    gs_no_key: bool,
    gs_no_value: bool,
    gs_loss_on_input: bool,
    gs_lr_scheduler: str,
    gs_lora: bool,
    gs_lora_rank: int,
    gs_lora_alpha: int,
    gs_lora_lr: float,
    gs_lora_beta1: float,
    gs_lora_beta2: float,
    gs_random_kv: bool,
    gs_num_permute: int,
    gs_permute_batch_size: int,
    gs_permute_back: bool,
    ttt_iters: int,
    ttt_batch_size: int,
    ttt_grad_accum_steps: int,
    ttt_lr: float,
    ttt_weight_decay: float,
    ttt_optimizer: str,
    ttt_max_grad_norm: float,
    ttt_lr_scheduler: str,
    ttt_lora_rank: int,
    ttt_lora_alpha: int,
    ttt_permute_n: int,
    output_dir: str,
):
    model.eval()

    # get modules in case of DDP
    model = model.module if isinstance(model, DistributedDataParallel) else model
    model.generation_config.pad_token_id = dataset.tokenizer.pad_token_id

    # We perform test-time adaptation in 2 stages.
    # First stage produces KV cache, num trainable params, runtime, num data. Second stage simply performs model loss/generation

    # STAGE 1: demonstration pair processing
    ttt_num_data_list, gs_num_data_list = [], []
    ttt_num_params_list, gs_num_params_list = [], []
    ttt_time_list, gs_time_list = [], []
    init_kv_time_list = []

    # outputs
    task_id_and_text_list = []
    task_id_and_inverter_grids = []
    exact_acc_list = []
    valid_grid_list = []
    correct_grid_dim_list = []
    token_acc_list = []
    relaxed_token_acc_list = []

    # we need to cache model in case of ttt or gs with trainable lora
    cached_model_path = os.path.join(output_dir, 'eval_cached_model')
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(cached_model_path)

    # group tasks based on task name
    task_name_to_eval_idxs = defaultdict(list)
    for task_i, task in enumerate(dataset.eval_tasks):
        assert task_i not in task_name_to_eval_idxs[task.name]
        task_name_to_eval_idxs[task.name].append(task_i)
    task_names = sorted(task_name_to_eval_idxs.keys())
    assert all(len(v) == 1 for v in task_name_to_eval_idxs.values())

    distributed_state = PartialState()
    with distributed_state.split_between_processes(task_names) as process_task_names:
        assert isinstance(process_task_names, list)

        # for now, let's not worry about voting
        for task_name in tqdm(process_task_names, desc='Task'):
            # data: get demonstration ids, assume no permute or augmentation
            task_idx = task_name_to_eval_idxs[task_name][0]
            data = dataset[task_idx]
            assert data is not None
            demon_input_ids = data['demon_input_ids'].unsqueeze(0).to(accelerator.device)
            demon_start_idxs = data["demon_start_idxs"]
            if dataset.debug_max_len:
                demon_len = dataset.max_seq_len * 6 // 8
                demon_input_ids = torch.randint(0, 30, (1, demon_len), dtype=torch.int64, device=accelerator.device)
                demon_start_idxs = [x * (demon_len // 8) for x in range(8)]
            # dataset.tokenizer.decode(demon_input_ids[0, demon_start_idxs[0]: demon_start_idxs[1]], False)

            # model: load cached for fresh model
            del model
            model = AutoModelForCausalLM.from_pretrained(cached_model_path).to(accelerator.device)
            model.eval()

            # logging
            ttt_num_data, gs_num_data = 0, 0
            ttt_num_params, gs_num_params = 0, 0
            ttt_time, gs_time = 0.0, 0.0

            # use ttt to refine model
            if ttt_iters > 0:
                with accelerator.no_sync(model):
                    start_time = time.time()
                    model, ttt_num_data, ttt_num_params = run_ttt(
                        task=dataset.eval_tasks[task_idx],
                        eval_dataset=dataset,
                        accelerator=accelerator,
                        model=model,
                        iters=ttt_iters,
                        batch_size=ttt_batch_size,
                        grad_accum_steps=ttt_grad_accum_steps,
                        lr=ttt_lr,
                        weight_decay=ttt_weight_decay,
                        optimizer=ttt_optimizer,
                        max_grad_norm=ttt_max_grad_norm,
                        lr_scheduler=ttt_lr_scheduler,
                        lora_rank=ttt_lora_rank,
                        lora_alpha=ttt_lora_alpha,
                        permute_n=ttt_permute_n,
                    )
                    ttt_time = time.time() - start_time
                    torch.cuda.empty_cache()
                    gc.collect()

            # initialize kv
            start_time = time.time()
            past_key_values = initialize_kv(
                model=model,
                demon_input_ids=demon_input_ids,
                demon_start_idxs=demon_start_idxs,
                accelerator=accelerator,
                gs_iters=gs_iters,
                gs_random_kv=gs_random_kv,
                gs_num_permute=gs_num_permute,
                gs_permute_batch_size=gs_permute_batch_size,
                gs_permute_back=gs_permute_back,
            )
            init_kv_time = time.time() - start_time
            torch.cuda.empty_cache()
            gc.collect()

            # use gs to refine kv
            if gs_iters > 0:
                with accelerator.no_sync(model):
                    assert past_key_values is not None
                    assert past_key_values[0][0].shape[0] == 1
                    if not no_flash_attn:
                        past_key_values = tuple(
                            (
                                layer_k.to(NBIT_TO_DTYPE[trainable_nbit]),
                                layer_v.to(NBIT_TO_DTYPE[trainable_nbit]),
                            )
                            for layer_k, layer_v in past_key_values
                        )

                    start_time = time.time()
                    model, past_key_values, gs_num_data, gs_num_params = run_gs(
                        task=dataset.eval_tasks[task_idx],
                        eval_dataset=dataset,
                        accelerator=accelerator,
                        model=model,
                        # inputs
                        past_key_values=past_key_values, # type: ignore
                        # config
                        iters=gs_iters,
                        lr=gs_lr,
                        beta1=gs_beta1,
                        beta2=gs_beta2,
                        weight_decay=gs_weight_decay,
                        batch_size=gs_batch_size,
                        grad_accum_steps=gs_grad_accum_steps,
                        optimizer=gs_optimizer,
                        max_grad_norm=gs_max_grad_norm,
                        no_key=gs_no_key,
                        no_value=gs_no_value,
                        loss_on_input=gs_loss_on_input,
                        lr_scheduler=gs_lr_scheduler,
                        lora=gs_lora,
                        lora_rank=gs_lora_rank,
                        lora_alpha=gs_lora_alpha,
                        lora_lr=gs_lora_lr,
                        lora_beta1=gs_lora_beta1,
                        lora_beta2=gs_lora_beta2,
                    )
                    gs_time = time.time() - start_time
                    torch.cuda.empty_cache()
                    gc.collect()

            # logging
            ttt_num_data_list.append(ttt_num_data)
            gs_num_data_list.append(gs_num_data)
            ttt_num_params_list.append(ttt_num_params)
            gs_num_params_list.append(gs_num_params)
            ttt_time_list.append(ttt_time)
            gs_time_list.append(gs_time)
            init_kv_time_list.append(init_kv_time)

            # STAGE 2: apply model and kv to all tests
            arbitrary_increase = 5
            out_token_length = data['out_token_length']
            gen_input_ids = data['gen_input_ids'].unsqueeze(0).to(accelerator.device)
            gen_attention_mask = data['gen_attention_mask'].unsqueeze(0).to(accelerator.device)

            with accelerator.autocast():
                # second step to generate
                # add past key values portion to input_ids and attention mask
                # the padding of input_ids is ignored
                gen_input_ids = torch.cat([
                    torch.zeros((1, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64),
                    gen_input_ids,
                ], dim=1)
                gen_attention_mask = torch.cat([
                    torch.ones((1, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64),
                    gen_attention_mask,
                ], dim=1)

                # print('generate', gen_attention_mask.shape)
                gen_tokens = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    past_key_values=past_key_values,
                    max_new_tokens=out_token_length + arbitrary_increase,
                    num_return_sequences=1,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False,
                    eos_token_id=[dataset.tokenizer.eos_token_id],
                )
                assert gen_tokens.shape[0] == 1
                gen_tokens = gen_tokens[0, gen_input_ids.shape[1]:]

            gen_tokens[out_token_length + arbitrary_increase:] = dataset.tokenizer.pad_token_id
            gen_texts = dataset.tokenizer.batch_decode(
                gen_tokens.unsqueeze(0),
                skip_special_tokens=True,
                no_separate_color_tokens=dataset.no_separate_color_tokens,
            )
            assert len(gen_texts) == 1
            gen_text = gen_texts[0]

            # print(gen_texts)
            # breakpoint()

            task_id = data["task_id"]
            inverter = data["inverter"]
            label_text = data["label_texts"]

            # Compare each gen_text with label_texts
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
    init_kv_time_list = gather_object(init_kv_time_list)
    # other logging
    # results
    ttt_num_data_list = gather_object(ttt_num_data_list)
    gs_num_data_list = gather_object(gs_num_data_list)
    ttt_num_params_list = gather_object(ttt_num_params_list)
    gs_num_params_list = gather_object(gs_num_params_list)
    ttt_time_list = gather_object(ttt_time_list)
    gs_time_list = gather_object(gs_time_list)

    assert len(task_id_and_text_list) == len(dataset), (len(task_id_and_text_list), len(dataset))
    assert len(exact_acc_list) == len(dataset), (len(exact_acc_list), len(dataset))
    assert len(valid_grid_list) == len(dataset), (len(valid_grid_list), len(dataset))
    assert len(correct_grid_dim_list) == len(dataset), (len(correct_grid_dim_list), len(dataset))
    assert len(token_acc_list) == len(dataset), (len(token_acc_list), len(dataset))
    assert len(relaxed_token_acc_list) == len(dataset), (len(relaxed_token_acc_list), len(dataset))
    assert len(ttt_num_data_list) == len(task_name_to_eval_idxs), (len(ttt_num_data_list), len(task_name_to_eval_idxs))
    assert len(gs_num_data_list) == len(task_name_to_eval_idxs), (len(gs_num_data_list), len(task_name_to_eval_idxs))
    assert len(ttt_num_params_list) == len(task_name_to_eval_idxs), (len(ttt_num_params_list), len(task_name_to_eval_idxs))
    assert len(gs_num_params_list) == len(task_name_to_eval_idxs), (len(gs_num_params_list), len(task_name_to_eval_idxs))
    assert len(ttt_time_list) == len(task_name_to_eval_idxs), (len(ttt_time_list), len(task_name_to_eval_idxs))
    assert len(gs_time_list) == len(task_name_to_eval_idxs), (len(gs_time_list), len(task_name_to_eval_idxs))
    assert len(init_kv_time_list) == len(task_name_to_eval_idxs), (len(init_kv_time_list), len(task_name_to_eval_idxs))

    # average metrics
    # note these are all computed without accounting for skipped eval grids
    exact_acc = sum(exact_acc_list) / len(dataset)
    valid_grid = sum(valid_grid_list) / len(dataset)
    correct_grid_dim = sum(correct_grid_dim_list) / len(dataset)
    token_acc = sum(token_acc_list) / len(dataset)
    relaxed_token_acc = sum(relaxed_token_acc_list) / len(dataset)

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

    # average others
    ttt_num_data = sum(ttt_num_data_list) / len(ttt_num_data_list)
    gs_num_data = sum(gs_num_data_list) / len(gs_num_data_list)
    ttt_num_params = sum(ttt_num_params_list) / len(ttt_num_params_list)
    gs_num_params = sum(gs_num_params_list) / len(gs_num_params_list)
    ttt_time = sum(ttt_time_list) / len(ttt_time_list)
    gs_time = sum(gs_time_list) / len(gs_time_list)
    init_kv_time = sum(init_kv_time_list) / len(init_kv_time_list)

    # remove cache model
    if accelerator.is_main_process:
        os.system(f"rm -rf {cached_model_path}")

    return exact_acc, valid_grid, correct_grid_dim, token_acc, relaxed_token_acc, task_id_to_texts, \
        votes, competition_sub_acc, competition_all_acc, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, init_kv_time


@torch.enable_grad()
def run_ttt(
    task: Task,
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    # config
    iters: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    max_grad_norm: float,
    lr_scheduler: str,
    lora_rank: int,
    lora_alpha: int,
    permute_n: int,
) -> Tuple[nn.Module, int, int]:

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config) # type: ignore
    # model.print_trainable_parameters()

    # get program parameters
    lora_params = []
    for n, p in model.named_parameters():
        assert p.requires_grad == ('lora' in n)
        if p.requires_grad:
            lora_params.append(p)
    num_params = sum(p.numel() for p in lora_params)

    # dataset and dataloader
    ttt_dataset = TTTDataset(
        task=task,
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        max_seq_len=eval_dataset.max_seq_len,
        permute_n=permute_n,
        seed=eval_dataset.seed,
        no_separate_color_tokens=eval_dataset.no_separate_color_tokens,
        no_bos=eval_dataset.no_bos,
    )
    if len(ttt_dataset) == 0:
        return model, 0, num_params

    batch_size = min(batch_size, len(ttt_dataset))
    ttt_collate_fn = partial(collate_fn_ttt, dataset=ttt_dataset)
    if eval_dataset.debug_max_len:
        ttt_collate_fn = partial(collate_fn_ttt_dummy, dataset=ttt_dataset)
    ttt_loader = DataLoader(
        ttt_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ttt_collate_fn,
        drop_last=True,
        num_workers=0,
    )

    # optimizer
    if optimizer == 'adamw':
        optim = torch.optim.AdamW(lora_params, weight_decay=weight_decay, lr=lr) # type: ignore
    else:
        optim = torch.optim.SGD(lora_params, lr=lr) # type: ignore
    optim.zero_grad()

    # lr scheduler
    if lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=iters)
    else:
        scheduler = get_constant_schedule(optim)

    # prepare stuff (no difference on singlegpu, havent tested on multigpu)
    # model, optim, ttt_loader = accelerator.prepare(model, optim, ttt_loader)

    # prepare some stuff
    model.train()

    # train!
    curr_iter = 0
    while curr_iter < iters:
        for batch in ttt_loader:
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            label_ids = batch["label_ids"].to(accelerator.device)
            device, dtype = input_ids.device, input_ids.dtype

            # necessary
            with accelerator.autocast():
                # build position ids
                position_ids = torch.zeros((batch_size, input_ids.shape[1]), device=device, dtype=torch.int64)
                mask_lens = attention_mask.sum(dim=1)
                for task_position_ids, mask_len in zip(position_ids, mask_lens):
                    assert mask_len > 0
                    new_positions = torch.tensor(range(mask_len), device=device, dtype=dtype)
                    if ttt_dataset.pad_side == "right":
                        task_position_ids[:mask_len] = new_positions
                    else:
                        task_position_ids[-mask_len:] = new_positions

                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids,
                    position_ids=position_ids,
                ).loss

                # print(loss.item())
                # breakpoint()

            accelerator.backward(loss)

            if (curr_iter + 1) % grad_accum_steps == 0 or curr_iter == iters - 1:
                accelerator.clip_grad_norm_(lora_params, max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad()

            curr_iter += 1
            if curr_iter >= iters:
                break

    model.eval()
    model = merge_lora(model)

    return model, len(ttt_dataset), num_params


@torch.enable_grad()
def run_gs(
    task: Task,
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    # inputs
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    # config
    iters: int,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    batch_size: int,
    grad_accum_steps: int,
    optimizer: str,
    max_grad_norm: float,
    no_key: bool,
    no_value: bool,
    loss_on_input: bool,
    lr_scheduler: str,
    lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_lr: float,
    lora_beta1: float,
    lora_beta2: float,
) -> Tuple[nn.Module, Tuple[Tuple[torch.Tensor, torch.Tensor]], int, int]:

    # optional lora
    if lora:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config) # type: ignore
        # model.print_trainable_parameters()

    # this copying is necessary because torch is dumb as hell
    assert past_key_values[0][0].shape[0] == 1
    past_key_values = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values
    ) # type: ignore

    # get program parameters
    program_params = []
    for layer_k, layer_v in past_key_values:
        if not no_key:
            program_params.append(layer_k)
        if not no_value:
            program_params.append(layer_v)
    num_params = sum(p.numel() for p in program_params)

    # get lora parameters
    lora_params = []
    if lora:
        lora_params = []
        for n, p in model.named_parameters():
            assert p.requires_grad == ('lora' in n)
            if p.requires_grad:
                lora_params.append(p)
        num_params += sum(p.numel() for p in lora_params)

    # dataset and dataloader
    gs_dataset = GSDataset(
        task=task,
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        no_separate_color_tokens=eval_dataset.no_separate_color_tokens,
        no_bos=eval_dataset.no_bos,
        max_seq_len=eval_dataset.max_seq_len,
        loss_on_input=loss_on_input,
    )
    if len(gs_dataset) == 0:
        if lora:
            model = merge_lora(model)
        return model, past_key_values, 0, num_params

    batch_size = min(batch_size, len(gs_dataset))
    gs_collate_fn = partial(collate_fn_gs, dataset=gs_dataset)
    if eval_dataset.debug_max_len:
        gs_collate_fn = partial(collate_fn_gs_dummy, dataset=gs_dataset)
    gs_loader = DataLoader(
        gs_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gs_collate_fn,
        drop_last=True,
        num_workers=0,
    )

    # set requires grad
    assert all(not p.requires_grad for p in program_params)
    for p in program_params:
        p.requires_grad = True

    # optimizer
    optimizer_grouped_params = [
        {"params": program_params, "lr": lr, 'betas': (beta1, beta2)},
        {"params": lora_params, "lr": lora_lr, 'betas': (lora_beta1, lora_beta2)},
    ]
    all_params = program_params + lora_params
    if optimizer == 'adamw':
        optim = torch.optim.AdamW(optimizer_grouped_params, weight_decay=weight_decay) # type: ignore
    else:
        optim = torch.optim.SGD(optimizer_grouped_params, lr=lr) # type: ignore
    optim.zero_grad()

    # lr scheduler
    if lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=iters)
    else:
        scheduler = get_constant_schedule(optim)

    # prepare stuff (no difference on singlegpu, havent tested on multigpu)
    # model, optim, gs_loader = accelerator.prepare(model, optim, gs_loader)

    # prepare some stuff
    model.train()

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if not lora else module.model.model.embed_tokens

    # expand to match predicted program with batch size
    past_key_values = tuple(
        (
            layer_k.expand(batch_size, *layer_k.shape[1:]),
            layer_v.expand(batch_size, *layer_v.shape[1:]),
        )
        for layer_k, layer_v in past_key_values
    ) # type: ignore
    past_kv_len = past_key_values[0][0].shape[2]

    # train!
    curr_iter = 0
    while curr_iter < iters:
        for batch in gs_loader:
            pair_input_ids = batch["input_ids"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_label_ids = batch["label_ids"].to(accelerator.device)
            device, dtype = pair_input_ids.device, pair_input_ids.dtype

            with accelerator.autocast():
                # build position ids
                position_ids = torch.zeros((batch_size, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
                mask_lens = pair_attention_mask.sum(dim=1)
                for task_position_ids, mask_len in zip(position_ids, mask_lens):
                    assert mask_len > 0
                    new_positions = torch.tensor(range(past_kv_len, past_kv_len + mask_len), device=device, dtype=dtype)
                    if gs_dataset.pad_side == "right":
                        task_position_ids[:mask_len] = new_positions
                    else:
                        task_position_ids[-mask_len:] = new_positions

                pair_inputs_embeds = embed_tokens(pair_input_ids)
                pair_attention_mask = torch.cat([
                    torch.ones((batch_size, past_kv_len), device=accelerator.device, dtype=torch.int64),
                    pair_attention_mask
                ], dim=1)

                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
                    "labels": pair_label_ids,
                    "use_cache": True,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }

                # get ce loss
                loss = model(**model_kwargs).loss

                # print(loss.item())
                # breakpoint()

            accelerator.backward(loss)

            if (curr_iter + 1) % grad_accum_steps == 0 or curr_iter == iters - 1:
                accelerator.clip_grad_norm_(all_params, max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad()

            curr_iter += 1
            if curr_iter >= iters:
                break

    model.eval()
    if lora:
        model = merge_lora(model)

    # shrink to bs1
    if batch_size > 1:
        assert torch.equal(past_key_values[0][0][0], past_key_values[0][0][1])
    past_key_values = tuple(
        (layer_k[:1].detach().clone(), layer_v[:1].detach().clone())
        for layer_k, layer_v in past_key_values
    ) # type: ignore

    return model, past_key_values, len(gs_dataset), num_params


def merge_lora(model: nn.Module) -> nn.Module:
    model.merge_and_unload() # required for the case of gs with lora
    model = model.model # no peftmodel class
    del model.peft_config # type: ignore
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs_eval")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    # Debug
    parser.add_argument("--debug_max_len", action='store_true')
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--no_tf32", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_epoch", type=int, required=True)

    # Evaluation & data
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--no_separate_color_tokens", action='store_true')
    parser.add_argument("--no_bos", action='store_true')

    # eval data
    parser.add_argument("--data_dir", type=str, default="./data/re-arc/arc_original/evaluation")
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--leave_ns_inc", action="store_true")
    parser.add_argument("--permute_n", type=int, default=0)
    parser.add_argument("--augment_n", type=int, default=0)
    parser.add_argument("--permute_iters", type=int, default=0)

    # ttt
    parser.add_argument("--ttt_iters", type=int, default=0)
    parser.add_argument("--ttt_lr", type=float, default=1e-4)
    parser.add_argument("--ttt_weight_decay", type=float, default=0.0)
    parser.add_argument("--ttt_batch_size", type=int, default=4)
    parser.add_argument("--ttt_grad_accum_steps", type=int, default=1)
    parser.add_argument("--ttt_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--ttt_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--ttt_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ttt_permute_n", type=int, default=40)
    parser.add_argument("--ttt_lora_rank", type=int, default=128)
    parser.add_argument("--ttt_lora_alpha", type=int, default=16)

    # gradient search
    parser.add_argument("--gs_iters", type=int, default=0)
    parser.add_argument("--gs_lr", type=float, default=1e-3)
    parser.add_argument("--gs_beta1", type=float, default=0.9)
    parser.add_argument("--gs_beta2", type=float, default=0.999)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0)
    parser.add_argument("--gs_batch_size", type=int, default=100000)
    parser.add_argument("--gs_grad_accum_steps", type=int, default=1)
    parser.add_argument("--gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--gs_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_no_key", action='store_true')
    parser.add_argument("--gs_no_value", action='store_true')
    parser.add_argument("--gs_loss_on_input", action='store_true')

    # gradient search model initialization
    parser.add_argument("--gs_random_kv", action='store_true')
    parser.add_argument("--gs_num_permute", type=int, default=1) # 1024
    parser.add_argument("--gs_permute_batch_size", type=int, default=16)
    parser.add_argument("--gs_permute_back", action='store_true')

    # gradient search with lora (NOT COMPATIBLE WITH TTT)
    parser.add_argument("--gs_lora", action='store_true')
    parser.add_argument("--gs_lora_rank", type=int, default=128)
    parser.add_argument("--gs_lora_alpha", type=int, default=16)
    parser.add_argument("--gs_lora_lr", type=float, default=1e-4)
    parser.add_argument("--gs_lora_beta1", type=float, default=0.9)
    parser.add_argument("--gs_lora_beta2", type=float, default=0.999)

    args = parser.parse_args()

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    args.gs_iters *= args.gs_grad_accum_steps
    args.ttt_iters *= args.ttt_grad_accum_steps

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_up_main_process_logger(accelerator, logger)
    set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_tf32:
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

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        **from_pretrained_kwargs,
    )
    if args.untrainable_nbit in [4, 8]:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False,
        )

    logger.info("Base models loaded.")

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

    # load weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    model_weight_path = os.path.join(weight_dir, f"lora_epoch_{args.weight_epoch}")

    model = PeftModel.from_pretrained(base_model, model_weight_path)
    logger.info("loaded model weights")

    # convert lora weights to trainable nbit
    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # Prepare with accelerator
    model = accelerator.prepare(model)

    # we dont train model so
    for p in model.parameters():
        p.requires_grad = False

    # merge lora because arc uses it
    model = merge_lora(model)

    # Build evaluation dataset
    dataset = EvalDataset(
        args.data_dir,
        select_tasks_path=args.select_tasks_path,
        leave_ns=args.leave_ns,
        leave_ns_inc=args.leave_ns_inc,
        permute_n=args.permute_n,
        augment_n=args.augment_n,
        permute_iters=args.permute_iters,
        seed=args.seed,
        tokenizer=tokenizer,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        pad_side=args.pad_side,
        debug_max_len=args.debug_max_len,
        no_separate_color_tokens=args.no_separate_color_tokens,
        max_seq_len=args.max_seq_len,
        no_bos=args.no_bos,
    )

    # evaluate
    exact_acc, valid_grid, correct_grid_dim, token_acc, relaxed_token_acc, texts, votes, competition_sub_acc, competition_all_acc, \
         ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, init_kv_time = test_time_evaluate(
        model=model,
        dataset=dataset,
        accelerator=accelerator,
        trainable_nbit=args.trainable_nbit,
        no_flash_attn=not args.flash_attn,
        gs_iters=args.gs_iters,
        gs_lr=args.gs_lr,
        gs_beta1=args.gs_beta1,
        gs_beta2=args.gs_beta2,
        gs_weight_decay=args.gs_weight_decay,
        gs_batch_size=args.gs_batch_size,
        gs_grad_accum_steps=args.gs_grad_accum_steps,
        gs_optimizer=args.gs_optimizer,
        gs_max_grad_norm=args.gs_max_grad_norm,
        gs_no_key=args.gs_no_key,
        gs_no_value=args.gs_no_value,
        gs_loss_on_input=args.gs_loss_on_input,
        gs_lr_scheduler=args.gs_lr_scheduler,
        gs_lora=args.gs_lora,
        gs_lora_rank=args.gs_lora_rank,
        gs_lora_alpha=args.gs_lora_alpha,
        gs_lora_lr=args.gs_lora_lr,
        gs_lora_beta1=args.gs_lora_beta1,
        gs_lora_beta2=args.gs_lora_beta2,
        gs_random_kv=args.gs_random_kv,
        gs_num_permute=args.gs_num_permute,
        gs_permute_batch_size=args.gs_permute_batch_size,
        gs_permute_back=args.gs_permute_back,
        ttt_iters=args.ttt_iters,
        ttt_lr=args.ttt_lr,
        ttt_weight_decay=args.ttt_weight_decay,
        ttt_batch_size=args.ttt_batch_size,
        ttt_grad_accum_steps=args.ttt_grad_accum_steps,
        ttt_optimizer=args.ttt_optimizer,
        ttt_lr_scheduler=args.ttt_lr_scheduler,
        ttt_max_grad_norm=args.ttt_max_grad_norm,
        ttt_lora_rank=args.ttt_lora_rank,
        ttt_lora_alpha=args.ttt_lora_alpha,
        ttt_permute_n=args.ttt_permute_n,
        output_dir=args.output_dir,
    )

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {
            "eval/exact_acc": exact_acc,
            "eval/valid_grid": valid_grid,
            "eval/correct_grid_dim": correct_grid_dim,
            "eval/token_acc": token_acc,
            "eval/relaxed_token_acc": relaxed_token_acc,
            "eval/competition_sub_acc": competition_sub_acc,
            "eval/competition_all_acc": competition_all_acc,
            # others
            "eval/ttt_num_data": ttt_num_data,
            "eval/gs_num_data": gs_num_data,
            "eval/ttt_num_params": ttt_num_params,
            "eval/gs_num_params": gs_num_params,
            "eval/ttt_time": ttt_time,
            "eval/gs_time": gs_time,
            "eval/init_kv_time": init_kv_time,
        }
        logger.info(f'Evaluation results:\n{pprint.pformat(metric_dict, indent=4)}')

        # save outputs
        save_pred_gt_path = os.path.join(args.output_dir, f"eval_pred_gt.json")
        with open(save_pred_gt_path, 'w') as f:
            json.dump(texts, f)
        logger.info(f"Saved eval train pred gt to {save_pred_gt_path}")

        # save votes
        save_vote_path = os.path.join(args.output_dir, f"eval_vote.json")
        with open(save_vote_path, 'w') as f:
            json.dump(votes, f)
        logger.info(f"Saved vote to {save_vote_path}")

    logger.info("All done evaluating.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
