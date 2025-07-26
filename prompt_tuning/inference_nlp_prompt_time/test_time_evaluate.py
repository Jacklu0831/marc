import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import copy
import gc
import time
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
from typing import Union, Callable, List, Tuple, Dict, Optional
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
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    GPT2LMHeadModel,
)
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from peft import LoraConfig, TaskType, get_peft_model # type: ignore

from data_utils import (
    EvalDataset,
    GSDataset,
    TTTDataset,
    collate_fn_eval,
    collate_fn_eval_dummy,
    collate_fn_gs,
    collate_fn_gs_dummy,
    collate_fn_ttt,
    collate_fn_ttt_dummy,
)

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import pprint
import json
from functools import partial
import argparse
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from data_utils import EvalDataset, collate_fn_eval
from train import (
    set_up_main_process_logger,
    get_individual_loss,
    compute_macrof1_or_accuracy,
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
    "gpt2": "openai-community/gpt2-large",
}
NBIT_TO_DTYPE = {
    16: torch.bfloat16,
    32: torch.float32,
}


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
def initialize_prompt(
    model: nn.Module,
    demon_input_ids: torch.Tensor,
    demon_start_idxs: List[int],
    accelerator: Accelerator,
    # init
    random_prompt: str,
    random_prompt_ntokens: int,
    separate_kv: bool,
    num_permute: int,
    permute_batch_size: int,
    permute_back: bool,
    permute_concat: bool,
    # dt
    dt_iters: int,
    dt_lr: int,
) -> torch.Tensor:

    if separate_kv:
        raise NotImplementedError()

    elif dt_iters > 0:
        raise NotImplementedError()

    elif random_prompt != 'none':
        # random kv initialization
        random_prompt_ntokens = random_prompt_ntokens if random_prompt_ntokens != -1 else demon_input_ids.shape[1]
        if random_prompt == 'uniform':
            prompt = torch.rand((1, random_prompt_ntokens, model.config.hidden_size), device=accelerator.device, dtype=torch.float32) - 0.5
        else:
            # initialize from first 1000 tokens distribution
            dummy_input_ids = torch.arange(0, random_prompt_ntokens)[None, ...].to(accelerator.device)
            with accelerator.autocast():
                prompt = model.transformer.wte(dummy_input_ids) # (1, ntoken, hiddensize)
                assert tuple(prompt.shape) == (1, dummy_input_ids.shape[1], model.config.hidden_size)

    elif num_permute == 1:
        with accelerator.autocast():
            prompt = model.transformer.wte(demon_input_ids) # (1, ntoken, hiddensize)
            assert tuple(prompt.shape) == (1, demon_input_ids.shape[1], model.config.hidden_size)

    elif not permute_concat:
        raise NotImplementedError()

    else:
        raise NotImplementedError()

    assert demon_input_ids.shape[0] == 1
    return prompt


@torch.no_grad()
def test_time_evaluate(
    model: Union[nn.Module, DistributedDataParallel],
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    trainable_nbit: int,
    log_every: int,
    output_dir: str,
    # gs
    gs_epochs: int,
    gs_batch_size: int,
    gs_lr: float,
    gs_beta1: float,
    gs_beta2: float,
    gs_weight_decay: float,
    gs_optimizer: str,
    gs_lr_scheduler: str,
    gs_max_grad_norm: float,
    gs_num_layer: int,
    gs_loss_on_input: bool,
    gs_dropout: str,
    gs_token_dropout: float,
    gs_detach: bool,
    gs_ntokens: int,
    gs_log_attention: bool,
    gs_final_tokens: int,
    # gs lora
    gs_lora: bool,
    gs_lora_rank: int,
    gs_lora_alpha: int,
    gs_lora_lr: float,
    gs_lora_beta1: float,
    gs_lora_beta2: float,
    gs_lora_dropout: float,
    gs_lora_rslora: bool,
    # gs init
    random_prompt: str,
    random_prompt_ntokens: int,
    separate_kv: bool,
    num_permute: int,
    permute_batch_size: int,
    permute_back: bool,
    permute_concat: bool,
    # ttt
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
    ttt_lora_dropout: float,
    ttt_lora_rslora: bool,
    ttt_loss_type: str,
    ttt_permute_n: int,
    # dt
    dt_iters: int,
    dt_lr: int,
) -> Tuple[float, float, float, float, float, float, float, float, List]:

    model.eval()

    # get modules in case of DDP
    model = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = model.transformer.wte
    model.generation_config.pad_token_id = dataset.tokenizer.pad_token_id

    # We perform test-time adaptation in 2 stages.
    # First stage produces KV cache, num trainable params, runtime, num data. Second stage simply performs model loss/generation

    # STAGE 1: demonstration pair processing
    ttt_num_data_list, gs_num_data_list = [], []
    ttt_num_params_list, gs_num_params_list = [], []
    ttt_time_list, gs_time_list = [], []
    init_prompt_time_list = []

    # outputs
    output_list = []

    assert set(len(v) for v in dataset.task_to_demonstrations.values()) == {dataset.num_demonstrations}
    assert len(dataset.tasks) >= accelerator.num_processes # avoid padding issue

    # we need to cache model in case of ttt or gs with trainable lora
    cached_model = copy.deepcopy(model).cpu()

    distributed_state = PartialState()
    with distributed_state.split_between_processes(dataset.tasks) as process_tasks:
        assert isinstance(process_tasks, list)

        for task in tqdm(process_tasks, desc='Task'):
            # data: get demonstration ids and indices, hacky
            if dataset.debug_max_len:
                demon_len = dataset.max_seq_len - dataset.max_pair_len
                demon_input_ids = torch.randint(0, 30, (1, demon_len), dtype=torch.int64, device=accelerator.device)
                demon_start_idxs = [x * (demon_len // 16) for x in range(16)]
            else:
                demon_input_ids = None
                demon_start_idxs = None
                for data in dataset.data:
                    if data['task'] == task:
                        demon_input_ids = data["demon_input_ids"].unsqueeze(0).to(accelerator.device)
                        demon_start_idxs = data["demon_start_idxs"]
                        break
            assert demon_input_ids is not None and demon_start_idxs is not None
            assert demon_input_ids.shape[1] <= dataset.max_seq_len

            # model: load cached for fresh model
            model = copy.deepcopy(cached_model).to(accelerator.device)
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
                        demonstration_pairs=dataset.task_to_demonstrations[task],
                        eval_dataset=dataset,
                        accelerator=accelerator,
                        model=model,
                        trainable_nbit=trainable_nbit,
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
                        lora_dropout=ttt_lora_dropout,
                        lora_rslora=ttt_lora_rslora,
                        loss_type=ttt_loss_type,
                        permute_n=ttt_permute_n,
                    )
                    ttt_time = time.time() - start_time
                    torch.cuda.empty_cache()
                    gc.collect()

            # initialize kv
            start_time = time.time()
            prompt = initialize_prompt(
                model=model,
                demon_input_ids=demon_input_ids,
                demon_start_idxs=demon_start_idxs,
                accelerator=accelerator,
                # init
                random_prompt=random_prompt,
                random_prompt_ntokens=random_prompt_ntokens,
                separate_kv=separate_kv,
                num_permute=num_permute,
                permute_batch_size=permute_batch_size,
                permute_back=permute_back,
                permute_concat=permute_concat,
                # dt
                dt_iters=dt_iters,
                dt_lr=dt_lr,
            )
            init_prompt_time = time.time() - start_time
            torch.cuda.empty_cache()
            gc.collect()

            # use gs to refine kv
            if gs_epochs > 0:
                with accelerator.no_sync(model):

                    start_time = time.time()
                    saved_gradckpt = model.transformer.gradient_checkpointing
                    model.transformer.gradient_checkpointing = False
                    model, prompt, gs_num_data, gs_num_params, attn_logger = run_gs(
                        demonstration_pairs=dataset.task_to_demonstrations[task],
                        eval_dataset=dataset,
                        accelerator=accelerator,
                        model=model,
                        # inputs
                        demon_start_idxs=demon_start_idxs,
                        prompt=prompt,
                        demon_input_ids_len=demon_input_ids.shape[1] if (random_prompt == 'none' or random_prompt_ntokens == -1) else random_prompt_ntokens,
                        # config
                        epochs=gs_epochs,
                        lr=gs_lr,
                        beta1=gs_beta1,
                        beta2=gs_beta2,
                        weight_decay=gs_weight_decay,
                        batch_size=gs_batch_size,
                        optimizer=gs_optimizer,
                        max_grad_norm=gs_max_grad_norm,
                        num_layer=gs_num_layer,
                        loss_on_input=gs_loss_on_input,
                        dropout=gs_dropout,
                        token_dropout=gs_token_dropout,
                        detach=gs_detach,
                        log_attention=gs_log_attention,
                        final_tokens=gs_final_tokens,
                        lr_scheduler=gs_lr_scheduler,
                        lora=gs_lora,
                        lora_rank=gs_lora_rank,
                        lora_alpha=gs_lora_alpha,
                        lora_lr=gs_lora_lr,
                        lora_beta1=gs_lora_beta1,
                        lora_beta2=gs_lora_beta2,
                        lora_dropout=gs_lora_dropout,
                        lora_rslora=gs_lora_rslora,
                        ntokens=gs_ntokens,
                    )
                    model.transformer.gradient_checkpointing = saved_gradckpt
                    gs_time = time.time() - start_time
                    torch.cuda.empty_cache()
                    gc.collect()

                    if attn_logger is not None:
                        os.makedirs(os.path.join(output_dir, 'attn'), exist_ok=True)
                        save_path = os.path.join(output_dir, 'attn', task) + '.jpg'
                        iters = range(len(attn_logger.instruct_attn))

                        plt.figure()
                        plt.plot(iters, attn_logger.instruct_attn, label='instruct')
                        plt.plot(iters, attn_logger.self_attn, label='self')
                        plt.plot(iters, attn_logger.other_demon_attn, label='other demon')
                        plt.plot(iters, attn_logger.self_demon_attn, label='self demon')
                        plt.legend()
                        plt.savefig(save_path)
                        plt.close()

                        logger.info(f'saved attention scores to {save_path}')

            # logging
            ttt_num_data_list.append(ttt_num_data)
            gs_num_data_list.append(gs_num_data)
            ttt_num_params_list.append(ttt_num_params)
            gs_num_params_list.append(gs_num_params)
            ttt_time_list.append(ttt_time)
            gs_time_list.append(gs_time)
            init_prompt_time_list.append(init_prompt_time)

            # STAGE 2: apply model and kv to all tests
            process_data_idxs = [i for i, d in enumerate(dataset.data) if d['task'] == task]
            assert isinstance(process_data_idxs, list)
            n_batches = math.ceil(len(process_data_idxs) / batch_size)
            data_idxs = [idxs for idxs in chunks(process_data_idxs, batch_size)]
            assert len(data_idxs) == n_batches

            progress_bar = tqdm(
                range(len(data_idxs)),
                desc=f"{task} tests",
                disable=not accelerator.is_local_main_process,
            )

            for eval_step, batch_idxs in enumerate(data_idxs):
                batch_data = [dataset[i] for i in batch_idxs]
                bs = len(batch_data)
                batch = collate_fn(batch_data)

                # get tensors
                task = batch['task']
                test_idx = batch['test_idx']
                option = batch['option']
                correct_option = batch['correct_option']
                assert len(set(task)) == 1 # same task!

                # for gs
                gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
                gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
                gen_label_ids = batch["gen_label_ids"].to(accelerator.device)

                with accelerator.autocast():
                    # expand past key values
                    assert prompt.shape[0] == 1
                    batch_prompt = prompt.detach().clone().expand(bs, *prompt.shape[1:])
                    batch_prompt_attention_mask = torch.ones(
                        (bs, prompt.shape[1]),
                        device=accelerator.device,
                        dtype=torch.int64
                    )

                    gen_inputs_embeds = embed_tokens(gen_input_ids)
                    gen_attention_mask = torch.cat([batch_prompt_attention_mask, gen_attention_mask], dim=1)

                    if random_prompt == 'none' or random_prompt_ntokens == -1:
                        position_start = demon_input_ids.shape[1]
                    else:
                        position_start = prompt.shape[1]

                    # build position ids (does NOT depend on dropout)
                    attention_mask_after_kv = gen_attention_mask[:, batch_prompt.shape[1]:]
                    position_ids = []
                    for mask_after_kv in attention_mask_after_kv:
                        sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=accelerator.device, dtype=torch.int64)
                        n_new_positions = mask_after_kv.sum()
                        new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=accelerator.device, dtype=torch.int64)
                        if dataset.pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)

                    assert position_ids.max() < dataset.max_seq_len

                    # full versions
                    gen_inputs_embeds = torch.cat([batch_prompt, gen_inputs_embeds], dim=1)
                    gen_label_ids = torch.cat([
                        torch.full((batch_prompt.shape[0], batch_prompt.shape[1]), -100, device=accelerator.device, dtype=torch.int64),
                        gen_label_ids,
                    ], dim=1)
                    position_ids = torch.cat([
                        torch.arange(batch_prompt.shape[1], device=accelerator.device)[None, ...].expand(batch_prompt.shape[0], -1),
                        position_ids,
                    ], dim=1)

                    model_out = model(
                        inputs_embeds=gen_inputs_embeds,
                        attention_mask=gen_attention_mask,
                        position_ids=position_ids,
                    )
                    losses = get_individual_loss(lm_logits=model_out.logits.half(), label_ids=gen_label_ids)

                # if gen_attention_mask.sum() < gen_attention_mask.numel():
                #     print(losses.tolist())
                #     breakpoint()

                assert isinstance(losses, torch.Tensor)
                assert losses.shape[0] == len(task) == len(test_idx) == len(option) == len(correct_option) == bs
                for x0, x1, x2, x3, x4 in zip(losses, task, test_idx, option, correct_option):
                    output_list.append((x0.item(), x1, x2, x3, x4))
                if (eval_step + 1) % log_every == 0:
                    progress_bar.update(log_every)

                torch.cuda.empty_cache()
                gc.collect()

    distributed_state.wait_for_everyone()
    # results
    ttt_num_data_list = gather_object(ttt_num_data_list)
    gs_num_data_list = gather_object(gs_num_data_list)
    ttt_num_params_list = gather_object(ttt_num_params_list)
    gs_num_params_list = gather_object(gs_num_params_list)
    ttt_time_list = gather_object(ttt_time_list)
    gs_time_list = gather_object(gs_time_list)
    init_prompt_time_list = gather_object(init_prompt_time_list)
    output_list = gather_object(output_list)
    assert len(ttt_num_data_list) == len(dataset.tasks), (len(ttt_num_data_list), len(dataset.tasks))
    assert len(gs_num_data_list) == len(dataset.tasks), (len(gs_num_data_list), len(dataset.tasks))
    assert len(ttt_num_params_list) == len(dataset.tasks), (len(ttt_num_params_list), len(dataset.tasks))
    assert len(gs_num_params_list) == len(dataset.tasks), (len(gs_num_params_list), len(dataset.tasks))
    assert len(ttt_time_list) == len(dataset.tasks), (len(ttt_time_list), len(dataset.tasks))
    assert len(gs_time_list) == len(dataset.tasks), (len(gs_time_list), len(dataset.tasks))
    assert len(init_prompt_time_list) == len(dataset.tasks), (len(init_prompt_time_list), len(dataset.tasks))
    assert len(output_list) == len(dataset), (len(output_list), len(dataset)) # dataset length, not num task

    # determine which tasks are classification (for macro-f1)
    task_to_is_clf = {}
    for task in dataset.tasks:
        meta_data_path = os.path.join('data/MetaICL/config/tasks', f'{task}.json')
        task_meta_data = json.load(open(meta_data_path, 'r'))
        task_to_is_clf[task] = task_meta_data['task_type'] == "classification"

    # metrics
    task_to_score = {}
    for task in dataset.tasks:
        task_outs = [x for x in output_list if x[1] == task]
        if len(task_outs) == 0:
            logger.info(f'[WARNING] {task} is not evaluated (likely due max_seq_len)')
            continue

        preds, gts = [], []
        test_idxs = set(x[2] for x in task_outs)
        for test_i in test_idxs:
            task_test_outs = [x for x in task_outs if x[2] == test_i]
            correct_option = task_test_outs[0][4]
            assert all(x[4] == correct_option for x in task_test_outs)
            # choose option with lowest loss
            lowest_loss = float('inf')
            chosen_option = None
            for x in task_test_outs:
                if x[0] < lowest_loss:
                    lowest_loss = x[0]
                    chosen_option = x[3]
            assert chosen_option is not None
            # record
            preds.append(chosen_option)
            gts.append(correct_option)

        task_to_score[task] = compute_macrof1_or_accuracy(preds, gts, task_to_is_clf[task])

    # average scores
    sorted_tasks = sorted(task_to_score.keys())
    for task in sorted_tasks:
        logger.info(f"{task} clf {task_to_is_clf[task]} has a score {task_to_score[task]}")
    score = sum(v for v in task_to_score.values()) / len(task_to_score)

    # average others
    ttt_num_data = sum(ttt_num_data_list) / len(ttt_num_data_list)
    gs_num_data = sum(gs_num_data_list) / len(gs_num_data_list)
    ttt_num_params = sum(ttt_num_params_list) / len(ttt_num_params_list)
    gs_num_params = sum(gs_num_params_list) / len(gs_num_params_list)
    ttt_time = sum(ttt_time_list) / len(ttt_time_list)
    gs_time = sum(gs_time_list) / len(gs_time_list)
    init_prompt_time = sum(init_prompt_time_list) / len(init_prompt_time_list)

    return score, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, init_prompt_time, output_list


@torch.enable_grad()
def run_ttt(
    demonstration_pairs: List[Dict],
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
    lora_dropout: float,
    lora_rslora: bool,
    loss_type: str,
    permute_n: int,
    trainable_nbit: int,
) -> Tuple[nn.Module, int, int]:

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=lora_rslora,
        target_modules=['c_attn', 'c_proj', 'c_fc'],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config) # type: ignore
    # model.print_trainable_parameters()

    # convert model weights to float16 (as in ttt paper)
    for name, param in model.named_parameters():
        assert param.requires_grad == ('lora' in name)
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[trainable_nbit])
        else:
            assert param.data.dtype == NBIT_TO_DTYPE[trainable_nbit]

    # get program parameters
    lora_params = []
    for n, p in model.named_parameters():
        assert p.requires_grad == ('lora' in n)
        if p.requires_grad:
            lora_params.append(p)
    num_params = sum(p.numel() for p in lora_params)

    # dataset and dataloader
    ttt_dataset = TTTDataset(
        demonstration_pairs=demonstration_pairs,
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        max_seq_len=eval_dataset.max_seq_len,
        max_pair_len=eval_dataset.max_pair_len,
        allow_truncate=eval_dataset.allow_truncate,
        delimiter=eval_dataset.delimiter,
        permute_n=permute_n,
        seed=eval_dataset.seed,
        loss_type=loss_type,
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
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=iters // grad_accum_steps)
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


class AttentionLogger:
    def __init__(self, demon_input_ids_len: int, demon_start_idxs: List[int]):
        self.demon_input_ids_len = demon_input_ids_len
        self.demon_start_idxs = demon_start_idxs

        self.instruct_attn = []
        self.self_attn = []
        self.self_demon_attn = []
        self.other_demon_attn = []

    def update(
        self,
        attentions: Tuple[torch.Tensor],
        pair_attention_mask: torch.Tensor,
        pair_example_idx: List[int],
    ) -> None:

        # attention formatted in tuple of layers, each (bs, nhead, pair_len, pair_len + past_kv_len)
        # assume batchsize1, averaged across heads and layers -> (pair_len, pair_len + past_kv_len)
        assert len(pair_example_idx) == 1
        attns = torch.stack([attn.detach().squeeze(0).mean(dim=0) for attn in attentions]).mean(dim=0)
        assert attns.shape[0] == pair_attention_mask.shape[1] - self.demon_input_ids_len and attns.shape[1] == pair_attention_mask.shape[1]
        attns = attns.mean(dim=0) # (pair_len + past_kv_len,)

        # compute average attention of query to each demonstration pair
        instruct_attn = attns[:self.demon_start_idxs[0]].mean().item()
        self_attn = attns[self.demon_input_ids_len:].mean().item()
        self_demon_attn = None
        other_demon_attns = []
        for idx in range(len(self.demon_start_idxs)):
            start = self.demon_start_idxs[idx]
            end = self.demon_start_idxs[idx + 1] if idx < len(self.demon_start_idxs) - 1 else self.demon_input_ids_len
            if idx == pair_example_idx[0]:
                self_demon_attn = attns[start: end].mean().item()
            else:
                other_demon_attns.append(attns[start: end].mean().item())
        other_demon_attn = sum(other_demon_attns) / len(other_demon_attns)

        # update
        self.instruct_attn.append(instruct_attn)
        self.self_attn.append(self_attn)
        self.self_demon_attn.append(self_demon_attn)
        self.other_demon_attn.append(other_demon_attn)


@torch.enable_grad()
def run_gs(
    demonstration_pairs: List[Dict],
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    # inputs
    demon_start_idxs: List[int],
    prompt: torch.Tensor,
    demon_input_ids_len: int,
    # config
    epochs: int,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    batch_size: int,
    optimizer: str,
    max_grad_norm: float,
    num_layer: int,
    loss_on_input: bool,
    dropout: str,
    token_dropout: float,
    detach: bool,
    log_attention: bool,
    final_tokens: int,
    lr_scheduler: str,
    lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_lr: float,
    lora_beta1: float,
    lora_beta2: float,
    lora_dropout: float,
    lora_rslora: bool,
    ntokens: int,
) -> Tuple[nn.Module, Tuple[torch.Tensor, ...], int, int, Optional[AttentionLogger]]:

    # optional lora
    if lora:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=lora_rslora,
            target_modules=['c_attn', 'c_proj', 'c_fc'],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config) # type: ignore
        # model.print_trainable_parameters()

    # this copying is necessary because torch is dumb as hell
    assert demon_start_idxs[0] == 0
    prompt = prompt.detach().clone()

    # get program parameters
    program_params = []
    if ntokens != -1:
        raise NotImplementedError()
    else:
        # full tuning of initialized KV
        prefix_prompt = None # no prefix tuning
        program_params.append(prompt)
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

    # dataset
    gs_dataset = GSDataset(
        demonstration_pairs={i: p for i, p in enumerate(demonstration_pairs)},
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        past_kv_len=demon_input_ids_len,
        max_seq_len=eval_dataset.max_seq_len,
        max_pair_len=eval_dataset.max_pair_len,
        allow_truncate=eval_dataset.allow_truncate,
        delimiter=eval_dataset.delimiter,
        loss_on_input=loss_on_input,
    )
    assert len(gs_dataset) > 0

    # dataloader
    batch_size = min(batch_size, len(gs_dataset))
    gs_collate_fn = partial(collate_fn_gs, dataset=gs_dataset)
    if eval_dataset.debug_max_len:
        gs_collate_fn = partial(collate_fn_gs_dummy, dataset=gs_dataset)
    gs_loader = DataLoader(
        gs_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gs_collate_fn,
        drop_last=False, # no drop last to ensure we get loss for every sample at least once
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
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=epochs)
    else:
        scheduler = get_constant_schedule(optim)

    # prepare stuff (no difference on singlegpu, havent tested on multigpu)
    # model, optim, gs_loader = accelerator.prepare(model, optim, gs_loader)

    # prepare some stuff
    model.train()

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.transformer.wte if not lora else module.model.transformer.wte

    attn_logger = None
    if log_attention:
        attn_logger = AttentionLogger(
            demon_input_ids_len=demon_input_ids_len,
            demon_start_idxs=demon_start_idxs,
        )

    # debug: save for assertions
    kv_len_before_dropout = prompt.shape[1]

    # train!
    curr_epoch = 0
    while curr_epoch < epochs:
        for batch in gs_loader:
            pair_input_ids = batch["input_ids"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_label_ids = batch["label_ids"].to(accelerator.device)
            pair_example_idx = batch["example_idx"]
            device, dtype = pair_input_ids.device, pair_input_ids.dtype
            bs = pair_input_ids.shape[0]

            # construct full attention mask for past key values first
            batch_prompt_attention_mask = torch.ones((batch_size, prompt.shape[1]), device=accelerator.device, dtype=torch.int64)

            if detach:
                raise NotImplementedError()
            else:
                # use the same past key values across batch, but adjust attention mask for dropping
                batch_prompt = prompt.expand(bs, -1, -1)

                # only drop training kv
                if dropout == 'train':
                    assert prompt.shape[1] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        start = demon_start_idxs[idx]
                        end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
                        batch_prompt_attention_mask[batch_i, start:end] = 0

                # drop training kv and drop suffix
                elif dropout == 'suffix':
                    assert prompt.shape[1] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        start = demon_start_idxs[idx]
                        batch_prompt_attention_mask[batch_i, start:] = 0

                # drop training kv and only keep power set
                elif dropout in ['power', 'power_with_train']:
                    assert prompt.shape[1] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        # figure out a non-empty set of kv to keep
                        choices = set(range(len(demon_start_idxs)))
                        if dropout == 'power':
                            choices -= {idx}
                        power_set = set(itertools.chain.from_iterable(itertools.combinations(choices, r) for r in range(len(choices) + 1))) - {()}
                        to_keep = random.choice(list(power_set))
                        assert len(to_keep) > 0
                        # remove
                        to_remove = [idx for idx in range(len(demon_start_idxs)) if idx not in to_keep]
                        for idx in to_remove:
                            start = demon_start_idxs[idx]
                            end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
                            batch_prompt_attention_mask[batch_i, start:end] = 0

            assert tuple(batch_prompt_attention_mask.shape) == (bs, kv_len_before_dropout)

            # tune the final few tokens only
            if final_tokens > -1:
                raise NotImplementedError()

            # tune the prefix only
            if prefix_prompt is not None:
                raise NotImplementedError()

            # token dropout
            if token_dropout != 0.0:
                drop_mask = (torch.rand_like(batch_prompt_attention_mask, dtype=torch.float) > token_dropout).float()
                batch_prompt_attention_mask = (batch_prompt_attention_mask * drop_mask).long()

            with accelerator.autocast():
                # build position ids
                position_ids = torch.zeros((batch_size, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
                new_lens = pair_attention_mask.sum(dim=1)
                for task_position_ids, new_len in zip(position_ids, new_lens):
                    new_positions = torch.tensor(range(demon_input_ids_len, demon_input_ids_len + new_len), device=device, dtype=dtype)
                    if gs_dataset.pad_side == "right":
                        task_position_ids[:new_len] = new_positions
                    else:
                        task_position_ids[-new_len:] = new_positions

                pair_inputs_embeds = embed_tokens(pair_input_ids)
                pair_attention_mask = torch.cat([batch_prompt_attention_mask, pair_attention_mask], dim=1)

                # full versions
                pair_inputs_embeds = torch.cat([batch_prompt, pair_inputs_embeds], dim=1)
                pair_label_ids = torch.cat([
                    torch.full((batch_prompt.shape[0], batch_prompt.shape[1]), -100, device=accelerator.device, dtype=torch.int64),
                    pair_label_ids,
                ], dim=1)
                position_ids = torch.cat([
                    torch.arange(demon_input_ids_len, device=accelerator.device)[None, ...].expand(batch_prompt.shape[0], -1),
                    position_ids,
                ], dim=1)

                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
                    "labels": pair_label_ids,
                    "position_ids": position_ids,
                    "output_attentions": log_attention,
                }

                # get ce loss
                model_out = model(**model_kwargs)
                loss = model_out.loss * bs / batch_size # not doing droplast, so scale by relative batchsize

                if attn_logger is not None:
                    attn_logger.update(
                        attentions=model_out.attentions,
                        pair_attention_mask=pair_attention_mask,
                        pair_example_idx=pair_example_idx,
                    )

                # if pair_attention_mask.sum() < pair_attention_mask.numel():
                #     print(loss.item())
                #     breakpoint()

            # print(loss.item())
            accelerator.backward(loss)

            # only at the end of epoch do we backprop
            accelerator.clip_grad_norm_(all_params, max_grad_norm)

            optim.step()
            scheduler.step()
            optim.zero_grad()

            curr_epoch += 1
            # print(curr_epoch, '/', epochs)
            if curr_epoch >= epochs:
                break

    model.eval()
    if lora:
        model = merge_lora(model)

    prompt = prompt.detach().clone()

    # add back instruction
    assert prompt.shape[1] == kv_len_before_dropout

    # add prefix
    if prefix_prompt is not None:
        raise NotImplementedError()

    return model, prompt, len(gs_dataset), num_params, attn_logger


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

    # debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_max_len", action='store_true')

    # Model
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--no_tf32", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_epoch", type=int, required=True)

    # Evaluation & data
    parser.add_argument("--config_file", type=str, default="data/MetaICL/config/hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="data/MetaICL/data")
    parser.add_argument("--num_demonstrations", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_pair_len", type=int, default=256)
    parser.add_argument('--eval_seeds', type=str, nargs="+", default=['13', '21', '42', '87', '100'])
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="right") # slightly more accurate
    parser.add_argument("--allow_truncate", action='store_true')
    parser.add_argument("--delimiter", type=str, choices=['space', 'newline'], default='space')

    # limit eval
    parser.add_argument('--eval_test_per_task', type=int, default=10000000)
    parser.add_argument('--eval_ratio', type=float, default=1.0)
    parser.add_argument('--eval_on_demonstrations', action='store_true')

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
    parser.add_argument("--ttt_lora_rank", type=int, default=64)
    parser.add_argument("--ttt_lora_alpha", type=int, default=64)
    parser.add_argument("--ttt_lora_dropout", type=float, default=0.05)
    parser.add_argument("--ttt_lora_rslora", action='store_true')
    parser.add_argument("--ttt_loss_type", type=str, choices=['only_last', 'all', 'exclude_first'], default='all')

    # gradient search
    parser.add_argument("--gs_epochs", type=int, default=0)
    parser.add_argument("--gs_lr", type=float, default=1e-3)
    parser.add_argument("--gs_beta1", type=float, default=0.9)
    parser.add_argument("--gs_beta2", type=float, default=0.999)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0)
    parser.add_argument("--gs_batch_size", type=int, default=4)
    parser.add_argument("--gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--gs_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_num_layer", type=int, default=-1) # tune top layers only
    parser.add_argument("--gs_loss_on_input", action='store_true')
    parser.add_argument("--gs_dropout", choices=['none', 'train', 'suffix', 'power', 'power_with_train'], type=str, default='train')
    parser.add_argument("--gs_token_dropout", type=float, default=0.0)
    parser.add_argument("--gs_detach", action='store_true')
    parser.add_argument("--gs_ntokens", type=int, default=-1)
    parser.add_argument("--gs_log_attention", action='store_true')
    parser.add_argument("--gs_final_tokens", type=int, default=-1)

    # gradient search model initialization
    parser.add_argument("--random_prompt", type=str, choices=['none', 'uniform', 'token'], default='none')
    parser.add_argument("--random_prompt_ntokens", type=int, default=-1)
    parser.add_argument("--separate_kv", action='store_true')
    parser.add_argument("--num_permute", type=int, default=1) # 1024
    parser.add_argument("--permute_batch_size", type=int, default=16)
    parser.add_argument("--permute_back", action='store_true')
    parser.add_argument("--permute_concat", action='store_true')

    # gradient search with lora
    parser.add_argument("--gs_lora", action='store_true')
    parser.add_argument("--gs_lora_rank", type=int, default=64)
    parser.add_argument("--gs_lora_alpha", type=int, default=64)
    parser.add_argument("--gs_lora_lr", type=float, default=1e-4)
    parser.add_argument("--gs_lora_beta1", type=float, default=0.9)
    parser.add_argument("--gs_lora_beta2", type=float, default=0.999)
    parser.add_argument("--gs_lora_dropout", type=float, default=0.05)
    parser.add_argument("--gs_lora_rslora", action='store_true')

    # deeeeeeeeeeep thinking
    parser.add_argument("--dt_iters", type=int, default=0)
    parser.add_argument("--dt_lr", type=float, default=1e-2) # eta in the paper

    args = parser.parse_args()

    if args.debug:
        args.tag = 'test'
        args.eval_seeds = ['100']
        args.eval_ratio = 0.01

    args.delimiter = " " if args.delimiter == 'space' else "\n"

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

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
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizers loaded and pad tokens handled.")

    # load weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    model_weight_path = os.path.join(weight_dir, f"lora_epoch_{args.weight_epoch}")

    # load model (no dropout when model.train() for ttt or gs)
    model = GPT2LMHeadModel.from_pretrained(
        model_weight_path,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        summary_first_dropout=0.0,
        torch_dtype=NBIT_TO_DTYPE[args.trainable_nbit],
        cache_dir="./encoder_decoder_cache",
        _attn_implementation='eager' if args.gs_log_attention else 'sdpa',
    )

    # number of parameters
    print_trainable_parameters(model)

    # model size
    logger.info(f'model size {round(model.get_memory_footprint() / 1024 ** 3, 2)}GB')

    # Prepare with accelerator
    model = accelerator.prepare(model)

    # we dont train model so
    for p in model.parameters():
        p.requires_grad = False

    # Build evaluation dataset
    datasets = [
        EvalDataset(
            data_dir=args.data_dir,
            config_file=args.config_file,
            seed=args.seed,
            eval_seed=eval_seed,
            tokenizer=tokenizer,
            debug_random_pad=False,
            debug_pad_len=-1,
            debug_max_len=args.debug_max_len,
            pad_side=args.pad_side,
            max_seq_len=args.max_seq_len,
            max_pair_len=args.max_pair_len,
            eval_test_per_task=args.eval_test_per_task,
            eval_ratio=args.eval_ratio,
            split='test',
            allow_truncate=args.allow_truncate,
            delimiter=args.delimiter,
            num_demonstrations=args.num_demonstrations,
            eval_on_demonstrations=args.eval_on_demonstrations,
        )
        for eval_seed in args.eval_seeds
    ]
    collate_fn = partial(collate_fn_eval, dataset=datasets[0])
    if args.debug_max_len:
        collate_fn = partial(collate_fn_eval_dummy, dataset=datasets[0])

    # Eval Datasets
    score_list = []
    ttt_num_data_list, gs_num_data_list = [], []
    ttt_num_params_list, gs_num_params_list = [], []
    ttt_time_list, gs_time_list = [], []
    init_prompt_time_list = []
    output_list = None

    for dataset in datasets:
        score, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, init_prompt_time, output_list = test_time_evaluate(
            model=model,
            dataset=dataset,
            accelerator=accelerator,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            trainable_nbit=args.trainable_nbit,
            log_every=args.log_every,
            output_dir=args.output_dir,
            # gs
            gs_epochs=args.gs_epochs,
            gs_lr=args.gs_lr,
            gs_beta1=args.gs_beta1,
            gs_beta2=args.gs_beta2,
            gs_weight_decay=args.gs_weight_decay,
            gs_batch_size=args.gs_batch_size,
            gs_optimizer=args.gs_optimizer,
            gs_lr_scheduler=args.gs_lr_scheduler,
            gs_max_grad_norm=args.gs_max_grad_norm,
            gs_num_layer=args.gs_num_layer,
            gs_loss_on_input=args.gs_loss_on_input,
            gs_dropout=args.gs_dropout,
            gs_token_dropout=args.gs_token_dropout,
            gs_detach=args.gs_detach,
            gs_ntokens=args.gs_ntokens,
            gs_log_attention=args.gs_log_attention,
            gs_final_tokens=args.gs_final_tokens,
            # gs lora
            gs_lora=args.gs_lora,
            gs_lora_rank=args.gs_lora_rank,
            gs_lora_alpha=args.gs_lora_alpha,
            gs_lora_lr=args.gs_lora_lr,
            gs_lora_beta1=args.gs_lora_beta1,
            gs_lora_beta2=args.gs_lora_beta2,
            gs_lora_dropout=args.gs_lora_dropout,
            gs_lora_rslora=args.gs_lora_rslora,
            # gs init
            random_prompt=args.random_prompt,
            random_prompt_ntokens=args.random_prompt_ntokens,
            separate_kv=args.separate_kv,
            num_permute=args.num_permute,
            permute_batch_size=args.permute_batch_size,
            permute_back=args.permute_back,
            permute_concat=args.permute_concat,
            # ttt
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
            ttt_lora_dropout=args.ttt_lora_dropout,
            ttt_lora_rslora=args.ttt_lora_rslora,
            ttt_loss_type=args.ttt_loss_type,
            ttt_permute_n=args.ttt_permute_n,
            # dt
            dt_iters=args.dt_iters,
            dt_lr=args.dt_lr,
        )

        score_list.append(score)
        ttt_num_data_list.append(ttt_num_data)
        gs_num_data_list.append(gs_num_data)
        ttt_num_params_list.append(ttt_num_params)
        gs_num_params_list.append(gs_num_params)
        ttt_time_list.append(ttt_time)
        gs_time_list.append(gs_time)
        init_prompt_time_list.append(init_prompt_time)

    score = sum(score_list) / len(score_list)
    ttt_num_data = sum(ttt_num_data_list) / len(ttt_num_data_list)
    gs_num_data = sum(gs_num_data_list) / len(gs_num_data_list)
    ttt_num_params = sum(ttt_num_params_list) / len(ttt_num_params_list)
    gs_num_params = sum(gs_num_params_list) / len(gs_num_params_list)
    ttt_time = sum(ttt_time_list) / len(ttt_time_list)
    gs_time = sum(gs_time_list) / len(gs_time_list)
    init_prompt_time = sum(init_prompt_time_list) / len(init_prompt_time_list)

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {
            "eval/score": score,
            "eval/score_std": np.std(score_list),
            "eval/ttt_num_data": ttt_num_data,
            "eval/gs_num_data": gs_num_data,
            "eval/ttt_num_params": ttt_num_params,
            "eval/gs_num_params": gs_num_params,
            "eval/ttt_time": ttt_time,
            "eval/gs_time": gs_time,
            "eval/init_prompt_time": init_prompt_time,
        }
        logger.info(f'Evaluation results:\n{pprint.pformat(metric_dict, indent=4)}')

        # Save outputs
        save_pred_gt_path = os.path.join(args.output_dir, f"eval_pred_gt.json")
        with open(save_pred_gt_path, 'w') as f:
            json.dump(output_list, f)
        logger.info(f"Saved eval pred gt to {save_pred_gt_path}")

    logger.info("All done evaluating.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
