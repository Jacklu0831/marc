import gc
import time
import gc
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
from typing import Union, Callable, List, Tuple, Dict
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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training # type: ignore

from transformers import BitsAndBytesConfig

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
from peft import prepare_model_for_kbit_training # type: ignore

from transformers import BitsAndBytesConfig, AutoModelForCausalLM

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


@torch.no_grad()
def test_time_evaluate(
    model: Union[nn.Module, DistributedDataParallel],
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    trainable_nbit: int,
    no_flash_attn: bool,
    log_every: int,
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
    gs_lr_scheduler: str,
    gs_lora: bool,
    gs_lora_rank: int,
    gs_lora_alpha: int,
    gs_lora_lr: float,
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
) -> Tuple[float, float, float, float, float, float, float, List]:

    model.eval()

    # get modules in case of DDP
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.transformer.wte

    # We perform test-time adaptation in 2 stages.
    # First stage produces KV cache, num trainable params, runtime, num data. Second stage simply performs model loss/generation

    # STAGE 1: demonstration pair processing
    ttt_num_data_list, gs_num_data_list = [], []
    ttt_num_params_list, gs_num_params_list = [], []
    ttt_time_list, gs_time_list = [], []
    output_list = []

    assert set(len(v) for v in dataset.task_to_demonstrations.values()) == {16}
    assert len(dataset.tasks) >= accelerator.num_processes # avoid padding issue

    # we need to cache model in case of ttt or gs with trainable lora
    cached_model_path = os.path.join(output_dir, 'eval_cached_model')
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(cached_model_path)

    distributed_state = PartialState()
    with distributed_state.split_between_processes(dataset.tasks) as process_tasks:
        assert isinstance(process_tasks, list)

        for task in tqdm(process_tasks, desc='Task'):
            # data: get demonstration ids, hacky
            if dataset.debug_max_len:
                demon_input_ids = torch.randint(0, 30, (1, dataset.max_seq_len - dataset.max_pair_len), dtype=torch.int64, device=accelerator.device)
            else:
                demon_input_ids = None
                for data in dataset.data:
                    if data['task'] == task:
                        demon_input_ids = data["demon_input_ids"].unsqueeze(0).to(accelerator.device)
                        break
            assert demon_input_ids is not None

            # model: load cached for fresh model
            del model
            model = AutoModelForCausalLM.from_pretrained(cached_model_path).to(accelerator.device)

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

            # compute kv with model (ttt or not)
            with accelerator.autocast():
                assert demon_input_ids.shape[1] <= dataset.max_seq_len
                past_key_values = model(input_ids=demon_input_ids, output_hidden_states=True).past_key_values

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
                        demonstration_pairs=dataset.task_to_demonstrations[task],
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
                        lr_scheduler=gs_lr_scheduler,
                        lora=gs_lora,
                        lora_rank=gs_lora_rank,
                        lora_alpha=gs_lora_alpha,
                        lora_lr=gs_lora_lr,
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

                # expand past key values
                assert past_key_values[0][0].shape[0] == 1
                batch_past_key_values = [
                    (
                        layer_k.detach().clone().expand(bs, *layer_k.shape[1:]),
                        layer_v.detach().clone().expand(bs, *layer_v.shape[1:])
                    )
                    for layer_k, layer_v in past_key_values
                ]
                batch_past_key_values_attention_mask = torch.ones(
                    (bs, batch_past_key_values[0][0].shape[2]),
                    device=accelerator.device,
                    dtype=torch.int64
                )

                # cast if necessary
                if not no_flash_attn:
                    batch_past_key_values = tuple(
                        (
                            layer_k.to(NBIT_TO_DTYPE[trainable_nbit]),
                            layer_v.to(NBIT_TO_DTYPE[trainable_nbit]),
                        )
                        for layer_k, layer_v in batch_past_key_values
                    )

                with accelerator.autocast():
                    # second step to generate
                    gen_inputs_embeds = embed_tokens(gen_input_ids)
                    if not no_flash_attn:
                        gen_inputs_embeds = gen_inputs_embeds.to(NBIT_TO_DTYPE[trainable_nbit])

                    # add past key values portion to attention mask
                    gen_attention_mask = torch.cat([batch_past_key_values_attention_mask, gen_attention_mask], dim=1)

                    # build position ids (does NOT depend on dropout)
                    attention_mask_just_for_kv = gen_attention_mask[:, :batch_past_key_values[0][0].shape[2]]
                    attention_mask_after_kv = gen_attention_mask[:, batch_past_key_values[0][0].shape[2]:]
                    position_ids = []
                    for mask_for_kv, mask_after_kv in zip(attention_mask_just_for_kv, attention_mask_after_kv):
                        sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=accelerator.device, dtype=torch.int64)
                        position_start = mask_for_kv.sum()
                        n_new_positions = mask_after_kv.sum()
                        new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=accelerator.device, dtype=torch.int64)
                        if dataset.pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)

                    assert position_ids.max() < dataset.max_seq_len
                    model_out = model(
                        inputs_embeds=gen_inputs_embeds,
                        attention_mask=gen_attention_mask,
                        past_key_values=batch_past_key_values,
                        position_ids=position_ids,
                    )
                    losses = get_individual_loss(lm_logits=model_out.logits.half(), label_ids=gen_label_ids)

                # print(losses.tolist())
                # breakpoint()

                assert isinstance(losses, torch.Tensor)
                assert losses.shape[0] == len(task) == len(test_idx) == len(option) == len(correct_option) == bs
                for x0, x1, x2, x3, x4 in zip(losses, task, test_idx, option, correct_option):
                    output_list.append((x0.item(), x1, x2, x3, x4))
                if (eval_step + 1) % log_every == 0:
                    progress_bar.update(log_every)

    distributed_state.wait_for_everyone()
    # results
    ttt_num_data_list = gather_object(ttt_num_data_list)
    gs_num_data_list = gather_object(gs_num_data_list)
    ttt_num_params_list = gather_object(ttt_num_params_list)
    gs_num_params_list = gather_object(gs_num_params_list)
    ttt_time_list = gather_object(ttt_time_list)
    gs_time_list = gather_object(gs_time_list)
    output_list = gather_object(output_list)
    assert len(ttt_num_data_list) == len(dataset.tasks), (len(ttt_num_data_list), len(dataset.tasks))
    assert len(gs_num_data_list) == len(dataset.tasks), (len(gs_num_data_list), len(dataset.tasks))
    assert len(ttt_num_params_list) == len(dataset.tasks), (len(ttt_num_params_list), len(dataset.tasks))
    assert len(gs_num_params_list) == len(dataset.tasks), (len(gs_num_params_list), len(dataset.tasks))
    assert len(ttt_time_list) == len(dataset.tasks), (len(ttt_time_list), len(dataset.tasks))
    assert len(gs_time_list) == len(dataset.tasks), (len(gs_time_list), len(dataset.tasks))
    assert len(output_list) == len(dataset), (len(output_list), len(dataset)) # dataset length, not num task

    # determine which tasks are classification (for macro-f1)
    task_to_is_clf = {}
    for task in dataset.tasks:
        meta_data_path = os.path.join('MetaICL/config/tasks', f'{task}.json')
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

    # remove cache model
    if accelerator.is_main_process:
        os.system(f"rm -rf {cached_model_path}")

    return score, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, output_list


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
    permute_n: int,
) -> Tuple[nn.Module, int, int]:

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=['c_attn', 'c_proj', 'c_fc'],
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
    demonstration_pairs: List[Dict],
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
    lr_scheduler: str,
    lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_lr: float,
) -> Tuple[nn.Module, Tuple[Tuple[torch.Tensor, torch.Tensor]], int, int]:

    # optional lora
    if lora:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=['c_attn', 'c_proj', 'c_fc'],
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
        demonstration_pairs=demonstration_pairs,
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        past_kv_len=past_key_values[0][0].shape[2],
        max_seq_len=eval_dataset.max_seq_len,
        max_pair_len=eval_dataset.max_pair_len,
        allow_truncate=eval_dataset.allow_truncate,
        delimiter=eval_dataset.delimiter,
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
        {"params": program_params, "lr": lr},
        {"params": lora_params, "lr": lora_lr},
    ]
    all_params = program_params + lora_params
    if optimizer == 'adamw':
        optim = torch.optim.AdamW(optimizer_grouped_params, weight_decay=weight_decay, betas=(beta1, beta2)) # type: ignore
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
    embed_tokens = module.transformer.wte if not lora else module.model.transformer.wte

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
                assert position_ids.max() < eval_dataset.max_seq_len
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

    # debug
    parser.add_argument("--debug_max_len", action='store_true')

    # Model
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--no_tf32", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_epoch", type=int, required=True)

    # Evaluation & data
    parser.add_argument("--config_file", type=str, default="MetaICL/config/hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="MetaICL/data")
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

    # gradient search
    parser.add_argument("--gs_iters", type=int, default=0)
    parser.add_argument("--gs_lr", type=float, default=1e-3)
    parser.add_argument("--gs_beta1", type=float, default=0.9)
    parser.add_argument("--gs_beta2", type=float, default=0.999)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0)
    parser.add_argument("--gs_batch_size", type=int, default=16)
    parser.add_argument("--gs_grad_accum_steps", type=int, default=1)
    parser.add_argument("--gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--gs_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_no_key", action='store_true')
    parser.add_argument("--gs_no_value", action='store_true')

    # gradient search with lora (NOT COMPATIBLE WITH TTT)
    parser.add_argument("--gs_lora", action='store_true')
    parser.add_argument("--gs_lora_rank", type=int, default=64)
    parser.add_argument("--gs_lora_alpha", type=int, default=64)
    parser.add_argument("--gs_lora_lr", type=float, default=1e-4)

    args = parser.parse_args()
    args.delimiter = " " if args.delimiter == 'space' else "\n"

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

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

    logger.info("Base models loaded.")

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
    )
    if args.untrainable_nbit in [4, 8]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
    for param in model.parameters():
        param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])

    # Prepare with accelerator
    model = accelerator.prepare(model)

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
    output_list = None

    for dataset in datasets:
        score, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, output_list = test_time_evaluate(
            model=model,
            dataset=dataset,
            accelerator=accelerator,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            trainable_nbit=args.trainable_nbit,
            no_flash_attn=not args.flash_attn,
            log_every=args.log_every,
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
            gs_lr_scheduler=args.gs_lr_scheduler,
            gs_lora=args.gs_lora,
            gs_lora_rank=args.gs_lora_rank,
            gs_lora_alpha=args.gs_lora_alpha,
            gs_lora_lr=args.gs_lora_lr,
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

        score_list.append(score)
        ttt_num_data_list.append(ttt_num_data)
        gs_num_data_list.append(gs_num_data)
        ttt_num_params_list.append(ttt_num_params)
        gs_num_params_list.append(gs_num_params)
        ttt_time_list.append(ttt_time)
        gs_time_list.append(gs_time)

    score = sum(score_list) / len(score_list)
    ttt_num_data = sum(ttt_num_data_list) / len(ttt_num_data_list)
    gs_num_data = sum(gs_num_data_list) / len(gs_num_data_list)
    ttt_num_params = sum(ttt_num_params_list) / len(ttt_num_params_list)
    gs_num_params = sum(gs_num_params_list) / len(gs_num_params_list)
    ttt_time = sum(ttt_time_list) / len(ttt_time_list)
    gs_time = sum(gs_time_list) / len(gs_time_list)

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {
            "eval/score": score,
            "eval/ttt_num_data": ttt_num_data,
            "eval/gs_num_data": gs_num_data,
            "eval/ttt_num_params": ttt_num_params,
            "eval/gs_num_params": gs_num_params,
            "eval/ttt_time": ttt_time,
            "eval/gs_time": gs_time,
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
