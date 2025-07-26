import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import random
import itertools
import copy
import datasets
import logging
from tasks import TASKS
import gc
import time
from datetime import timedelta
from typing import Union, Callable, List, Tuple, Dict, Any, Iterator, Optional
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

import transformers
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    LlamaConfig,
)
from custom_llama import MyLlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training # type: ignore

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

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from datetime import timedelta
import pprint
import json
from functools import partial
import argparse
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training # type: ignore

from data_utils import EvalDataset, collate_fn_eval


import os
os.system('nvidia-smi')
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
    "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
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


def post_process_answer(answer: str, answer_format: str) -> str:
    """
    Post-process answers to comply with the formatting rules.
    """
    answer = answer.strip().lower()
    if not answer:
        return answer
    if answer_format == "Answer with only a sequence of words." or answer_format == "Answer with only a sequence of space-separated parentheses.":
        if '\n' in answer:
            answer = answer.split('\n')[0]
        if '-' in answer:
            answer = answer.split('-')[0]
    else:
        answer = answer.split()[0]

    if '.' in answer:
        answer = answer.split('.')[0].strip()

    if answer_format == "Answer with only the corresponding letter (e.g. (A)).":
        if answer and answer[0] != "(":
            answer = "(" + answer[0] + ")"

    return answer


def compute_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compare predictions and targets (case-insensitive match).
    Return accuracy as a percentage (float).
    """
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip().lower() == target.strip().lower():
            correct += 1
    return (correct / len(targets)) * 100 if len(targets) > 0 else 0.0


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
    trainable_nbit: int,
    # init
    random_kv: str,
    random_kv_ntokens: int,
    separate_kv: bool,
    num_permute: int,
    permute_batch_size: int,
    permute_back: bool,
    permute_back_strip_position: bool,
    permute_concat: bool,
    # dt
    dt_iters: int,
    dt_lr: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:

    if separate_kv:
        # too lazy to figure out how positional ids should work with instructions, lets see if work on arc/nlp first
        raise NotImplementedError('havent figured out how to do separate kv with instruction')

    elif dt_iters > 0:
        # first initialize KV, no shinanigans
        with accelerator.autocast():
            past_key_values = model(input_ids=demon_input_ids, output_hidden_states=True).past_key_values

        # now the iterative part
        demonstration_len = past_key_values[0][0].shape[2]
        for _ in range(dt_iters):
            with accelerator.autocast():
                past_key_values = model(
                    input_ids=demon_input_ids,
                    past_key_values=past_key_values,
                    position_ids=torch.arange(demon_input_ids.shape[1], device=accelerator.device)[None, ...],
                    output_hidden_states=True,
                ).past_key_values # first demonstration_len are old unmodified kv, then demonstration_len for new kv

            # some kind of learning
            assert past_key_values[0][0].shape[2] == demonstration_len * 2
            past_key_values = tuple(
                (
                    layer_k[:, :, :demonstration_len, :] * (1.0 - dt_lr) + layer_k[:, :, demonstration_len:, :] * dt_lr,
                    layer_v[:, :, :demonstration_len, :] * (1.0 - dt_lr) + layer_v[:, :, demonstration_len:, :] * dt_lr,
                )
                for (layer_k, layer_v) in past_key_values
            )
        assert past_key_values[0][0].shape[2] == demonstration_len

    elif random_kv != 'none':
        # random_kv_ntokens = random_kv_ntokens + demon_start_idxs[0] if random_kv_ntokens != -1 else demon_input_ids.shape[1]
        random_kv_ntokens = random_kv_ntokens if random_kv_ntokens != -1 else demon_input_ids.shape[1]
        assert random_kv == 'mlp'
        # assert random_kv_ntokens > demon_start_idxs[0]

        token_dim = 512
        past_key_values_seed = torch.rand((random_kv_ntokens, token_dim), device=accelerator.device, dtype=torch.float32)
        past_key_values_transform = torch.nn.Sequential(
            torch.nn.Linear(token_dim, token_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(token_dim, model.config.num_hidden_layers * 2 * model.config.head_dim * model.config.num_key_value_heads),
        ).to(accelerator.device)
        past_key_values_seed.requires_grad = False
        for p in past_key_values_transform.parameters():
            p.requires_grad = False

        with accelerator.autocast():
            past_key_values_instruct = model(input_ids=demon_input_ids[:, :demon_start_idxs[0]], output_hidden_states=True).past_key_values

    elif num_permute == 1:
        # only one kv is needed
        with accelerator.autocast():
            past_key_values = model(input_ids=demon_input_ids, output_hidden_states=True).past_key_values

    elif not permute_concat:
        # generate batches of permutations of them and average all
        permute_masks = generate_unique_permute_masks(demon_input_ids[0], demon_start_idxs, num_permute)
        permute_masks = [torch.cat([torch.arange(0, demon_start_idxs[0]), m]) for m in permute_masks] # add instruction

        past_key_values = tuple(
            (
                torch.zeros((1, model.config.num_key_value_heads, demon_input_ids.shape[1], model.config.head_dim), device=accelerator.device, dtype=torch.float32),
                torch.zeros((1, model.config.num_key_value_heads, demon_input_ids.shape[1], model.config.head_dim), device=accelerator.device, dtype=torch.float32),
            ) for _ in range(model.config.num_hidden_layers)
        )
        for batch_permute_masks in chunks(permute_masks, permute_batch_size):
            # get batch of permuted demon input ids
            batch_demon_input_ids = []
            for permute_mask in batch_permute_masks:
                batch_demon_input_ids.append(demon_input_ids.squeeze(0)[permute_mask])
            batch_demon_input_ids = torch.stack(batch_demon_input_ids)

            # get kv of each
            with accelerator.autocast():
                model_out = model(
                    input_ids=batch_demon_input_ids,
                    output_hidden_states=True,
                    return_key_states_no_pos=permute_back_strip_position,
                )
                batch_past_key_values = model_out.past_key_values
            assert len(batch_permute_masks) == batch_past_key_values[0][0].shape[0]

            # strip position for aggregation
            if permute_back_strip_position:
                # debug: sanity check that nopos is correct
                # position_ids = torch.arange(0, batch_demon_input_ids.shape[1], device=accelerator.device, dtype=torch.int64).unsqueeze(0)
                # cos, sin = model.model.rotary_emb(
                #     x=torch.tensor(0, device=accelerator.device, dtype=NBIT_TO_DTYPE[trainable_nbit]),
                #     position_ids=position_ids,
                # )
                # for layer_i in range(model.config.num_hidden_layers):
                #     k_no_pos = model_out.key_states_no_pos[layer_i].to(NBIT_TO_DTYPE[trainable_nbit])
                #     k_pos, k_pos_copy = apply_rotary_pos_emb(k_no_pos, k_no_pos, cos, sin)
                #     assert torch.equal(k_pos, k_pos_copy)
                #     assert torch.equal(k_pos, batch_past_key_values[layer_i][0])
                assert len(model_out.key_states_no_pos) == model.config.num_hidden_layers and not (None in model_out.key_states_no_pos)
                for layer_i in range(model.config.num_hidden_layers):
                    k_no_pos = model_out.key_states_no_pos[layer_i].to(NBIT_TO_DTYPE[trainable_nbit])
                    assert batch_past_key_values[layer_i][0].shape == k_no_pos.shape
                    batch_past_key_values[layer_i][0].copy_(k_no_pos)
                del model_out.key_states_no_pos

            # optionally permute kv back
            inverse_mask = None
            if permute_back:
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

        # add back position for aggregation
        if permute_back_strip_position:
            position_ids = torch.arange(0, demon_input_ids.shape[1], device=accelerator.device, dtype=torch.int64).unsqueeze(0)
            cos, sin = model.model.rotary_emb(
                x=torch.tensor(0, device=accelerator.device, dtype=NBIT_TO_DTYPE[trainable_nbit]),
                position_ids=position_ids,
            )
            assert past_key_values[0][0].shape[-2:] == cos.shape[-2:] == sin.shape[-2:]
            for layer_i in range(model.config.num_hidden_layers):
                k_pos, _ = apply_rotary_pos_emb(past_key_values[layer_i][0], past_key_values[layer_i][0], cos, sin)
                past_key_values[layer_i][0].copy_(k_pos)

    else:
        # generate batches of permutations of them and average all
        permute_masks = generate_unique_permute_masks(demon_input_ids[0], demon_start_idxs, num_permute)
        permute_masks = [torch.cat([torch.arange(0, demon_start_idxs[0]), m]) for m in permute_masks] # add instruction

        past_key_values = [[[], []] for _ in range(model.config.num_hidden_layers)]
        for batch_permute_masks in chunks(permute_masks, permute_batch_size):
            # get batch of permuted demon input ids
            batch_demon_input_ids = []
            for permute_mask in batch_permute_masks:
                batch_demon_input_ids.append(demon_input_ids.squeeze(0)[permute_mask])
            batch_demon_input_ids = torch.stack(batch_demon_input_ids)

            # get kv of each
            with accelerator.autocast():
                batch_past_key_values = model(input_ids=batch_demon_input_ids, output_hidden_states=True).past_key_values
            assert len(batch_permute_masks) == batch_past_key_values[0][0].shape[0]

            # add batch sum to a total sum of past_key_values
            assert batch_past_key_values[0][0].shape[0] == len(batch_permute_masks)
            for layer_i in range(len(batch_past_key_values)):
                for kv_i in range(2):
                    for kv in batch_past_key_values[layer_i][kv_i]:
                        past_key_values[layer_i][kv_i].append(kv)
        # concat
        past_key_values = tuple(
            (
                torch.cat(layer_k, dim=1).unsqueeze(0),
                torch.cat(layer_v, dim=1).unsqueeze(0),
            )
            for layer_k, layer_v in past_key_values
        )

    return past_key_values_seed, past_key_values_transform, past_key_values_instruct # type: ignore


def l2_compress(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    keep_ratio: float,
    skip_layers: List[int],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # assert all kv are same size
    assert len(set(tuple(x[0].shape for x in past_key_values)).union(set(tuple(x[0].shape for x in past_key_values)))) == 1
    assert skip_layers == []

    past_key_values = list(past_key_values) # type: ignore
    tokens_to_keep = math.ceil(keep_ratio * past_key_values[0][0].size(2))

    for layer_i, (layer_k, layer_v) in enumerate(past_key_values):
        if layer_i in skip_layers:
            continue

        key_norms = torch.norm(layer_k, p=2, dim=-1)
        sorted_indices = key_norms.squeeze(-1).argsort(dim=-1)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, layer_k.shape[-1])

        # apply sort
        sorted_layer_k = torch.gather(layer_k, dim=2, index=sorted_indices_expanded)
        sorted_values = torch.gather(layer_v, dim=2, index=sorted_indices_expanded)

        past_key_values[layer_i] = (sorted_layer_k[:, :, :tokens_to_keep, :], sorted_values[:, :, :tokens_to_keep, :]) # type: ignore

    return tuple(past_key_values) # type: ignore


def plot_heatmap(data1: np.ndarray, data2: np.ndarray, task: str, output_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    # Top heatmap: auto‐scaled
    im1 = ax1.imshow(
        data1,
        aspect='auto',
        interpolation='nearest',
        cmap='RdBu_r',       # or any colormap you like
        vmin=data1.min(),
        vmax=data1.max()
    )
    ax1.set_title(f'fisher values for keys of {task}')
    ax2.set_xlabel('sequence')
    ax1.set_ylabel('layer')
    fig.colorbar(im1, ax=ax1, orientation='vertical', label='fisher value')

    # Bottom heatmap: auto‐scaled independently
    im2 = ax2.imshow(
        data2,
        aspect='auto',
        interpolation='nearest',
        cmap='RdBu_r',
        vmin=data2.min(),
        vmax=data2.max()
    )
    ax1.set_title(f'fisher values for values of {task}')
    ax2.set_xlabel('sequence')
    ax1.set_ylabel('layer')
    fig.colorbar(im2, ax=ax2, orientation='vertical', label='fisher value')

    plt.savefig(output_path)
    plt.close()


def plot_log_heatmap(data1: np.ndarray, data2: np.ndarray, task: str, output_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

    # Top heatmap: per-plot log scale
    im1 = ax1.imshow(
        data1,
        aspect='auto',
        interpolation='nearest',
        norm=LogNorm(vmin=data1.min(), vmax=data1.max()),
        cmap='viridis'
    )
    ax1.set_title(f'fisher values for keys of {task}')
    ax2.set_xlabel('sequence')
    ax1.set_ylabel('layer')
    fig.colorbar(im1, ax=ax1, orientation='vertical', label='fisher key')

    # Bottom heatmap: per-plot log scale
    im2 = ax2.imshow(
        data2,
        aspect='auto',
        interpolation='nearest',
        norm=LogNorm(vmin=data2.min(), vmax=data2.max()),
        cmap='viridis'
    )
    ax1.set_title(f'fisher values for values of {task}')
    ax2.set_xlabel('sequence')
    ax1.set_ylabel('layer')
    fig.colorbar(im2, ax=ax2, orientation='vertical', label='fisher value')

    plt.savefig(output_path)
    plt.close()


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
    output_dir: str,
    ttt_weight_root_dir: str,
    ttt_weight_dir: Optional[str],
    zero_shot: bool,
    # compression
    compression_ratio: float,
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
    gs_no_key: bool,
    gs_no_value: bool,
    gs_num_layer: int,
    gs_loss_on_input: bool,
    gs_dropout: str,
    gs_token_dropout: float,
    gs_detach: bool,
    gs_freeze_instruct: bool,
    gs_ntokens: int,
    gs_log_attention: bool,
    gs_final_tokens: int,
    # gs prefix lora
    gs_prefix_lora: bool,
    gs_prefix_lora_rank: int,
    gs_prefix_lora_share_head: bool,
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
    random_kv: str,
    random_kv_ntokens: int,
    separate_kv: bool,
    num_permute: int,
    permute_batch_size: int,
    permute_back: bool,
    permute_back_strip_position: bool,
    permute_concat: bool,
    # gs curriculum
    gs_curriculum_epochs: int,
    gs_curriculum_lr: float,
    gs_curriculum_beta1: float,
    gs_curriculum_beta2: float,
    gs_curriculum_weight_decay: float,
    gs_curriculum_batch_size: int,
    gs_curriculum_optimizer: str,
    gs_curriculum_max_grad_norm: float,
    gs_curriculum_no_key: bool,
    gs_curriculum_no_value: bool,
    gs_curriculum_num_layer: int,
    gs_curriculum_loss_on_input: bool,
    # gs regularization
    gs_lambda_param_sqr: float,
    gs_fisher: bool,
    gs_log_fisher: bool,
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
    ttt_save: bool,
    ttt_permute_n: int,
    # ttt regularization
    ttt_lambda_param_sqr: float,
    ttt_fisher: bool,
    ttt_fisher_iters: int,
    # dt
    dt_iters: int,
    dt_lr: int,
) -> Tuple[float, float, float, float, float, float, float, float, List]:

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
    output_list = []
    acc_list = []

    assert set(len(v) for v in dataset.task_to_demonstrations.values()) == {dataset.num_demonstrations}
    assert len(TASKS) >= accelerator.num_processes # avoid padding issue

    # we need to cache model in case of ttt or gs with trainable lora
    cached_model = copy.deepcopy(model).cpu()

    # if loading ttt, make sure ckpts actually exist
    task_to_ttt_paths = {}
    if ttt_weight_dir is not None:
        ttt_weight_dir = os.path.join(ttt_weight_root_dir, ttt_weight_dir)
        for task in list(TASKS.keys()):
            task_to_ttt_paths[task] = os.path.join(ttt_weight_dir, f'{task}.pt')
            # assert os.path.exists(task_to_ttt_paths[task]), task_to_ttt_paths[task]

    distributed_state = PartialState()
    with distributed_state.split_between_processes(list(TASKS.keys())) as process_tasks:
        assert isinstance(process_tasks, list)

        for task in tqdm(process_tasks, desc='Task'):
            # data: get demonstration ids and indices, hacky
            if dataset.debug_max_len:
                demon_len = dataset.max_seq_len * 8 // dataset.num_demonstrations
                demon_input_ids = torch.randint(0, 30, (1, demon_len), dtype=torch.int64, device=accelerator.device)
                demon_start_idxs = [x * (demon_len // dataset.num_demonstrations) for x in range(dataset.num_demonstrations)]
            else:
                demon_input_ids = None
                demon_start_idxs = None
                for data in dataset.data:
                    if data['task'] == task:
                        demon_input_ids = data["demon_input_ids"].unsqueeze(0).to(accelerator.device)
                        demon_start_idxs = data["demon_start_idxs"]
                        break

            if demon_input_ids == None:
                continue

            assert demon_input_ids is not None and demon_start_idxs is not None
            assert demon_input_ids.shape[1] <= dataset.max_seq_len

            answer_format = TASKS[task]['answer_format']

            # load cached for fresh model
            model = copy.deepcopy(cached_model).to(accelerator.device)
            model.eval()

            if ttt_weight_dir is not None:
                # construct lora
                peft_config = LoraConfig(
                    r=ttt_lora_rank,
                    lora_alpha=ttt_lora_alpha,
                    lora_dropout=ttt_lora_dropout,
                    use_rslora=ttt_lora_rslora,
                    target_modules=['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, peft_config) # type: ignore
                for name, param in model.named_parameters():
                    assert param.requires_grad == ('lora' in name)
                    if param.requires_grad:
                        param.data = param.data.to(NBIT_TO_DTYPE[trainable_nbit])
                    else:
                        assert param.data.dtype == NBIT_TO_DTYPE[trainable_nbit]
                # load ttt checkpoint
                ttt_state_dict = torch.load(
                    task_to_ttt_paths[task],
                    weights_only=True,
                    map_location=accelerator.device,
                )
                model.load_state_dict(ttt_state_dict, strict=False)
                model = merge_lora(model)
                model.eval()
                logger.info(f'loaded ttt weights from {task_to_ttt_paths[task]}')
                # for n, p in model.named_parameters(): print(n, p.dtype, p.requires_grad)
                # breakpoint()

            # logging
            ttt_num_data, gs_num_data = 0, 0
            ttt_num_params, gs_num_params = 0, 0
            ttt_time, gs_time = 0.0, 0.0

            # use ttt to refine model
            if ttt_iters > 0:
                with accelerator.no_sync(model):
                    start_time = time.time()
                    ttt_save_path = os.path.join(output_dir, f'{task}.pt') if ttt_save else None
                    model, ttt_num_data, ttt_num_params = run_ttt(
                        task=task,
                        demonstration_pairs=dataset.task_to_demonstrations[task],
                        eval_dataset=dataset,
                        accelerator=accelerator,
                        model=model,
                        save_path=ttt_save_path,
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
                        # ttt regularization
                        lambda_param_sqr=ttt_lambda_param_sqr,
                        fisher=ttt_fisher,
                        fisher_iters=ttt_fisher_iters,
                    )
                    ttt_time = time.time() - start_time
                    torch.cuda.empty_cache()
                    gc.collect()

            # initialize kv
            start_time = time.time()

            if gs_curriculum_epochs > 0:
                past_key_values = run_gs_curriculum(
                    demonstration_pairs=dataset.task_to_demonstrations[task],
                    eval_dataset=dataset,
                    accelerator=accelerator,
                    model=model,
                    # inputs
                    demon_input_ids=demon_input_ids,
                    demon_start_idxs=demon_start_idxs,
                    # config
                    epochs=gs_curriculum_epochs,
                    lr=gs_curriculum_lr,
                    beta1=gs_curriculum_beta1,
                    beta2=gs_curriculum_beta2,
                    weight_decay=gs_curriculum_weight_decay,
                    batch_size=gs_curriculum_batch_size,
                    optimizer=gs_curriculum_optimizer,
                    max_grad_norm=gs_curriculum_max_grad_norm,
                    no_key=gs_curriculum_no_key,
                    no_value=gs_curriculum_no_value,
                    num_layer=gs_curriculum_num_layer,
                    loss_on_input=gs_curriculum_loss_on_input,
                    freeze_instruct=gs_freeze_instruct,
                )

            else:
                past_key_values_seed, past_key_values_transform, past_key_values_instruct = initialize_kv(
                    model=model,
                    demon_input_ids=demon_input_ids,
                    demon_start_idxs=demon_start_idxs,
                    accelerator=accelerator,
                    trainable_nbit=trainable_nbit,
                    # init
                    random_kv=random_kv,
                    random_kv_ntokens=random_kv_ntokens,
                    separate_kv=separate_kv,
                    num_permute=num_permute,
                    permute_batch_size=permute_batch_size,
                    permute_back=permute_back,
                    permute_back_strip_position=permute_back_strip_position,
                    permute_concat=permute_concat,
                    # dt
                    dt_iters=dt_iters,
                    dt_lr=dt_lr,
                )

            # if random_kv != 'none' and random_kv_ntokens == -1:
            #     assert past_key_values[0][0].shape[2] == demon_input_ids.shape[1]
            # elif random_kv != 'none' and random_kv_ntokens > -1:
            #     assert past_key_values[0][0].shape[2] == random_kv_ntokens + demon_start_idxs[0]
            # elif not permute_concat:
            #     assert past_key_values[0][0].shape[2] == demon_input_ids.shape[1]
            # else:
            #     assert past_key_values[0][0].shape[2] % num_permute == 0
            #     assert past_key_values[0][0].shape[2] // num_permute == demon_input_ids.shape[1]

            init_kv_time = time.time() - start_time
            torch.cuda.empty_cache()
            gc.collect()

            # compression
            if compression_ratio < 1.0:
                past_key_values = l2_compress(
                    past_key_values=past_key_values,
                    keep_ratio=compression_ratio,
                    skip_layers=[],
                )

            # use gs to refine kv
            if gs_epochs > 0:
                with accelerator.no_sync(model):
                    assert past_key_values_seed is not None
                    # assert past_key_values[0][0].shape[0] == 1

                    start_time = time.time()
                    saved_gradckpt = model.model.gradient_checkpointing
                    model.model.gradient_checkpointing = False
                    model, past_key_values_seed, past_key_values_transform, past_key_values_instruct, gs_num_data, gs_num_params, attn_logger, fisher_vals = run_gs(
                        demonstration_pairs=dataset.task_to_demonstrations[task],
                        eval_dataset=dataset,
                        accelerator=accelerator,
                        model=model,
                        # inputs
                        demon_start_idxs=demon_start_idxs,
                        past_key_values_seed=past_key_values_seed, # type: ignore
                        past_key_values_transform=past_key_values_transform, # type: ignore
                        past_key_values_instruct=past_key_values_instruct,
                        demon_input_ids_len=demon_input_ids.shape[1] if (random_kv == 'none' or random_kv_ntokens == -1) else random_kv_ntokens + demon_start_idxs[0],
                        # config
                        epochs=gs_epochs,
                        lr=gs_lr,
                        beta1=gs_beta1,
                        beta2=gs_beta2,
                        weight_decay=gs_weight_decay,
                        batch_size=gs_batch_size,
                        optimizer=gs_optimizer,
                        max_grad_norm=gs_max_grad_norm,
                        no_key=gs_no_key,
                        no_value=gs_no_value,
                        num_layer=gs_num_layer,
                        loss_on_input=gs_loss_on_input,
                        dropout=gs_dropout,
                        token_dropout=gs_token_dropout,
                        detach=gs_detach,
                        freeze_instruct=gs_freeze_instruct,
                        log_attention=gs_log_attention,
                        final_tokens=gs_final_tokens,
                        lr_scheduler=gs_lr_scheduler,
                        prefix_lora=gs_prefix_lora,
                        prefix_lora_rank=gs_prefix_lora_rank,
                        prefix_lora_share_head=gs_prefix_lora_share_head,
                        lora=gs_lora,
                        lora_rank=gs_lora_rank,
                        lora_alpha=gs_lora_alpha,
                        lora_lr=gs_lora_lr,
                        lora_beta1=gs_lora_beta1,
                        lora_beta2=gs_lora_beta2,
                        lora_dropout=gs_lora_dropout,
                        lora_rslora=gs_lora_rslora,
                        ntokens=gs_ntokens,
                        lambda_param_sqr=gs_lambda_param_sqr,
                        fisher=gs_fisher,
                        log_fisher=gs_log_fisher,
                    )
                    model.model.gradient_checkpointing = saved_gradckpt
                    gs_time = time.time() - start_time
                    torch.cuda.empty_cache()
                    gc.collect()

                    past_key_values_flat = past_key_values_transform(past_key_values_seed)
                    past_key_values_flat = past_key_values_flat.view(model.config.num_hidden_layers, 2, 1, model.config.num_key_value_heads, -1, model.config.head_dim)
                    past_key_values = tuple((x[0], x[1]) for x in past_key_values_flat)
                    past_key_values = tuple(
                        (
                            torch.cat([k1, k2], dim=2),
                            torch.cat([v1, v2], dim=2),
                        ) for (k1, v1), (k2, v2) in zip(past_key_values_instruct, past_key_values)
                    )

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

                    if fisher_vals is not None:
                        # average over heads and hidden dimension
                        assert fisher_vals[0][0].dim() == 4
                        heatmap_k = torch.stack([x[0].squeeze(0) for x in fisher_vals]).mean(dim=(1, 3)) # (layer, seq)
                        heatmap_v = torch.stack([x[1].squeeze(0) for x in fisher_vals]).mean(dim=(1, 3)) # (layer, seq)

                        # # remove instruction
                        # heatmap_k = heatmap_k[:, demon_start_idxs[0]:]
                        # heatmap_v = heatmap_v[:, demon_start_idxs[0]:]

                        os.makedirs(os.path.join(output_dir, 'fisher'), exist_ok=True)
                        save_path = os.path.join(output_dir, 'fisher', task) + '.jpg'
                        plot_heatmap(heatmap_k.cpu().numpy(), heatmap_v.cpu().numpy(), task, save_path)
                        logger.info(f'saved fisher to {save_path}')

                        os.makedirs(os.path.join(output_dir, 'log_fisher'), exist_ok=True)
                        save_path = os.path.join(output_dir, 'log_fisher', task) + '.jpg'
                        plot_log_heatmap(heatmap_k.cpu().numpy(), heatmap_v.cpu().numpy(), task, save_path)
                        logger.info(f'saved log fisher to {save_path}')

            # logging
            ttt_num_data_list.append(ttt_num_data)
            gs_num_data_list.append(gs_num_data)
            ttt_num_params_list.append(ttt_num_params)
            gs_num_params_list.append(gs_num_params)
            ttt_time_list.append(ttt_time)
            gs_time_list.append(gs_time)
            init_kv_time_list.append(init_kv_time)

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

            all_predictions = []
            all_labels = []

            for eval_step, batch_idxs in enumerate(data_idxs):
                batch_data = [dataset[i] for i in batch_idxs]
                bs = len(batch_data)
                batch = collate_fn(batch_data)

                # get tensors
                task = batch['task']
                test_idx = batch['test_idx']
                label = batch['label']
                generation_length = batch['generation_length']
                assert len(set(task)) == 1 # same task
                assert len(set(generation_length)) == 1 # same generation_length

                # for gs
                gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
                gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)

                with accelerator.autocast():
                    # expand past key values
                    assert past_key_values[0][0].shape[0] == 1
                    batch_past_key_values = [
                        (
                            layer_k.detach().clone().expand(bs, *layer_k.shape[1:]),
                            layer_v.detach().clone().expand(bs, *layer_v.shape[1:]),
                        )
                        for layer_k, layer_v in past_key_values
                    ]
                    batch_past_key_values_attention_mask = torch.ones(
                        (bs, batch_past_key_values[0][0].shape[2]),
                        device=accelerator.device,
                        dtype=torch.int64
                    )

                    if not zero_shot:
                        gen_input_ids = torch.cat([
                            torch.zeros((bs, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64),
                            gen_input_ids,
                        ], dim=1)
                        gen_attention_mask = torch.cat([batch_past_key_values_attention_mask, gen_attention_mask], dim=1)
                    else:
                        # if zero-shot, keep the past key values instruction
                        instruct_len = demon_start_idxs[0]
                        gen_input_ids = torch.cat([
                            torch.zeros((bs, instruct_len), device=accelerator.device, dtype=torch.int64),
                            gen_input_ids,
                        ], dim=1)
                        batch_past_key_values = tuple(
                            (layer_k[:, :, :instruct_len, :], layer_v[:, :, :instruct_len, :])
                            for layer_k, layer_v in batch_past_key_values
                        )
                        batch_past_key_values_attention_mask = batch_past_key_values_attention_mask[:, :instruct_len]
                        gen_attention_mask = torch.cat([batch_past_key_values_attention_mask, gen_attention_mask], dim=1)

                    # i truly dont know why this is necessary, but this is necessary
                    assert batch_past_key_values[0][0].dtype == torch.float32
                    if not no_flash_attn:
                        batch_past_key_values = tuple(
                            (
                                layer_k.to(NBIT_TO_DTYPE[trainable_nbit]),
                                layer_v.to(NBIT_TO_DTYPE[trainable_nbit]),
                            )
                            for layer_k, layer_v in batch_past_key_values
                        )

                    if zero_shot:
                        model.subtract_position_ids_by = 0
                    elif random_kv == 'none' or random_kv_ntokens == -1:
                        model.subtract_position_ids_by = past_key_values[0][0].shape[2] - demon_input_ids.shape[1] # HACK for prefix # type: ignore
                    else:
                        model.subtract_position_ids_by = 0 # HACK for prefix # type: ignore
                    gen_tokens = model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        past_key_values=batch_past_key_values,
                        max_new_tokens=generation_length[0],
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                    )
                    model.subtract_position_ids_by = 0 # HACK for prefix # type: ignore
                    assert gen_tokens.shape[0] == bs
                    gen_tokens = gen_tokens[:, gen_input_ids.shape[1]:]

                assert len(gen_tokens) == len(generation_length)
                gen_texts = dataset.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                predictions = [post_process_answer(t.strip(), answer_format) for t in gen_texts]

                # if gen_attention_mask.sum() < gen_attention_mask.numel():
                #     print(gen_texts)
                #     print(predictions)
                #     breakpoint()

                all_predictions += predictions
                all_labels += label

                assert len(predictions) == len(task) == len(test_idx) == bs
                for x0, x1, x2 in zip(predictions, task, test_idx):
                    output_list.append((x0, x1, x2))
                if (eval_step + 1) % log_every == 0:
                    progress_bar.update(log_every)

                torch.cuda.empty_cache()
                gc.collect()

            # after all batches of test samples, compute acc for this task
            acc_list.append(compute_accuracy(all_predictions, all_labels))

    distributed_state.wait_for_everyone()
    # results
    ttt_num_data_list = gather_object(ttt_num_data_list)
    gs_num_data_list = gather_object(gs_num_data_list)
    ttt_num_params_list = gather_object(ttt_num_params_list)
    gs_num_params_list = gather_object(gs_num_params_list)
    ttt_time_list = gather_object(ttt_time_list)
    gs_time_list = gather_object(gs_time_list)
    init_kv_time_list = gather_object(init_kv_time_list)
    output_list = gather_object(output_list)
    acc_list = gather_object(acc_list)
    # assert len(ttt_num_data_list) == len(TASKS), (len(ttt_num_data_list), len(TASKS))
    # assert len(gs_num_data_list) == len(TASKS), (len(gs_num_data_list), len(TASKS))
    # assert len(ttt_num_params_list) == len(TASKS), (len(ttt_num_params_list), len(TASKS))
    # assert len(gs_num_params_list) == len(TASKS), (len(gs_num_params_list), len(TASKS))
    # assert len(ttt_time_list) == len(TASKS), (len(ttt_time_list), len(TASKS))
    # assert len(gs_time_list) == len(TASKS), (len(gs_time_list), len(TASKS))
    # assert len(init_kv_time_list) == len(TASKS), (len(init_kv_time_list), len(TASKS))
    # assert len(output_list) == len(dataset), (len(output_list), len(dataset)) # dataset length, not num task
    # assert len(acc_list) == len(TASKS), (len(acc_list), len(TASKS))

    # average others
    ttt_num_data = sum(ttt_num_data_list) / len(ttt_num_data_list)
    gs_num_data = sum(gs_num_data_list) / len(gs_num_data_list)
    ttt_num_params = sum(ttt_num_params_list) / len(ttt_num_params_list)
    gs_num_params = sum(gs_num_params_list) / len(gs_num_params_list)
    ttt_time = sum(ttt_time_list) / len(ttt_time_list)
    gs_time = sum(gs_time_list) / len(gs_time_list)
    init_kv_time = sum(init_kv_time_list) / len(init_kv_time_list)
    score = sum(acc_list) / len(acc_list)

    return score, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, init_kv_time, output_list


def save_lora(model: nn.Module, save_path: str) -> None:
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_weights[name] = param.data
    torch.save(lora_weights, save_path)
    logger.info(f'saved {len(lora_weights)} ttt lora weights to {save_path}')


@torch.enable_grad()
def run_ttt(
    task: str,
    demonstration_pairs: List[Dict],
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    save_path: Optional[str],
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
    # ttt regularization
    lambda_param_sqr: float,
    fisher: bool,
    fisher_iters: int,
) -> Tuple[nn.Module, int, int]:

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=lora_rslora,
        target_modules=['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
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
        task=task,
        demonstration_pairs=demonstration_pairs,
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        max_seq_len=eval_dataset.max_seq_len,
        permute_n=permute_n,
        seed=eval_dataset.seed,
        loss_type=loss_type,
    )
    if len(ttt_dataset) == 0:
        # save and merge lora
        if save_path is not None:
            save_lora(model, save_path)
        model = merge_lora(model)
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

    # regularization
    saved_params = None

    # get program parameters
    fisher_vals = {}
    for n, p in model.named_parameters():
        assert p.requires_grad == ('lora' in n)
        if p.requires_grad:
            fisher_vals[n] = torch.tensor(1.0, device=accelerator.device)

    if lambda_param_sqr > 0:
        saved_params = {n: p.detach().clone() for n, p in model.named_parameters() if n in fisher_vals}
        assert all(not p.requires_grad for p in saved_params.values())

        if fisher:
            fisher_vals = compute_ttt_fisher(
                accelerator=accelerator,
                model=model,
                optim=optim,
                batch_size=batch_size,
                fisher_iters=fisher_iters,
                ttt_dataset=ttt_dataset,
                ttt_collate_fn=ttt_collate_fn,
            )

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

                # get regularization loss
                param_sqr_penalty = torch.tensor(0.0, device=accelerator.device)
                if saved_params is not None:
                    for n, p in model.named_parameters():
                        if n in saved_params:
                            param_sqr_penalty += (fisher_vals[n] * (p - saved_params[n]).pow(2)).sum()
                reg_loss = lambda_param_sqr / 2.0 * param_sqr_penalty

                # print(loss.item())
                # breakpoint()

            accelerator.backward(loss + reg_loss)

            if (curr_iter + 1) % grad_accum_steps == 0 or curr_iter == iters - 1:
                accelerator.clip_grad_norm_(lora_params, max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad()

            curr_iter += 1
            if curr_iter >= iters:
                break

    model.eval()

    # save and merge lora
    if save_path is not None:
        save_lora(model, save_path)
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


def compute_gs_fisher(
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    gs_dataset: GSDataset,
    demon_input_ids_len: int,
    optim: Any,
    gs_collate_fn: Any,
    embed_tokens: nn.Module,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # for each demon pair (1 by 1), compute grad, average over number of pairs

    assert past_key_values[0][0].shape[0] == 1
    optim.zero_grad()

    # dataloader
    gs_loader = DataLoader(
        gs_dataset,
        batch_size=1, # estimate over each data for fidelity
        shuffle=True,
        collate_fn=gs_collate_fn,
        drop_last=False, # no drop last to ensure we get loss for every sample at least once
        num_workers=0,
    )

    # initialize fisher
    fisher_vals = tuple(
        (
            torch.zeros_like(layer_k, device=accelerator.device, dtype=layer_k.dtype),
            torch.zeros_like(layer_v, device=accelerator.device, dtype=layer_v.dtype),
        ) for layer_k, layer_v in past_key_values
    )

    # get fisher
    for batch in gs_loader:
        pair_input_ids = batch["input_ids"].to(accelerator.device)
        pair_attention_mask = batch["attention_mask"].to(accelerator.device)
        pair_label_ids = batch["label_ids"].to(accelerator.device)
        device, dtype = pair_input_ids.device, pair_input_ids.dtype

        with accelerator.autocast():
            # build position ids
            position_ids = torch.zeros((1, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
            new_lens = pair_attention_mask.sum(dim=1)
            for task_position_ids, new_len in zip(position_ids, new_lens):
                new_positions = torch.tensor(range(demon_input_ids_len, demon_input_ids_len + new_len), device=device, dtype=dtype)
                if gs_dataset.pad_side == "right":
                    task_position_ids[:new_len] = new_positions
                else:
                    task_position_ids[-new_len:] = new_positions

            pair_inputs_embeds = embed_tokens(pair_input_ids)
            batch_past_key_values_attention_mask = torch.ones((1, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64)
            pair_attention_mask = torch.cat([batch_past_key_values_attention_mask, pair_attention_mask], dim=1)

            model_kwargs = {
                "inputs_embeds": pair_inputs_embeds,
                "attention_mask": pair_attention_mask,
                "labels": pair_label_ids,
                "use_cache": True,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "output_attentions": False,
            }

            # get ce loss
            model_out = model(**model_kwargs)

        accelerator.backward(model_out.loss)

        # update fisher
        for layer_i, (layer_k, layer_v) in enumerate(past_key_values):
            fisher_vals[layer_i][0].add_(layer_k.grad.data.pow(2) / len(gs_loader)) # type: ignore
            fisher_vals[layer_i][1].add_(layer_v.grad.data.pow(2) / len(gs_loader)) # type: ignore

        optim.zero_grad()

    return fisher_vals # type: ignore


def compute_ttt_fisher(
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    optim: Any,
    batch_size: int,
    fisher_iters: int,
    ttt_dataset: TTTDataset,
    ttt_collate_fn: Any,
) -> Dict[str, torch.Tensor]:

    optim.zero_grad()

    # dataloader
    ttt_loader = DataLoader(
        ttt_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ttt_collate_fn,
        drop_last=False,
        num_workers=0,
    )

    # initialize fisher
    fisher_vals = {}
    for n, p in model.named_parameters():
        assert p.requires_grad == ('lora' in n)
        if p.requires_grad:
            fisher_vals[n] = torch.zeros_like(p, device=accelerator.device, dtype=p.dtype)

    # get fisher
    curr_iter = 0

    fisher_iters = min(fisher_iters, len(ttt_loader))
    for batch in ttt_loader:
        if curr_iter >= fisher_iters:
            break
        input_ids = batch["input_ids"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)
        label_ids = batch["label_ids"].to(accelerator.device)
        bs = input_ids.shape[0]
        device, dtype = input_ids.device, input_ids.dtype

        # necessary
        with accelerator.autocast():
            # build position ids
            position_ids = torch.zeros((bs, input_ids.shape[1]), device=device, dtype=torch.int64)
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

        accelerator.backward(loss)

        # update fisher
        for n, p in model.named_parameters():
            assert p.requires_grad == ('lora' in n)
            if p.requires_grad:
                fisher_vals[n].add_(p.grad.data.pow(2) / fisher_iters) # type: ignore

        optim.zero_grad()

        curr_iter += 1

    return fisher_vals


class KVLora(nn.Module):
    def __init__(
        self,
        num_layer: int,
        num_head: int,
        num_token: int,
        token_dim: int,
        rank: int,
        share_head: bool,
    ):
        super().__init__()

        self.num_layer = num_layer
        self.num_head = num_head
        self.num_token = num_token
        self.token_dim = token_dim
        self.rank = rank
        self.share_head = share_head

        self.key_lora_A, self.key_lora_B = self.init_layers()
        self.value_lora_A, self.value_lora_B = self.init_layers()

    def init_layers(self):
        # init layers
        lora_As = []
        lora_Bs = []
        for _ in range(self.num_layer):
            if self.share_head:
                # num_token -> num_head * token_dim
                lora_A = nn.Linear(self.num_token, self.rank, bias=False)
                lora_B = nn.Linear(self.rank, self.num_head * self.token_dim, bias=False)
                nn.init.zeros_(lora_B.weight)
                lora_As.append(lora_A)
                lora_Bs.append(lora_B)
            else:
                lora_As_for_head = []
                lora_Bs_for_head = []
                for _ in range(self.num_head):
                    lora_A = nn.Linear(self.num_token, self.rank, bias=False)
                    lora_B = nn.Linear(self.rank, self.token_dim, bias=False)
                    nn.init.zeros_(lora_B.weight)
                    lora_As_for_head.append(lora_A)
                    lora_Bs_for_head.append(lora_B)
                lora_As.append(nn.ModuleList(lora_As_for_head))
                lora_Bs.append(nn.ModuleList(lora_Bs_for_head))

        lora_As = nn.ModuleList(lora_As)
        lora_Bs = nn.ModuleList(lora_Bs)
        assert len(lora_As) == self.num_layer
        assert len(lora_Bs) == self.num_layer
        return lora_As, lora_Bs

    def forward(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:

        assert tuple(past_key_values[0][0].shape[1:]) == tuple([self.num_head, self.num_token, self.token_dim])

        updated = []
        for layer_idx, (key, val) in enumerate(past_key_values):
            # get each updated kv with delta
            updated_kv = []
            for key_or_val, (lora_A, lora_B) in zip(
                [key, val],
                [[self.key_lora_A, self.key_lora_B], [self.value_lora_A, self.value_lora_B]]
            ):
                if self.share_head:
                    delta_flat = lora_B[layer_idx].weight @ lora_A[layer_idx].weight # (num_head * token_dim, num_token)
                    delta = delta_flat.view(self.num_head, self.token_dim, self.num_token)
                    delta = delta.permute(0, 2, 1) # (num_head, num_token, token_dim)
                else:
                    head_deltas = []
                    for h in range(self.num_head):
                        delta = (lora_B[layer_idx][h].weight @ lora_A[layer_idx][h].weight).t() # (num_token, token_dim) # type: ignore
                        head_deltas.append(delta)
                    delta = torch.stack(head_deltas, dim=0) # (num_head, num_token, token_dim)
                updated_kv.append(key_or_val + delta.unsqueeze(0).expand_as(key_or_val))
            # accumulate
            updated.append(tuple(updated_kv))

        return tuple(updated)


@torch.enable_grad()
def run_gs(
    demonstration_pairs: List[Dict],
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    # inputs
    demon_start_idxs: List[int],
    past_key_values_seed: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    past_key_values_transform,
    past_key_values_instruct,
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
    no_key: bool,
    no_value: bool,
    num_layer: int,
    loss_on_input: bool,
    dropout: str,
    token_dropout: float,
    detach: bool,
    freeze_instruct: bool,
    log_attention: bool,
    final_tokens: int,
    lr_scheduler: str,
    prefix_lora: bool,
    prefix_lora_rank: int,
    prefix_lora_share_head: bool,
    lora: bool,
    lora_rank: int,
    lora_alpha: int,
    lora_lr: float,
    lora_beta1: float,
    lora_beta2: float,
    lora_dropout: float,
    lora_rslora: bool,
    ntokens: int,
    lambda_param_sqr: float,
    fisher: bool,
    log_fisher: bool,
) -> Tuple[nn.Module, Tuple[Tuple[torch.Tensor, torch.Tensor]], int, int, Optional[AttentionLogger], Optional[Any]]:

    # optional lora
    if lora:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=lora_rslora,
            target_modules=['q_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config) # type: ignore
        # model.print_trainable_parameters()

    # this copying is necessary because torch is dumb as hell
    assert demon_start_idxs[0] > 0
    past_key_values_seed = past_key_values_seed.detach().clone()
    past_key_values_instruct = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values_instruct
    ) # type: ignore

    # get program parameters
    program_params = []
    prefix_past_key_values = None # no prefix tuning
    lora_module = None

    if prefix_lora:
        lora_module = KVLora(
            num_layer=len(past_key_values),
            num_head=past_key_values[0][0].shape[1],
            num_token=past_key_values[0][0].shape[2],
            token_dim=past_key_values[0][0].shape[3],
            rank=prefix_lora_rank,
            share_head=prefix_lora_share_head,
        ).to(accelerator.device)

        for param in lora_module.parameters():
            program_params.append(param)

        # key/value and double for loraA/B
        if prefix_lora_share_head:
            assert len(program_params) == 2 * 2 * len(past_key_values)
        else:
            assert len(program_params) == 2 * 2 * len(past_key_values) * past_key_values[0][0].shape[1]

    elif ntokens != -1:
        # additional prefix tuning
        prefix_past_key_values = tuple(
            (
                layer_k.mean(dim=2, keepdim=True).repeat(1, 1, ntokens, 1),
                layer_v.mean(dim=2, keepdim=True).repeat(1, 1, ntokens, 1),
            )
            for layer_k, layer_v in past_key_values # type: ignore
        )
        # add some noise so not all tokens end up optimized to the same
        prefix_past_key_values = tuple(
            (
                layer_k + 0.02 * torch.randn_like(layer_k, device=accelerator.device, dtype=layer_k.dtype),
                layer_v + 0.02 * torch.randn_like(layer_v, device=accelerator.device, dtype=layer_v.dtype),
            )
            for layer_k, layer_v in prefix_past_key_values
        )
        # add these to optimized params
        assert not no_key and not no_value
        for layer_k, layer_v in prefix_past_key_values:
            program_params.append(layer_k)
            program_params.append(layer_v)

    else:
        # # full tuning of initialized KV
        # num_layer = model.config.num_hidden_layers if num_layer == -1 else num_layer # type: ignore
        # for layer_k, layer_v in past_key_values[-num_layer:]:
        #     if not no_key:
        #         program_params.append(layer_k)
        #     if not no_value:
        #         program_params.append(layer_v)
        program_params.append(past_key_values_seed)
        for p in past_key_values_transform.parameters():
            program_params.append(p)
        for layer_k, layer_v in past_key_values_instruct:
            program_params.append(layer_k)
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

    # dataset
    gs_dataset = GSDataset(
        demonstration_pairs={i: p for i, p in enumerate(demonstration_pairs)},
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
        max_seq_len=eval_dataset.max_seq_len,
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
    if prefix_lora:
        assert all(p.requires_grad for p in program_params)
    else:
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
    embed_tokens = module.model.embed_tokens if not lora else module.model.model.embed_tokens

    attn_logger = None
    if log_attention:
        attn_logger = AttentionLogger(
            demon_input_ids_len=demon_input_ids_len,
            demon_start_idxs=demon_start_idxs,
        )

    # debug: save for assertions
    kv_len_before_dropout = past_key_values_seed.shape[0] + past_key_values_instruct[0][0].shape[2]

    # # regularization
    # saved_past_key_values = None
    # fisher_vals = tuple(
    #     (torch.tensor(1.0, device=accelerator.device), torch.tensor(1.0, device=accelerator.device))
    #     for _, _ in past_key_values
    # )

    # if lambda_param_sqr > 0:
    #     assert ntokens <= 0 # assume task A is the original ICL fast weights
    #     saved_past_key_values = tuple(
    #         (layer_k.detach().clone(), layer_v.detach().clone())
    #         for layer_k, layer_v in past_key_values
    #     )
    #     assert not saved_past_key_values[0][0].requires_grad

    #     if fisher:
    #         fisher_vals = compute_gs_fisher(
    #             accelerator=accelerator,
    #             model=model,
    #             past_key_values=past_key_values,
    #             gs_dataset=gs_dataset,
    #             demon_input_ids_len=demon_input_ids_len,
    #             optim=optim,
    #             gs_collate_fn=gs_collate_fn,
    #             embed_tokens=embed_tokens,
    #         )

    #         # avg_fisher = 0
    #         # for fisher_k, fisher_v in fisher_vals:
    #         #     avg_fisher += fisher_k.mean().item() + fisher_v.mean().item()
    #         # avg_fisher /= (len(fisher_vals) * 2)

    # train!
    for _ in range(epochs):
        for batch in gs_loader:
            pair_input_ids = batch["input_ids"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_label_ids = batch["label_ids"].to(accelerator.device)
            pair_example_idx = batch["example_idx"]
            device, dtype = pair_input_ids.device, pair_input_ids.dtype
            bs = pair_input_ids.shape[0]

            past_key_values_flat = past_key_values_transform(past_key_values_seed)
            past_key_values_flat = past_key_values_flat.view(model.config.num_hidden_layers, 2, 1, model.config.num_key_value_heads, -1, model.config.head_dim)
            past_key_values = tuple((x[0], x[1]) for x in past_key_values_flat)
            past_key_values = tuple(
                (
                    torch.cat([k1, k2], dim=2),
                    torch.cat([v1, v2], dim=2),
                ) for (k1, v1), (k2, v2) in zip(past_key_values_instruct, past_key_values)
            )

            # construct full attention mask for past key values first
            batch_past_key_values_attention_mask = torch.ones((batch_size, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64)

            if detach:
                # use the same past key values and attention mask, but detach untrained parts
                batch_past_key_values = tuple(
                    (layer_k.repeat(bs, 1, 1, 1), layer_v.repeat(bs, 1, 1, 1)) # repeat here
                    for layer_k, layer_v in past_key_values
                )

                # only drop training kv
                if dropout == 'train':
                    assert past_key_values[0][0].shape[2] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        start = demon_start_idxs[idx]
                        end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
                        for layer_i, (layer_k, layer_v) in enumerate(batch_past_key_values):
                            batch_past_key_values[layer_i][0][batch_i] = torch.cat([layer_k[batch_i, :, :start], layer_k[batch_i, :, start: end].detach().clone(), layer_k[batch_i, :, end:]], dim=1)
                            batch_past_key_values[layer_i][1][batch_i] = torch.cat([layer_v[batch_i, :, :start], layer_v[batch_i, :, start: end].detach().clone(), layer_v[batch_i, :, end:]], dim=1)

                # drop training kv and drop suffix
                elif dropout == 'suffix':
                    assert past_key_values[0][0].shape[2] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        start = demon_start_idxs[idx]
                        for layer_i, (layer_k, layer_v) in enumerate(batch_past_key_values):
                            batch_past_key_values[layer_i][0][batch_i] = torch.cat([layer_k[batch_i, :, :start], layer_k[batch_i, :, start:].detach().clone()], dim=1)
                            batch_past_key_values[layer_i][1][batch_i] = torch.cat([layer_v[batch_i, :, :start], layer_v[batch_i, :, start:].detach().clone()], dim=1)

                # drop training kv and only keep power set
                elif dropout in ['power', 'power_with_train']:
                    assert past_key_values[0][0].shape[2] == demon_input_ids_len # make sure demon_start_idxs are correct
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
                        to_keep, to_remove = set(to_keep), set(to_remove)

                        for layer_i, (layer_k, layer_v) in enumerate(batch_past_key_values):
                            new_layer_k = [layer_k[batch_i, :, :demon_start_idxs[0]]] # instruction
                            new_layer_v = [layer_v[batch_i, :, :demon_start_idxs[0]]] # instruction
                            for idx in range(len(demon_start_idxs)):
                                start = demon_start_idxs[idx]
                                end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
                                new_layer_k.append(layer_k[batch_i, :, start: end].detach().clone() if (idx in to_remove) else layer_k[batch_i, :, start: end])
                                new_layer_v.append(layer_v[batch_i, :, start: end].detach().clone() if (idx in to_remove) else layer_v[batch_i, :, start: end])
                            batch_past_key_values[layer_i][0][batch_i] = torch.cat(new_layer_k, dim=1)
                            batch_past_key_values[layer_i][1][batch_i] = torch.cat(new_layer_v, dim=1)

            else:
                # use the same past key values across batch, but adjust attention mask for dropping
                batch_past_key_values = tuple(
                    (layer_k.expand(bs, -1, -1, -1), layer_v.expand(bs, -1, -1, -1)) # expand here because no modifications
                    for layer_k, layer_v in past_key_values
                )

                # only drop training kv
                if dropout == 'train':
                    assert past_key_values[0][0].shape[2] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        start = demon_start_idxs[idx]
                        end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else demon_input_ids_len
                        batch_past_key_values_attention_mask[batch_i, start:end] = 0

                # drop training kv and drop suffix
                elif dropout == 'suffix':
                    assert past_key_values[0][0].shape[2] == demon_input_ids_len # make sure demon_start_idxs are correct
                    for batch_i, idx in enumerate(pair_example_idx):
                        start = demon_start_idxs[idx]
                        batch_past_key_values_attention_mask[batch_i, start:] = 0

                # drop training kv and only keep power set
                elif dropout in ['power', 'power_with_train']:
                    assert past_key_values[0][0].shape[2] == demon_input_ids_len # make sure demon_start_idxs are correct
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
                            batch_past_key_values_attention_mask[batch_i, start:end] = 0

            # debug: check lengths are correct
            for layer_k, layer_v in batch_past_key_values:
                assert (layer_k.shape[0], layer_k.shape[2]) == (layer_v.shape[0], layer_v.shape[2]) == (bs, kv_len_before_dropout)
            assert tuple(batch_past_key_values_attention_mask.shape) == (bs, kv_len_before_dropout)

            # tune the final few tokens only
            if final_tokens > -1:
                batch_past_key_values = tuple(
                    (
                        torch.cat([layer_k[:, :, :-final_tokens, :].detach().clone(), layer_k[:, :, -final_tokens:, :]], dim=2),
                        torch.cat([layer_v[:, :, :-final_tokens, :].detach().clone(), layer_v[:, :, -final_tokens:, :]], dim=2),
                    )
                    for layer_k, layer_v in batch_past_key_values
                )

            # tune the prefix only
            if prefix_past_key_values is not None:
                batch_past_key_values = tuple(
                    (
                        torch.cat([prefix_layer_k.expand(batch_size, *prefix_layer_k.shape[1:]), layer_k], dim=2),
                        torch.cat([prefix_layer_v.expand(batch_size, *prefix_layer_v.shape[1:]), layer_v], dim=2),
                    )
                    for (prefix_layer_k, prefix_layer_v), (layer_k, layer_v) in zip(prefix_past_key_values, batch_past_key_values)
                )
                batch_past_key_values_attention_mask = torch.cat([
                    torch.ones((bs, ntokens), device=accelerator.device, dtype=torch.int64),
                    batch_past_key_values_attention_mask,
                ], dim=1)

            # apply lora
            if lora_module is not None:
                old_shape = batch_past_key_values[0][0].shape
                batch_past_key_values = lora_module(batch_past_key_values)
                assert batch_past_key_values[0][0].shape == old_shape

            # token dropout
            if token_dropout != 0.0:
                drop_mask = (torch.rand_like(batch_past_key_values_attention_mask, dtype=torch.float) > token_dropout).float()
                batch_past_key_values_attention_mask = (batch_past_key_values_attention_mask * drop_mask).long()

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
                pair_attention_mask = torch.cat([batch_past_key_values_attention_mask, pair_attention_mask], dim=1)

                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
                    "labels": pair_label_ids,
                    "use_cache": True,
                    "past_key_values": batch_past_key_values,
                    "position_ids": position_ids,
                    "output_attentions": log_attention,
                }

                # get ce loss
                model_out = model(**model_kwargs)
                loss = model_out.loss * bs / batch_size # not doing droplast, so scale by relative batchsize

                # # get regularization loss
                # param_sqr_penalty = torch.tensor(0.0, device=accelerator.device)
                # if saved_past_key_values is not None:
                #     assert len(past_key_values) == len(saved_past_key_values) == len(fisher_vals) == model.config.num_hidden_layers # type: ignore
                #     assert past_key_values[0][0].shape[0] == saved_past_key_values[0][0].shape[0] == 1
                #     for (saved_layer_k, saved_layer_v), (layer_k, layer_v), (fisher_k, fisher_v) in zip(saved_past_key_values, past_key_values, fisher_vals):
                #         param_sqr_penalty += (fisher_k * (layer_k - saved_layer_k).pow(2)).sum()
                #         param_sqr_penalty += (fisher_v * (layer_v - saved_layer_v).pow(2)).sum()
                # reg_loss = lambda_param_sqr / 2.0 * param_sqr_penalty

                if attn_logger is not None:
                    attn_logger.update(
                        attentions=model_out.attentions,
                        pair_attention_mask=pair_attention_mask,
                        pair_example_idx=pair_example_idx,
                    )

                # if pair_attention_mask.sum() < pair_attention_mask.numel():
                #     print(loss.item() + reg_loss.item())
                #     breakpoint()

            # print(loss.item(), reg_loss.item())
            accelerator.backward(loss)

        # only at the end of epoch do we backprop
        accelerator.clip_grad_norm_(all_params, max_grad_norm)

        # ignore instruction
        if freeze_instruct:
            for layer_k, layer_v in past_key_values:
                layer_k.grad[:, :, :demon_start_idxs[0]] = 0.0 # type: ignore
                layer_v.grad[:, :, :demon_start_idxs[0]] = 0.0 # type: ignore

        optim.step()
        scheduler.step()
        optim.zero_grad()

    model.eval()
    if lora:
        model = merge_lora(model)

    past_key_values_seed = past_key_values_seed.detach().clone()
    past_key_values_instruct = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values_instruct
    ) # type: ignore

    # add back instruction
    # assert past_key_values[0][0].shape[2] == kv_len_before_dropout

    # add prefix
    if prefix_past_key_values is not None:
        past_key_values = tuple(
            (
                torch.cat([prefix_layer_k.detach().clone(), layer_k], dim=2),
                torch.cat([prefix_layer_v.detach().clone(), layer_v], dim=2),
            )
            for (prefix_layer_k, prefix_layer_v), (layer_k, layer_v) in zip(prefix_past_key_values, past_key_values)
        ) # type: ignore

    # apply lora
    if lora_module is not None:
        old_shape = past_key_values[0][0].shape
        past_key_values = lora_module(past_key_values)
        assert past_key_values[0][0].shape == old_shape

    ret_fisher_vals = fisher_vals if log_fisher else None
    return model, past_key_values_seed, past_key_values_transform, past_key_values_instruct, len(gs_dataset), num_params, attn_logger, ret_fisher_vals


def get_individual_loss(lm_logits: torch.Tensor, label_ids: torch.Tensor) -> List[float]:
    # move labels to correct device to enable model parallelism
    labels = label_ids.to(lm_logits.device)
    # Shift so that tokens < n predict n
    assert lm_logits.shape[0] == labels.shape[0]
    losses = []
    for logs, labs in zip(lm_logits, labels):
        shift_logits = logs[:-1, :].contiguous()
        shift_labels = labs[1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        assert loss.shape == labs[1:].shape
        loss = loss[labs[1:] != -100].mean()
        losses.append(loss)

    # debugging, should match crossentropy reduction (currently just match to 4 decimals)
    # ns = [(l != -100).sum() for l in labels]
    # ns = [n / sum(ns) for n in ns]
    # m = 0
    # for loss, n in zip(losses, ns):
    #     m += loss * n
    # print(m.item())

    return torch.stack(losses).tolist()


@torch.enable_grad()
def run_gs_curriculum(
    demonstration_pairs: List[Dict],
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: Union[nn.Module, DistributedDataParallel],
    # inputs
    demon_input_ids: torch.Tensor,
    demon_start_idxs: List[int],
    # config
    epochs: int,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    batch_size: int,
    optimizer: str,
    max_grad_norm: float,
    no_key: bool,
    no_value: bool,
    num_layer: int,
    loss_on_input: bool,
    freeze_instruct: bool,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:

    # this copying is necessary because torch is dumb as hell
    assert demon_start_idxs[0] > 0

    # prepare some stuff
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens

    demon_input_ids_len = demon_input_ids.shape[1]
    past_key_values = None
    pair_i_candidates = set(range(-1, len(demon_start_idxs)))
    curr_pair_i = -1 # always start with the instruction

    while True:
        # update kv with new pair
        model.eval()
        start = demon_start_idxs[curr_pair_i] if curr_pair_i >= 0 else 0
        end = demon_start_idxs[curr_pair_i + 1] if curr_pair_i < len(demon_start_idxs) - 1 else demon_input_ids.shape[1]
        pair_input_ids = demon_input_ids[:, start: end]
        past_key_values = model(
            inputs_embeds=embed_tokens(pair_input_ids),
            past_key_values=past_key_values,
            use_cache=True,
        ).past_key_values

        # no optimization for the last pair
        pair_i_candidates.remove(curr_pair_i)
        if len(pair_i_candidates) == 0:
            break

        # get program parameters
        # full tuning of initialized KV
        program_params = []
        num_layer = model.config.num_hidden_layers if num_layer == -1 else num_layer # type: ignore
        for layer_k, layer_v in past_key_values[-num_layer:]:
            if not no_key:
                program_params.append(layer_k)
            if not no_value:
                program_params.append(layer_v)

        # set requires grad
        assert all(not p.requires_grad for p in program_params)
        for p in program_params:
            p.requires_grad = True

        # optimizer
        optimizer_grouped_params = [
            {"params": program_params, "lr": lr, 'betas': (beta1, beta2)},
        ]
        if optimizer == 'adamw':
            optim = torch.optim.AdamW(optimizer_grouped_params, weight_decay=weight_decay) # type: ignore
        else:
            optim = torch.optim.SGD(optimizer_grouped_params, lr=lr) # type: ignore
        optim.zero_grad()

        # dataset
        gs_dataset = GSDataset(
            demonstration_pairs={i: demonstration_pairs[i] for i in pair_i_candidates},
            tokenizer=eval_dataset.tokenizer,
            debug_random_pad=eval_dataset.debug_random_pad,
            debug_pad_len=eval_dataset.debug_pad_len,
            pad_side=eval_dataset.pad_side,
            max_seq_len=eval_dataset.max_seq_len,
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

        # lr scheduler
        scheduler = get_constant_schedule(optim)

        model.train()

        # train!
        example_idx_to_losses = {pair_i: float('inf') for pair_i in pair_i_candidates}
        for _ in range(epochs):
            for batch in gs_loader:
                pair_input_ids = batch["input_ids"].to(accelerator.device)
                pair_attention_mask = batch["attention_mask"].to(accelerator.device)
                pair_label_ids = batch["label_ids"].to(accelerator.device)
                pair_example_idx = batch["example_idx"]
                device, dtype = pair_input_ids.device, pair_input_ids.dtype
                bs = pair_input_ids.shape[0]

                # construct full attention mask for past key values first
                batch_past_key_values_attention_mask = torch.ones((bs, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64)
                batch_past_key_values = tuple(
                    (layer_k.expand(bs, -1, -1, -1), layer_v.expand(bs, -1, -1, -1)) # expand here because no modifications
                    for layer_k, layer_v in past_key_values
                )

                with accelerator.autocast():
                    # build position ids
                    position_ids = torch.zeros((bs, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
                    new_lens = pair_attention_mask.sum(dim=1)
                    for task_position_ids, new_len in zip(position_ids, new_lens):
                        new_positions = torch.tensor(range(demon_input_ids_len, demon_input_ids_len + new_len), device=device, dtype=dtype)
                        if gs_dataset.pad_side == "right":
                            task_position_ids[:new_len] = new_positions
                        else:
                            task_position_ids[-new_len:] = new_positions

                    pair_inputs_embeds = embed_tokens(pair_input_ids)
                    pair_attention_mask = torch.cat([batch_past_key_values_attention_mask, pair_attention_mask], dim=1)

                    model_kwargs = {
                        "inputs_embeds": pair_inputs_embeds,
                        "attention_mask": pair_attention_mask,
                        "labels": pair_label_ids,
                        "use_cache": True,
                        "past_key_values": batch_past_key_values,
                        "position_ids": position_ids,
                    }

                    # get ce loss
                    model_out = model(**model_kwargs)

                    # update losses
                    detached_losses = get_individual_loss(lm_logits=model_out.logits.detach(), label_ids=pair_label_ids)
                    assert len(detached_losses) == len(pair_example_idx)
                    for i, l in zip(pair_example_idx, detached_losses):
                        assert i in example_idx_to_losses
                        example_idx_to_losses[i] = l

                    # if pair_attention_mask.sum() < pair_attention_mask.numel():
                    #     print(model_out.loss.item())
                    #     breakpoint()

                accelerator.backward(model_out.loss * bs / batch_size) # not doing droplast, so scale by relative batchsize

            # only at the end of epoch do we backprop
            accelerator.clip_grad_norm_(program_params, max_grad_norm)

            # ignore instruction
            if freeze_instruct:
                for layer_k, layer_v in past_key_values:
                    layer_k.grad[:, :, :demon_start_idxs[0]] = 0.0 # type: ignore
                    layer_v.grad[:, :, :demon_start_idxs[0]] = 0.0 # type: ignore

            optim.step()
            scheduler.step()
            optim.zero_grad()

        # after one pair of optimization, select easiest pair i
        assert max(example_idx_to_losses.values()) < float('inf')
        curr_pair_i = min(example_idx_to_losses, key=example_idx_to_losses.get) # type: ignore

        past_key_values = tuple(
            (layer_k.detach().clone(), layer_v.detach().clone())
            for layer_k, layer_v in past_key_values
        ) # type: ignore

    assert past_key_values[0][0].shape[2] == demon_input_ids.shape[1] # type: ignore
    return past_key_values # type: ignore


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
    parser.add_argument("--model_name", type=str, default="llama8b")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--no_tf32", action="store_true")

    # Weights
    parser.add_argument("--ttt_weight_root_dir", type=str, default="./encoder_decoder/outputs_eval")
    parser.add_argument("--ttt_weight_dir", type=str, default=None)

    # Evaluation & data
    parser.add_argument("--data_dir", type=str, default="./BIG-Bench-Hard")
    parser.add_argument("--num_demonstrations", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="left")

    # limit eval
    parser.add_argument('--eval_ratio', type=float, default=1.0)
    parser.add_argument('--eval_on_demonstrations', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')

    # compress
    parser.add_argument("--compression_ratio", type=float, default=1.0)

    # ttt
    parser.add_argument("--ttt_iters", type=int, default=0)
    parser.add_argument("--ttt_lr", type=float, default=1e-4)
    parser.add_argument("--ttt_weight_decay", type=float, default=0.0)
    parser.add_argument("--ttt_batch_size", type=int, default=5)
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
    parser.add_argument("--ttt_save", action='store_true')
    parser.add_argument("--ttt_gradient_checkpointing", action='store_true')

    # ttt regularization
    parser.add_argument("--ttt_lambda_param_sqr", type=float, default=0.0)
    parser.add_argument("--ttt_fisher", action='store_true')
    parser.add_argument("--ttt_fisher_iters", type=int, default=25)

    # gradient search
    parser.add_argument("--gs_epochs", type=int, default=0)
    parser.add_argument("--gs_lr", type=float, default=1e-3)
    parser.add_argument("--gs_beta1", type=float, default=0.9)
    parser.add_argument("--gs_beta2", type=float, default=0.999)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0)
    parser.add_argument("--gs_batch_size", type=int, default=2)
    parser.add_argument("--gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--gs_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_no_key", action='store_true')
    parser.add_argument("--gs_no_value", action='store_true')
    parser.add_argument("--gs_num_layer", type=int, default=-1) # tune top layers only
    parser.add_argument("--gs_loss_on_input", action='store_true')
    parser.add_argument("--gs_dropout", choices=['none', 'train', 'suffix', 'power', 'power_with_train'], type=str, default='train')
    parser.add_argument("--gs_token_dropout", type=float, default=0.0)
    parser.add_argument("--gs_detach", action='store_true')
    parser.add_argument("--gs_freeze_instruct", action='store_true')
    parser.add_argument("--gs_ntokens", type=int, default=-1)
    parser.add_argument("--gs_log_attention", action='store_true')
    parser.add_argument("--gs_final_tokens", type=int, default=-1)

    # gradient search prefix lora
    parser.add_argument("--gs_prefix_lora", action='store_true')
    parser.add_argument("--gs_prefix_lora_rank", type=int, default=64)
    parser.add_argument("--gs_prefix_lora_share_head", action='store_true')

    # gradient search model initialization
    parser.add_argument("--random_kv", type=str, choices=['none', 'uniform', 'token', 'mlp'], default='none')
    parser.add_argument("--random_kv_ntokens", type=int, default=-1)
    parser.add_argument("--separate_kv", action='store_true')
    parser.add_argument("--num_permute", type=int, default=1) # 1024
    parser.add_argument("--permute_batch_size", type=int, default=16)
    parser.add_argument("--permute_back", action='store_true')
    parser.add_argument("--permute_back_strip_position", action='store_true')
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

    # gradient search curriculum
    parser.add_argument("--gs_curriculum_epochs", type=int, default=0)
    parser.add_argument("--gs_curriculum_lr", type=float, default=1e-3)
    parser.add_argument("--gs_curriculum_beta1", type=float, default=0.9)
    parser.add_argument("--gs_curriculum_beta2", type=float, default=0.999)
    parser.add_argument("--gs_curriculum_weight_decay", type=float, default=0.0)
    parser.add_argument("--gs_curriculum_batch_size", type=int, default=10)
    parser.add_argument("--gs_curriculum_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_curriculum_max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_curriculum_no_key", action='store_true')
    parser.add_argument("--gs_curriculum_no_value", action='store_true')
    parser.add_argument("--gs_curriculum_num_layer", type=int, default=-1) # tune top layers only
    parser.add_argument("--gs_curriculum_loss_on_input", action='store_true')

    # gradient search regularization
    parser.add_argument("--gs_lambda_param_sqr", type=float, default=0.0)
    parser.add_argument("--gs_fisher", action='store_true')
    parser.add_argument("--gs_log_fisher", action='store_true')

    # deeeeeeeeeeep thinking
    parser.add_argument("--dt_iters", type=int, default=0)
    parser.add_argument("--dt_lr", type=float, default=1e-2) # eta in the paper

    args = parser.parse_args()

    if args.debug:
        args.tag = 'test'
        args.eval_ratio = 0.01

    args.tag = f"eval_{args.tag}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    args.ttt_iters *= args.ttt_grad_accum_steps

    if args.ttt_weight_dir is not None:
        assert args.ttt_iters == 0
        assert not args.ttt_save

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

    # load config in case using naive attention for attention map
    config = LlamaConfig.from_pretrained(MODEL_NAME_TO_PATH[args.model_name])
    if args.gs_log_attention:
        config._attn_implementation_autoset = False
        config._attn_implementation = 'eager'

    # load model
    model = MyLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        config=config,
        **from_pretrained_kwargs,
    )
    model.subtract_position_ids_by = 0 # HACK for prefix # type: ignore
    if args.untrainable_nbit in [4, 8]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.ttt_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        if args.ttt_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logger.info("Base models loaded.")

    # convert model weights
    for param in model.parameters():
        param.data = param.data.to(NBIT_TO_DTYPE[args.untrainable_nbit])

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
    dataset = EvalDataset(
        data_dir=args.data_dir,
        seed=args.seed,
        tokenizer=tokenizer,
        debug_random_pad=False,
        debug_pad_len=-1,
        debug_max_len=args.debug_max_len,
        pad_side=args.pad_side,
        max_seq_len=args.max_seq_len,
        num_demonstrations=args.num_demonstrations,
        eval_ratio=args.eval_ratio,
        eval_on_demonstrations=args.eval_on_demonstrations,
    )
    collate_fn = partial(collate_fn_eval, dataset=dataset)
    if args.debug_max_len:
        collate_fn = partial(collate_fn_eval_dummy, dataset=dataset)

    # update this, no redundant memory
    args.max_seq_len = dataset.max_seq_len

    # Eval Datasets
    score, ttt_num_data, gs_num_data, ttt_num_params, gs_num_params, ttt_time, gs_time, init_kv_time, output_list = test_time_evaluate(
        model=model,
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        trainable_nbit=args.trainable_nbit,
        no_flash_attn=not args.flash_attn,
        log_every=args.log_every,
        output_dir=args.output_dir,
        ttt_weight_root_dir=args.ttt_weight_root_dir,
        ttt_weight_dir=args.ttt_weight_dir,
        zero_shot=args.zero_shot,
        # eval
        compression_ratio=args.compression_ratio,
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
        gs_no_key=args.gs_no_key,
        gs_no_value=args.gs_no_value,
        gs_num_layer=args.gs_num_layer,
        gs_loss_on_input=args.gs_loss_on_input,
        gs_dropout=args.gs_dropout,
        gs_token_dropout=args.gs_token_dropout,
        gs_detach=args.gs_detach,
        gs_freeze_instruct=args.gs_freeze_instruct,
        gs_ntokens=args.gs_ntokens,
        gs_log_attention=args.gs_log_attention,
        gs_final_tokens=args.gs_final_tokens,
        # gs prefix lora
        gs_prefix_lora=args.gs_prefix_lora,
        gs_prefix_lora_rank=args.gs_prefix_lora_rank,
        gs_prefix_lora_share_head=args.gs_prefix_lora_share_head,
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
        random_kv=args.random_kv,
        random_kv_ntokens=args.random_kv_ntokens,
        separate_kv=args.separate_kv,
        num_permute=args.num_permute,
        permute_batch_size=args.permute_batch_size,
        permute_back=args.permute_back,
        permute_back_strip_position=args.permute_back_strip_position,
        permute_concat=args.permute_concat,
        # gs curriculum
        gs_curriculum_epochs=args.gs_curriculum_epochs,
        gs_curriculum_lr=args.gs_curriculum_lr,
        gs_curriculum_beta1=args.gs_curriculum_beta1,
        gs_curriculum_beta2=args.gs_curriculum_beta2,
        gs_curriculum_weight_decay=args.gs_curriculum_weight_decay,
        gs_curriculum_batch_size=args.gs_curriculum_batch_size,
        gs_curriculum_optimizer=args.gs_curriculum_optimizer,
        gs_curriculum_max_grad_norm=args.gs_curriculum_max_grad_norm,
        gs_curriculum_no_key=args.gs_curriculum_no_key,
        gs_curriculum_no_value=args.gs_curriculum_no_value,
        gs_curriculum_num_layer=args.gs_curriculum_num_layer,
        gs_curriculum_loss_on_input=args.gs_curriculum_loss_on_input,
        # gs regularization
        gs_lambda_param_sqr=args.gs_lambda_param_sqr,
        gs_fisher=args.gs_fisher,
        gs_log_fisher=args.gs_log_fisher,
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
        ttt_save=args.ttt_save,
        ttt_permute_n=args.ttt_permute_n,
        # ttt regularization
        ttt_lambda_param_sqr=args.ttt_lambda_param_sqr,
        ttt_fisher=args.ttt_fisher,
        ttt_fisher_iters=args.ttt_fisher_iters,
        # dt
        dt_iters=args.dt_iters,
        dt_lr=args.dt_lr,
    )

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
            "eval/init_kv_time": init_kv_time,
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
