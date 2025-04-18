import random
from custom_llama import MyLlamaForCausalLM
import shutil
import wandb
import gc
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
from datetime import timedelta
import arclib # required
import numpy as np
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Optional, Iterator, Dict, Set, Any
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

from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training # type: ignore
import logging
import datasets
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LlamaConfig,
)
import bitsandbytes as bnb

from data_utils import (
    TrainDataset,
    EvalDataset,
    GSDataset,
    collate_fn_train,
    collate_fn_train_invar,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
    collate_fn_gs,
    ARCTokenizer,
)

import os
os.system('nvidia-smi')
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


class ProgramCache:
    def __init__(self, cache_size_per_task: int, seed: int):
        assert cache_size_per_task > 0
        self.cache_size_per_task = cache_size_per_task
        self.cache = defaultdict(list)
        self.select_prior_rng = np.random.RandomState(seed + 1234) # all gpus need to sample the same here

    def update(self, task_identifier: str, program: torch.Tensor) -> None:
        # FIFO queue
        if len(self.cache[task_identifier]) == self.cache_size_per_task:
            self.cache[task_identifier] = self.cache[task_identifier][1:]
        self.cache[task_identifier].append(program)

    def get_num_items_in_cache(self) -> int:
        return sum(len(v) for v in self.cache.values())

    def sample_from_cache(self, task_identifier: str, device: torch.device) -> torch.Tensor:
        assert task_identifier in self.cache
        return random.choice(self.cache[task_identifier]).to(device) # dont need to be same across gpu

    def get_memory_footprint(self) -> int:
        size = 0
        for programs in self.cache.values():
            size += len(programs) * programs[0].nelement() * programs[0].element_size()
        return size

    def get_average_cache_len(self) -> float:
        lens = []
        for programs in self.cache.values():
            lens.append(len(programs))
        return sum(lens) / len(lens) if lens else 0.0

    def validate(self):
        dtype, shape = None, None
        for programs in self.cache.values():
            assert len(programs) <= self.cache_size_per_task
            for program in programs:
                if dtype == None:
                    dtype, shape = program.dtype, program.shape
                else:
                    assert dtype == program.dtype and shape == program.shape

    def debug_full_cache(self, ntokens: int, hidden_size: int, num_task_identifiers: int):
        for task_identifier in range(num_task_identifiers):
            print('initializing dummy cache', task_identifier)
            for _ in range(self.cache_size_per_task):
                self.cache[str(task_identifier)].append(torch.randn((ntokens, hidden_size), dtype=torch.float32, device='cpu'))


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float):
        """
        Args:
            margin (float): The cosine margin δ for negative pairs.
            aggregation (str): Method to aggregate program embeddings.
                               Options: 'average', 'max'. (Here we use 'average')
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, programs1: torch.Tensor, programs2: torch.Tensor, is_same: bool):
        programs1_flat = programs1.mean(dim=0)[None, ...].flatten(start_dim=1)
        programs2_flat = programs2.mean(dim=0)[None, ...].flatten(start_dim=1)
        is_same = 1 if is_same else -1 # type: ignore
        is_same = torch.tensor([is_same], device=programs1.device) # type: ignore
        return self.cosine_loss(programs1_flat, programs2_flat, is_same)


class ProgramEmbeddings(nn.Module):
    def __init__(self, embedding: torch.Tensor):
        super(ProgramEmbeddings, self).__init__()
        self.embedding = nn.Parameter(embedding)

    def forward(self, program_i: int) -> torch.Tensor:
        del program_i
        return self.embedding


class Quantizer(nn.Module):
    def __init__(
            self,
            codebook_size: int,
            hidden_size: int,
            fsq_L: List[int],
            device: torch.device,
        ):
        super(Quantizer, self).__init__()

        self.quantizer_type = 'vqvae' if fsq_L == [] else 'fsq'
        self.device = device

        if self.quantizer_type == 'vqvae':
            self.embedding = nn.Parameter(torch.randn(
                (codebook_size, hidden_size),
                device=device,
            ))
            self.num_embeddings, self.embedding_dim = tuple(self.embedding.shape)
        else:
            self.fsq_L = torch.tensor(fsq_L, device=device)
            self.fsq_hidden_to_latent = nn.Linear(hidden_size, len(fsq_L))
            self.fsq_latent_to_hidden = nn.Linear(len(fsq_L), hidden_size)

    def forward(
            self,
            program: torch.Tensor,
            train_codebook_only: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, ntokens, embedding_dim = program.shape
        if self.quantizer_type == 'fsq':
            # quantize
            latent = self.fsq_hidden_to_latent(program)
            latent_bounded = (self.fsq_L - 1.0) / 2.0 * torch.tanh(latent)
            latent_quantized = latent_bounded + (torch.round(latent_bounded) - latent_bounded).detach()
            program_quantized = self.fsq_latent_to_hidden(latent_quantized)
            # check and return
            assert program_quantized.shape == program.shape
            dummy = torch.tensor(0.0, device=self.device)
            return program_quantized, dummy, dummy, dummy

        assert embedding_dim == self.embedding_dim

        # when only training codebook, match codebook to ntokens
        if train_codebook_only:
            detached_program = program.detach()
            flat_program = detached_program.flatten(start_dim=0, end_dim=1) # (batch_size x ntokens, embedding_dim)
            distances = torch.sum(self.embedding ** 2.0, dim=1, keepdim=True) \
                         + torch.sum(flat_program ** 2.0, dim=1) \
                         - 2 * self.embedding @ flat_program.t() # (nembeddings, batch_size x ntokens)
            encoding_indices = torch.argmin(distances, dim=1) # (nembeddings,)
            encodings = torch.nn.functional.one_hot(encoding_indices, batch_size * ntokens).type(flat_program.dtype) #(nembeddings, batch_size x ntokens)
            quantized = torch.matmul(encodings, flat_program) # (nembeddings, embedding_dim)
            quantized = quantized.reshape(self.num_embeddings, embedding_dim)
            codebook_loss = torch.nn.functional.mse_loss(quantized, self.embedding)
            # dummy and return
            commitment_loss = torch.tensor(0.0, device=program.device)
            perplexity = torch.tensor(0.0, device=program.device)
            return program, codebook_loss, commitment_loss, perplexity

        # quantize
        flat_program = program.flatten(start_dim=0, end_dim=1) # (batch_size x ntokens, embedding_dim)
        distances = torch.sum(flat_program ** 2.0, dim=1, keepdim=True) \
                     + torch.sum(self.embedding ** 2.0, dim=1) \
                     - 2 * flat_program @ self.embedding.t() # (batch_size x ntokens, nembeddings)
        encoding_indices = torch.argmin(distances, dim=1) # (batch_size x ntokens,)
        encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).type(flat_program.dtype) #(batch_size x ntokens, nembeddings)
        quantized = torch.matmul(encodings, self.embedding) # (batch_size x ntokens, embedding_dim)
        quantized = quantized.reshape(batch_size, ntokens, embedding_dim)

        # losses
        codebook_loss = torch.nn.functional.mse_loss(quantized, program.detach())
        commitment_loss = torch.nn.functional.mse_loss(quantized.detach(), program)

        # get quantized program (straight-through estimator)
        quantized = program + (quantized - program).detach()

        # calculate perplexity for debugging
        avg_probs = torch.mean(encodings.detach(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) / ntokens
        return quantized, codebook_loss, commitment_loss, perplexity


class LambdaScheduler:
    def __init__(
            self,
            loss_lambda: float,
            start_epoch: int,
            linear_epochs: int,
            steps_per_epoch: int,
        ):

        self.loss_lambda = loss_lambda
        self.start_step = start_epoch * steps_per_epoch
        self.total_warmup_steps = linear_epochs * steps_per_epoch

    def get_lambda(self, step: int) -> float:
        step += 1
        if step < self.start_step:
            # stage 1, before start epoch
            return 0.0
        elif step < self.start_step + self.total_warmup_steps:
            # stage 2: during linear warmup phase
            weight = (step - self.start_step) / self.total_warmup_steps
            return weight * self.loss_lambda
        else:
            # stage 3: after warmup
            return self.loss_lambda

    def visualize(self, total_steps: int, path: str = "temp.jpg"):
        lambdas = [self.get_lambda(s) for s in range(total_steps)]
        plt.figure()
        plt.plot(lambdas)
        plt.xlabel('step')
        plt.savefig(path)
        plt.close()


class VaeProjection(nn.Module):
    def __init__(
            self,
            mlp_factor: int,
            latent_dim: int,
            device: torch.device,
        ):
        super(VaeProjection, self).__init__()

        self.device = device
        self.mlp_mu = self.get_projection(mlp_factor, latent_dim)
        self.mlp_logvar = self.get_projection(mlp_factor, latent_dim)

    def get_projection(self, mlp_factor: int, latent_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent_dim, mlp_factor * latent_dim),
            nn.SiLU(),
            nn.Linear(mlp_factor * latent_dim, latent_dim)
        )

    def forward(self, predicted_program: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mlp_mu(predicted_program), self.mlp_logvar(predicted_program)


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


def extract_program_from_right_side(data: torch.Tensor, lens: List[int], ntokens: int, pad_side: str):
    # extract program from the right side of data
    if pad_side == "right":
        program = []
        for x, l in zip(data, lens):
            program.append(x[l:l+ntokens])
        return torch.stack(program)
    else:
        if ntokens == 0:
            return data[:, 0:0, :]
        else:
            return data[:, -ntokens:, :]


def sample_from_vae(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def compute_kl_loss(mu1: torch.Tensor, mu2: torch.Tensor, logvar1: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
    # test if mu1 logvar1 in mu2 logvar2
    batch_size = mu1.shape[0]
    assert mu1.shape == mu2.shape == logvar1.shape == logvar2.shape
    mu1 = mu1.flatten()
    mu2 = mu2.flatten()
    logvar1 = logvar1.flatten()
    logvar2 = logvar2.flatten()
    logvar1 = torch.clamp(logvar1, min=-10, max=10)
    logvar2 = torch.clamp(logvar2, min=-10, max=10)
    kl_loss = ((logvar2 - logvar1) + torch.exp(logvar1 - logvar2) - 1 + (mu1 - mu2) ** 2.0 * torch.exp(-logvar2))
    kl_loss = kl_loss.sum() * 0.5
    return kl_loss / batch_size


def get_memory_footprint(module: nn.Module):
    return sum(p.nelement() * p.element_size() for p in module.parameters()) + \
        sum(p.nelement() * p.element_size() for p in module.buffers())


def get_predicted_program(
    # model
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    tokenizer: ARCTokenizer,
    # data
    input_ids: List[torch.Tensor],
    attention_mask: List[torch.Tensor],
    input_ids_lens: List[List[int]],
    num_pairs: List[int],
    # others
    ntokens: int,
    pad_side: str,
    no_residual: bool,
    no_discrete_prior: bool,
    train_codebook_only: bool,
    weird_cast: bool,
    kv_pad_side: str,
    short_context: bool,
    attention_reduction_ratio: float,
) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]], Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]], Optional[torch.Tensor]]:

    def update_based_on_avail(all_x: List[Any], new_x: List[Any], avail_mask: torch.Tensor, concat: bool) -> None:
        # inplace
        assert len(all_x) == len(avail_mask)
        assert len(new_x) == sum(avail_mask)
        new_idx = 0
        for i, m in enumerate(avail_mask):
            if m:
                if concat:
                    if isinstance(all_x[i], List):
                        all_x[i].append(new_x[new_idx])
                    else:
                        all_x[i] = torch.cat([all_x[i], new_x[new_idx]])
                else:
                    all_x[i] = new_x[new_idx]
                new_idx += 1

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # samples do not have to be the same number of pairs in evaluation, but try to parallelize it anyway
    # assert min(num_pairs) >= 2 # at least 2 train
    assert max(num_pairs) == len(input_ids)

    assert len(set(len(ids) for ids in input_ids)) == 1 # batch size uniform across pair_idx
    batch_size = len(input_ids[0])
    dtype, device = input_ids[0].dtype, input_ids[0].device

    # reused tensors
    pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device))
    inputs_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in input_ids]

    # apply program norm to prior
    prior_inputs_embeds = prior_embeddings("dummy")[None, ...]
    if program_norm is not None:
        prior_inputs_embeds = program_norm(prior_inputs_embeds)
    # quantize prior
    if (quantizer is not None) and not no_discrete_prior:
        prior_inputs_embeds, _, _, _ = quantizer(prior_inputs_embeds, train_codebook_only=train_codebook_only)
    # expand prior
    prior_inputs_embeds = prior_inputs_embeds.expand(batch_size, -1, -1)

    # previous program of each sample in batch
    all_prev_programs = [p for p in prior_inputs_embeds] # batchsize x (ntokens, hiddendim)
    all_prev_past_key_values = None # batchsize x numlayer x 2, (nhead, seqlen, hiddendim)
    all_prev_past_key_values_attention_mask = None # batchsize x (seqlen,)

    # separately, we save ALL intermediate programs for generation because huggingface doesn't allow me to generate with kv cache + input embeds
    all_intermediate_programs = [[all_prev_programs[batch_i]] for batch_i in range(batch_size)] # batchsize x num-program-in-each-task x (ntokens, hiddendim)
    # NOTE: num-program-in-task is one more than the number of input_ids pairs

    # precompute demonstration intervals for these three tries
    demonstration_intervals = [[] for _ in range(batch_size)]
    if attention_reduction_ratio != 1.0:
        start = ntokens # keep track of start of each demonstration pair
        for pair_j, lens in enumerate(input_ids_lens):
            if pair_j > 0:
                start += attention_mask[pair_j - 1].shape[1] + ntokens
            max_l = attention_mask[pair_j].shape[1]
            for batch_i, l in enumerate(lens):
                if pair_j < num_pairs[batch_i]:
                    s = start if pad_side == 'right' else start + max_l - l
                    demonstration_intervals[batch_i].append((s, s + l))

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_input_ids_lens) in enumerate(zip(inputs_embeds, attention_mask, input_ids_lens)):

        # STEP 0: filter out samples that are not this many pairs (for eval)
        avail_mask = torch.tensor([int(pair_i < n) for n in num_pairs], dtype=torch.bool)
        pair_inputs_embeds = pair_inputs_embeds[avail_mask]
        pair_attention_mask = pair_attention_mask[avail_mask]
        pair_input_ids_lens = [l for l, m in zip(pair_input_ids_lens, avail_mask) if m]
        select_demonstration_intervals = [l for l, m in zip(demonstration_intervals, avail_mask) if m]

        # do the same for all_prev_programs
        prev_programs = [p for p, m in zip(all_prev_programs, avail_mask) if m]
        assert len(set(p.shape[0] for p in prev_programs)) == 1
        prev_programs = torch.stack(prev_programs)

        # do the same for prev_past_key_values and attention
        prev_past_key_values = None
        prev_past_key_values_attention_mask = None
        if pair_i > 0 and not short_context:
            assert all_prev_past_key_values is not None
            assert all_prev_past_key_values_attention_mask is not None
            # get avail
            prev_past_key_values = [p for p, m in zip(all_prev_past_key_values, avail_mask) if m]
            prev_past_key_values_attention_mask = [p for p, m in zip(all_prev_past_key_values_attention_mask, avail_mask) if m]
            # ensure same length
            assert len(set(m.shape[0] for m in prev_past_key_values_attention_mask)) == 1 # same seq len
            assert len(set(x[0][0].shape[1] for x in prev_past_key_values)) == 1 # same seq len across batch
            # stack these and format key values
            num_layer = len(prev_past_key_values[0])
            prev_past_key_values = tuple(
                (
                    torch.stack([kv[layer_i][0] for kv in prev_past_key_values]),
                    torch.stack([kv[layer_i][1] for kv in prev_past_key_values]),
                )
                for layer_i in range(num_layer)
            )
            prev_past_key_values_attention_mask = torch.stack(prev_past_key_values_attention_mask)

        # some more checks on avail
        n_avail = int(avail_mask.sum().item())
        assert avail_mask.shape[0] == batch_size
        if pair_i == 0:
            assert avail_mask.sum() == batch_size

        # STEP 1: prepend the last predicted program for all pairs except the first
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((n_avail, ntokens), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )

        # update lens to reflect the extra program
        pair_input_ids_lens = [l + ntokens for l in pair_input_ids_lens]

        # save attention mask
        pair_attention_mask_no_program_embed = pair_attention_mask.detach().clone()

        # STEP 2: append new program input to the right
        emb = program_embeddings("dummy")[None, ...].expand(n_avail, -1, -1)
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=(emb.to(torch.bfloat16) if weird_cast else emb),
            lens=pair_input_ids_lens,
            insert_side="right",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((n_avail, ntokens), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="right",
            pad_side=pad_side,
            pad_id=0,
        )

        # attention mask should span to the past kvs
        if pair_i > 0 and not short_context:
            assert prev_past_key_values is not None
            assert prev_past_key_values_attention_mask is not None
            assert prev_past_key_values_attention_mask.shape[1] == prev_past_key_values[0][0].shape[2]
            assert prev_past_key_values_attention_mask.shape[0] == pair_attention_mask.shape[0] == n_avail
            pair_attention_mask = torch.cat([prev_past_key_values_attention_mask, pair_attention_mask], dim=1)

        model_kwargs = {
            "inputs_embeds": pair_inputs_embeds,
            "attention_mask": pair_attention_mask,
            "output_hidden_states": True,
            "use_cache": not short_context,
        }
        if pair_i > 0 and not short_context:
            assert prev_past_key_values is not None
            model_kwargs["past_key_values"] = prev_past_key_values

            # build position ids
            attention_mask_just_for_kv = pair_attention_mask[:, :prev_past_key_values[0][0].shape[2]]
            attention_mask_after_kv = pair_attention_mask[:, prev_past_key_values[0][0].shape[2]:]
            position_ids = []
            for mask_for_kv, mask_after_kv in zip(attention_mask_just_for_kv, attention_mask_after_kv):
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=dtype)
                position_start = mask_for_kv.sum()
                n_new_positions = mask_after_kv.sum()
                new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=dtype)
                if pad_side == "right":
                    sequence_position_ids[:n_new_positions] = new_positions
                else:
                    sequence_position_ids[-n_new_positions:] = new_positions
                position_ids.append(sequence_position_ids)
            position_ids = torch.stack(position_ids)
            model_kwargs["position_ids"] = position_ids

            # apply attention reduction
            if attention_reduction_ratio != 1.0:
                assert all(pair_i < len(x) for x in select_demonstration_intervals) # there is one more for gen_input_ids
                model_kwargs['demonstration_intervals'] = [x[:pair_i] for x in select_demonstration_intervals]
                model_kwargs['attention_reduction_ratio'] = attention_reduction_ratio

        else:
            # necessary for padside left and does not change when padside is right, idky
            position_ids = []
            for m in pair_attention_mask:
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=dtype)
                n_new_positions = m.sum()
                new_positions = torch.tensor(range(n_new_positions), device=device, dtype=dtype)
                if pad_side == "right":
                    sequence_position_ids[:n_new_positions] = new_positions
                else:
                    sequence_position_ids[-n_new_positions:] = new_positions
                position_ids.append(sequence_position_ids)
            position_ids = torch.stack(position_ids)
            model_kwargs["position_ids"] = position_ids

        # refine program
        model_out = model(**model_kwargs)
        assert model_out is not None

        if not short_context:
            # update program intervals
            update_based_on_avail(demonstration_intervals, select_demonstration_intervals, avail_mask, concat=False)

            # remove the end program of past key values
            old_key_values_len = prev_past_key_values[0][0].shape[2] if prev_past_key_values is not None else 0
            if pad_side == "right":
                new_past_key_values = tuple(
                    (
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[0], pair_input_ids_lens)]),
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[1], pair_input_ids_lens)]),
                    )
                    for layer_kv in model_out.past_key_values
                )
            else:
                if ntokens == 0:
                    new_past_key_values = model_out.past_key_values
                else:
                    new_past_key_values = tuple(
                        (
                            layer_kv[0][:, :, :-ntokens, :],
                            layer_kv[1][:, :, :-ntokens, :],
                        )
                        for layer_kv in model_out.past_key_values
                    )

            # format kv to batchsize x numlayer x 2, (nhead, seqlen, hiddendim)
            new_past_key_values = [
                [
                    (kv[0][batch_i], kv[1][batch_i])
                    for kv in new_past_key_values
                ]
                for batch_i in range(sum(avail_mask))
            ]
            # update key values
            if pair_i == 0:
                all_prev_past_key_values = new_past_key_values
            else:
                assert all_prev_past_key_values is not None
                update_based_on_avail(all_prev_past_key_values, new_past_key_values, avail_mask, concat=False)

            # format kv attention mask to batchsize x (seqlen,)
            new_prev_past_key_values_attention_mask = [x for x in pair_attention_mask_no_program_embed]
            # update kv attention mask
            if pair_i == 0:
                all_prev_past_key_values_attention_mask = new_prev_past_key_values_attention_mask
            else:
                assert all_prev_past_key_values_attention_mask is not None
                update_based_on_avail(all_prev_past_key_values_attention_mask, new_prev_past_key_values_attention_mask, avail_mask, concat=True)

            # check length
            for batch_i in range(batch_size):
                assert all_prev_past_key_values[batch_i][0][0].shape[1] == all_prev_past_key_values_attention_mask[batch_i].shape[0]
                n_pair = min(pair_i + 1, num_pairs[batch_i])
                assert all_prev_past_key_values[batch_i][0][0].shape[1] == n_pair * ntokens + sum(x[batch_i].shape[0] for x in inputs_embeds[:n_pair])

        # projection then sample new program prediction for all pairs except the last
        # sample program
        hidden_states = model_out.hidden_states[-1] # (batch_size, seq_len, hidden_dim)
        new_programs = extract_program_from_right_side(
            data=hidden_states,
            lens=pair_input_ids_lens,
            ntokens=ntokens,
            pad_side=pad_side,
        )
        # vae projection (no sample in eval)
        if vae_projection is not None:
            new_programs, _ = vae_projection(new_programs)
        # optionally use residual connection
        if not no_residual:
            new_programs = new_programs + prev_programs
        # optionally apply norm
        if program_norm is not None:
            new_programs = program_norm(new_programs)
        # optionally quantize
        if quantizer is not None:
            new_programs, _, _, _ = quantizer(program=new_programs, train_codebook_only=train_codebook_only)
        # update new program
        assert len(avail_mask) == len(all_prev_programs)
        assert sum(avail_mask) == len(new_programs)

        # format and update programs
        new_programs = [x for x in new_programs]
        update_based_on_avail(all_prev_programs, new_programs, avail_mask, concat=False)

        # save all intermediate programs of all tasks in batch
        update_based_on_avail(all_intermediate_programs, new_programs, avail_mask, concat=True)

    assert len(all_prev_programs) == batch_size
    assert len(set(p.shape[0] for p in all_prev_programs)) == 1

    if short_context:
        padded_past_key_values = None
        padded_past_key_values_attention_mask = None
    else:
        # prepare well padded past key values for generation, very ugly but it is what it is
        assert all_prev_past_key_values is not None and all_prev_past_key_values_attention_mask is not None
        max_seq_len = max(len(x) for x in all_prev_past_key_values_attention_mask)

        # pad key values
        padded_past_key_values = []
        n_layers = len(all_prev_past_key_values[0])
        for layer_i in range(n_layers):
            padded_kv = []
            for kv_i in range(2):
                layer_data = [x[layer_i][kv_i] for x in all_prev_past_key_values] # batchsize x (nhead, seqlen, hiddendim)
                for batch_i, layer_data_i in enumerate(layer_data):
                    pad_len = max_seq_len - layer_data_i.shape[1]
                    if pad_len > 0:
                        pads = torch.zeros(
                            (layer_data_i.shape[0], pad_len, layer_data_i.shape[2]),
                            device=layer_data_i.device, dtype=layer_data_i.dtype
                        )
                        if kv_pad_side == 'left':
                            layer_data[batch_i] = torch.cat([pads, layer_data_i], dim=1)
                        else:
                            layer_data[batch_i] = torch.cat([layer_data_i, pads], dim=1)
                padded_kv.append(torch.stack(layer_data))
            padded_past_key_values.append(tuple(padded_kv))
        padded_past_key_values = tuple(padded_past_key_values)

        # pad key values attention mask (adjust pad program intervals here too)
        for batch_i, mask in enumerate(all_prev_past_key_values_attention_mask):
            pad_len = max_seq_len - len(mask)
            if pad_len > 0:
                pads = torch.zeros((pad_len,), device=mask.device, dtype=mask.dtype)
                if kv_pad_side == 'left':
                    all_prev_past_key_values_attention_mask[batch_i] = torch.cat([pads, mask], dim=0)
                    demonstration_intervals[batch_i] = [(s + pad_len, e + pad_len) for s, e in demonstration_intervals[batch_i]]
                else:
                    all_prev_past_key_values_attention_mask[batch_i] = torch.cat([mask, pads], dim=0)
        padded_past_key_values_attention_mask = torch.stack(all_prev_past_key_values_attention_mask)

    # get final programs
    final_programs = [x[-1] for x in all_intermediate_programs]
    final_programs = torch.stack(final_programs)

    return final_programs, demonstration_intervals, padded_past_key_values, padded_past_key_values_attention_mask


def model_loss(
    # model
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    program_dropout: nn.Dropout,
    tokenizer: ARCTokenizer,
    accelerator: Accelerator,
    # data
    input_ids: List[torch.Tensor],
    attention_mask: List[torch.Tensor],
    label_ids: List[torch.Tensor],
    input_ids_lens: List[List[int]],
    num_pairs: List[int],
    is_same: bool,
    task_identifiers: List[str],
    # others
    ntokens: int,
    pad_side: str,
    kl_loss_lambda_scheduler: LambdaScheduler,
    codebook_loss_lambda_scheduler: LambdaScheduler,
    commitment_loss_lambda_scheduler: LambdaScheduler,
    program_loss_lambda_scheduler: LambdaScheduler,
    consistency_loss_lambda_scheduler: LambdaScheduler,
    invar_loss_lambda_scheduler: LambdaScheduler,
    global_step: int,
    no_residual: bool,
    no_discrete_prior: bool,
    program_type: str,
    train_codebook_only: bool,
    ar_gradient_checkpointing: bool,
    program_noise_std: float,
    subset_kl: bool,
    long_context_checkpointing_threshold: int,
    token_weighted_loss: bool,
    weird_cast: bool,
    full_demonstration_dropout: bool,
    partial_demonstration_dropout: bool,
    loss_on_first: bool,
    debug: bool,
    contrastive_loss: ContrastiveLoss,
    short_context: bool,
    attention_reduction_ratio: float,
    program_cache: Optional[ProgramCache],
    prior_embed_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:

    # # debugging: make sure middle attention 0s do not affect hidden states -> works for both float32 and 16 on rtx8000 without flashattn
    # batch_size = len(input_ids[0])
    # dtype, device = input_ids[0].dtype, input_ids[0].device

    # dummy_inputs_embeds = torch.randn((batch_size, 5, 2048), dtype=torch.float32, device=device)
    # dummy_attention_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long, device=device)[None, ...].expand(batch_size, -1)

    # dummy_out1 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )

    # dummy_inputs_embeds[:, -2:, :] = torch.randn_like(dummy_inputs_embeds[:, -2:, :], dtype=torch.float32, device=device)
    # dummy_out2 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )
    # print('right pad dummy1 vs dummy2 hidden diff', (dummy_out1.hidden_states[-1][:, :-2, :] - dummy_out2.hidden_states[-1][:, :-2, :]).abs().max().item())

    # dummy_inputs_embeds = dummy_inputs_embeds[:, :-2, :]
    # dummy_attention_mask = torch.tensor([1, 1, 1], dtype=torch.long, device=device)[None, ...].expand(batch_size, -1)
    # dummy_out3 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )
    # print('right pad dummy1 vs dummy3 hidden diff', (dummy_out1.hidden_states[-1][:, :-2, :] - dummy_out3.hidden_states[-1]).abs().max().item())









    # # debugging: make sure middle attention 0s do not affect hidden states -> works for both float32 and 16 on rtx8000 without flashattn
    # batch_size = len(input_ids[0])
    # dtype, device = input_ids[0].dtype, input_ids[0].device

    # dummy_inputs_embeds = torch.randn((batch_size, 5, 2048), dtype=torch.float32, device=device)
    # dummy_attention_mask = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device)[None, ...].expand(batch_size, -1)

    # dummy_out1 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )

    # dummy_inputs_embeds[:, :2, :] = torch.randn_like(dummy_inputs_embeds[:, :2, :], dtype=torch.float32, device=device)
    # dummy_out2 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )
    # print('left pad dummy1 vs dummy2 hidden diff', (dummy_out1.hidden_states[-1][:, 2:, :] - dummy_out2.hidden_states[-1][:, 2:, :]).abs().max().item())

    # dummy_inputs_embeds = dummy_inputs_embeds[:, 2:, :]
    # dummy_attention_mask = torch.tensor([1, 1, 1], dtype=torch.long, device=device)[None, ...].expand(batch_size, -1)
    # dummy_out3 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )
    # print('left pad dummy1 vs dummy3 hidden diff', (dummy_out1.hidden_states[-1][:, 2:, :] - dummy_out3.hidden_states[-1]).abs().max().item())






    # # debugging: make sure middle attention 0s do not affect hidden states -> works for both float32 and 16 on rtx8000 without flashattn
    # batch_size = len(input_ids[0])
    # dtype, device = input_ids[0].dtype, input_ids[0].device

    # dummy_inputs_embeds = torch.randn((batch_size, 5, 2048), dtype=torch.float32, device=device)
    # dummy_attention_mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long, device=device)[None, ...].expand(batch_size, -1)
    # position_ids = torch.tensor([0, 0, 1, 0, 2], dtype=torch.long, device=device)[None, ...]

    # dummy_out1 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     position_ids=position_ids,
    #     output_hidden_states=True,
    # )

    # dummy_inputs_embeds[:, 1, :] = torch.randn_like(dummy_inputs_embeds[:, 1, :], dtype=torch.float32, device=device)
    # dummy_inputs_embeds[:, 3, :] = torch.randn_like(dummy_inputs_embeds[:, 3, :], dtype=torch.float32, device=device)
    # dummy_out2 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     position_ids=position_ids,
    #     output_hidden_states=True,
    # )
    # print('mid pad dummy1 vs dummy2 hidden diff', (dummy_out1.hidden_states[-1][:, torch.tensor([0, 2, 4]), :] - dummy_out2.hidden_states[-1][:, torch.tensor([0, 2, 4]), :]).abs().max().item())

    # dummy_inputs_embeds = dummy_inputs_embeds[:, torch.tensor([0, 2, 4]), :]
    # dummy_attention_mask = torch.tensor([1, 1, 1], dtype=torch.long, device=device)[None, ...].expand(batch_size, -1)
    # dummy_out3 = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     output_hidden_states=True,
    # )
    # print('mid pad dummy1 vs dummy3 hidden diff', (dummy_out1.hidden_states[-1][:, torch.tensor([0, 2, 4]), :] - dummy_out3.hidden_states[-1]).abs().max().item())
    # # finally ~1e-7






    # # debugging: simulate left padding's effect on kv cache, conclusion is that position_ids IS needed
    # dtype, device = input_ids[0].dtype, input_ids[0].device

    # dummy_inputs_embeds1 = torch.randn((2, 7, 2048), dtype=torch.float32, device=device)
    # dummy_attention_mask1 = torch.tensor([[0, 0, 1, 0, 1, 0, 1],
    #                                       [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long, device=device)
    # position_ids = torch.tensor([[0, 0, 0, 0, 1, 0, 2],
    #                              [0, 1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
    # dummy_out1 = model(
    #     inputs_embeds=dummy_inputs_embeds1,
    #     attention_mask=dummy_attention_mask1,
    #     position_ids=position_ids,
    #     output_hidden_states=True,
    # )

    # dummy_inputs_embeds2 = torch.randn((2, 7, 2048), dtype=torch.float32, device=device)
    # dummy_inputs_embeds2[1] = dummy_inputs_embeds1[1]
    # dummy_inputs_embeds2[0, 4] = dummy_inputs_embeds1[0, 2]
    # dummy_inputs_embeds2[0, 5] = dummy_inputs_embeds1[0, 4]
    # dummy_inputs_embeds2[0, 6] = dummy_inputs_embeds1[0, 6]
    # dummy_attention_mask2 = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long, device=device)
    # position_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 2], [0, 1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
    # dummy_out2 = model(
    #     inputs_embeds=dummy_inputs_embeds2,
    #     attention_mask=dummy_attention_mask2,
    #     position_ids=position_ids,
    #     output_hidden_states=True,
    # )

    # embeds1 = dummy_inputs_embeds1.masked_select(dummy_attention_mask1.bool()[:, :, None])
    # embeds2 = dummy_inputs_embeds2.masked_select(dummy_attention_mask2.bool()[:, :, None])
    # assert (embeds1 - embeds2).max() == 0.0

    # assert dummy_out1.past_key_values[0][0].shape == dummy_out2.past_key_values[0][0].shape
    # max_kv_diff = 0.0
    # for i in range(len(dummy_out1.past_key_values)):
    #     for j in range(2):
    #         kv1 = dummy_out1.past_key_values[i][j]
    #         kv2 = dummy_out2.past_key_values[i][j]
    #         kv1 = kv1.masked_select(dummy_attention_mask1.bool()[:, None, :, None])
    #         kv2 = kv2.masked_select(dummy_attention_mask2.bool()[:, None, :, None])
    #         max_kv_diff = max(max_kv_diff, (kv1 - kv2).abs().max().item())
    #         print(i, j, (kv1 - kv2).abs().max().item())
    # print('mid pad dummy1 vs dummy2 kv diff', max_kv_diff)
    # breakpoint()
    # # finally ~1e-7




    # # debugging: simulate right padding's effect on kv cache, conclusion is that position_ids is NOT needed
    # dtype, device = input_ids[0].dtype, input_ids[0].device

    # dummy_inputs_embeds = torch.randn((2, 7, 2048), dtype=torch.float32, device=device)
    # dummy_attention_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0],
    #                                      [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long, device=device)
    # position_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 2],
    #                              [0, 1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
    # dummy_out = model(
    #     inputs_embeds=dummy_inputs_embeds,
    #     attention_mask=dummy_attention_mask,
    #     position_ids=position_ids,
    #     output_hidden_states=True,
    # )

    # dummy_inputs_embeds2 = torch.randn((2, 7, 2048), dtype=torch.float32, device=device)
    # dummy_inputs_embeds2[1] = dummy_inputs_embeds[1]
    # dummy_inputs_embeds2[0, 0] = dummy_inputs_embeds[0, 0]
    # dummy_inputs_embeds2[0, 1] = dummy_inputs_embeds[0, 2]
    # dummy_inputs_embeds2[0, 2] = dummy_inputs_embeds[0, 4]
    # dummy_attention_mask2 = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long, device=device)
    # position_ids = torch.tensor([[0, 1, 2, 2, 2, 2, 2], [0, 1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
    # dummy_out2 = model(
    #     inputs_embeds=dummy_inputs_embeds2,
    #     attention_mask=dummy_attention_mask2,
    #     position_ids=position_ids,
    #     output_hidden_states=True,
    # )

    # embeds1 = dummy_inputs_embeds.masked_select(dummy_attention_mask.bool()[:, :, None])
    # embeds2 = dummy_inputs_embeds2.masked_select(dummy_attention_mask2.bool()[:, :, None])
    # assert (embeds1 - embeds2).max() == 0.0

    # assert dummy_out.past_key_values[0][0].shape == dummy_out2.past_key_values[0][0].shape
    # max_kv_diff = 0.0
    # for i in range(len(dummy_out.past_key_values)):
    #     for j in range(2):
    #         kv1 = dummy_out.past_key_values[i][j]
    #         kv2 = dummy_out2.past_key_values[i][j]
    #         kv1 = kv1.masked_select(dummy_attention_mask.bool()[:, None, :, None])
    #         kv2 = kv2.masked_select(dummy_attention_mask2.bool()[:, None, :, None])
    #         max_kv_diff = max(max_kv_diff, (kv1 - kv2).abs().max().item())
    #         print(i, j, (kv1 - kv2).abs().max().item())
    # print('mid pad dummy1 vs dummy2 kv diff', max_kv_diff)
    # breakpoint()
    # # finally ~1e-7




    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # all samples should have same number of pairs
    assert len(set(num_pairs)) == 1
    n_pairs = num_pairs[0]
    # assert n_pairs >= 3 # at least 2 train and 1 test
    assert n_pairs == len(input_ids) # input_ids

    assert len(set(len(ids) for ids in input_ids)) == 1 # batch size uniform across pair_idx
    batch_size = len(input_ids[0])
    dtype, device = input_ids[0].dtype, input_ids[0].device

    # reused tensors
    pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device))
    inputs_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in input_ids]

    # losses
    ce_losses = []
    kl_losses = []
    codebook_losses = []
    commitment_losses = []
    perplexitys = []

    # loss lambdas
    kl_loss_lambda = kl_loss_lambda_scheduler.get_lambda(step=global_step)
    codebook_loss_lambda = codebook_loss_lambda_scheduler.get_lambda(step=global_step)
    commitment_loss_lambda = commitment_loss_lambda_scheduler.get_lambda(step=global_step)
    program_loss_lambda = program_loss_lambda_scheduler.get_lambda(step=global_step)
    consistency_loss_lambda = consistency_loss_lambda_scheduler.get_lambda(step=global_step)
    invar_loss_lambda = invar_loss_lambda_scheduler.get_lambda(step=global_step)

    def get_prior():
        # apply program norm to prior
        prior_inputs_embeds = prior_embeddings("dummy")[None, ...]
        if program_norm is not None:
            prior_inputs_embeds = program_norm(prior_inputs_embeds)
        # NOTE: no dropout or noise injection to prior
        # quantize prior
        if (quantizer is not None) and not no_discrete_prior:
            prior_inputs_embeds, codebook_loss, commitment_loss, perplexity = quantizer(prior_inputs_embeds, train_codebook_only=train_codebook_only)
            codebook_losses.append(codebook_loss)
            commitment_losses.append(commitment_loss)
            perplexitys.append(perplexity)
        return prior_inputs_embeds[0]

    # for training, initial program don't have to be prior embeddding is we cache them
    if program_cache is None:
        prev_programs = get_prior()[None, ...].expand(batch_size, -1, -1)
    else:
        prev_programs = []
        prior_i = -1
        for task_i, task_identifier in enumerate(task_identifiers):
            if (task_identifier not in program_cache.cache) or (program_cache.select_prior_rng.uniform(0, 1) < prior_embed_ratio):
                # use prior (make sure get prior is only called once)
                if prior_i > -1:
                    prev_programs.append(prev_programs[prior_i])
                else:
                    prev_programs.append(get_prior())
                    prior_i = task_i
            else:
                # sample from cache
                prev_programs.append(program_cache.sample_from_cache(task_identifier, accelerator.device))
        prev_programs = torch.stack(prev_programs)

    # previous program of each sample in batch
    prev_program_mus = torch.zeros_like(prev_programs, device=device, dtype=prev_programs.dtype)
    prev_program_logvars = torch.zeros_like(prev_programs, device=device, dtype=prev_programs.dtype)
    prev_past_key_values = None
    prev_past_key_values_attention_mask = None

    # save all programs
    do_save_programs = (debug or program_type != 'none' or invar_loss_lambda > 0.0 or consistency_loss_lambda > 0.0 or (program_cache is not None))
    saved_all_programs = []
    if do_save_programs:
        saved_all_programs = [prev_programs]

    # precompute demonstration intervals for these three tries
    demonstration_intervals = [[] for _ in range(batch_size)]
    if full_demonstration_dropout > 0.0 or partial_demonstration_dropout > 0.0 or attention_reduction_ratio != 1.0:
        start = ntokens # keep track of start of each demonstration pair
        for pair_j, lens in enumerate(input_ids_lens[:-1]):
            if pair_j > 0:
                start += attention_mask[pair_j - 1].shape[1] + ntokens
            max_l = attention_mask[pair_j].shape[1]
            for batch_i, l in enumerate(lens):
                s = start if pad_side == 'right' else start + max_l - l
                demonstration_intervals[batch_i].append((s, s + l))

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_label_ids, pair_input_ids_lens) in enumerate(zip(inputs_embeds, attention_mask, label_ids, input_ids_lens)):
        # STEP 1: prepend the last predicted program for all pairs except the first
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((batch_size, ntokens), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )
        if pair_i > 0 or loss_on_first:
            pair_label_ids = insert_based_on_sides(
                data=pair_label_ids,
                to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=dtype),
                lens=pair_input_ids_lens,
                insert_side="left",
                pad_side=pad_side,
                pad_id=-100,
            )

        # update lens to reflect the extra program
        pair_input_ids_lens = [l + ntokens for l in pair_input_ids_lens]

        # save attention mask
        pair_attention_mask_no_program_embed = pair_attention_mask.detach().clone()

        # STEP 2: append new program input to the right for all pairs except the last
        if (pair_i < n_pairs - 1) or (program_cache is not None):
            emb = program_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
            pair_inputs_embeds = insert_based_on_sides(
                data=pair_inputs_embeds,
                to_insert=(emb.to(torch.bfloat16) if weird_cast else emb),
                lens=pair_input_ids_lens,
                insert_side="right",
                pad_side=pad_side,
                pad_id=pad_embeds,
            )
            pair_attention_mask = insert_based_on_sides(
                data=pair_attention_mask,
                to_insert=torch.full((batch_size, ntokens), 1, device=device, dtype=dtype),
                lens=pair_input_ids_lens,
                insert_side="right",
                pad_side=pad_side,
                pad_id=0,
            )
            if pair_i > 0 or loss_on_first:
                pair_label_ids = insert_based_on_sides(
                    data=pair_label_ids,
                    to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=dtype),
                    lens=pair_input_ids_lens,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=-100,
                )

        # STEP 3: refine program (no output grid label for first pair)

        # attention mask should span to the past kvs
        if pair_i > 0 and not short_context: # no loss on first here because we just keep pair_attention_mask
            assert prev_past_key_values is not None
            assert prev_past_key_values_attention_mask is not None
            assert prev_past_key_values_attention_mask.shape[1] == prev_past_key_values[0][0].shape[2]
            pair_attention_mask = torch.cat([prev_past_key_values_attention_mask, pair_attention_mask], dim=1)

        # model forward and get loss
        model_kwargs = {
            "inputs_embeds": pair_inputs_embeds,
            "attention_mask": pair_attention_mask,
            "output_hidden_states": (pair_i < n_pairs - 1) or (program_cache is not None), # not the last pair
            "use_cache": not short_context, # need to generate or use kv cache
        }
        if pair_i > 0 or loss_on_first:
            model_kwargs["labels"] = pair_label_ids

        if pair_i > 0 and not short_context:
            assert prev_past_key_values is not None
            model_kwargs["past_key_values"] = prev_past_key_values

            # program dropout just before passing into the model, ugly
            if full_demonstration_dropout > 0.0 or partial_demonstration_dropout > 0.0:
                pair_attention_mask_with_dropout = pair_attention_mask.detach().clone()
                for batch_i, (m, intervals) in enumerate(zip(pair_attention_mask_with_dropout, demonstration_intervals)):
                    for s, e in intervals[:pair_i]:
                        # full demonstration dropout
                        if torch.rand(1) < full_demonstration_dropout:
                            m[s: e] = 0
                        # partial demonstration dropout
                        dropout_mask = torch.rand(m[s: e].shape, device=device) < partial_demonstration_dropout
                        m[s: e] *= dropout_mask
                # replace mask
                model_kwargs["attention_mask"] = pair_attention_mask_with_dropout

            # build position ids (does NOT depend on dropout)
            attention_mask_just_for_kv = pair_attention_mask[:, :prev_past_key_values[0][0].shape[2]]
            attention_mask_after_kv = pair_attention_mask[:, prev_past_key_values[0][0].shape[2]:]
            position_ids = []
            for mask_for_kv, mask_after_kv in zip(attention_mask_just_for_kv, attention_mask_after_kv):
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=dtype)
                position_start = mask_for_kv.sum()
                n_new_positions = mask_after_kv.sum()
                new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=dtype)
                if pad_side == "right":
                    sequence_position_ids[:n_new_positions] = new_positions
                else:
                    sequence_position_ids[-n_new_positions:] = new_positions
                position_ids.append(sequence_position_ids)
            position_ids = torch.stack(position_ids)
            model_kwargs["position_ids"] = position_ids

            # apply attention reduction
            if attention_reduction_ratio != 1.0:
                model_kwargs['demonstration_intervals'] = [x[:pair_i] for x in demonstration_intervals]
                model_kwargs['attention_reduction_ratio'] = attention_reduction_ratio

        else:
            # necessary for padside left and does not change when padside is right, idky
            position_ids = []
            for m in pair_attention_mask:
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=dtype)
                n_new_positions = m.sum()
                new_positions = torch.tensor(range(n_new_positions), device=device, dtype=dtype)
                if pad_side == "right":
                    sequence_position_ids[:n_new_positions] = new_positions
                else:
                    sequence_position_ids[-n_new_positions:] = new_positions
                position_ids.append(sequence_position_ids)
            position_ids = torch.stack(position_ids)
            model_kwargs["position_ids"] = position_ids

        can_checkpoint = sum(x.shape[1] for x in inputs_embeds[:pair_i+1]) > long_context_checkpointing_threshold
        if ar_gradient_checkpointing and can_checkpoint:
            model_out = checkpoint.checkpoint(model, **model_kwargs, use_reentrant=False)
        else:
            model_out = model(**model_kwargs)

        if pair_i > 0 or loss_on_first:
            ce_losses.append(model_out.loss) # type: ignore

        # STEP 4: update kv
        if (pair_i < n_pairs - 1) and not short_context:
            assert model_out is not None

            # remove end program from kv cache
            old_key_values_len = prev_past_key_values[0][0].shape[2] if prev_past_key_values is not None else 0
            if pad_side == "right":
                new_past_key_values = tuple(
                    (
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[0], pair_input_ids_lens)]),
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[1], pair_input_ids_lens)]),
                    )
                    for layer_kv in model_out.past_key_values
                )
            else:
                if ntokens == 0:
                    new_past_key_values = model_out.past_key_values
                else:
                    new_past_key_values = tuple(
                        (
                            layer_kv[0][:, :, :-ntokens, :],
                            layer_kv[1][:, :, :-ntokens, :],
                        )
                        for layer_kv in model_out.past_key_values
                    )
            # update kv cache
            if prev_past_key_values is not None:
                assert prev_past_key_values[0][0].shape[2] + inputs_embeds[pair_i].shape[1] + ntokens == new_past_key_values[0][0].shape[2]
                # debugging: check if past key values are changed
                # seq_len_to_check = prev_past_key_values[0][0].shape[2]
                # mean_kv_relative_diff = 0.0
                # for i in range(len(prev_past_key_values)):
                #     for j in range(2):
                #         assert torch.equal(prev_past_key_values[i][j], new_past_key_values[i][j][:, :, :seq_len_to_check, :])
            prev_past_key_values = new_past_key_values

            # update attention mask
            if pair_i == 0:
                prev_past_key_values_attention_mask = pair_attention_mask_no_program_embed
            else:
                assert prev_past_key_values_attention_mask is not None
                prev_past_key_values_attention_mask = torch.cat([prev_past_key_values_attention_mask, pair_attention_mask_no_program_embed], dim=1)
            # check length
            assert prev_past_key_values[0][0].shape[2] == prev_past_key_values_attention_mask.shape[1]
            assert prev_past_key_values[0][0].shape[2] == (pair_i + 1) * ntokens + sum(x.shape[1] for x in inputs_embeds[:pair_i+1])

        # STEP 5: projection then sample new program prediction for all pairs except the last
        if (pair_i < n_pairs - 1) or (program_cache is not None):
            # extract and sample program
            hidden_states = model_out.hidden_states[-1] # (batch_size, seq_len, hidden_dim) # type: ignore
            new_programs = extract_program_from_right_side(
                data=hidden_states,
                lens=pair_input_ids_lens,
                ntokens=ntokens,
                pad_side=pad_side,
            )

            # vae projection (no sample in eval)
            new_program_mus, new_program_logvars = None, None
            if vae_projection is not None:
                new_program_mus, new_program_logvars = vae_projection(new_programs)
                new_programs = sample_from_vae(mu=new_program_mus, logvar=new_program_logvars)
            # dropout
            new_programs = program_dropout(new_programs)
            # inject noise
            new_programs = new_programs + program_noise_std * torch.randn_like(new_programs, dtype=new_programs.dtype, device=device)
            # use residual connection
            if not no_residual:
                new_programs = new_programs + prev_programs
            # apply norm
            if program_norm is not None:
                new_programs = program_norm(new_programs)
            # quantize
            if quantizer is not None:
                new_programs, codebook_loss, commitment_loss, perplexity = quantizer(program=new_programs, train_codebook_only=train_codebook_only)
                codebook_losses.append(codebook_loss)
                commitment_losses.append(commitment_loss)
                perplexitys.append(perplexity)

            # save and update new program
            prev_programs = new_programs
            if do_save_programs:
                saved_all_programs.append(prev_programs)

            # kl loss
            if kl_loss_lambda != 0:
                assert new_program_mus is not None and new_program_logvars is not None
                kl_losses.append(compute_kl_loss(
                    mu1=new_program_mus,
                    mu2=prev_program_mus,
                    logvar1=new_program_logvars,
                    logvar2=prev_program_logvars,
                ))
                if subset_kl:
                    # update new program mus and logvars
                    prev_program_mus = new_program_mus
                    prev_program_logvars = new_program_logvars

    # program loss
    program_loss = torch.tensor(0.0, device=device)
    if program_type != 'none':
        # e.g., p0 x0 y0 p1 x1 y1 p2 x2 y2 (n_pairs=3)
        assert len(saved_all_programs) == n_pairs and len(saved_all_programs) >= 2
        # select program and pair
        if program_type == 'concat':
            # concatenate all programs and a random pair
            select_program = torch.cat(saved_all_programs, dim=1)
            select_idx = int(torch.randint(low=0, high=n_pairs, size=(1,)).item())
        else:
            # select random program and a random pair AFTER it
            program_idx = int(torch.randint(low=2, high=n_pairs, size=(1,)).item()) # do not select the first two
            select_program = saved_all_programs[program_idx]
            select_idx = int(torch.randint(low=program_idx, high=n_pairs, size=(1,)).item())
        # insert program
        program_len = select_program.shape[1]
        select_inputs_embeds = insert_based_on_sides(
            data=inputs_embeds[select_idx],
            to_insert=(select_program.to(torch.bfloat16) if weird_cast else select_program),
            lens=input_ids_lens[select_idx],
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        select_attention_mask = insert_based_on_sides(
            data=attention_mask[select_idx],
            to_insert=torch.full((batch_size, program_len), 1, device=device, dtype=dtype),
            lens=input_ids_lens[select_idx],
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )
        select_label_ids = insert_based_on_sides(
            data=label_ids[select_idx],
            to_insert=torch.full((batch_size, program_len), -100, device=device, dtype=dtype),
            lens=input_ids_lens[select_idx],
            insert_side="left",
            pad_side=pad_side,
            pad_id=-100,
        )
        # get program_loss (no checkpointing for now)
        model_kwargs = {
            "inputs_embeds": select_inputs_embeds,
            "attention_mask": select_attention_mask,
            "labels": select_label_ids,
        }
        model_out = model(**model_kwargs)
        program_loss = model_out.loss
        program_loss /= len(ce_losses) # normalize based on num pairs to not dominate

    # consistency loss
    consistency_loss = torch.tensor(0.0, device=device)
    if consistency_loss_lambda > 0.0:
        for batch_i in range(batch_size):
            task_programs = torch.stack([x[batch_i] for x in saved_all_programs[1:]]) # ignore prior program
            task_programs = task_programs.flatten(start_dim=1) # (nprogram, program numel)
            task_mean_program = task_programs.mean(dim=0)[None, ...]
            cos_sim = torch.cosine_similarity(task_programs, task_mean_program, dim=-1)
            consistency_loss += (1.0 - cos_sim).mean() / batch_size

    # invar loss
    invar_loss = torch.tensor(0.0, device=device)
    if invar_loss_lambda > 0.0:
        assert all(x.shape[0] == 2 for x in saved_all_programs)
        programs1 = torch.stack([x[0] for x in saved_all_programs[1:]]) # skip prior
        programs2 = torch.stack([x[1] for x in saved_all_programs[1:]]) # skip prior
        invar_loss = contrastive_loss(programs1, programs2, is_same)

    # ce loss might be token weighted
    if token_weighted_loss:
        if loss_on_first:
            token_weights = [(pair_label_ids != -100).sum().item() for pair_label_ids in label_ids]
        else:
            token_weights = [(pair_label_ids != -100).sum().item() for pair_label_ids in label_ids[1:]]
        token_weights = torch.tensor(token_weights, device=device) / sum(token_weights)
        assert len(ce_losses) == len(token_weights)
        ce_loss = sum(ce_loss * token_weight for ce_loss, token_weight in zip(ce_losses, token_weights))
    else:
        ce_loss = sum(ce_losses) / len(ce_losses)

    # aggregate losses
    kl_loss = sum(kl_losses) / len(kl_losses) if len(kl_losses) > 0 else torch.tensor(0.0, device=device)
    codebook_loss = sum(codebook_losses) / len(codebook_losses) if len(codebook_losses) > 0 else torch.tensor(0.0, device=device)
    commitment_loss = sum(commitment_losses) / len(commitment_losses) if len(commitment_losses) > 0 else torch.tensor(0.0, device=device)
    total_loss = ce_loss + kl_loss_lambda * kl_loss + codebook_loss * codebook_loss_lambda + commitment_loss_lambda * commitment_loss + program_loss_lambda * program_loss + invar_loss_lambda * invar_loss + consistency_loss_lambda * consistency_loss

    # logging perplexity for debugging vqvae
    perplexity = sum(perplexitys) / len(perplexitys) if len(perplexitys) > 0 else torch.tensor(0.0, device=device)







    # # DEBUG: now we have all the programs along the way, let's aggregate all context into a single long context
    # #        and make sure the output is the same, try this for train_pad_side left and right
    # assert len(saved_all_programs) == len(inputs_embeds) == len(attention_mask) == len(label_ids)
    # single_inputs_embeds = []
    # single_attention_mask = []
    # single_label_ids = []
    # for pair_i, (predicted_program, pair_inputs_embeds, pair_attention_mask, pair_label_ids, pair_input_ids_lens) in enumerate(zip(saved_all_programs, inputs_embeds, attention_mask, label_ids, input_ids_lens)):
    #     single_inputs_embeds.append(insert_based_on_sides(
    #         data=pair_inputs_embeds,
    #         to_insert=predicted_program,
    #         lens=pair_input_ids_lens,
    #         insert_side="left",
    #         pad_side=pad_side,
    #         pad_id=pad_embeds,
    #     ))
    #     single_attention_mask.append(insert_based_on_sides(
    #         data=pair_attention_mask,
    #         to_insert=torch.full((batch_size, ntokens), 1, device=device, dtype=dtype),
    #         lens=pair_input_ids_lens,
    #         insert_side="left",
    #         pad_side=pad_side,
    #         pad_id=0,
    #     ))
    #     if pair_i == 0 and not loss_on_first:
    #         pair_label_ids = torch.full(pair_label_ids.shape, -100, dtype=dtype, device=device)
    #     single_label_ids.append(insert_based_on_sides(
    #         data=pair_label_ids,
    #         to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=dtype),
    #         lens=pair_input_ids_lens,
    #         insert_side="left",
    #         pad_side=pad_side,
    #         pad_id=-100,
    #     ))

    # single_inputs_embeds1 = torch.cat(single_inputs_embeds, dim=1)
    # single_attention_mask1 = torch.cat(single_attention_mask, dim=1)
    # single_label_ids1 = torch.cat(single_label_ids, dim=1)
    # assert single_inputs_embeds1.shape[:2] == single_attention_mask1.shape[:2] == single_label_ids1.shape[:2]

    # assert prev_past_key_values is not None
    # assert prev_past_key_values_attention_mask is not None

    # # assertions
    # seq_len_to_check = prev_past_key_values_attention_mask.shape[1] # does not have final pair
    # assert seq_len_to_check == sum(x.shape[1] for x in inputs_embeds[:-1]) + (len(inputs_embeds) - 1) * ntokens
    # assert prev_past_key_values[0][0].shape[2] == seq_len_to_check
    # assert torch.equal(single_attention_mask1[:, :seq_len_to_check], prev_past_key_values_attention_mask)

    # # construct position ids
    # position_ids1 = []
    # for m in single_attention_mask1:
    #     sequence_position_ids1 = torch.cumsum(m, dim=0) - 1
    #     sequence_position_ids1[sequence_position_ids1 < 0] = 0
    #     position_ids1.append(sequence_position_ids1)
    # position_ids1 = torch.stack(position_ids1)
    # assert position_ids1.shape == single_attention_mask1.shape

    # # single model forward
    # model_out_1 = model(
    #     inputs_embeds=single_inputs_embeds1,
    #     attention_mask=single_attention_mask1,
    #     labels=single_label_ids1,
    #     position_ids=position_ids1,
    #     output_hidden_states=True,
    # )
    # single_past_key_values = model_out_1.past_key_values
    # assert seq_len_to_check + inputs_embeds[-1].shape[1] + ntokens == single_past_key_values[0][0].shape[2]

    # kv_max_diff = 0.0
    # for i in range(len(prev_past_key_values)):
    #     for j in range(2):
    #         old_kv = prev_past_key_values[i][j]
    #         new_kv = single_past_key_values[i][j].detach().clone()[:, :, :seq_len_to_check, :]
    #         old_kv = old_kv.masked_select(prev_past_key_values_attention_mask.bool()[:, None, :, None])
    #         new_kv = new_kv.masked_select(prev_past_key_values_attention_mask.bool()[:, None, :, None])
    #         kv_max_diff = max(kv_max_diff, (new_kv - old_kv).abs().max().item())

    # print('kv max diff', kv_max_diff)
    # print('loss diff', (model_out_1.loss - ce_loss).abs().item())







    # # DEBUG: even more extreme than above, we strip all intermediate paddings in the long context and just
    # #        do a big manual padding at the end, try this for train_pad_side left and right
    # assert len(saved_all_programs) == len(input_ids) == len(attention_mask) == len(label_ids)
    # single_inputs_embeds = []
    # single_attention_mask = []
    # single_label_ids = []

    # new_program_idxs = [[] for _ in range(batch_size)]
    # for batch_i in range(batch_size):
    #     # get programs and inputs for task
    #     task_programs = [x[batch_i] for x in saved_all_programs]
    #     task_input_ids = [x[batch_i] for x in input_ids]
    #     task_attention_mask = [x[batch_i] for x in attention_mask]
    #     task_label_ids = [x[batch_i] for x in label_ids]
    #     task_input_ids_lens = [x[batch_i] for x in input_ids_lens]
    #     assert len(task_programs) == len(task_input_ids) == len(task_attention_mask) == len(task_label_ids) == len(task_input_ids_lens) == n_pairs
    #     assert sum(x.shape[0] for x in task_input_ids) == sum(x.shape[0] for x in task_attention_mask)

    #     # no loss on first
    #     if not loss_on_first:
    #         label1 = task_label_ids[0]
    #         task_label_ids[0] = torch.full(label1.shape, -100, dtype=label1.dtype, device=label1.device)

    #     # strip individual train paddings (make sure they actually are paddings)
    #     if pad_side == 'right':
    #         input_ids_pads = torch.cat([x[l:] for x, l in zip(task_input_ids, task_input_ids_lens)])
    #         assert set(input_ids_pads.tolist()).issubset({tokenizer.pad_token_id})
    #         attention_mask_pads = torch.cat([x[l:] for x, l in zip(task_attention_mask, task_input_ids_lens)])
    #         assert set(attention_mask_pads.tolist()).issubset({0})
    #         label_ids_pads = torch.cat([x[l:] for x, l in zip(task_label_ids, task_input_ids_lens)])
    #         assert set(label_ids_pads.tolist()).issubset({-100})
    #         task_input_ids = [x[:l] for x, l in zip(task_input_ids, task_input_ids_lens)]
    #         task_attention_mask = [x[:l] for x, l in zip(task_attention_mask, task_input_ids_lens)]
    #         task_label_ids = [x[:l] for x, l in zip(task_label_ids, task_input_ids_lens)]
    #     else:
    #         input_ids_pads = torch.cat([x[:-l] for x, l in zip(task_input_ids, task_input_ids_lens)])
    #         assert set(input_ids_pads.tolist()).issubset({tokenizer.pad_token_id})
    #         attention_mask_pads = torch.cat([x[:-l] for x, l in zip(task_attention_mask, task_input_ids_lens)])
    #         assert set(attention_mask_pads.tolist()).issubset({0})
    #         label_ids_pads = torch.cat([x[:-l] for x, l in zip(task_label_ids, task_input_ids_lens)])
    #         assert set(label_ids_pads.tolist()).issubset({-100})
    #         task_input_ids = [x[-l:] for x, l in zip(task_input_ids, task_input_ids_lens)]
    #         task_attention_mask = [x[-l:] for x, l in zip(task_attention_mask, task_input_ids_lens)]
    #         task_label_ids = [x[-l:] for x, l in zip(task_label_ids, task_input_ids_lens)]
    #     assert all(x.shape == y.shape == z.shape for x, y, z in zip(task_input_ids, task_attention_mask, task_label_ids))

    #     # get task embeds
    #     task_inputs_embeds = [embed_tokens(x) for x in task_input_ids]

    #     # finally now without padding, concat
    #     k = 0
    #     for x in task_inputs_embeds:
    #         new_program_idxs[batch_i].append(k)
    #         k += ntokens + x.shape[0]

    #     # interleave programs and pad 1 attention
    #     assert len(task_programs) == len(task_inputs_embeds) == len(task_attention_mask) == len(task_label_ids)
    #     program_attentions = [torch.full((ntokens,), 1, device=x.device, dtype=x.dtype) for x in task_attention_mask]
    #     program_label_ids = [torch.full((ntokens,), -100, device=x.device, dtype=x.dtype) for x in task_label_ids]
    #     task_inputs_embeds = [item for pair in zip(task_programs, task_inputs_embeds) for item in pair]
    #     task_attention_mask = [item for pair in zip(program_attentions, task_attention_mask) for item in pair]
    #     task_label_ids = [item for pair in zip(program_label_ids, task_label_ids) for item in pair]
    #     assert len(task_inputs_embeds) == len(task_attention_mask) == len(task_label_ids) == 2 * n_pairs

    #     task_inputs_embeds = torch.cat(task_inputs_embeds)
    #     task_attention_mask = torch.cat(task_attention_mask)
    #     task_label_ids = torch.cat(task_label_ids)
    #     assert task_inputs_embeds.shape[:1] == task_attention_mask.shape == task_label_ids.shape
    #     assert (task_attention_mask == 1).sum() == task_attention_mask.numel() # no padding so full attention
    #     assert len(task_inputs_embeds) == ntokens * n_pairs + sum(task_input_ids_lens)

    #     single_inputs_embeds.append(task_inputs_embeds)
    #     single_attention_mask.append(task_attention_mask)
    #     single_label_ids.append(task_label_ids)

    # # now pad based on pad side
    # single_attention_mask2 = pad_sequence_with_side(single_attention_mask, padding_value=0, side=pad_side)
    # single_label_ids2 = pad_sequence_with_side(single_label_ids, padding_value=-100, side=pad_side)
    # assert single_attention_mask2.shape == single_label_ids2.shape

    # # pad single_inputs_embeds batchsize x (task-seqlen, hiddendim)
    # max_task_len = max(x.shape[0] for x in single_inputs_embeds)
    # for i, x in enumerate(single_inputs_embeds):
    #     if max_task_len > x.shape[0]:
    #         task_pad_embeds = pad_embeds.unsqueeze(0).expand(max_task_len - x.shape[0], -1)
    #         if pad_side == 'left':
    #             single_inputs_embeds[i] = torch.cat([task_pad_embeds, single_inputs_embeds[i]])
    #         else:
    #             single_inputs_embeds[i] = torch.cat([single_inputs_embeds[i], task_pad_embeds])
    # single_inputs_embeds2 = torch.stack(single_inputs_embeds)
    # assert single_inputs_embeds2.shape[:2] == single_label_ids2.shape

    # assert prev_past_key_values is not None
    # assert prev_past_key_values_attention_mask is not None

    # # construct position ids
    # position_ids2 = []
    # for m in single_attention_mask2:
    #     sequence_position_ids2 = torch.cumsum(m, dim=0) - 1
    #     sequence_position_ids2[sequence_position_ids2 < 0] = 0
    #     position_ids2.append(sequence_position_ids2)
    # position_ids2 = torch.stack(position_ids2)
    # assert position_ids2.shape == single_attention_mask2.shape

    # # single model forward
    # model_out_2 = model(
    #     inputs_embeds=single_inputs_embeds2,
    #     attention_mask=single_attention_mask2,
    #     labels=single_label_ids2,
    #     position_ids=position_ids2,
    #     output_hidden_states=True,
    # )
    # single_past_key_values = model_out_2.past_key_values

    # # compare
    # kv_max_diff = 0.0
    # for i in range(len(prev_past_key_values)):
    #     for j in range(2):
    #         old_kv = prev_past_key_values[i][j].detach().clone()
    #         new_kv = single_past_key_values[i][j].detach().clone()
    #         assert old_kv.shape[:2] == new_kv.shape[:2] and old_kv.shape[3] == new_kv.shape[3]
    #         assert prev_past_key_values_attention_mask.shape[0] == single_attention_mask2.shape[0] == batch_size
    #         assert prev_past_key_values_attention_mask.shape[1] == old_kv.shape[2] and single_attention_mask2.shape[1] == new_kv.shape[2]
    #         for batch_i, (task_old_kv, task_new_kv, task_old_mask, task_new_mask, task_new_program_idxs) in enumerate(zip(old_kv, new_kv, prev_past_key_values_attention_mask, single_attention_mask2, new_program_idxs)):
    #             task_input_ids_lens = [x[batch_i] for x in input_ids_lens]
    #             # # remove program from old
    #             # assert task_old_kv.shape[1] == sum(x.shape[1] for x in inputs_embeds[:-1]) + ntokens * (len(inputs_embeds) - 1)
    #             # k = 0
    #             # for task_inputs_embeds in inputs_embeds[:-1]:
    #             #     task_old_kv[:, k: k+ntokens, :] = 0.0
    #             #     k += task_inputs_embeds.shape[1] + ntokens
    #             # # remove programs from new
    #             # for k in task_new_program_idxs:
    #             #     task_new_kv[:, k: k+ntokens, :] = 0.0
    #             # remove paddings
    #             task_old_kv = task_old_kv[:, task_old_mask.bool(), :]
    #             task_new_kv = task_new_kv[:, task_new_mask.bool(), :][:, :task_old_kv.shape[1], :]
    #             # n_is_zero = 0
    #             # for l in range(task_old_kv.shape[1]):
    #             #     old_is_zero = task_old_kv[:, l, :].abs().sum() == 0
    #             #     new_is_zero = task_new_kv[:, l, :].abs().sum() == 0
    #             #     assert old_is_zero == new_is_zero
    #             #     n_is_zero += old_is_zero
    #             # assert n_is_zero == ntokens * (len(inputs_embeds) - 1)
    #             assert task_old_kv.shape[1] == task_new_kv.shape[1] == sum(task_input_ids_lens[:-1]) + (n_pairs - 1) * ntokens
    #             kv_max_diff = max(kv_max_diff, (task_old_kv - task_new_kv).abs().max().item())

    # print('kv max diff', kv_max_diff)
    # print('loss diff', (model_out_2.loss - ce_loss).abs().item()) # type: ignore




    # # compare model_out_1 and model_out_2 for sanity check
    # embeds1 = single_inputs_embeds1.masked_select(single_attention_mask1.bool()[:, :, None])
    # embeds2 = single_inputs_embeds2.masked_select(single_attention_mask2.bool()[:, :, None])
    # assert (embeds1 - embeds2).max() == 0.0
    # labs1 = single_label_ids1.masked_select(single_attention_mask1.bool())
    # labs2 = single_label_ids2.masked_select(single_attention_mask2.bool())
    # assert (labs1 - labs2).max() == 0.0

    # max_kv_diff = 0.0
    # assert len(model_out_1.past_key_values) == len(model_out_2.past_key_values)
    # for i in range(len(model_out_1.past_key_values)):
    #     assert len(model_out_1.past_key_values[i]) == len(model_out_2.past_key_values[i])
    #     for j in range(2):
    #         kv1 = model_out_1.past_key_values[i][j]
    #         kv2 = model_out_2.past_key_values[i][j]
    #         kv1 = kv1.masked_select(single_attention_mask1.bool()[:, None, :, None])
    #         kv2 = kv2.masked_select(single_attention_mask2.bool()[:, None, :, None])
    #         assert kv1.shape == kv2.shape
    #         max_kv_diff = max(max_kv_diff, (kv1 - kv2).abs().max().item())
    # print('kv max diff', max_kv_diff)
    # print('loss diff', (model_out_2.loss - model_out_1.loss).abs().item()) # type: ignore
    # breakpoint()



    # print(ce_loss.item())
    # breakpoint()

    inferred_programs = None
    if program_cache is not None:
        inferred_programs = torch.stack(saved_all_programs[1:]) # all programs except the prior
        inferred_programs = inferred_programs.permute(1, 0, 2, 3) # (batch_size, num_program, ntoken, hiddendim)

    return ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, program_loss, consistency_loss, invar_loss, total_loss, ce_losses, \
        inferred_programs # type: ignore


def insert_based_on_sides(
        data: torch.Tensor,
        to_insert: torch.Tensor,
        lens: List[int],
        insert_side: str,
        pad_side: str,
        pad_id: Union[int, torch.Tensor],
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


@torch.enable_grad()
def gradient_search(
        batch_idx: int,
        eval_dataset: EvalDataset,
        accelerator: Accelerator,
        model: Union[nn.Module, DistributedDataParallel],
        weird_cast: bool,
        short_context: bool,
        # inputs
        prev_programs: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]],
        past_key_values_attention_mask: Optional[torch.Tensor],
        # config
        iters: int,
        lr: float,
        beta1: float,
        beta2: float,
        batch_size: int,
        optimizer: str,
        lr_scheduler: str,
        max_grad_norm: float,
        take_best: bool,
        train_past_kv: bool,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]]:
    # NOTE: demonstration interval

    # note gradient checkpointing does not matter because we are freezing the model here
    # however, if we choose to tune the decoder as well, then gradient checkpointing might be desired
    # didnt use grad accum, dont think needed

    assert prev_programs.shape[0] == 1
    if past_key_values is not None:
        assert past_key_values[0][0].shape[0] == 1
    if past_key_values_attention_mask is not None:
        assert past_key_values_attention_mask.shape[0] == 1

    # dataset and dataloader
    gs_dataset = GSDataset(
        task=eval_dataset.eval_tasks[batch_idx],
        tokenizer=eval_dataset.tokenizer,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        train_pad_side=eval_dataset.train_pad_side,
        no_dim=eval_dataset.no_dim,
        no_separate_color_tokens=eval_dataset.no_separate_color_tokens,
        no_bos=eval_dataset.no_bos,
        only_first_bos=eval_dataset.only_first_bos,
    )
    if take_best:
        assert batch_size >= len(gs_dataset)
    batch_size = min(batch_size, len(gs_dataset))
    gs_collate_fn = partial(collate_fn_gs, dataset=gs_dataset)
    gs_loader = DataLoader(
        gs_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gs_collate_fn,
        drop_last=False,
        num_workers=0,
    )

    # get program parameters
    program_params = [prev_programs]
    if train_past_kv:
        assert past_key_values is not None
        for layer_k, layer_v in past_key_values:
            program_params.append(layer_k)
            program_params.append(layer_v)

    # set requires grad
    assert all(not p.requires_grad for p in program_params)
    for p in program_params:
        p.requires_grad = True

    # expand to match predicted program with batch size
    prev_programs = prev_programs.expand(batch_size, *prev_programs.shape[1:])
    if past_key_values is not None:
        past_key_values = tuple(
            (
                layer_k.expand(batch_size, *layer_k.shape[1:]),
                layer_v.expand(batch_size, *layer_v.shape[1:]),
            )
            for layer_k, layer_v in past_key_values
        ) # type: ignore
    if past_key_values_attention_mask is not None:
        past_key_values_attention_mask = past_key_values_attention_mask.expand(batch_size, *past_key_values_attention_mask.shape[1:])

    # optimizer
    if optimizer == 'adamw':
        optim = torch.optim.AdamW(program_params, weight_decay=0.0, lr=lr, betas=(beta1, beta2)) # type: ignore
    else:
        optim = torch.optim.SGD(program_params, lr=lr) # type: ignore

    # lr scheduler
    if lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=iters)
    else:
        scheduler = get_constant_schedule(optim)

    # prepare stuff (no difference on singlegpu, havent tested on multigpu)
    # model, optim, gs_loader = accelerator.prepare(model, optim, gs_loader)

    # prepare some stuff
    curr_iter = 0
    best_loss = float("inf")
    best_program = prev_programs
    best_past_key_values = past_key_values
    model.train()

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens
    pad_embeds = embed_tokens(torch.tensor(eval_dataset.tokenizer.pad_token_id, device=accelerator.device))

    # train!
    while curr_iter < iters:
        for batch in gs_loader:
            pair_input_ids = batch["input_ids"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_label_ids = batch["label_ids"].to(accelerator.device)
            pair_input_ids_lens = batch["input_ids_lens"]
            device, dtype = pair_input_ids.device, pair_input_ids.dtype

            with accelerator.autocast():
                pair_inputs_embeds = embed_tokens(pair_input_ids)
                pair_inputs_embeds = insert_based_on_sides(
                    data=pair_inputs_embeds,
                    to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
                    lens=pair_input_ids_lens,
                    insert_side="left",
                    pad_side=gs_dataset.train_pad_side,
                    pad_id=pad_embeds,
                )
                pair_attention_mask = insert_based_on_sides(
                    data=pair_attention_mask,
                    to_insert=torch.full((batch_size, eval_dataset.ntokens), 1, device=device, dtype=dtype),
                    lens=pair_input_ids_lens,
                    insert_side="left",
                    pad_side=gs_dataset.train_pad_side,
                    pad_id=0,
                )
                pair_label_ids = insert_based_on_sides(
                    data=pair_label_ids,
                    to_insert=torch.full((batch_size, eval_dataset.ntokens), -100, device=device, dtype=dtype),
                    lens=pair_input_ids_lens,
                    insert_side="left",
                    pad_side=gs_dataset.train_pad_side,
                    pad_id=-100,
                )

                if not short_context:
                    assert past_key_values_attention_mask is not None
                    pair_attention_mask = torch.cat([past_key_values_attention_mask, pair_attention_mask], dim=1)

                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
                    "labels": pair_label_ids,
                    "use_cache": not short_context, # need to generate or use kv cache
                }
                if not short_context:
                    assert past_key_values is not None
                    model_kwargs["past_key_values"] = past_key_values

                    # build position ids (does NOT depend on dropout)
                    attention_mask_just_for_kv = pair_attention_mask[:, :past_key_values[0][0].shape[2]]
                    attention_mask_after_kv = pair_attention_mask[:, past_key_values[0][0].shape[2]:]
                    position_ids = []
                    for mask_for_kv, mask_after_kv in zip(attention_mask_just_for_kv, attention_mask_after_kv):
                        sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=dtype)
                        position_start = mask_for_kv.sum()
                        n_new_positions = mask_after_kv.sum()
                        new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=dtype)
                        if gs_dataset.train_pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)
                    model_kwargs["position_ids"] = position_ids

                else:
                    # necessary for padside left and does not change when padside is right, idky
                    position_ids = []
                    for m in pair_attention_mask:
                        sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=dtype)
                        n_new_positions = m.sum()
                        new_positions = torch.tensor(range(n_new_positions), device=device, dtype=dtype)
                        if gs_dataset.train_pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)
                    model_kwargs["position_ids"] = position_ids

                # get ce loss
                loss = model(**model_kwargs).loss
                # print(loss.item())
                # breakpoint()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(program_params, max_grad_norm)
            optim.step()
            scheduler.step()
            optim.zero_grad()

            if take_best and loss.item() < best_loss:
                best_loss = loss.item()
                best_program = prev_programs.detach().clone()
                if train_past_kv:
                    assert past_key_values is not None
                    best_past_key_values = tuple(
                        (layer_k.detach().clone(), layer_v.detach().clone())
                        for layer_k, layer_v in past_key_values
                    )

            curr_iter += 1
            if curr_iter >= iters:
                break

    model.eval()

    if take_best:
        prev_programs = best_program
        if train_past_kv:
            past_key_values = best_past_key_values  # type: ignore

    # shrink to match predicted program with batch size 1
    if batch_size > 1:
        assert torch.equal(prev_programs[0], prev_programs[1])
        prev_programs = prev_programs[:1]
        if past_key_values is not None:
            past_key_values = tuple(
                (layer_k[:1], layer_v[:1])
                for layer_k, layer_v in past_key_values
            ) # type: ignore

    return prev_programs, past_key_values


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    desc: str,
    task_to_ttt_path: Optional[Dict[str, Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]]],
    ttt_param_names: Optional[Set[str]],
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    trainable_nbit: int,
    no_flash_attn: bool,
    dry_eval_run: bool,
    no_residual: bool,
    no_discrete_prior: bool,
    output_dir: str,
    gs_iters: int,
    gs_batch_size: int,
    gs_lr: float,
    gs_beta1: float,
    gs_beta2: float,
    gs_optimizer: str,
    gs_max_grad_norm: float,
    gs_lr_scheduler: str,
    gs_take_best: bool,
    gs_train_past_kv: bool,
    no_codebook: bool,
    weird_cast: bool,
    short_context: bool,
    kv_pad_side: str,
    attention_reduction_ratio: float,
):
    model.eval()
    prior_embeddings.eval()
    program_embeddings.eval()
    if vae_projection is not None:
        vae_projection.eval()
    if quantizer is not None:
        quantizer.eval()
    if program_norm is not None:
        program_norm.eval()

    # get modules in case of DDP
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # if ttt provided, same model weights for the missing ttt task weights
    cached_model_weights_path = None
    cached_prior_embeddings_weights_path = None
    cached_program_embeddings_weights_path = None
    cached_vae_projection_weights_path = None
    cached_quantizer_weights_path = None
    cached_program_norm_weights_path = None
    curr_ttt_task_name = None
    if task_to_ttt_path is not None: # run on both processes
        # save model for default when ttt is missing
        cached_model_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_cache.pt")
        assert isinstance(ttt_param_names, set)
        names = set(name for name, _ in module.named_parameters())
        assert all(name in names for name in ttt_param_names), f"process{accelerator.process_index}\n\n{ttt_param_names}\n\n{names}"
        cache_weights = {name: param for name, param in module.named_parameters() if name in ttt_param_names}
        torch.save(cache_weights, cached_model_weights_path)
        logger.info(f"ttt provided, cached {len(cache_weights)} model weights to {cached_model_weights_path}")
        # save prior embeddings
        cached_prior_embeddings_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_prior_embeddings_cache.pt")
        torch.save(prior_embeddings, cached_prior_embeddings_weights_path)
        logger.info(f"ttt provided, cached prior embeddings weights to {cached_prior_embeddings_weights_path}")
        # save program embeddings
        cached_program_embeddings_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_program_embeddings_cache.pt")
        torch.save(program_embeddings, cached_program_embeddings_weights_path)
        logger.info(f"ttt provided, cached program embeddings weights to {cached_program_embeddings_weights_path}")
        # save vae projection
        if vae_projection is not None:
            cached_vae_projection_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_vae_projection_cache.pt")
            torch.save(vae_projection, cached_vae_projection_weights_path)
            logger.info(f"ttt provided, cached vae projection weights to {cached_vae_projection_weights_path}")
        # save quantizer
        if quantizer is not None:
            cached_quantizer_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_quantizer_cache.pt")
            torch.save(quantizer, cached_quantizer_weights_path)
            logger.info(f"ttt provided, cached quantizer weights to {cached_quantizer_weights_path}")
        # save program norm
        if program_norm is not None:
            cached_program_norm_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_program_norm_cache.pt")
            torch.save(program_norm, cached_program_norm_weights_path)
            logger.info(f"ttt provided, cached program norm weights to {cached_program_norm_weights_path}")
        # save default to model paths and set current ttt weights to default
        task_to_ttt_path["default"] = (
            cached_model_weights_path,
            cached_prior_embeddings_weights_path,
            cached_program_embeddings_weights_path,
            cached_vae_projection_weights_path,
            cached_quantizer_weights_path,
            cached_program_norm_weights_path,
        )
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
            data_idx_iterator = tqdm(chunks_uniform_batch(task_ids, process_data_idxs, batch_size), total=n_batches, desc=desc) # type: ignore
        else:
            data_idx_iterator = tqdm(chunks(process_data_idxs, batch_size), total=n_batches, desc=desc)  # type: ignore

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
                    (
                        ttt_model_weights_path,
                        ttt_prior_embeddings_weights_path,
                        ttt_program_embeddings_weights_path,
                        ttt_vae_projection_weights_path,
                        ttt_quantizer_weights_path,
                        ttt_program_norm_weights_path,
                    ) = task_to_ttt_path[task_name]
                    # load model
                    model_ttt_state_dict = torch.load(
                        ttt_model_weights_path,
                        weights_only=True,
                        map_location=accelerator.device
                    )
                    assert set(model_ttt_state_dict.keys()) == ttt_param_names
                    module.load_state_dict(model_ttt_state_dict, strict=False)
                    del model_ttt_state_dict
                    # load prior embeddings
                    prior_embeddings = torch.load(ttt_prior_embeddings_weights_path, weights_only=False, map_location=accelerator.device)
                    # load program embeddings
                    program_embeddings = torch.load(ttt_program_embeddings_weights_path, weights_only=False, map_location=accelerator.device)
                    # load vae projection
                    if ttt_vae_projection_weights_path is not None:
                        vae_projection = torch.load(ttt_vae_projection_weights_path, weights_only=False, map_location=accelerator.device)
                    # load quantizer
                    if ttt_quantizer_weights_path is not None:
                        quantizer = torch.load(ttt_quantizer_weights_path, weights_only=False, map_location=accelerator.device)
                    # load program norm
                    if ttt_program_norm_weights_path is not None:
                        program_norm = torch.load(ttt_program_norm_weights_path, weights_only=False, map_location=accelerator.device)
                    curr_ttt_task_name = task_name # set current task name
                    model.eval() # another eval after loading weight just in case
                    prior_embeddings.eval()
                    program_embeddings.eval()
                    if vae_projection is not None:
                        vae_projection.eval()
                    if quantizer is not None:
                        quantizer.eval()
                    if program_norm is not None:
                        program_norm.eval()
            ttt_provided_list += ttt_provided

            batch_data = [dataset[i] for i in batch_idxs]
            batch = collate_fn(batch_data)

            if dry_eval_run:
                continue

            # get tensors
            task_ids = batch["task_ids"]
            inverters = batch["inverters"]
            input_ids = [x.to(accelerator.device) for x in batch["input_ids"]]
            attention_mask = [x.to(accelerator.device) for x in batch["attention_mask"]]
            gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            out_token_length = batch["out_token_length"]
            label_texts = batch["label_texts"]
            input_ids_lens = batch["input_ids_lens"]
            gen_input_ids_lens = batch["gen_input_ids_lens"]
            num_pairs = batch["num_pairs"] # not including test pair

            with accelerator.autocast():
                # STEP 1: get predicted programs and kv cache
                prev_programs, demonstration_intervals, past_key_values, past_key_values_attention_mask = get_predicted_program(
                    # model
                    model=model,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    vae_projection=vae_projection,
                    quantizer=quantizer,
                    program_norm=program_norm,
                    tokenizer=dataset.tokenizer,
                    # data
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_ids_lens=input_ids_lens,
                    num_pairs=num_pairs,
                    # others
                    ntokens=dataset.ntokens,
                    pad_side=dataset.train_pad_side,
                    no_residual=no_residual,
                    no_discrete_prior=no_discrete_prior,
                    train_codebook_only=no_codebook,
                    weird_cast=weird_cast,
                    short_context=short_context,
                    kv_pad_side=kv_pad_side,
                    attention_reduction_ratio=attention_reduction_ratio,
                )

            # gradient search is OUTSIDE of autocast
            if gs_iters > 0:
                # construct new programs and kv to be filled
                new_prev_programs = []
                new_past_key_values = None
                assert prev_programs.shape[0] == bs
                if not short_context:
                    assert past_key_values is not None
                    new_past_key_values = tuple([[], []] for _ in past_key_values)
                    assert past_key_values[0][0].shape[0] == bs

                # gradient search is done individually for simplicity
                with accelerator.no_sync(model):
                    for batch_i, batch_idx in enumerate(batch_idxs):

                        # extract the batchsize1 task inputs
                        task_prev_programs = prev_programs[batch_i:batch_i+1].detach().clone()
                        task_past_key_values = None
                        task_past_key_values_attention_mask = None
                        if not short_context:
                            assert past_key_values is not None and past_key_values_attention_mask is not None
                            task_past_key_values = tuple(
                                (layer_k[batch_i: batch_i+1].detach().clone(), layer_v[batch_i: batch_i+1].detach().clone())
                                for layer_k, layer_v in past_key_values
                            )
                            task_past_key_values_attention_mask = past_key_values_attention_mask[batch_i:batch_i+1].detach().clone()

                        # search!
                        task_prev_programs, task_past_key_values = gradient_search(
                            batch_idx=batch_idx,
                            eval_dataset=dataset,
                            accelerator=accelerator,
                            model=model,
                            weird_cast=weird_cast,
                            short_context=short_context,
                            # inputs
                            prev_programs=task_prev_programs,
                            past_key_values=task_past_key_values, # type: ignore
                            past_key_values_attention_mask=task_past_key_values_attention_mask,
                            # config
                            iters=gs_iters,
                            lr=gs_lr,
                            beta1=gs_beta1,
                            beta2=gs_beta2,
                            batch_size=gs_batch_size,
                            optimizer=gs_optimizer,
                            lr_scheduler=gs_lr_scheduler,
                            max_grad_norm=gs_max_grad_norm,
                            take_best=gs_take_best,
                            train_past_kv=gs_train_past_kv,
                        )

                        # fill new programs and kv, no need to update kv attention mask
                        new_prev_programs.append(task_prev_programs)
                        if not short_context:
                            assert task_past_key_values is not None and new_past_key_values is not None
                            for layer_i, (layer_k, layer_v) in enumerate(task_past_key_values):
                                new_past_key_values[layer_i][0].append(layer_k)
                                new_past_key_values[layer_i][1].append(layer_v)

                    # finally, tuple-lize kv and rename
                    prev_programs = torch.cat(new_prev_programs)
                    if not short_context:
                        assert new_past_key_values is not None
                        past_key_values = tuple(
                            (torch.cat(layer_k), torch.cat(layer_v))
                            for layer_k, layer_v in new_past_key_values
                        )

            with accelerator.autocast():
                # insert programs to inputs_embeds
                assert len(gen_input_ids) == bs
                inputs_embeds = embed_tokens(gen_input_ids)
                pad_embeds = embed_tokens(torch.tensor(dataset.tokenizer.pad_token_id, device=accelerator.device))
                inputs_embeds = insert_based_on_sides(
                    data=inputs_embeds,
                    to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
                    lens=gen_input_ids_lens,
                    insert_side="left",
                    pad_side=dataset.gen_pad_side,
                    pad_id=pad_embeds,
                )

                # insert programs to attention mask
                attention_mask = insert_based_on_sides(
                    data=gen_attention_mask,
                    to_insert=torch.full((bs, dataset.ntokens), 1, device=accelerator.device, dtype=gen_attention_mask.dtype),
                    lens=gen_input_ids_lens,
                    insert_side="left",
                    pad_side=dataset.gen_pad_side,
                    pad_id=0,
                )

                arbitrary_increase = 5
                if not no_flash_attn:
                    inputs_embeds = inputs_embeds.to(NBIT_TO_DTYPE[trainable_nbit])

                if short_context:
                    # generate somehow needs this conversion done beforehand
                    gen_tokens = module.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=max(out_token_length) + arbitrary_increase,
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                    )
                else:
                    assert past_key_values is not None and past_key_values_attention_mask is not None
                    # add past key values portion to inputs_embeds and attention mask
                    # the padding of inputs_embeds is ignored
                    pad_len = past_key_values_attention_mask.shape[1]
                    inputs_embeds = torch.cat([
                        torch.zeros((bs, pad_len, inputs_embeds.shape[2]), device=accelerator.device, dtype=inputs_embeds.dtype),
                        inputs_embeds,
                    ], dim=1)
                    attention_mask = torch.cat([past_key_values_attention_mask, attention_mask], dim=1)

                    if not no_flash_attn:
                        past_key_values = tuple(
                            (
                                layer_k.to(NBIT_TO_DTYPE[trainable_nbit]),
                                layer_v.to(NBIT_TO_DTYPE[trainable_nbit]),
                            )
                            for layer_k, layer_v in past_key_values
                        )

                    gen_tokens = module.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        max_new_tokens=max(out_token_length) + arbitrary_increase,
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                        demonstration_intervals=demonstration_intervals,
                        attention_reduction_ratio=attention_reduction_ratio,
                    )

                assert len(gen_tokens) == len(out_token_length)
                for t, l in zip(gen_tokens, out_token_length):
                    t[l + arbitrary_increase:] = dataset.tokenizer.pad_token_id
                gen_texts = dataset.tokenizer.batch_decode(
                    gen_tokens,
                    skip_special_tokens=True,
                    no_separate_color_tokens=dataset.no_separate_color_tokens,
                )
                # print(gen_texts)
                # breakpoint()

            # Compare each gen_text with label_texts
            assert len(task_ids) == len(inverters) == bs, (len(task_ids), len(inverters), bs)
            assert len(gen_texts) == len(label_texts) == bs, (len(gen_texts), len(label_texts), bs)

            for task_id, inverter, gen_text, label_text in zip(task_ids, inverters, gen_texts, label_texts):
                relaxed_token_acc_list.append(best_match_count(gen_text, label_text) / len(label_text))
                # is valid grid
                gen_grid, gen_is_grid = text_to_2d_grid(text=gen_text, no_dim=dataset.no_dim)
                label_grid, label_is_grid = text_to_2d_grid(text=label_text, no_dim=dataset.no_dim)
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
                gen_text = grid_2d_to_text(gen_grid, no_dim=dataset.no_dim)
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

    if cached_model_weights_path is not None:
        os.remove(cached_model_weights_path)
    if cached_prior_embeddings_weights_path is not None:
        os.remove(cached_prior_embeddings_weights_path)
    if cached_program_embeddings_weights_path is not None:
        os.remove(cached_program_embeddings_weights_path)
    if cached_vae_projection_weights_path is not None:
        os.remove(cached_vae_projection_weights_path)
    if cached_quantizer_weights_path is not None:
        os.remove(cached_quantizer_weights_path)
    if cached_program_norm_weights_path is not None:
        os.remove(cached_program_norm_weights_path)

    return exact_acc, valid_grid, correct_grid_dim, token_acc, relaxed_token_acc, task_id_to_texts, votes, competition_sub_acc, competition_all_acc, ttt_provided


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


def grid_2d_to_text(grid: list[List[int]], no_dim: bool):
    height, width = len(grid), len(grid[0])
    lines = [] if no_dim else [f"{str(height)}{str(width)}"]
    for row in grid:
        lines.append("".join([str(x) for x in row]))
    return "\n".join(lines)


def text_to_2d_grid(text: str, no_dim: bool) -> Tuple[Optional[List[List[int]]], bool]:
    try:
        text = text.strip() # label is appended by \n
        grid_lines = text.split('\n')
        grid = []
        row_lens = []
        if not no_dim:
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
        prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
        program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
        vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
        quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
        program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
        output_dir: str,
        epoch: int,
    ) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]:

    # model
    save_model_path = os.path.join(output_dir, f"lora_epoch_{epoch+1}")
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.save_pretrained(save_model_path, save_embedding_layers=True)
    logger.info(f"Saved model to {save_model_path}")

    # prior embeddings
    save_prior_embeddings_path = os.path.join(output_dir, f"prior_embeddings_epoch_{epoch+1}.pt")
    prior_embeddings_module = prior_embeddings
    if isinstance(prior_embeddings, DistributedDataParallel):
        prior_embeddings_module = prior_embeddings.module
    torch.save(prior_embeddings_module, save_prior_embeddings_path)
    logger.info(f"Saved prior embeddings to {save_prior_embeddings_path}")

    # program embeddings
    save_program_embeddings_path = os.path.join(output_dir, f"program_embeddings_epoch_{epoch+1}.pt")
    program_embeddings_module = program_embeddings
    if isinstance(program_embeddings, DistributedDataParallel):
        program_embeddings_module = program_embeddings.module
    torch.save(program_embeddings_module, save_program_embeddings_path)
    logger.info(f"Saved program embeddings to {save_program_embeddings_path}")

    # vae projection
    save_vae_projection_path = None
    if vae_projection is not None:
        save_vae_projection_path = os.path.join(output_dir, f"vae_projection_epoch_{epoch+1}.pt")
        vae_projection_module = vae_projection
        if isinstance(vae_projection, DistributedDataParallel):
            vae_projection_module = vae_projection.module
        torch.save(vae_projection_module, save_vae_projection_path)
        logger.info(f"vae projection to {save_vae_projection_path}")

    # quantizer
    save_quantizer_path = None
    if quantizer is not None:
        save_quantizer_path = os.path.join(output_dir, f"quantizer_epoch_{epoch+1}.pt")
        quantizer_module = quantizer
        if isinstance(quantizer, DistributedDataParallel):
            quantizer_module = quantizer.module
        torch.save(quantizer_module, save_quantizer_path)
        logger.info(f"Saved program norm to {save_quantizer_path}")

    # program norm
    save_program_norm_path = None
    if program_norm is not None:
        save_program_norm_path = os.path.join(output_dir, f"program_norm_epoch_{epoch+1}.pt")
        program_norm_module = program_norm
        if isinstance(program_norm, DistributedDataParallel):
            program_norm_module = program_norm.module
        torch.save(program_norm_module, save_program_norm_path)
        logger.info(f"Saved program norm to {save_program_norm_path}")

    return save_model_path, save_prior_embeddings_path, save_program_embeddings_path, \
        save_vae_projection_path, save_quantizer_path, save_program_norm_path


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
        cov_scale: float,
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
    distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean_embeddings, covariance_matrix=cov_scale * covariance)
    return distribution.sample(sample_shape=(ntokens,)).to(device).to(dtype) # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--tracker_project_name", type=str, default="arc")
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_len", type=int, default=-1) # two grid -> 1867
    parser.add_argument("--debug_fixed_order", action="store_true")
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)
    parser.add_argument("--debug_train_data", action="store_true")
    parser.add_argument("--debug_no_resume", action='store_true')

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ar_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--token_weighted_loss", action="store_true")
    parser.add_argument("--weird_cast", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--full_demonstration_dropout", type=float, default=0.0)
    parser.add_argument("--partial_demonstration_dropout", type=float, default=0.0)
    parser.add_argument("--loss_on_first", action="store_true")
    parser.add_argument("--attention_reduction_ratio", type=float, default=1.0)

    # invar loss
    parser.add_argument("--invar_loss_margin", type=float, default=0.5)

    # program loss
    parser.add_argument("--program_type", type=str, choices=["none", "random", "concat"], default="none")

    # long context
    parser.add_argument("--long_context_checkpointing_threshold", type=int, default=0)

    # vqvae
    parser.add_argument("--codebook_size", type=int, default=-1)
    parser.add_argument("--fsq_L", metavar='N', type=int, nargs='+', default=[])
    parser.add_argument("--no_discrete_prior", action="store_true")
    parser.add_argument("--warmup_cookbook_only_epochs", type=int, default=0)

    # vae
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--subset_kl", action="store_true")
    parser.add_argument("--mlp_factor", type=int, default=4)

    # Training
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=2e-5)
    parser.add_argument("--lr_program", type=float, default=2e-4)
    parser.add_argument("--lr_prior", type=float, default=2e-4)
    parser.add_argument("--lr_other", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")
    parser.add_argument("--dry_train_run", action="store_true")
    parser.add_argument("--dry_eval_run", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--full_attention_dropout", type=float, default=0.0)
    parser.add_argument("--demonstration_attention_dropout", type=float, default=0.0) # activate naive selfattn but oom
    parser.add_argument("--program_dropout", type=float, default=0.0)
    parser.add_argument("--program_noise_std", type=float, default=0.0)
    parser.add_argument("--short_context", action='store_true')

    # program caching
    parser.add_argument("--cache_size_per_task", type=int, default=0)
    parser.add_argument("--prior_embed_ratio", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--extra_inference_pairs", type=int, default=0)
    parser.add_argument("--limit_inference_pairs", action='store_true')
    parser.add_argument("--limit_inference_pairs_strict", action='store_true') # overrides limit_inference_pairs

    # scheduled extra losses
    parser.add_argument("--consistency_loss_lambda", type=float, default=0.0)
    parser.add_argument("--consistency_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--consistency_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.0)
    parser.add_argument("--invar_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--invar_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--program_loss_lambda", type=float, default=1.0)
    parser.add_argument("--program_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--program_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--commitment_loss_lambda", type=float, default=0.1)
    parser.add_argument("--commitment_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--commitment_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--codebook_loss_lambda", type=float, default=1.0)
    parser.add_argument("--codebook_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--codebook_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--kl_loss_lambda", type=float, default=0.0)
    parser.add_argument("--kl_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--kl_loss_linear_epochs", type=int, default=0)

    # both data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_num_pair", type=int, default=8) # includes test pair
    parser.add_argument("--max_num_pair", type=int, default=8) # includes test pair
    parser.add_argument("--train_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--gen_pad_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--kv_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--no_dim", action='store_true')
    parser.add_argument("--no_separate_color_tokens", action='store_true')
    parser.add_argument("--no_color_permute", action="store_true")
    parser.add_argument("--no_pair_permute", action="store_true")
    parser.add_argument("--no_d8", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--no_bos", action="store_true")
    parser.add_argument("--only_first_bos", action="store_true")

    # re-arc train data
    parser.add_argument("--train_data_dir", type=str, default="./data/re-arc/train_data/tasks")
    parser.add_argument("--verifier_file", type=str, default="./data/re-arc/verifiers.py") # for re-arc and train-original invar loss
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

    # gradient search train
    parser.add_argument("--train_gs_iters", type=int, default=0)
    parser.add_argument("--train_gs_lr", type=float, default=1.0)
    parser.add_argument("--train_gs_beta1", type=float, default=0.9)
    parser.add_argument("--train_gs_beta2", type=float, default=0.9)
    parser.add_argument("--train_gs_batch_size", type=int, default=2)
    parser.add_argument("--train_gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--train_gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--train_gs_max_grad_norm", default=1e8, type=float, help="Max gradient norm.")
    parser.add_argument("--train_gs_take_best", action="store_true")
    parser.add_argument("--train_gs_train_past_kv", action="store_true")

    # gradient search eval
    parser.add_argument("--eval_gs_iters", type=int, default=0)
    parser.add_argument("--eval_gs_lr", type=float, default=1.0)
    parser.add_argument("--eval_gs_beta1", type=float, default=0.9)
    parser.add_argument("--eval_gs_beta2", type=float, default=0.9)
    parser.add_argument("--eval_gs_batch_size", type=int, default=2)
    parser.add_argument("--eval_gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--eval_gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--eval_gs_max_grad_norm", default=1e8, type=float, help="Max gradient norm.")
    parser.add_argument("--eval_gs_take_best", action="store_true")
    parser.add_argument("--eval_gs_train_past_kv", action="store_true")

    # Lora
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=float, default=24.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument('--lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head'
    ])
    parser.add_argument("--no_rslora", action='store_true')

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # override to debug stuff
    if args.debug:
        args.tag = 'test'
        args.wandb = False
        args.samples_per_epoch = 8
        args.log_every = 1
        args.debug_no_resume = True

    # check args
    assert args.gen_pad_side == 'left' # literally doesnt work otherwise
    assert not (args.no_train_original and args.only_train_original)
    if args.warmup_cookbook_only_epochs:
        assert args.codebook_loss_offset_epochs == 0
        assert args.codebook_loss_linear_epochs == 0
    assert args.commitment_loss_offset_epochs >= args.warmup_cookbook_only_epochs
    assert args.extra_inference_pairs == 0
    if args.demonstration_attention_dropout:
        assert args.no_flash_attn
    if args.cache_size_per_task > 0:
        assert args.debug_no_resume # don't allow resuming from cold cache yet
        assert args.min_num_pair == args.max_num_pair
        # not asserting here, but dont add any kl, consistency, program loss, invar, etc

    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable

    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    os.environ["WANDB_API_KEY"]="faf21d9ff65ee150697c7e96f070616f6b662134"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        project_config=project_config,
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_up_main_process_logger(accelerator, logger)
    set_seed(args.seed + accelerator.process_index)

    # recovery_state_file is only not none if it exists has the valid keys
    # state file is saved after all accelerator state, so if state file is valid then so is everything before
    recovery_checkpoint_dir = os.path.join(args.output_dir, "recovery_checkpoint")
    recovery_state_file_path = os.path.join(recovery_checkpoint_dir, "training_state.json")
    recovery_state_file = None
    if not args.debug_no_resume:
        try:
            recovery_state_file = json.load(open(recovery_state_file_path, 'r'))
            if args.wandb:
                assert set(recovery_state_file.keys()) == {"run_id", "global_step", "batch_idx", "epoch"}, 'wrong state keys'
            else:
                assert set(recovery_state_file.keys()) == {"global_step", "batch_idx", "epoch"}, 'wrong state keys'
            logger.info(f'loaded state from {recovery_state_file_path}')
        except Exception as e:
            recovery_state_file = None
            logger.info(f'could not load state file due to {e}')

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tracker_config = dict(vars(args))

        # recovery get runid
        wandb_init_args = {"name": args.tag}
        if (recovery_state_file is not None) and args.wandb:
            wandb_init_args['id'] = recovery_state_file["run_id"]
            wandb_init_args['resume'] = 'allow'

        accelerator.init_trackers(
            args.tracker_project_name,
            tracker_config,
            init_kwargs={"wandb": wandb_init_args}
        )
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
    if not args.no_flash_attn:
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

    # load config here to set attention dropout params
    config = LlamaConfig.from_pretrained(MODEL_NAME_TO_PATH[args.model_name])
    config.attention_dropout = args.full_attention_dropout
    config.demonstration_attention_dropout = args.demonstration_attention_dropout
    if args.demonstration_attention_dropout > 0.0:
        config._attn_implementation_autoset = False
        config._attn_implementation = 'eager'

    base_model = MyLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        config=config,
        **from_pretrained_kwargs,
    )

    # program dropout
    program_dropout = nn.Dropout(p=args.program_dropout)

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

    # initialize program embeddings
    prior_embeddings = ProgramEmbeddings(
        embedding=initialize_program_embeddings(
            base_model.model.embed_tokens.weight.data.detach().clone(),
            accelerator,
            ntokens=args.ntokens,
            cov_scale=1e-9,
        ),
    )
    program_embeddings = ProgramEmbeddings(
        embedding=initialize_program_embeddings(
            base_model.model.embed_tokens.weight.data.detach().clone(),
            accelerator,
            ntokens=args.ntokens,
            cov_scale=1e-9,
        ),
    )
    logger.info("Prior & Program embeddings initialized.")

    # vqvae codebook
    quantizer: Optional[Quantizer] = None
    if args.codebook_size > 0 or args.fsq_L != []:
        quantizer = Quantizer(
            codebook_size=args.codebook_size,
            hidden_size=base_model.config.hidden_size,
            fsq_L=args.fsq_L,
            device=accelerator.device,
        )
        logger.info("Codebook initialized.")

    # only keep these tokens, resize model embedding (eos == pad)
    # we do not include program tokens here, those are added later during training and inference
    if not args.no_separate_color_tokens:
        keep_tokens = [str(i) for i in range(31)]
        if args.no_dim:
            keep_tokens = []
    else:
        keep_tokens = [str(i) for i in range(31)]
        if args.no_dim:
            keep_tokens = [str(i) for i in range(10)]
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
        base_model.model.embed_tokens.num_embeddings = len(keep_token_ids)
        assert base_model.lm_head.bias is None
        base_model.lm_head.weight = nn.Parameter(base_model.lm_head.weight[keep_token_ids])
        base_model.lm_head.out_features = len(keep_token_ids)
        base_model.config.tie_word_embeddings = False

        if not args.no_separate_color_tokens:
            assert isinstance(color_embeddings, torch.Tensor)
            base_model.model.embed_tokens.weight = nn.Parameter(torch.cat([color_embeddings, base_model.model.embed_tokens.weight]))
            base_model.model.embed_tokens.num_embeddings += 10
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

    # vae projection (empty module if vae=False)
    vae_projection = None
    if args.vae:
        vae_projection = VaeProjection(
            mlp_factor=args.mlp_factor,
            latent_dim=model.config.hidden_size, # type: ignore
            device=accelerator.device,
        )
        logger.info("vae projection initialized")

    # shared norm
    program_norm = None
    if not args.no_normalize:
        program_norm = LlamaRMSNorm(model.config.hidden_size, eps=model.config.rms_norm_eps) # type: ignore
        logger.info("norm layer initialized")

    # ensure requires grad
    for param in prior_embeddings.parameters():
        param.requires_grad = True
    for param in program_embeddings.parameters():
        param.requires_grad = True
    if vae_projection is not None:
        for param in vae_projection.parameters():
            param.requires_grad = True
    if quantizer is not None:
        for param in quantizer.parameters():
            param.requires_grad = True
    if program_norm is not None:
        for param in program_norm.parameters():
            param.requires_grad = True

    # convert model weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for name, param in prior_embeddings.named_parameters():
        assert param.requires_grad
        param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for name, param in program_embeddings.named_parameters():
        assert param.requires_grad
        param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if vae_projection is not None:
        for name, param in vae_projection.named_parameters():
            assert param.requires_grad
            param.data = param.data.to(torch.float32) # keep vaeproj at float32
    if quantizer is not None:
        for param in quantizer.parameters():
            assert param.requires_grad
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if program_norm is not None:
        for param in program_norm.parameters():
            assert param.requires_grad
            param.data = param.data.to(torch.float32) # keep norm at float32
    logger.info(f'converted most trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # number of parameters
    print_trainable_parameters(model)
    prior_embeddings_n_params = sum(p.numel() for p in prior_embeddings.parameters())
    program_embeddings_n_params = sum(p.numel() for p in program_embeddings.parameters())
    logger.info(f'prior embedding params {three_commas(prior_embeddings_n_params)}')
    logger.info(f'program embedding params {three_commas(program_embeddings_n_params)}')
    if vae_projection is not None:
        vae_projection_n_params = sum(p.numel() for p in vae_projection.parameters())
        logger.info(f'vae projection params {three_commas(vae_projection_n_params)}')
    if quantizer is not None:
        quantizer_n_params = sum(p.numel() for p in quantizer.parameters())
        logger.info(f'quantizer params {three_commas(quantizer_n_params)}')
    if program_norm is not None:
        program_norm_n_params = sum(p.numel() for p in program_norm.parameters())
        logger.info(f'program norm params {three_commas(program_norm_n_params)}')

    # model size
    logger.info(f'model size {round(model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    logger.info(f'prior embeddings size {round(get_memory_footprint(prior_embeddings) / 1024 ** 3, 2)}GB')
    logger.info(f'program embeddings size {round(get_memory_footprint(program_embeddings) / 1024 ** 3, 2)}GB')
    if vae_projection is not None:
        logger.info(f'vae projection size {round(get_memory_footprint(vae_projection) / 1024 ** 3, 2)}GB')
    if quantizer is not None:
        logger.info(f'quantizer size {round(get_memory_footprint(quantizer) / 1024 ** 3, 2)}GB')
    if program_norm is not None:
        logger.info(f'program norm size {round(get_memory_footprint(program_norm) / 1024 ** 3, 2)}GB')

    # Build training dataset
    train_dataset = TrainDataset(
        train_data_dir=args.train_data_dir,
        eval_train_dir=args.eval_train_dir,
        verifier_file=args.verifier_file,
        re_arc_ratio=args.re_arc_ratio,
        concept_arc_ratio=args.concept_arc_ratio,
        arc_heavy_ratio=args.arc_heavy_ratio,
        tokenizer=tokenizer,
        total_steps=args.samples_per_epoch,
        extra_augment_ratio=args.extra_augment_ratio,
        extra_augment_single_grid=args.extra_augment_single_grid,
        seed=args.seed,
        process_index=accelerator.process_index,
        ntokens=args.ntokens,
        debug_fixed_order=args.debug_fixed_order,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        train_pad_side=args.train_pad_side,
        debug_train_data=args.debug_train_data,
        no_color_permute=args.no_color_permute,
        no_pair_permute=args.no_pair_permute,
        no_d8=args.no_d8,
        min_num_pair=args.min_num_pair,
        max_num_pair=args.max_num_pair,
        no_train_original=args.no_train_original,
        only_train_original=args.only_train_original,
        debug_len=args.debug_len,
        num_workers=args.num_workers,
        no_dim=args.no_dim,
        no_separate_color_tokens=args.no_separate_color_tokens,
        max_seq_len=args.max_seq_len,
        no_bos=args.no_bos,
        only_first_bos=args.only_first_bos,
        same_task_identifier_across_gpus=args.cache_size_per_task > 0, # when using cache + multigpu, need same task identifier
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
    if args.invar_loss_lambda > 0.0:
        train_collate_fn = partial(collate_fn_train_invar, dataset=train_dataset)
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

    if args.debug_train_data:
        os.system("rm -rf ./debug_train_data")
        os.makedirs("./debug_train_data")
        os.system("chmod -R 777 ./debug_train_data")

    # Param groups for LoRA
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "embed" in name or "lm_head" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    if vae_projection is not None:
        for param in vae_projection.parameters():
            other_params.append(param)
    if quantizer is not None:
        for param in quantizer.parameters():
            other_params.append(param)
    if program_norm is not None:
        for param in program_norm.parameters():
            other_params.append(param)
    prior_params = [param for param in prior_embeddings.parameters()]
    program_params = [param for param in program_embeddings.parameters()]

    optimizer_grouped_params = [
        {"params": embedding_params, "lr": args.lr_embedding},
        {"params": prior_params, "lr": args.lr_prior},
        {"params": program_params, "lr": args.lr_program},
        {"params": other_params, "lr": args.lr_other},
    ]
    all_params = embedding_params + prior_params + program_params + other_params
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay) # type: ignore
    elif args.optimizer == 'adamw8bit':
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_params, weight_decay=args.weight_decay)
        # optimizer = bnb.optim.PagedAdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_params) # type: ignore
    logger.info(f"Optimizer with {len(embedding_params)} embed-params lr={args.lr_embedding}")
    logger.info(f"Optimizer with {len(prior_params)} prior-params lr={args.lr_prior}")
    logger.info(f"Optimizer with {len(program_params)} program-params lr={args.lr_program}")
    logger.info(f"Optimizer with {len(other_params)} other-params lr={args.lr_other}")

    # LR schedule
    steps_per_epoch = args.samples_per_epoch // (args.train_batch_size * args.grad_accum_steps * accelerator.num_processes)
    num_training_steps = steps_per_epoch * args.num_epochs
    if args.lr_scheduler == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps * args.warmup_epochs,
            num_training_steps=num_training_steps * args.grad_accum_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps * args.warmup_epochs,
        )

    # lr scheduler is not automatically registered, do that
    accelerator.register_for_checkpointing(lr_scheduler)

    # lambda schedulers
    kl_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.kl_loss_lambda,
        start_epoch=args.kl_loss_offset_epochs,
        linear_epochs=args.kl_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    codebook_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.codebook_loss_lambda,
        start_epoch=args.codebook_loss_offset_epochs,
        linear_epochs=args.codebook_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    commitment_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.commitment_loss_lambda,
        start_epoch=args.commitment_loss_offset_epochs,
        linear_epochs=args.commitment_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    program_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.program_loss_lambda,
        start_epoch=args.program_loss_offset_epochs,
        linear_epochs=args.program_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    consistency_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.consistency_loss_lambda,
        start_epoch=args.consistency_loss_offset_epochs,
        linear_epochs=args.consistency_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    invar_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.invar_loss_lambda,
        start_epoch=args.invar_loss_offset_epochs,
        linear_epochs=args.invar_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    # kl_loss_lambda_scheduler.visualize(num_training_steps, 'kl.jpg')
    # codebook_loss_lambda_scheduler.visualize(num_training_steps, 'codebook.jpg')
    # commitment_loss_lambda_scheduler.visualize(num_training_steps, 'commitment.jpg')
    # program_loss_lambda_scheduler.visualize(num_training_steps, 'program.jpg')
    # consistency_loss_lambda_scheduler.visualize(num_training_steps, 'consistency.jpg')
    # invar_loss_lambda_scheduler.visualize(num_training_steps, 'invar.jpg')

    contrastive_loss = ContrastiveLoss(margin=args.invar_loss_margin)

    # Prepare with accelerator
    (
        model,
        prior_embeddings,
        program_embeddings,
        vae_projection,
        quantizer,
        program_norm,
        optimizer,
        train_loader,
    ) = accelerator.prepare(
        model,
        prior_embeddings,
        program_embeddings,
        vae_projection,
        quantizer,
        program_norm,
        optimizer,
        train_loader,
    )

    # recovery
    start_epoch = 0
    global_step = 0
    resume_batch_idx = 0
    if recovery_state_file is not None:
        logger.info(f"Loading checkpoint from {recovery_checkpoint_dir}")
        accelerator.load_state(recovery_checkpoint_dir)
        start_epoch = recovery_state_file["epoch"]
        global_step = recovery_state_file["global_step"]
        resume_batch_idx = recovery_state_file["batch_idx"]

    assert isinstance(model, (nn.Module, DistributedDataParallel))
    assert isinstance(prior_embeddings, (ProgramEmbeddings, DistributedDataParallel))
    assert isinstance(program_embeddings, (ProgramEmbeddings, DistributedDataParallel))
    if vae_projection is not None:
        assert isinstance(vae_projection, (VaeProjection, DistributedDataParallel))
    if quantizer is not None:
        assert isinstance(quantizer, (Quantizer, DistributedDataParallel))
    if program_norm is not None:
        assert isinstance(program_norm, (LlamaRMSNorm, DistributedDataParallel))

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
        ntokens=args.ntokens,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        train_pad_side=args.train_pad_side,
        gen_pad_side=args.gen_pad_side,
        debug_len=args.debug_len,
        no_dim=args.no_dim,
        no_separate_color_tokens=args.no_separate_color_tokens,
        extra_inference_pairs=args.extra_inference_pairs,
        limit_inference_pairs=args.limit_inference_pairs,
        limit_inference_pairs_strict=args.limit_inference_pairs_strict,
        max_num_train_pair=args.max_num_pair - 1,
        max_seq_len=args.max_seq_len,
        no_bos=args.no_bos,
        only_first_bos=args.only_first_bos,
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
        ntokens=args.ntokens,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        train_pad_side=args.train_pad_side,
        gen_pad_side=args.gen_pad_side,
        debug_len=args.debug_len,
        no_dim=args.no_dim,
        no_separate_color_tokens=args.no_separate_color_tokens,
        extra_inference_pairs=args.extra_inference_pairs,
        limit_inference_pairs=args.limit_inference_pairs,
        limit_inference_pairs_strict=args.limit_inference_pairs_strict,
        max_num_train_pair=args.max_num_pair - 1,
        max_seq_len=args.max_seq_len,
        no_bos=args.no_bos,
        only_first_bos=args.only_first_bos,
    )
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_train_dataset) # only use tokenizer, padding info
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

    progress_bar = tqdm(
        range(num_training_steps),
        desc="Train Steps",
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Total Train Steps")

    # model saving
    last_save_model_path = None
    epoch_to_eval_exact_acc = {}

    # recovery
    logger.info(f"start/resume training from epoch {start_epoch} global_step {global_step} batch {resume_batch_idx}")
    if global_step > 0:
        progress_bar.update(global_step)

    # program cache
    program_cache: Optional[ProgramCache] = None
    if args.cache_size_per_task > 0:
        program_cache = ProgramCache(cache_size_per_task=args.cache_size_per_task, seed=args.seed)

        # debug: allocate full cache here, make sure to comment out
        # program_cache.debug_full_cache(args.ntokens, model.config.hidden_size, num_task_identifiers=400 * 8)

    # train!
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        prior_embeddings.train()
        program_embeddings.train()
        if vae_projection is not None:
            vae_projection.train()
        if quantizer is not None:
            quantizer.train()
        if program_norm is not None:
            program_norm.train()
        program_dropout.train()

        ce_loss_accum = 0.0
        kl_loss_accum = 0.0
        codebook_loss_accum = 0.0
        commitment_loss_accum = 0.0
        perplexity_accum = 0.0
        program_loss_accum = 0.0
        consistency_loss_accum = 0.0
        invar_loss_accum = 0.0
        total_loss_accum = 0.0
        grad_norm_accum = 0.0
        if args.loss_on_first:
            ce_losses_accum = [0.0 for _ in range(args.max_num_pair)]
        else:
            ce_losses_accum = [0.0 for _ in range(args.max_num_pair - 1)]

        train_dataset.set_rngs(epoch)
        for batch_idx, batch_data in enumerate(train_loader):
            # skip batch idx if recovered run already encountered it
            if epoch == start_epoch and batch_idx < resume_batch_idx:
                continue

            input_ids = [x.to(accelerator.device) for x in batch_data["input_ids"]]
            attention_mask = [x.to(accelerator.device) for x in batch_data["attention_mask"]]
            label_ids = [x.to(accelerator.device) for x in batch_data["label_ids"]]
            input_ids_lens = batch_data["input_ids_lens"]
            num_pairs = batch_data["num_pairs"]
            is_same = batch_data["is_same"]
            task_identifiers = batch_data["task_identifiers"]

            train_codebook_only = (global_step < args.warmup_cookbook_only_epochs * steps_per_epoch)

            with accelerator.accumulate(model, prior_embeddings, program_embeddings, vae_projection, quantizer, program_norm):
                with accelerator.autocast():
                    ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, program_loss, consistency_loss, \
                        invar_loss, total_loss, log_ce_losses, inferred_programs = model_loss(
                        # model
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        vae_projection=vae_projection,
                        quantizer=quantizer,
                        program_norm=program_norm,
                        program_dropout=program_dropout,
                        tokenizer=tokenizer,
                        accelerator=accelerator,
                        # data
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        label_ids=label_ids,
                        input_ids_lens=input_ids_lens,
                        num_pairs=num_pairs,
                        task_identifiers=task_identifiers,
                        # others
                        ntokens=args.ntokens,
                        pad_side=args.train_pad_side,
                        kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
                        codebook_loss_lambda_scheduler=codebook_loss_lambda_scheduler,
                        commitment_loss_lambda_scheduler=commitment_loss_lambda_scheduler,
                        program_loss_lambda_scheduler=program_loss_lambda_scheduler,
                        consistency_loss_lambda_scheduler=consistency_loss_lambda_scheduler,
                        invar_loss_lambda_scheduler=invar_loss_lambda_scheduler,
                        global_step=global_step,
                        no_residual=args.no_residual,
                        no_discrete_prior=args.no_discrete_prior,
                        program_type=args.program_type,
                        train_codebook_only=train_codebook_only,
                        ar_gradient_checkpointing=args.ar_gradient_checkpointing,
                        program_noise_std=args.program_noise_std,
                        subset_kl=args.subset_kl,
                        long_context_checkpointing_threshold=args.long_context_checkpointing_threshold,
                        token_weighted_loss=args.token_weighted_loss,
                        weird_cast=args.weird_cast,
                        full_demonstration_dropout=args.full_demonstration_dropout,
                        partial_demonstration_dropout=args.partial_demonstration_dropout,
                        loss_on_first=args.loss_on_first,
                        debug=args.debug,
                        contrastive_loss=contrastive_loss,
                        is_same=is_same,
                        short_context=args.short_context,
                        attention_reduction_ratio=args.attention_reduction_ratio,
                        program_cache=program_cache,
                        prior_embed_ratio=args.prior_embed_ratio,
                    )

                # only log one process
                ce_loss_accum += ce_loss.item() / args.grad_accum_steps
                kl_loss_accum += kl_loss.item() / args.grad_accum_steps
                codebook_loss_accum += codebook_loss.item() / args.grad_accum_steps
                commitment_loss_accum += commitment_loss.item() / args.grad_accum_steps
                perplexity_accum += perplexity.item() / args.grad_accum_steps
                program_loss_accum += program_loss.item() / args.grad_accum_steps
                consistency_loss_accum += consistency_loss.item() / args.grad_accum_steps
                invar_loss_accum += invar_loss.item() / args.grad_accum_steps
                total_loss_accum += total_loss.item() / args.grad_accum_steps
                for pair_i, pair_ce_loss in enumerate(log_ce_losses):
                    ce_losses_accum[pair_i] += pair_ce_loss.item() / args.grad_accum_steps

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm_accum += accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item() # type: ignore
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if program_cache is not None:
                # gather task identifiers and programs
                if accelerator.num_processes > 1:
                    # gather task, list of strings
                    all_task_identifiers = [None] * accelerator.num_processes
                    torch.distributed.all_gather_object(all_task_identifiers, task_identifiers) # type: ignore
                    assert len(set(tuple(ids) for ids in all_task_identifiers)) == 1
                    all_task_identifiers = [x for sub in all_task_identifiers for x in sub] # type: ignore
                    # gather programs as a single tensor
                    all_inferred_programs = accelerator.gather(inferred_programs)
                else:
                    all_task_identifiers = task_identifiers
                    all_inferred_programs = inferred_programs

                # update program cache
                assert isinstance(all_task_identifiers, list)
                assert isinstance(all_inferred_programs, torch.Tensor)
                assert len(all_task_identifiers) == all_inferred_programs.shape[0]
                for task_identifier, inferred_programs in zip(all_task_identifiers, all_inferred_programs):
                    for program in inferred_programs:
                        program_cache.update(task_identifier, program.detach().clone().cpu()) # VRAM is expensive, use RAM # type: ignore

            if accelerator.sync_gradients:
                global_step += 1

                # debug: make sure program cache has the correct number of items (only true until cache is full), make sure to comment
                # if program_cache is not None:
                #     assert program_cache.get_num_items_in_cache() == accelerator.num_processes * args.min_num_pair * args.grad_accum_steps * args.train_batch_size * global_step
                #     print(f"processindex{accelerator.process_index} cache size is {round(program_cache.get_memory_footprint() / 1024 ** 3, 2)}GB)")
                #     program_cache.validate()

                if global_step % args.log_every == 0:
                    progress_bar.update(args.log_every)
                    train_metrics = {
                        "train/ce_loss": ce_loss_accum,
                        "train/kl_loss": kl_loss_accum,
                        "train/codebook_loss": codebook_loss_accum,
                        "train/commitment_loss": commitment_loss_accum,
                        "train/perplexity_accum": perplexity_accum,
                        "train/program_loss_accum": program_loss_accum,
                        "train/consistency_loss_accum": consistency_loss_accum,
                        "train/invar_loss_accum": invar_loss_accum,
                        "train/total_loss": total_loss_accum,
                        "train/grad_norm": grad_norm_accum,
                        "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                        "train/lr_prior": lr_scheduler.get_last_lr()[1],
                        "train/lr_program": lr_scheduler.get_last_lr()[2],
                        "train/lr_other": lr_scheduler.get_last_lr()[3],
                        **{f"train/ce_loss_pair_{pair_i}": pair_ce_loss_accum for pair_i, pair_ce_loss_accum in enumerate(ce_losses_accum)},
                    }
                    if program_cache is not None:
                        train_metrics["train/avg_cache_len"] = program_cache.get_average_cache_len()

                    try:
                        accelerator.log(train_metrics, step=global_step)
                    except:
                        logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                ce_loss_accum = 0.0
                kl_loss_accum = 0.0
                codebook_loss_accum = 0.0
                commitment_loss_accum = 0.0
                perplexity_accum = 0.0
                program_loss_accum = 0.0
                consistency_loss_accum = 0.0
                invar_loss_accum = 0.0
                total_loss_accum = 0.0
                grad_norm_accum = 0.0
                if args.loss_on_first:
                    ce_losses_accum = [0.0 for _ in range(args.max_num_pair)]
                else:
                    ce_losses_accum = [0.0 for _ in range(args.max_num_pair - 1)]

                # recovery
                if global_step % args.save_every == 0:
                    if accelerator.is_main_process:
                        if os.path.exists(recovery_checkpoint_dir):
                            shutil.rmtree(recovery_checkpoint_dir)
                        os.makedirs(recovery_checkpoint_dir, exist_ok=True)
                        accelerator.save_state(recovery_checkpoint_dir)
                        # must save state AFTER everything else
                        # we use it determine whether the save is valid (not interrupted in middle of saving)
                        state = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "batch_idx": batch_idx + 1,
                        }
                        if args.wandb:
                            assert wandb.run is not None
                            state['run_id'] = wandb.run.id
                        json.dump(state, open(recovery_state_file_path, "w"))
                        logger.info(f"saved training at epoch {epoch} global_step {global_step} batch_idx {batch_idx + 1}")
                        logger.info(f"saved state to {recovery_state_file_path}")

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
            torch.cuda.empty_cache()
            gc.collect()

            no_codebook = epoch < args.warmup_cookbook_only_epochs

            train_exact_acc, train_valid_grid, train_correct_grid_dim, train_token_acc, train_relaxed_token_acc, train_texts, \
                train_votes, train_competition_sub_acc, train_competition_all_acc, _ = evaluate(
                desc="eval_train",
                task_to_ttt_path=None,
                ttt_param_names=None,
                model=model,
                prior_embeddings=prior_embeddings,
                program_embeddings=program_embeddings,
                vae_projection=vae_projection,
                quantizer=quantizer,
                program_norm=program_norm,
                dataset=eval_train_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                dry_eval_run=args.dry_eval_run,
                no_residual=args.no_residual,
                no_discrete_prior=args.no_discrete_prior,
                output_dir=args.output_dir,
                gs_iters=args.train_gs_iters,
                gs_lr=args.train_gs_lr,
                gs_beta1=args.train_gs_beta1,
                gs_beta2=args.train_gs_beta2,
                gs_batch_size=args.train_gs_batch_size,
                gs_optimizer=args.train_gs_optimizer,
                gs_max_grad_norm=args.train_gs_max_grad_norm,
                gs_lr_scheduler=args.train_gs_lr_scheduler,
                gs_take_best=args.train_gs_take_best,
                gs_train_past_kv=args.train_gs_train_past_kv,
                no_codebook=no_codebook,
                weird_cast=args.weird_cast,
                short_context=args.short_context,
                kv_pad_side=args.kv_pad_side,
                attention_reduction_ratio=args.attention_reduction_ratio,
            )
            eval_exact_acc, eval_valid_grid, eval_correct_grid_dim, eval_token_acc, eval_relaxed_token_acc, eval_texts, \
                eval_votes, eval_competition_sub_acc, eval_competition_all_acc, _ = evaluate(
                desc="eval_eval",
                task_to_ttt_path=None,
                ttt_param_names=None,
                model=model,
                prior_embeddings=prior_embeddings,
                program_embeddings=program_embeddings,
                vae_projection=vae_projection,
                quantizer=quantizer,
                program_norm=program_norm,
                dataset=eval_eval_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                dry_eval_run=args.dry_eval_run,
                no_residual=args.no_residual,
                no_discrete_prior=args.no_discrete_prior,
                output_dir=args.output_dir,
                gs_iters=args.eval_gs_iters,
                gs_lr=args.eval_gs_lr,
                gs_beta1=args.eval_gs_beta1,
                gs_beta2=args.eval_gs_beta2,
                gs_batch_size=args.eval_gs_batch_size,
                gs_optimizer=args.eval_gs_optimizer,
                gs_max_grad_norm=args.eval_gs_max_grad_norm,
                gs_lr_scheduler=args.eval_gs_lr_scheduler,
                gs_take_best=args.eval_gs_take_best,
                gs_train_past_kv=args.eval_gs_train_past_kv,
                no_codebook=no_codebook,
                weird_cast=args.weird_cast,
                short_context=args.short_context,
                kv_pad_side=args.kv_pad_side,
                attention_reduction_ratio=args.attention_reduction_ratio,
            )

            torch.cuda.empty_cache()
            gc.collect()

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
                        save_model_path, save_prior_embeddings_path, save_program_embeddings_path, \
                            save_vae_projection_path, save_quantizer_path, save_program_norm_path = last_save_model_path
                        rm_cmd = f"rm -rf {save_model_path} {save_prior_embeddings_path}"
                        rm_cmd += f" {save_program_embeddings_path}"
                        if save_vae_projection_path is not None:
                            rm_cmd += f" {save_vae_projection_path}"
                        if save_quantizer_path is not None:
                            rm_cmd += f" {save_quantizer_path}"
                        if save_program_norm_path is not None:
                            rm_cmd += f" {save_program_norm_path}"
                        os.system(rm_cmd)
                    last_save_model_path = save_train_model(
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        vae_projection=vae_projection,
                        quantizer=quantizer,
                        program_norm=program_norm,
                        output_dir=args.output_dir,
                        epoch=epoch,
                    )
                epoch_to_eval_exact_acc[epoch] = eval_exact_acc

    # # debug: check if train eval and ttt load the same exact model
    # input_ids = torch.tensor([list(range(20)), list(range(20))], device=accelerator.device, dtype=torch.int64)
    # attention_mask = torch.full(input_ids.shape, 1, device=accelerator.device, dtype=torch.int64)
    # ce_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
    # print(ce_loss.item())
    # breakpoint()

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
