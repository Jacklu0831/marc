import random
from custom_llama import MyLlamaModel
import gc
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Optional, Iterator, Any
import pprint
import math
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
import logging
import datasets
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import (
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
    collate_fn_gs,
    get_torch_generator,
)
from oracle_fit import create_ground_truth_net

import os
os.system('nvidia-smi')
os.environ["NCCL_TIMEOUT"] = "28800" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "28800000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

logger = get_logger(__name__, log_level="INFO")


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
        return random.choice(self.cache[task_identifier]).to(device)

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
    net_input_dim: int,
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    # data
    inputs_embeds: List[torch.Tensor],
    attention_mask: List[torch.Tensor],
    inputs_embeds_lens: List[List[int]],
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

    # samples do not have to be the same number of pairs in evaluation, but try to parallelize it anyway
    # assert min(num_pairs) >= 2 # at least 2 train
    assert max(num_pairs) == len(inputs_embeds)

    assert len(set(len(ids) for ids in inputs_embeds)) == 1 # batch size uniform across pair_idx
    batch_size = len(inputs_embeds[0])
    device = inputs_embeds[0].device

    # reused tensors
    pad_embeds = torch.full((inputs_embeds[0].shape[-1],), 0.1234, device=device, dtype=torch.float32)

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
        for pair_j, lens in enumerate(inputs_embeds_lens):
            if pair_j > 0:
                start += attention_mask[pair_j - 1].shape[1] + ntokens
            max_l = attention_mask[pair_j].shape[1]
            for batch_i, l in enumerate(lens):
                if pair_j < num_pairs[batch_i]:
                    s = start if pad_side == 'right' else start + max_l - l
                    demonstration_intervals[batch_i].append((s, s + l))

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_inputs_embeds_lens) in enumerate(zip(inputs_embeds, attention_mask, inputs_embeds_lens)):

        # STEP 0: filter out samples that are not this many pairs (for eval)
        avail_mask = torch.tensor([int(pair_i < n) for n in num_pairs], dtype=torch.bool)
        pair_inputs_embeds = pair_inputs_embeds[avail_mask]
        pair_attention_mask = pair_attention_mask[avail_mask]
        pair_inputs_embeds_lens = [l for l, m in zip(pair_inputs_embeds_lens, avail_mask) if m]
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
            lens=pair_inputs_embeds_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((n_avail, ntokens), 1, device=device, dtype=torch.int64),
            lens=pair_inputs_embeds_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )

        # update lens to reflect the extra program
        pair_inputs_embeds_lens = [l + ntokens for l in pair_inputs_embeds_lens]

        # save attention mask
        pair_attention_mask_no_program_embed = pair_attention_mask.detach().clone()

        # STEP 2: append new program input to the right
        emb = program_embeddings("dummy")[None, ...].expand(n_avail, -1, -1)
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=(emb.to(torch.bfloat16) if weird_cast else emb),
            lens=pair_inputs_embeds_lens,
            insert_side="right",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((n_avail, ntokens), 1, device=device, dtype=torch.int64),
            lens=pair_inputs_embeds_lens,
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
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                position_start = mask_for_kv.sum()
                n_new_positions = mask_after_kv.sum()
                new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=torch.int64)
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
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                n_new_positions = m.sum()
                new_positions = torch.tensor(range(n_new_positions), device=device, dtype=torch.int64)
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
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[0], pair_inputs_embeds_lens)]),
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[1], pair_inputs_embeds_lens)]),
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
        hidden_states = hidden_states[:, :, :net_input_dim] # toy subset dimensions
        new_programs = extract_program_from_right_side(
            data=hidden_states,
            lens=pair_inputs_embeds_lens,
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


def compute_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    individual_loss: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:

    assert predictions.dim() == 2
    assert predictions.shape == labels.shape

    losses = []
    for pred, gt in zip(predictions, labels):
        gt_mask = (gt != -100)
        pred_mask = torch.roll(gt_mask, shifts=-1, dims=0)
        assert not pred_mask[-1]

        loss = ((pred[pred_mask] - gt[gt_mask]) ** 2.0).mean()
        losses.append(loss)

    if individual_loss:
        return losses

    return sum(losses) / len(losses) # type: ignore


def model_loss(
    # model
    model: Union[nn.Module, DistributedDataParallel],
    net_input_dim: int,
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    program_dropout: nn.Dropout,
    accelerator: Accelerator,
    # data
    inputs_embeds: List[torch.Tensor],
    attention_mask: List[torch.Tensor],
    labels: List[torch.Tensor],
    inputs_embeds_lens: List[List[int]],
    num_pairs: List[int],
    is_same: bool,
    task_identifiers: List[str],
    # others
    ntokens: int,
    pad_side: str,
    kl_loss_lambda_scheduler: LambdaScheduler,
    codebook_loss_lambda_scheduler: LambdaScheduler,
    commitment_loss_lambda_scheduler: LambdaScheduler,
    consistency_loss_lambda_scheduler: LambdaScheduler,
    invar_loss_lambda_scheduler: LambdaScheduler,
    global_step: int,
    no_residual: bool,
    no_discrete_prior: bool,
    train_codebook_only: bool,
    program_noise_std: float,
    subset_kl: bool,
    weird_cast: bool,
    full_demonstration_dropout: bool,
    partial_demonstration_dropout: bool,
    debug: bool,
    contrastive_loss: ContrastiveLoss,
    short_context: bool,
    attention_reduction_ratio: float,
    program_cache: Optional[ProgramCache],
    prior_embed_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:

    # all samples should have same number of pairs
    assert len(set(num_pairs)) == 1
    n_pairs = num_pairs[0]
    assert n_pairs == len(inputs_embeds)

    assert len(set(len(emb) for emb in inputs_embeds)) == 1 # batch size uniform across pair_idx
    batch_size = len(inputs_embeds[0])
    device = inputs_embeds[0].device

    # reused tensors
    pad_embeds = torch.full((inputs_embeds[0].shape[-1],), 0.1234, device=device, dtype=torch.float32)

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
    do_save_programs = (debug or invar_loss_lambda > 0.0 or consistency_loss_lambda > 0.0 or (program_cache is not None))
    saved_all_programs = []
    if do_save_programs:
        saved_all_programs = [prev_programs]

    # precompute demonstration intervals for these three tries
    demonstration_intervals = [[] for _ in range(batch_size)]
    if full_demonstration_dropout > 0.0 or partial_demonstration_dropout > 0.0 or attention_reduction_ratio != 1.0:
        start = ntokens # keep track of start of each demonstration pair
        for pair_j, lens in enumerate(inputs_embeds_lens[:-1]):
            if pair_j > 0:
                start += attention_mask[pair_j - 1].shape[1] + ntokens
            max_l = attention_mask[pair_j].shape[1]
            for batch_i, l in enumerate(lens):
                s = start if pad_side == 'right' else start + max_l - l
                demonstration_intervals[batch_i].append((s, s + l))

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_labels, pair_inputs_embeds_lens) in enumerate(zip(inputs_embeds, attention_mask, labels, inputs_embeds_lens)):
        # STEP 1: prepend the last predicted program for all pairs except the first
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
            lens=pair_inputs_embeds_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((batch_size, ntokens), 1, device=device, dtype=torch.int64),
            lens=pair_inputs_embeds_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )
        pair_labels = insert_based_on_sides(
            data=pair_labels,
            to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=torch.float32),
            lens=pair_inputs_embeds_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=-100,
        )

        # update lens to reflect the extra program
        pair_inputs_embeds_lens = [l + ntokens for l in pair_inputs_embeds_lens]

        # save attention mask
        pair_attention_mask_no_program_embed = pair_attention_mask.detach().clone()

        # STEP 2: append new program input to the right for all pairs except the last
        if (pair_i < n_pairs - 1) or (program_cache is not None):
            emb = program_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
            pair_inputs_embeds = insert_based_on_sides(
                data=pair_inputs_embeds,
                to_insert=(emb.to(torch.bfloat16) if weird_cast else emb),
                lens=pair_inputs_embeds_lens,
                insert_side="right",
                pad_side=pad_side,
                pad_id=pad_embeds,
            )
            pair_attention_mask = insert_based_on_sides(
                data=pair_attention_mask,
                to_insert=torch.full((batch_size, ntokens), 1, device=device, dtype=torch.int64),
                lens=pair_inputs_embeds_lens,
                insert_side="right",
                pad_side=pad_side,
                pad_id=0,
            )
            pair_labels = insert_based_on_sides(
                data=pair_labels,
                to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=torch.float32),
                lens=pair_inputs_embeds_lens,
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
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                position_start = mask_for_kv.sum()
                n_new_positions = mask_after_kv.sum()
                new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=torch.int64)
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
                sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                n_new_positions = m.sum()
                new_positions = torch.tensor(range(n_new_positions), device=device, dtype=torch.int64)
                if pad_side == "right":
                    sequence_position_ids[:n_new_positions] = new_positions
                else:
                    sequence_position_ids[-n_new_positions:] = new_positions
                position_ids.append(sequence_position_ids)
            position_ids = torch.stack(position_ids)
            model_kwargs["position_ids"] = position_ids

        model_out = model(**model_kwargs)
        pair_loss = compute_loss(model_out.predictions.squeeze(-1), pair_labels, individual_loss=False)
        ce_losses.append(pair_loss)

        # STEP 4: update kv
        if (pair_i < n_pairs - 1) and not short_context:
            assert model_out is not None

            # remove end program from kv cache
            old_key_values_len = prev_past_key_values[0][0].shape[2] if prev_past_key_values is not None else 0
            if pad_side == "right":
                new_past_key_values = tuple(
                    (
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[0], pair_inputs_embeds_lens)]),
                        torch.stack([torch.cat([kv[:, :old_key_values_len+l], kv[:, old_key_values_len+l+ntokens:]], dim=1) for kv, l in zip(layer_kv[1], pair_inputs_embeds_lens)]),
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
            hidden_states = hidden_states[:, :, :net_input_dim] # toy subset dimensions
            new_programs = extract_program_from_right_side(
                data=hidden_states,
                lens=pair_inputs_embeds_lens,
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

    # aggregate losses
    ce_loss = sum(ce_losses) / len(ce_losses)
    kl_loss = sum(kl_losses) / len(kl_losses) if len(kl_losses) > 0 else torch.tensor(0.0, device=device)
    codebook_loss = sum(codebook_losses) / len(codebook_losses) if len(codebook_losses) > 0 else torch.tensor(0.0, device=device)
    commitment_loss = sum(commitment_losses) / len(commitment_losses) if len(commitment_losses) > 0 else torch.tensor(0.0, device=device)
    total_loss = ce_loss + kl_loss_lambda * kl_loss + codebook_loss * codebook_loss_lambda + commitment_loss_lambda * commitment_loss + invar_loss_lambda * invar_loss + consistency_loss_lambda * consistency_loss

    # logging perplexity for debugging vqvae
    perplexity = sum(perplexitys) / len(perplexitys) if len(perplexitys) > 0 else torch.tensor(0.0, device=device)

    # print(ce_loss.item())
    # breakpoint()

    inferred_programs = None
    if program_cache is not None:
        inferred_programs = torch.stack(saved_all_programs[1:]) # all programs except the prior
        inferred_programs = inferred_programs.permute(1, 0, 2, 3) # (batch_size, num_program, ntoken, hiddendim)

    return ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, consistency_loss, invar_loss, total_loss, ce_losses, \
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

    # didnt use grad accum, dont think needed

    assert prev_programs.shape[0] == 1
    if past_key_values is not None:
        assert past_key_values[0][0].shape[0] == 1
    if past_key_values_attention_mask is not None:
        assert past_key_values_attention_mask.shape[0] == 1

    # dataset and dataloader
    gs_dataset = GSDataset(
        data=eval_dataset.data[batch_idx],
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        pad_side=eval_dataset.pad_side,
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

    pad_embeds = torch.full((eval_dataset.net_input_dim,), 0.1234, device=accelerator.device, dtype=torch.float32)

    # train!
    while curr_iter < iters:
        for batch in gs_loader:
            pair_inputs_embeds = batch["inputs_embeds"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_labels = batch["labels"].to(accelerator.device)
            pair_inputs_embeds_lens = batch["inputs_embeds_lens"]
            device = pair_inputs_embeds.device

            with accelerator.autocast():
                pair_inputs_embeds = insert_based_on_sides(
                    data=pair_inputs_embeds,
                    to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
                    lens=pair_inputs_embeds_lens,
                    insert_side="left",
                    pad_side=gs_dataset.pad_side,
                    pad_id=pad_embeds,
                )
                pair_attention_mask = insert_based_on_sides(
                    data=pair_attention_mask,
                    to_insert=torch.full((batch_size, eval_dataset.ntokens), 1, device=device, dtype=torch.int64),
                    lens=pair_inputs_embeds_lens,
                    insert_side="left",
                    pad_side=gs_dataset.pad_side,
                    pad_id=0,
                )
                pair_labels = insert_based_on_sides(
                    data=pair_labels,
                    to_insert=torch.full((batch_size, eval_dataset.ntokens), -100, device=device, dtype=torch.float32),
                    lens=pair_inputs_embeds_lens,
                    insert_side="left",
                    pad_side=gs_dataset.pad_side,
                    pad_id=-100,
                )

                if not short_context:
                    assert past_key_values_attention_mask is not None
                    pair_attention_mask = torch.cat([past_key_values_attention_mask, pair_attention_mask], dim=1)

                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
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
                        sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                        position_start = mask_for_kv.sum()
                        n_new_positions = mask_after_kv.sum()
                        new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=torch.int64)
                        if gs_dataset.pad_side == "right":
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
                        sequence_position_ids = torch.zeros(pair_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                        n_new_positions = m.sum()
                        new_positions = torch.tensor(range(n_new_positions), device=device, dtype=torch.int64)
                        if gs_dataset.pad_side == "right":
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
    no_residual: bool,
    no_discrete_prior: bool,
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

    distributed_state = PartialState()
    loss_list = []

    data_idxs = list(range(len(dataset)))
    assert len(data_idxs) >= accelerator.num_processes # avoid padding issue

    with distributed_state.split_between_processes(data_idxs) as process_data_idxs:
        assert isinstance(process_data_idxs, list)
        n_batches = math.ceil(len(process_data_idxs) / batch_size)
        data_idx_iterator = tqdm(chunks(process_data_idxs, batch_size), total=n_batches, desc=desc)  # type: ignore

        for batch_idxs in data_idx_iterator:
            bs = len(batch_idxs)
            batch_data = [dataset[i] for i in batch_idxs]
            batch = collate_fn(batch_data)

            # get tensors
            inputs_embeds = [x.to(accelerator.device) for x in batch["inputs_embeds"]]
            attention_mask = [x.to(accelerator.device) for x in batch["attention_mask"]]
            inputs_embeds_lens = batch["inputs_embeds_lens"]
            num_pairs = batch["num_pairs"] # not including test pair

            # gen stuff
            gen_inputs_embeds = batch["gen_inputs_embeds"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_labels = batch["gen_labels"].to(accelerator.device)
            gen_inputs_embeds_lens = batch["gen_inputs_embeds_lens"]

            device = gen_inputs_embeds.device

            with accelerator.autocast():
                # STEP 1: get predicted programs and kv cache
                prev_programs, demonstration_intervals, past_key_values, past_key_values_attention_mask = get_predicted_program(
                    # model
                    model=model,
                    net_input_dim=dataset.net_input_dim,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    vae_projection=vae_projection,
                    quantizer=quantizer,
                    program_norm=program_norm,
                    # data
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    inputs_embeds_lens=inputs_embeds_lens,
                    num_pairs=num_pairs,
                    # others
                    ntokens=dataset.ntokens,
                    pad_side=dataset.pad_side,
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
                pad_embeds = torch.full((inputs_embeds[0].shape[-1],), 0.1234, device=device, dtype=torch.float32)

                # prepend the last predicted program for all pairs except the first
                gen_inputs_embeds = insert_based_on_sides(
                    data=gen_inputs_embeds,
                    to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
                    lens=gen_inputs_embeds_lens,
                    insert_side="left",
                    pad_side=dataset.pad_side,
                    pad_id=pad_embeds,
                )
                gen_attention_mask = insert_based_on_sides(
                    data=gen_attention_mask,
                    to_insert=torch.full((bs, dataset.ntokens), 1, device=device, dtype=torch.int64),
                    lens=gen_inputs_embeds_lens,
                    insert_side="left",
                    pad_side=dataset.pad_side,
                    pad_id=0,
                )
                gen_labels = insert_based_on_sides(
                    data=gen_labels,
                    to_insert=torch.full((bs, dataset.ntokens), -100, device=device, dtype=torch.float32),
                    lens=gen_inputs_embeds_lens,
                    insert_side="left",
                    pad_side=dataset.pad_side,
                    pad_id=-100,
                )

                # attention mask should span to the past kvs
                if not short_context:
                    assert past_key_values is not None
                    assert past_key_values_attention_mask is not None
                    assert past_key_values_attention_mask.shape[1] == past_key_values[0][0].shape[2]
                    assert past_key_values_attention_mask.shape[0] == gen_attention_mask.shape[0] == bs
                    gen_attention_mask = torch.cat([past_key_values_attention_mask, gen_attention_mask], dim=1)

                model_kwargs = {
                    "inputs_embeds": gen_inputs_embeds,
                    "attention_mask": gen_attention_mask,
                    "use_cache": not short_context,
                }
                if not short_context:
                    assert past_key_values is not None
                    model_kwargs["past_key_values"] = past_key_values

                    # build position ids (does NOT depend on dropout)
                    attention_mask_just_for_kv = gen_attention_mask[:, :past_key_values[0][0].shape[2]]
                    attention_mask_after_kv = gen_attention_mask[:, past_key_values[0][0].shape[2]:]
                    position_ids = []
                    for mask_for_kv, mask_after_kv in zip(attention_mask_just_for_kv, attention_mask_after_kv):
                        sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                        position_start = mask_for_kv.sum()
                        n_new_positions = mask_after_kv.sum()
                        new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=torch.int64)
                        if dataset.pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)
                    model_kwargs["position_ids"] = position_ids

                    # apply attention reduction
                    if attention_reduction_ratio != 1.0:
                        model_kwargs['demonstration_intervals'] = demonstration_intervals
                        model_kwargs['attention_reduction_ratio'] = attention_reduction_ratio

                else:
                    # necessary for padside left and does not change when padside is right, idky
                    position_ids = []
                    for m in gen_attention_mask:
                        sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                        n_new_positions = m.sum()
                        new_positions = torch.tensor(range(n_new_positions), device=device, dtype=torch.int64)
                        if dataset.pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)
                    model_kwargs["position_ids"] = position_ids

                # get losses
                model_out = model(**model_kwargs)
                losses = compute_loss(model_out.predictions.squeeze(-1), gen_labels, individual_loss=True)

            losses = [l.item() / dataset.net_input_dim for l in losses]
            # print(losses)
            # breakpoint()

            loss_list += losses

    distributed_state.wait_for_everyone()

    loss_list = gather_object(loss_list)
    assert len(loss_list) == len(dataset), (len(loss_list), len(dataset))
    loss = sum(loss_list) / len(dataset)

    return loss


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--tracker_project_name", type=str, default="metaicl_toy")
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_len", type=int, default=-1) # two grid -> 1867
    parser.add_argument("--debug_fixed_order", action="store_true")
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)

    # Model
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--weird_cast", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--full_demonstration_dropout", type=float, default=0.0)
    parser.add_argument("--partial_demonstration_dropout", type=float, default=0.0)
    parser.add_argument("--attention_reduction_ratio", type=float, default=1.0)

    # invar loss
    parser.add_argument("--invar_loss_margin", type=float, default=0.5)

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
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--lr_program", type=float, default=4e-4)
    parser.add_argument("--lr_prior", type=float, default=4e-4)
    parser.add_argument("--lr_other", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--samples_per_epoch", type=int, default=100000)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--full_attention_dropout", type=float, default=0.0)
    parser.add_argument("--program_dropout", type=float, default=0.0)
    parser.add_argument("--program_noise_std", type=float, default=0.0)
    parser.add_argument("--short_context", action='store_true')

    # program caching
    parser.add_argument("--cache_size_per_task", type=int, default=0)
    parser.add_argument("--prior_embed_ratio", type=float, default=1.0)

    # scheduled extra losses
    parser.add_argument("--consistency_loss_lambda", type=float, default=0.0)
    parser.add_argument("--consistency_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--consistency_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.0)
    parser.add_argument("--invar_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--invar_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--commitment_loss_lambda", type=float, default=0.1)
    parser.add_argument("--commitment_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--commitment_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--codebook_loss_lambda", type=float, default=1.0)
    parser.add_argument("--codebook_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--codebook_loss_linear_epochs", type=int, default=0)
    parser.add_argument("--kl_loss_lambda", type=float, default=0.0)
    parser.add_argument("--kl_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--kl_loss_linear_epochs", type=int, default=0)

    # data
    parser.add_argument("--net_input_dim", type=int, default=20)
    parser.add_argument("--net_hidden_dim", type=int, default=100)
    parser.add_argument("--num_train_net", type=int, default=-1)
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--kv_pad_side", type=str, choices=["left", "right"], default="right")
    # train
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_train_num_pair", type=int, default=101) # includes test pair
    parser.add_argument("--max_train_num_pair", type=int, default=101) # includes test pair
    # eval
    parser.add_argument("--num_eval_net", type=int, default=10000) # includes test pair
    parser.add_argument("--min_eval_num_pair", type=int, default=101) # includes test pair
    parser.add_argument("--max_eval_num_pair", type=int, default=101) # includes test pair

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

    # model architecture
    parser.add_argument("--model_hidden_size", type=int, default=256) # 64, 128, 256
    parser.add_argument("--num_hidden_layers", type=int, default=12) # 3, 6, 12
    parser.add_argument("--num_attention_heads", type=int, default=8) # 2, 4, 8
    parser.add_argument("--intermediate_size", type=int, default=1024) # 2, 4, 8

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # override to debug stuff
    if args.debug:
        args.tag = 'test'
        args.wandb = False
        args.samples_per_epoch = 8
        args.log_every = 1

    # check args
    if args.warmup_cookbook_only_epochs:
        assert args.codebook_loss_offset_epochs == 0
        assert args.codebook_loss_linear_epochs == 0
    assert args.commitment_loss_offset_epochs >= args.warmup_cookbook_only_epochs
    if args.cache_size_per_task > 0:
        assert args.min_train_num_pair == args.max_train_num_pair
        # not asserting here, but dont add any kl, consistency, program loss, invar, etc

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

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        tracker_config = dict(vars(args))
        # get runid
        wandb_init_args = {"name": args.tag}
        accelerator.init_trackers(
            args.tracker_project_name,
            tracker_config,
            init_kwargs={"wandb": wandb_init_args}
        )
    logger.info("Accelerator and seed set up.")

    # log args
    logger.info("#### BEGIN ALL ARGUMENTS ####")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("#### END ALL ARGUMENTS ####\n")

    # build configurable, not pretrained llama model
    model_config = LlamaConfig(
        hidden_size=args.model_hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
    )
    model_config.net_input_dim = args.net_input_dim
    model_config._attn_implementation = 'sdpa'
    model_config.attention_dropout = args.full_attention_dropout
    model = MyLlamaModel(model_config)

    # for n, p in model.named_parameters(): print(n, p.numel())
    # print_trainable_parameters(model)
    # breakpoint()

    logger.info("Base models loaded.")

    # program dropout
    program_dropout = nn.Dropout(p=args.program_dropout)

    # initialize program embeddings
    prior_embeddings = ProgramEmbeddings(
        embedding=torch.randn((args.ntokens, args.net_input_dim), device=accelerator.device, dtype=torch.float32)
    )
    program_embeddings = ProgramEmbeddings(
        embedding=torch.randn((args.ntokens, args.net_input_dim), device=accelerator.device, dtype=torch.float32)
    )
    logger.info("Prior & Program embeddings initialized.")

    # vqvae codebook
    quantizer: Optional[Quantizer] = None
    if args.codebook_size > 0 or args.fsq_L != []:
        quantizer = Quantizer(
            codebook_size=args.codebook_size,
            hidden_size=args.net_input_dim,
            fsq_L=args.fsq_L,
            device=accelerator.device,
        )
        logger.info("Codebook initialized.")

    # vae projection (empty module if vae=False)
    vae_projection = None
    if args.vae:
        vae_projection = VaeProjection(
            mlp_factor=args.mlp_factor,
            latent_dim=args.net_input_dim, # type: ignore
            device=accelerator.device,
        )
        logger.info("vae projection initialized")

    # shared norm
    program_norm = None
    if not args.no_normalize:
        program_norm = LlamaRMSNorm(args.net_input_dim, eps=model.config.rms_norm_eps) # type: ignore
        logger.info("norm layer initialized")

    # ensure requires grad
    for param in model.parameters():
        assert param.requires_grad
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

    # generate train nn
    # each nn has 2100 params -> 8400 bytes, 1GB can support 120k
    train_net_rng = get_torch_generator(args.seed)
    train_groundtruth_nets = []
    if args.num_train_net != -1:
        train_groundtruth_nets = [
            create_ground_truth_net(args.net_input_dim, args.net_hidden_dim, generator=train_net_rng)
            for _ in range(args.num_train_net)
        ]
    # generate eval nn (3secs for 10000 nets)
    eval_net_rng = get_torch_generator(args.seed + 10)
    eval_groundtruth_nets = [
        create_ground_truth_net(args.net_input_dim, args.net_hidden_dim, generator=eval_net_rng)
        for _ in range(args.num_eval_net)
    ]

    # Build training dataset
    train_dataset = TrainDataset(
        total_steps=args.samples_per_epoch,
        seed=args.seed,
        pad_side=args.pad_side,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        process_index=accelerator.process_index,
        min_num_pair=args.min_train_num_pair,
        max_num_pair=args.max_train_num_pair,
        num_workers=args.num_workers,
        net_input_dim=args.net_input_dim,
        net_hidden_dim=args.net_hidden_dim,
        groundtruth_nets=train_groundtruth_nets,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
    if args.invar_loss_lambda > 0.0:
        train_collate_fn = partial(collate_fn_train_invar, dataset=train_dataset)
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
    other_params = [p for p in model.parameters()]
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
        {"params": prior_params, "lr": args.lr_prior},
        {"params": program_params, "lr": args.lr_program},
        {"params": other_params, "lr": args.lr_other},
    ]
    all_params = prior_params + program_params + other_params
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay) # type: ignore
    elif args.optimizer == 'adamw8bit':
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_params, weight_decay=args.weight_decay)
        # optimizer = bnb.optim.PagedAdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_params) # type: ignore
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

    start_epoch = 0
    global_step = 0

    assert isinstance(model, (nn.Module, DistributedDataParallel))
    assert isinstance(prior_embeddings, (ProgramEmbeddings, DistributedDataParallel))
    assert isinstance(program_embeddings, (ProgramEmbeddings, DistributedDataParallel))
    if vae_projection is not None:
        assert isinstance(vae_projection, (VaeProjection, DistributedDataParallel))
    if quantizer is not None:
        assert isinstance(quantizer, (Quantizer, DistributedDataParallel))
    if program_norm is not None:
        assert isinstance(program_norm, (LlamaRMSNorm, DistributedDataParallel))

    # Build evaluation datasets
    eval_train_dataset: Optional[EvalDataset] = None
    if len(train_groundtruth_nets) > 0:
        eval_train_dataset = EvalDataset(
            seed=args.seed,
            debug_random_pad=args.debug_random_pad,
            debug_pad_len=args.debug_pad_len,
            min_num_pair=args.min_eval_num_pair,
            max_num_pair=args.max_eval_num_pair,
            debug_len=args.debug_len,
            pad_side=args.pad_side,
            net_input_dim=args.net_input_dim,
            net_hidden_dim=args.net_hidden_dim,
            groundtruth_nets=train_groundtruth_nets,
            ntokens=args.ntokens,
        )
    eval_eval_dataset = EvalDataset(
        seed=args.seed,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        min_num_pair=args.min_eval_num_pair,
        max_num_pair=args.max_eval_num_pair,
        debug_len=args.debug_len,
        pad_side=args.pad_side,
        net_input_dim=args.net_input_dim,
        net_hidden_dim=args.net_hidden_dim,
        groundtruth_nets=eval_groundtruth_nets,
        ntokens=args.ntokens,
    )
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_eval_dataset)

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
    epoch_to_eval_loss = {}

    # program cache
    program_cache: Optional[ProgramCache] = None
    if args.cache_size_per_task > 0:
        program_cache = ProgramCache(cache_size_per_task=args.cache_size_per_task, seed=args.seed)

        # debug: allocate full cache here, make sure to comment out
        # program_cache.debug_full_cache(args.ntokens, model.config.net_input_dim, num_task_identifiers=400 * 8)

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
        consistency_loss_accum = 0.0
        invar_loss_accum = 0.0
        total_loss_accum = 0.0
        grad_norm_accum = 0.0
        ce_losses_accum = [0.0 for _ in range(args.max_train_num_pair)]

        train_dataset.set_rngs(epoch)
        for batch_data in train_loader:
            inputs_embeds = [x.to(accelerator.device) for x in batch_data["inputs_embeds"]]
            attention_mask = [x.to(accelerator.device) for x in batch_data["attention_mask"]]
            labels = [x.to(accelerator.device) for x in batch_data["labels"]]
            inputs_embeds_lens = batch_data["inputs_embeds_lens"]
            num_pairs = batch_data["num_pairs"]
            is_same = batch_data["is_same"]
            task_identifiers = batch_data["task_identifiers"]

            train_codebook_only = (global_step < args.warmup_cookbook_only_epochs * steps_per_epoch)

            with accelerator.accumulate(model, prior_embeddings, program_embeddings, vae_projection, quantizer, program_norm):
                with accelerator.autocast():
                    ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, consistency_loss, \
                        invar_loss, total_loss, log_ce_losses, inferred_programs = model_loss(
                        # model
                        model=model,
                        net_input_dim=args.net_input_dim,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        vae_projection=vae_projection,
                        quantizer=quantizer,
                        program_norm=program_norm,
                        program_dropout=program_dropout,
                        accelerator=accelerator,
                        # data
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        inputs_embeds_lens=inputs_embeds_lens,
                        num_pairs=num_pairs,
                        task_identifiers=task_identifiers,
                        # others
                        ntokens=args.ntokens,
                        pad_side=args.pad_side,
                        kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
                        codebook_loss_lambda_scheduler=codebook_loss_lambda_scheduler,
                        commitment_loss_lambda_scheduler=commitment_loss_lambda_scheduler,
                        consistency_loss_lambda_scheduler=consistency_loss_lambda_scheduler,
                        invar_loss_lambda_scheduler=invar_loss_lambda_scheduler,
                        global_step=global_step,
                        no_residual=args.no_residual,
                        no_discrete_prior=args.no_discrete_prior,
                        train_codebook_only=train_codebook_only,
                        program_noise_std=args.program_noise_std,
                        subset_kl=args.subset_kl,
                        weird_cast=args.weird_cast,
                        full_demonstration_dropout=args.full_demonstration_dropout,
                        partial_demonstration_dropout=args.partial_demonstration_dropout,
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
                #     assert program_cache.get_num_items_in_cache() == accelerator.num_processes * args.min_train_num_pair * args.grad_accum_steps * args.train_batch_size * global_step
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
                        "train/consistency_loss_accum": consistency_loss_accum,
                        "train/invar_loss_accum": invar_loss_accum,
                        "train/total_loss": total_loss_accum,
                        "train/grad_norm": grad_norm_accum,
                        "train/lr_prior": lr_scheduler.get_last_lr()[0],
                        "train/lr_program": lr_scheduler.get_last_lr()[1],
                        "train/lr_other": lr_scheduler.get_last_lr()[2],
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
                consistency_loss_accum = 0.0
                invar_loss_accum = 0.0
                total_loss_accum = 0.0
                grad_norm_accum = 0.0
                ce_losses_accum = [0.0 for _ in range(args.max_train_num_pair)]

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
            torch.cuda.empty_cache()
            gc.collect()

            no_codebook = epoch < args.warmup_cookbook_only_epochs

            train_loss = None
            if eval_train_dataset is not None:
                train_loss = evaluate(
                    desc="eval_train",
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
                    no_residual=args.no_residual,
                    no_discrete_prior=args.no_discrete_prior,
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
            eval_loss = evaluate(
                desc="eval_eval",
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
                no_residual=args.no_residual,
                no_discrete_prior=args.no_discrete_prior,
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
                eval_metric_dict = {"eval/eval_loss": eval_loss}
                if train_loss is not None:
                    eval_metric_dict["eval/train_loss"] = train_loss
                logger.info(f'Evaluation results:\n{pprint.pformat(eval_metric_dict, indent=4)}')
                try:
                    accelerator.log(eval_metric_dict, step=global_step)
                except:
                    logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                # Save model
                do_save_model = args.save_all_models
                if not args.save_all_models:
                    if (not epoch_to_eval_loss) or eval_loss < min(epoch_to_eval_loss.values()):
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
                epoch_to_eval_loss[epoch] = eval_loss

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
