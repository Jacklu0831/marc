import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
from datetime import timedelta
import copy
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
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)
import bitsandbytes as bnb

from data_utils import (
    TrainDataset,
    EvalDataset,
    GSDataset,
    collate_fn_train,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
    collate_fn_gs,
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


class Quantizer(nn.Module):
    def __init__(self, embedding: torch.Tensor):
        super(Quantizer, self).__init__()
        self.embedding = nn.Parameter(embedding)
        self.num_embeddings, self.embedding_dim = tuple(self.embedding.shape)

    def forward(
            self,
            program: torch.Tensor,
            train_codebook_only: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, ntokens, embedding_dim = program.shape
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
    concat_programs: bool,
    train_codebook_only: bool,
) -> torch.Tensor:

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # samples do not have to be the same number of pairs in evaluation, but try to parallelize it anyway
    assert min(num_pairs) >= 2 # at least 2 train
    assert max(num_pairs) == len(input_ids)

    assert len(set(len(ids) for ids in input_ids)) == 1 # batch size uniform across pair_idx
    batch_size = len(input_ids[0])
    dtype, device = input_ids[0].dtype, input_ids[0].device

    # reused tensors
    pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device))
    inputs_embeds = [embed_tokens(pair_input_ids) for pair_input_ids in input_ids]

    # apply program norm to prior
    prior_inputs_embeds = prior_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
    # norm
    if program_norm is not None:
        prior_inputs_embeds = program_norm(prior_inputs_embeds)
    # quantize prior
    if (quantizer is not None) and not no_discrete_prior:
        prior_inputs_embeds, _, _, _ = quantizer(prior_inputs_embeds, train_codebook_only=train_codebook_only)
        assert torch.allclose(prior_inputs_embeds[0], prior_inputs_embeds[1])

    # previous program of each sample in batch
    all_prev_programs = [p for p in prior_inputs_embeds]

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_input_ids_lens) in enumerate(zip(inputs_embeds, attention_mask, input_ids_lens)):

        # STEP 0: filter out samples that are not this long (for eval)
        avail_mask = torch.tensor([int(pair_i < n) for n in num_pairs], dtype=torch.bool)
        pair_inputs_embeds = pair_inputs_embeds[avail_mask]
        pair_attention_mask = pair_attention_mask[avail_mask]
        pair_input_ids_lens = [l for l, m in zip(pair_input_ids_lens, avail_mask) if m]
        n_avail = int(avail_mask.sum().item())

        # stack available programs that have to be the same length
        prev_programs = [p for p, m in zip(all_prev_programs, avail_mask) if m]
        assert len(set(p.shape[0] for p in prev_programs)) == 1
        prev_programs = torch.stack(prev_programs)
        program_len = prev_programs.shape[1]

        # STEP 1: prepend the last predicted program for all pairs except the first
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=prev_programs,
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((n_avail, program_len), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )

        # update lens to reflect the extra program
        pair_input_ids_lens = [l + program_len for l in pair_input_ids_lens]

        # STEP 2: append new program input to the right for all pairs except the last
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=program_embeddings("dummy")[None, ...].expand(n_avail, -1, -1),
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

        # refine program
        model_out = model(
            inputs_embeds=pair_inputs_embeds,
            attention_mask=pair_attention_mask,
            output_hidden_states=True,
        )

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
            new_programs += prev_programs
        # optionally apply norm
        if program_norm is not None:
            new_programs = program_norm(new_programs)
        # optionally quantize
        if quantizer is not None:
            new_programs, _, _, _ = quantizer(program=new_programs, train_codebook_only=train_codebook_only)
        # optionally concat program
        if concat_programs:
            new_programs = torch.cat([prev_programs, new_programs], dim=1)
        # update new program
        assert len(avail_mask) == len(all_prev_programs)
        assert sum(avail_mask) == len(new_programs)
        new_programs_idx = 0
        for i, m in enumerate(avail_mask):
            if m:
                all_prev_programs[i] = new_programs[new_programs_idx]
                new_programs_idx += 1

    assert len(all_prev_programs) == batch_size
    assert len(set(p.shape[0] for p in all_prev_programs)) == 1
    return torch.stack(all_prev_programs)


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
    # data
    input_ids: List[torch.Tensor],
    attention_mask: List[torch.Tensor],
    label_ids: List[torch.Tensor],
    input_ids_lens: List[List[int]],
    num_pairs: List[int],
    # others
    ntokens: int,
    pad_side: str,
    kl_loss_lambda_scheduler: LambdaScheduler,
    codebook_loss_lambda_scheduler: LambdaScheduler,
    commitment_loss_lambda_scheduler: LambdaScheduler,
    consistency_loss_lambda_scheduler: LambdaScheduler,
    global_step: int,
    no_residual: bool,
    no_discrete_prior: bool,
    consistency_type: str,
    concat_programs: bool,
    train_codebook_only: bool,
    ar_gradient_checkpointing: bool,
    program_noise_std: float,
    subset_kl: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:

    def forward_with_checkpoint(model, inputs_embeds, attention_mask, labels=None):
        def custom_forward(x, attn, lab=None):
            if lab is None:
                return model(
                    inputs_embeds=x,
                    attention_mask=attn,
                    output_hidden_states=True,
                )
            else:
                return model(
                    inputs_embeds=x,
                    attention_mask=attn,
                    labels=lab,
                    output_hidden_states=True,
                )

        if labels is None:
            return checkpoint.checkpoint(custom_forward, inputs_embeds, attention_mask, use_reentrant=False)
        else:
            return checkpoint.checkpoint(custom_forward, inputs_embeds, attention_mask, labels, use_reentrant=False)

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # all samples should have same number of pairs
    assert len(set(num_pairs)) == 1
    n_pairs = num_pairs[0]
    assert n_pairs >= 3 # at least 2 train and 1 test
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
    consistency_loss_lambda = consistency_loss_lambda_scheduler.get_lambda(step=global_step)

    # apply program norm to prior
    prior_inputs_embeds = prior_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
    if program_norm is not None:
        prior_inputs_embeds = program_norm(prior_inputs_embeds)
    # NOTE: no dropout or noise injection to prior
    # quantize prior
    if (quantizer is not None) and not no_discrete_prior:
        prior_inputs_embeds, codebook_loss, commitment_loss, perplexity = quantizer(prior_inputs_embeds, train_codebook_only=train_codebook_only)
        codebook_losses.append(codebook_loss)
        commitment_losses.append(commitment_loss)
        perplexitys.append(perplexity)
        assert torch.allclose(prior_inputs_embeds[0], prior_inputs_embeds[1])

    # previous program of each sample in batch
    prev_programs = prior_inputs_embeds
    prev_program_mus = torch.zeros_like(prior_inputs_embeds, device=device, dtype=prior_inputs_embeds.dtype)
    prev_program_logvars = torch.zeros_like(prior_inputs_embeds, device=device, dtype=prior_inputs_embeds.dtype)

    # save all programs for randomly selecting one for consistency loss
    saved_all_programs = []
    if consistency_type != 'none':
        saved_all_programs = [prev_programs]

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_label_ids, pair_input_ids_lens) in enumerate(zip(inputs_embeds, attention_mask, label_ids, input_ids_lens)):

        # STEP 1: prepend the last predicted program for all pairs except the first
        program_len = prev_programs.shape[1]
        pair_inputs_embeds = insert_based_on_sides(
            data=pair_inputs_embeds,
            to_insert=prev_programs,
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        pair_attention_mask = insert_based_on_sides(
            data=pair_attention_mask,
            to_insert=torch.full((batch_size, program_len), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )
        if pair_i > 0:
            pair_label_ids = insert_based_on_sides(
                data=pair_label_ids,
                to_insert=torch.full((batch_size, program_len), -100, device=device, dtype=dtype),
                lens=pair_input_ids_lens,
                insert_side="left",
                pad_side=pad_side,
                pad_id=-100,
            )

        # update lens to reflect the extra program
        pair_input_ids_lens = [l + program_len for l in pair_input_ids_lens]

        # STEP 2: append new program input to the right for all pairs except the last
        if pair_i < n_pairs - 1 or consistency_type != 'none':
            pair_inputs_embeds = insert_based_on_sides(
                data=pair_inputs_embeds,
                to_insert=program_embeddings("dummy")[None, ...].expand(batch_size, -1, -1),
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
            if pair_i > 0:
                pair_label_ids = insert_based_on_sides(
                    data=pair_label_ids,
                    to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=dtype),
                    lens=pair_input_ids_lens,
                    insert_side="right",
                    pad_side=pad_side,
                    pad_id=-100,
                )

        # STEP 4: refine program (no output grid label for first pair)
        if pair_i == 0:
            if ar_gradient_checkpointing:
                model_out = forward_with_checkpoint(model, pair_inputs_embeds, pair_attention_mask)
            else:
                model_out = model(
                    inputs_embeds=pair_inputs_embeds,
                    attention_mask=pair_attention_mask,
                    output_hidden_states=True,
                )
        else:
            if ar_gradient_checkpointing:
                model_out = forward_with_checkpoint(model, pair_inputs_embeds, pair_attention_mask, pair_label_ids)
            else:
                model_out = model(
                    inputs_embeds=pair_inputs_embeds,
                    attention_mask=pair_attention_mask,
                    labels=pair_label_ids,
                    output_hidden_states=True,
                )
            ce_losses.append(model_out.loss) # type: ignore

        # STEP 5: projection then sample new program prediction for all pairs except the last
        if pair_i < n_pairs - 1 or consistency_type != 'none':
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
            new_programs += program_noise_std * torch.randn_like(new_programs, dtype=new_programs.dtype, device=device)
            # use residual connection
            if not no_residual:
                new_programs += prev_programs
            # apply norm
            if program_norm is not None:
                new_programs = program_norm(new_programs)
            # quantize
            if quantizer is not None:
                new_programs, codebook_loss, commitment_loss, perplexity = quantizer(program=new_programs, train_codebook_only=train_codebook_only)
                codebook_losses.append(codebook_loss)
                commitment_losses.append(commitment_loss)
                perplexitys.append(perplexity)
            # concat program
            if concat_programs:
                new_programs = torch.cat([prev_programs, new_programs], dim=1)

            # save and update new program
            if consistency_type != 'none':
                saved_all_programs.append(new_programs)
            prev_programs = new_programs

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
    if consistency_type != 'none':
        assert len(saved_all_programs) == n_pairs + 1
        # select program index (as low as second pair's output program)
        program_idx = int(torch.randint(low=2, high=n_pairs + 1, size=(1,)).item())
        select_programs = saved_all_programs[program_idx]
        # select pair index
        if consistency_type == 'only_first':
            select_idx = 0
        elif consistency_type == 'exclude_last':
            select_idx = int(torch.randint(low=0, high=program_idx - 1, size=(1,)).item())
        elif consistency_type == 'only_last':
            select_idx = program_idx - 1
        else:
            select_idx = int(torch.randint(low=0, high=program_idx, size=(1,)).item())
        # get consistency loss
        select_inputs_embeds = insert_based_on_sides(
            data=inputs_embeds[select_idx],
            to_insert=select_programs,
            lens=input_ids_lens[select_idx],
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        select_attention_mask = insert_based_on_sides(
            data=attention_mask[select_idx],
            to_insert=torch.full((batch_size, ntokens), 1, device=device, dtype=dtype),
            lens=input_ids_lens[select_idx],
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )
        select_label_ids = insert_based_on_sides(
            data=label_ids[select_idx],
            to_insert=torch.full((batch_size, ntokens), -100, device=device, dtype=dtype),
            lens=input_ids_lens[select_idx],
            insert_side="left",
            pad_side=pad_side,
            pad_id=-100,
        )
        consistency_loss = model(
            inputs_embeds=select_inputs_embeds,
            attention_mask=select_attention_mask,
            labels=select_label_ids,
            output_hidden_states=True,
        ).loss
        consistency_loss /= len(ce_losses) # normalize based on num pairs to not dominate

    ce_loss = sum(ce_losses) / len(ce_losses)
    kl_loss = sum(kl_losses) / len(kl_losses) if len(kl_losses) > 0 else torch.tensor(0.0, device=device)
    codebook_loss = sum(codebook_losses) / len(codebook_losses) if len(codebook_losses) > 0 else torch.tensor(0.0, device=device)
    commitment_loss = sum(commitment_losses) / len(commitment_losses) if len(commitment_losses) > 0 else torch.tensor(0.0, device=device)
    total_loss = ce_loss + kl_loss_lambda * kl_loss + codebook_loss * codebook_loss_lambda + commitment_loss_lambda * commitment_loss + consistency_loss_lambda * consistency_loss

    # logging perplexity for debugging vqvae
    perplexity = sum(perplexitys) / len(perplexitys) if len(perplexitys) > 0 else torch.tensor(0.0, device=device)
    return ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, consistency_loss, total_loss, ce_losses # type: ignore


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


@torch.enable_grad()
def gradient_search(
        batch_idxs: List[int],
        eval_dataset: EvalDataset,
        batch_size: int,
        optimizer: str,
        lr_scheduler: str,
        lr: float,
        take_best: bool,
        beta1: float,
        beta2: float,
        accelerator: Accelerator,
        model: nn.Module,
        iters: int,
        predicted_program: torch.Tensor,
        max_grad_norm: float,
    ) -> torch.Tensor:
    # note gradient checkpointing does not matter because we are freezing the model here
    # however, if we choose to tune the decoder as well, then gradient checkpointing might be desired

    assert len(batch_idxs) == 1
    eval_task = eval_dataset.eval_tasks[batch_idxs[0]]

    # gradient search dataset and dataloader
    gs_dataset = GSDataset(
        task=eval_task,
        tokenizer=eval_dataset.tokenizer,
        ntokens=eval_dataset.ntokens,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        gen_pad_side=eval_dataset.gen_pad_side,
        no_dim=eval_dataset.no_dim,
        no_separate_color_tokens=eval_dataset.no_separate_color_tokens,
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
    program_params = [predicted_program]
    assert all(not p.requires_grad for p in program_params)
    for p in program_params:
        p.requires_grad = True

    # expand to match predicted program with batch size
    assert predicted_program.shape[0] == 1
    predicted_program = predicted_program.expand(batch_size, *predicted_program.shape[1:])

    # optimizer
    if optimizer == 'adamw':
        optim = torch.optim.AdamW(program_params, weight_decay=0.0, lr=lr, betas=(beta1, beta2)) # type: ignore
    else:
        optim = torch.optim.SGD(program_params, lr=lr) # type: ignore

    # lr scheduler
    if lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=0,
            num_training_steps=iters,
        )
    else:
        scheduler = get_constant_schedule(optim)
    # optim, gs_loader = accelerator.prepare(optim, gs_loader)

    curr_iter = 0
    best_loss = float("inf")
    best_program = None
    model.train() # is this useful? try without it???

    device = predicted_program.device
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens
    extra_program_attention_mask = torch.full((batch_size, eval_dataset.ntokens), 1, device=device, dtype=torch.int64)
    extra_program_label_ids = torch.full((batch_size, eval_dataset.ntokens), -100, device=device, dtype=torch.int64)
    pad_embeds = embed_tokens(torch.tensor(eval_dataset.tokenizer.pad_token_id, device=device))

    # train!
    while curr_iter < iters:
        for gs_batch_data in gs_loader:
            input_ids = gs_batch_data["input_ids"].to(accelerator.device)
            attention_mask = gs_batch_data["attention_mask"].to(accelerator.device)
            label_ids = gs_batch_data["label_ids"].to(accelerator.device)
            input_ids_lens = gs_batch_data["input_ids_lens"]

            with accelerator.autocast():
                inputs_embeds = embed_tokens(input_ids)
                # STEP 1: prepend the last predicted program
                inputs_embeds = insert_based_on_sides(
                    data=inputs_embeds,
                    to_insert=predicted_program,
                    lens=input_ids_lens,
                    insert_side="left",
                    pad_side=eval_dataset.train_pad_side,
                    pad_id=pad_embeds,
                )
                attention_mask = insert_based_on_sides(
                    data=attention_mask,
                    to_insert=extra_program_attention_mask,
                    lens=input_ids_lens,
                    insert_side="left",
                    pad_side=eval_dataset.train_pad_side,
                    pad_id=0,
                )
                label_ids = insert_based_on_sides(
                    data=label_ids,
                    to_insert=extra_program_label_ids,
                    lens=input_ids_lens,
                    insert_side="left",
                    pad_side=eval_dataset.train_pad_side,
                    pad_id=-100,
                )
                gs_loss = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=label_ids,
                    output_hidden_states=True,
                ).loss

            accelerator.backward(gs_loss)
            accelerator.clip_grad_norm_(program_params, max_grad_norm)
            optim.step()
            scheduler.step()
            optim.zero_grad()

            if take_best and gs_loss.item() < best_loss:
                best_loss = gs_loss.item()
                best_program = predicted_program.detach().clone()

            curr_iter += 1
            if curr_iter >= iters:
                break

    # set decoder back to eval mode
    model.eval()

    for p in program_params:
        p.requires_grad = False

    if take_best:
        assert best_program is not None
        predicted_program = best_program

    # shrink to match predicted program with batch size 1
    predicted_program = predicted_program[:1]
    # NOTE: we do not apply normalization here
    return predicted_program


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
    desc: str,
    task_to_ttt_path: Optional[Dict[str, Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]]],
    ttt_param_names: Optional[Set[str]],
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    tokenizer: ARCTokenizer,
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    trainable_nbit: int,
    no_flash_attn: bool,
    dry_eval_run: bool,
    gs_iters: int,
    gs_batch_size: int,
    gs_lr: float,
    gs_beta1: float,
    gs_beta2: float,
    gs_optimizer: str,
    gs_max_grad_norm: float,
    gs_lr_scheduler: str,
    gs_take_best: bool,
    no_residual: bool,
    no_discrete_prior: bool,
    output_dir: str,
    concat_programs: bool,
    no_codebook: bool,
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

            # predict multiple programs from i/o permutations
            predicted_program_accum = None
            avail_accum = [0] * bs
            first_batch = None

            if dry_eval_run:
                for batch_data, _ in dataset.get_io_permuted_batches(batch_idxs):
                    _ = collate_fn(batch_data)
                continue

            for batch_permute_i, (batch_data, batch_avail) in enumerate(dataset.get_io_permuted_batches(batch_idxs)):
                batch = collate_fn(batch_data)
                assert len(batch["task_ids"]) <= bs
                assert len(batch["task_ids"]) == sum(batch_avail)
                if batch_permute_i == 0:
                    assert len(batch["task_ids"]) == sum(batch_avail) == bs
                assert len(batch_avail) == bs
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
                num_pairs = batch["num_pairs"]

                # save first batch with complete tasks, rest might be incomplete
                if batch_permute_i == 0:
                    first_batch = {
                        "task_ids": copy.deepcopy(task_ids),
                        "inverters": copy.deepcopy(inverters),
                        "input_ids": copy.deepcopy(input_ids),
                        "attention_mask": copy.deepcopy(attention_mask),
                        "gen_input_ids": copy.deepcopy(gen_input_ids),
                        "gen_attention_mask": copy.deepcopy(gen_attention_mask),
                        "out_token_length": copy.deepcopy(out_token_length),
                        "label_texts": copy.deepcopy(label_texts),
                        "input_ids_lens": copy.deepcopy(input_ids_lens),
                        "gen_input_ids_lens": copy.deepcopy(gen_input_ids_lens),
                        "num_pairs": copy.deepcopy(num_pairs),
                    }

                with accelerator.autocast():
                    predicted_program = get_predicted_program(
                        # model
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        vae_projection=vae_projection,
                        quantizer=quantizer,
                        program_norm=program_norm,
                        tokenizer=tokenizer,
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
                        concat_programs=concat_programs,
                        train_codebook_only=no_codebook,
                    )

                # accumulate program
                if batch_permute_i == 0:
                    predicted_program_accum = predicted_program
                else:
                    assert predicted_program_accum is not None
                    assert len(batch_avail) == predicted_program_accum.shape[0]
                    assert sum(batch_avail) == predicted_program.shape[0]
                    avail_count = 0
                    for batch_i, avail in enumerate(batch_avail):
                        if avail:
                            predicted_program_accum[batch_i] += predicted_program[avail_count] # type: ignore
                            avail_count += 1

                # accumulate avail (count of permutations)
                assert len(batch_avail) == len(avail_accum) == bs
                for batch_i in range(bs):
                    avail_accum[batch_i] += batch_avail[batch_i]

            # average program
            assert all(avail > 0 for avail in avail_accum)
            assert predicted_program_accum.shape[0] == len(avail_accum) == bs # type: ignore
            for i, avail in enumerate(avail_accum):
                predicted_program_accum[i] /= avail # type: ignore
            predicted_program = predicted_program_accum

            # recover data from first batch (e.g. task_ids might be missing tasks due to permute_iters)
            assert isinstance(first_batch, dict)
            task_ids = first_batch["task_ids"]
            inverters = first_batch["inverters"]
            input_ids = first_batch["input_ids"]
            attention_mask = first_batch["attention_mask"]
            gen_input_ids = first_batch["gen_input_ids"]
            gen_attention_mask = first_batch["gen_attention_mask"]
            out_token_length = first_batch["out_token_length"]
            label_texts = first_batch["label_texts"]
            input_ids_lens = first_batch["input_ids_lens"]
            gen_input_ids_lens = first_batch["gen_input_ids_lens"]
            num_pairs = first_batch["num_pairs"]

            if gs_iters > 0:
                with accelerator.no_sync(model):
                    assert isinstance(predicted_program, torch.Tensor)
                    assert predicted_program.shape[0] == 1 # only support batch size 1
                    predicted_program = gradient_search(
                        batch_idxs=batch_idxs,
                        eval_dataset=dataset,
                        batch_size=gs_batch_size,
                        optimizer=gs_optimizer,
                        lr_scheduler=gs_lr_scheduler,
                        lr=gs_lr,
                        take_best=gs_take_best,
                        beta1=gs_beta1,
                        beta2=gs_beta2,
                        accelerator=accelerator,
                        model=model,
                        iters=gs_iters,
                        predicted_program=predicted_program,
                        max_grad_norm=gs_max_grad_norm,
                    )

            # compute accuracy
            with accelerator.autocast():
                assert isinstance(predicted_program, torch.Tensor)
                program_len = predicted_program.shape[1]
                device = gen_input_ids.device
                pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device))
                gen_inputs_embeds = embed_tokens(gen_input_ids)
                # pad decoder inputs embeds
                gen_inputs_embeds = insert_based_on_sides(
                    data=gen_inputs_embeds,
                    to_insert=predicted_program,
                    lens=gen_input_ids_lens,
                    insert_side="left",
                    pad_side=dataset.gen_pad_side,
                    pad_id=pad_embeds,
                )
                if not no_flash_attn:
                    gen_inputs_embeds = gen_inputs_embeds.to(NBIT_TO_DTYPE[trainable_nbit])
                # pad decoder attention masks
                extra_program_attention_mask = torch.full((bs, program_len), 1, device=device, dtype=gen_input_ids.dtype)
                gen_attention_mask = insert_based_on_sides(
                    data=gen_attention_mask,
                    to_insert=extra_program_attention_mask,
                    lens=gen_input_ids_lens,
                    insert_side="left",
                    pad_side=dataset.gen_pad_side,
                    pad_id=0,
                )
                # generate (limit generation length but also prevent information leak)
                arbitrary_increase = 5
                gen_tokens = module.generate(
                    inputs_embeds=gen_inputs_embeds,
                    attention_mask=gen_attention_mask,
                    max_new_tokens=max(out_token_length) + arbitrary_increase,
                    num_return_sequences=1,
                    temperature=1.0,
                    top_p=1.0,
                    do_sample=False,
                    eos_token_id=[dataset.tokenizer.eos_token_id],
                )
                assert len(gen_tokens) == len(out_token_length)
                for t, l in zip(gen_tokens, out_token_length):
                    t[l + arbitrary_increase:] = dataset.tokenizer.pad_token_id
                gen_texts = dataset.tokenizer.batch_decode(
                    gen_tokens,
                    skip_special_tokens=True,
                    no_separate_color_tokens=dataset.no_separate_color_tokens
                )

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
    parser.add_argument("--tracker_project_name", type=str, default="arc")
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_len", type=int, default=-1) # two grid -> 1867
    parser.add_argument("--debug_fixed_order", action="store_true")
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)
    parser.add_argument("--debug_train_data", action="store_true")

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
    parser.add_argument("--concat_programs", action="store_true")

    # Self-consistency
    parser.add_argument("--consistency_type", type=str, choices=["none", "all", "only_first", "exclude_last", "only_last"], default="none")

    # vqvae
    parser.add_argument("--codebook_size", type=int, default=-1)
    parser.add_argument("--no_discrete_prior", action="store_true")
    parser.add_argument("--warmup_cookbook_only_epochs", type=int, default=0)

    # vae
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--subset_kl", action="store_true")
    parser.add_argument("--mlp_factor", type=int, default=4)

    # Training
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=64)
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
    parser.add_argument("--warmup_epoch", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--program_dropout", type=float, default=0.0)
    parser.add_argument("--program_noise_std", type=float, default=0.0)

    # Evaluation
    parser.add_argument("--extra_inference_pairs", type=int, default=0)
    parser.add_argument("--limit_inference_pairs", action='store_true')
    parser.add_argument("--limit_inference_pairs_strict", action='store_true') # overrides limit_inference_pairs

    # scheduled extra losses
    parser.add_argument("--consistency_loss_lambda", type=float, default=1.0)
    parser.add_argument("--consistency_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--consistency_loss_linear_epochs", type=int, default=0)
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
    parser.add_argument("--curriculum_iters", type=int, default=-1) # grow from min_num_pair to max_num_pair
    parser.add_argument("--no_dim", action='store_true')
    parser.add_argument("--no_separate_color_tokens", action='store_true')
    parser.add_argument("--no_color_permute", action="store_true")
    parser.add_argument("--no_pair_permute", action="store_true")
    parser.add_argument("--no_d8", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")

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

    # Lora
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=float, default=24.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument('--lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head'
    ])
    parser.add_argument("--no_rslora", action='store_true')

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=16)
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
    if args.train_gs_iters > 0 or args.eval_gs_iters > 0:
        assert args.eval_batch_size == 1
    if args.concat_programs:
        assert args.consistency_type == "none"
        assert args.no_residual # cannot add programs
        assert args.eval_batch_size == 1
    if args.warmup_cookbook_only_epochs:
        assert args.codebook_loss_offset_epochs == 0
        assert args.codebook_loss_linear_epochs == 0
    assert args.commitment_loss_offset_epochs >= args.warmup_cookbook_only_epochs

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

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        **from_pretrained_kwargs,
    )

    # attention dropout
    for layer in base_model.model.layers:
        layer.self_attn.attention_dropout = args.attention_dropout
    base_model.config.attention_dropout = args.attention_dropout

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
    if args.codebook_size > 0:
        quantizer = Quantizer(
            embedding=torch.randn(
                (args.codebook_size, base_model.config.hidden_size),
                device=accelerator.device,
                dtype=NBIT_TO_DTYPE[args.trainable_nbit],
            ),
        )
        # quantizer = Quantizer(
        #     embedding=initialize_program_embeddings(
        #         base_model.lm_head.weight.data.detach().clone(), # lmhead weights
        #         accelerator,
        #         ntokens=args.codebook_size,
        #         cov_scale=1.0,
        #     ),
        # )
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
    logger.info(f'converted trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

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
    global_batch_size = args.train_batch_size * args.grad_accum_steps * accelerator.num_processes
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
        curriculum_iters=args.curriculum_iters,
        global_batch_size=global_batch_size,
        no_dim=args.no_dim,
        no_separate_color_tokens=args.no_separate_color_tokens,
        max_seq_len=args.max_seq_len,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
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
    steps_per_epoch = args.samples_per_epoch // global_batch_size
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
    # kl_loss_lambda_scheduler.visualize(num_training_steps, 'kl.jpg')
    # codebook_loss_lambda_scheduler.visualize(num_training_steps, 'codebook.jpg')
    # commitment_loss_lambda_scheduler.visualize(num_training_steps, 'commitment.jpg')
    # consistency_loss_lambda_scheduler.visualize(num_training_steps, 'consistency.jpg')

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

    global_step = 0
    progress_bar = tqdm(
        range(num_training_steps),
        desc="train",
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
        total_loss_accum = 0.0
        grad_norm_accum = 0.0
        ce_losses_accum = [0.0 for _ in range(args.max_num_pair - 1)]

        for batch_data in train_loader:
            input_ids = [x.to(accelerator.device) for x in batch_data["input_ids"]]
            attention_mask = [x.to(accelerator.device) for x in batch_data["attention_mask"]]
            label_ids = [x.to(accelerator.device) for x in batch_data["label_ids"]]
            input_ids_lens = batch_data["input_ids_lens"]
            num_pairs = batch_data["num_pairs"]

            train_codebook_only = (global_step < args.warmup_cookbook_only_epochs * steps_per_epoch)

            with accelerator.accumulate(model, prior_embeddings, program_embeddings, vae_projection, quantizer, program_norm):
                with accelerator.autocast():
                    ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, consistency_loss, total_loss, log_ce_losses = model_loss(
                        # model
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        vae_projection=vae_projection,
                        quantizer=quantizer,
                        program_norm=program_norm,
                        program_dropout=program_dropout,
                        tokenizer=tokenizer,
                        # data
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        label_ids=label_ids,
                        input_ids_lens=input_ids_lens,
                        num_pairs=num_pairs,
                        # others
                        ntokens=args.ntokens,
                        pad_side=args.train_pad_side,
                        kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
                        codebook_loss_lambda_scheduler=codebook_loss_lambda_scheduler,
                        commitment_loss_lambda_scheduler=commitment_loss_lambda_scheduler,
                        consistency_loss_lambda_scheduler=consistency_loss_lambda_scheduler,
                        global_step=global_step,
                        no_residual=args.no_residual,
                        no_discrete_prior=args.no_discrete_prior,
                        consistency_type=args.consistency_type,
                        concat_programs=args.concat_programs,
                        train_codebook_only=train_codebook_only,
                        ar_gradient_checkpointing=args.ar_gradient_checkpointing,
                        program_noise_std=args.program_noise_std,
                        subset_kl=args.subset_kl,
                    )

                # only log one process
                ce_loss_accum += ce_loss.item() / args.grad_accum_steps
                kl_loss_accum += kl_loss.item() / args.grad_accum_steps
                codebook_loss_accum += codebook_loss.item() / args.grad_accum_steps
                commitment_loss_accum += commitment_loss.item() / args.grad_accum_steps
                perplexity_accum += perplexity.item() / args.grad_accum_steps
                consistency_loss_accum += consistency_loss.item() / args.grad_accum_steps
                total_loss_accum += total_loss.item() / args.grad_accum_steps
                for pair_i, pair_ce_loss in enumerate(log_ce_losses):
                    ce_losses_accum[pair_i] += pair_ce_loss.item() / args.grad_accum_steps

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm_accum += accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item() # type: ignore
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
                            "train/kl_loss": kl_loss_accum,
                            "train/codebook_loss": codebook_loss_accum,
                            "train/commitment_loss": commitment_loss_accum,
                            "train/perplexity_accum": perplexity_accum,
                            "train/consistency_loss_accum": consistency_loss_accum,
                            "train/total_loss": total_loss_accum,
                            "train/grad_norm": grad_norm_accum,
                            "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                            "train/lr_prior": lr_scheduler.get_last_lr()[1],
                            "train/lr_program": lr_scheduler.get_last_lr()[2],
                            "train/lr_other": lr_scheduler.get_last_lr()[3],
                            **{f"train/ce_loss_pair_{pair_i}": pair_ce_loss_accum for pair_i, pair_ce_loss_accum in enumerate(ce_losses_accum)},
                        }, step=global_step)
                    except:
                        logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                ce_loss_accum = 0.0
                kl_loss_accum = 0.0
                codebook_loss_accum = 0.0
                commitment_loss_accum = 0.0
                perplexity_accum = 0.0
                consistency_loss_accum = 0.0
                total_loss_accum = 0.0
                grad_norm_accum = 0.0
                ce_losses_accum = [0.0 for _ in range(args.max_num_pair - 1)]

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
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
                tokenizer=tokenizer,
                dataset=eval_train_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                dry_eval_run=args.dry_eval_run,
                gs_iters=args.train_gs_iters,
                gs_lr=args.train_gs_lr,
                gs_beta1=args.train_gs_beta1,
                gs_beta2=args.train_gs_beta2,
                gs_batch_size=args.train_gs_batch_size,
                gs_optimizer=args.train_gs_optimizer,
                gs_max_grad_norm=args.train_gs_max_grad_norm,
                gs_lr_scheduler=args.train_gs_lr_scheduler,
                gs_take_best=args.train_gs_take_best,
                no_residual=args.no_residual,
                no_discrete_prior=args.no_discrete_prior,
                output_dir=args.output_dir,
                concat_programs=args.concat_programs,
                no_codebook=no_codebook,
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
                tokenizer=tokenizer,
                dataset=eval_eval_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                dry_eval_run=args.dry_eval_run,
                gs_iters=args.eval_gs_iters,
                gs_lr=args.eval_gs_lr,
                gs_beta1=args.eval_gs_beta1,
                gs_beta2=args.eval_gs_beta2,
                gs_batch_size=args.eval_gs_batch_size,
                gs_optimizer=args.eval_gs_optimizer,
                gs_max_grad_norm=args.eval_gs_max_grad_norm,
                gs_lr_scheduler=args.eval_gs_lr_scheduler,
                gs_take_best=args.eval_gs_take_best,
                no_residual=args.no_residual,
                no_discrete_prior=args.no_discrete_prior,
                output_dir=args.output_dir,
                concat_programs=args.concat_programs,
                no_codebook=no_codebook,
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

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
