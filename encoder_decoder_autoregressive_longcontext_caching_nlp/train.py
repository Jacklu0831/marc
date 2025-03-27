from custom_llama import MyLlamaForCausalLM
import shutil
import wandb
import gc
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Optional, Iterator, Any
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
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LlamaConfig,
)
import bitsandbytes as bnb

from data_utils import (
    TrainDataset,
    EvalDataset,
    collate_fn_train,
    collate_fn_train_invar,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
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


def compute_macrof1_or_accuracy(predictions, groundtruths, is_classification) -> float:
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for prediction, groundtruth in zip(predictions, groundtruths):
        prediction = prediction.strip()
        groundtruth = groundtruth.strip()
        is_correct = prediction==groundtruth
        accs.append(is_correct)
        if is_classification:
            recalls[groundtruth].append(is_correct)
            precisions[prediction].append(is_correct)

    if not is_classification:
        return float(np.mean(accs))

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return float(np.mean(f1s))


def get_individual_loss(lm_logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
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

    return torch.stack(losses)


def get_predicted_program(
    # model
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    program_embeddings: Union[ProgramEmbeddings, DistributedDataParallel],
    vae_projection: Optional[Union[VaeProjection, DistributedDataParallel]],
    quantizer: Optional[Union[Quantizer, DistributedDataParallel]],
    program_norm: Optional[Union[LlamaRMSNorm, DistributedDataParallel]],
    tokenizer: PreTrainedTokenizerFast,
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
    prior_inputs_embeds = prior_embeddings("dummy")[None, ...].expand(batch_size, -1, -1)
    if program_norm is not None:
        prior_inputs_embeds = program_norm(prior_inputs_embeds)
    # quantize prior
    if (quantizer is not None) and not no_discrete_prior:
        prior_inputs_embeds, _, _, _ = quantizer(prior_inputs_embeds, train_codebook_only=train_codebook_only)
        if batch_size > 0:
            assert torch.allclose(prior_inputs_embeds[0], prior_inputs_embeds[1])

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
    tokenizer: PreTrainedTokenizerFast,
    # data
    input_ids: List[torch.Tensor],
    attention_mask: List[torch.Tensor],
    label_ids: List[torch.Tensor],
    input_ids_lens: List[List[int]],
    num_pairs: List[int],
    is_same: bool,
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
    loss_type: str,
    debug: bool,
    contrastive_loss: ContrastiveLoss,
    short_context: bool,
    attention_reduction_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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
    prev_past_key_values = None
    prev_past_key_values_attention_mask = None

    # save all programs
    do_save_programs = (debug or program_type != 'none' or invar_loss_lambda > 0.0 or consistency_loss_lambda > 0.0)
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

    # pair indices that need losses
    pairs_that_have_losses = []

    for pair_i, (pair_inputs_embeds, pair_attention_mask, pair_label_ids, pair_input_ids_lens) in enumerate(zip(inputs_embeds, attention_mask, label_ids, input_ids_lens)):
        # check if we need to do losses
        do_loss = True
        if pair_i == 0 and loss_type == 'exclude_first':
            do_loss = False
        if pair_i < n_pairs - 1 and loss_type == 'only_last':
            do_loss = False
        if do_loss:
            pairs_that_have_losses.append(pair_i)

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
        if do_loss:
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
        if pair_i < n_pairs - 1:
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
            if do_loss:
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
            "output_hidden_states": (pair_i < n_pairs - 1), # not the last pair
            "use_cache": not short_context, # need to generate or use kv cache
        }
        if do_loss:
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

        if do_loss:
            ce_losses.append(model_out.loss) # type: ignore

        # STEP 4: update kv
        if pair_i < n_pairs - 1 and not short_context:
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
        if pair_i < n_pairs - 1:
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
        label_ids_with_losses = [label_ids[pair_i] for pair_i in pairs_that_have_losses]
        token_weights = [(pair_label_ids != -100).sum().item() for pair_label_ids in label_ids_with_losses]
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

    # print(ce_loss.item())
    # breakpoint()
    return ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, program_loss, consistency_loss, invar_loss, total_loss # type: ignore


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


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
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
    dry_eval_run: bool,
    no_residual: bool,
    no_discrete_prior: bool,
    no_codebook: bool,
    weird_cast: bool,
    short_context: bool,
    kv_pad_side: str,
    attention_reduction_ratio: float,
    log_every: int,
) -> Tuple[float, List]:

    model.eval()
    prior_embeddings.eval()
    program_embeddings.eval()
    if vae_projection is not None:
        vae_projection.eval()
    if quantizer is not None:
        quantizer.eval()
    if program_norm is not None:
        program_norm.eval()

    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    distributed_state = PartialState()
    output_list = []

    data_idxs = list(range(len(dataset)))
    assert len(data_idxs) >= accelerator.num_processes # avoid padding issue

    with distributed_state.split_between_processes(data_idxs) as process_data_idxs:
        assert isinstance(process_data_idxs, list)
        n_batches = math.ceil(len(process_data_idxs) / batch_size)
        data_idxs = [idxs for idxs in chunks(process_data_idxs, batch_size)]
        assert len(data_idxs) == n_batches

        progress_bar = tqdm(
            range(len(data_idxs)),
            desc="Eval Steps",
            disable=not accelerator.is_local_main_process
        )

        for eval_step, batch_idxs in enumerate(data_idxs):
            batch_data = [dataset[i] for i in batch_idxs]
            bs = len(batch_idxs)
            batch = collate_fn(batch_data)

            if dry_eval_run:
                continue

            # get tensors
            task = batch['task']
            test_idx = batch['test_idx']
            input_ids = [x.to(accelerator.device) for x in batch["input_ids"]]
            attention_mask = [x.to(accelerator.device) for x in batch["attention_mask"]]
            input_ids_lens = batch["input_ids_lens"]
            num_pairs = batch["num_pairs"] # not including test pair
            option = batch['option']
            correct_option = batch['correct_option']

            # gen stuff
            gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_label_ids = batch["gen_label_ids"].to(accelerator.device)
            gen_input_ids_lens = batch["gen_input_ids_lens"]

            device, dtype = gen_input_ids.device, gen_input_ids.dtype

            with accelerator.autocast():
                # get predicted programs and kv cache
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
                    pad_side=dataset.pad_side,
                    no_residual=no_residual,
                    no_discrete_prior=no_discrete_prior,
                    train_codebook_only=no_codebook,
                    weird_cast=weird_cast,
                    kv_pad_side=kv_pad_side,
                    short_context=short_context,
                    attention_reduction_ratio=attention_reduction_ratio,
                )

                pad_embeds = embed_tokens(torch.tensor(dataset.tokenizer.pad_token_id, device=device))
                gen_inputs_embeds = embed_tokens(gen_input_ids)

                # prepend the last predicted program for all pairs except the first
                gen_inputs_embeds = insert_based_on_sides(
                    data=gen_inputs_embeds,
                    to_insert=(prev_programs.to(torch.bfloat16) if weird_cast else prev_programs),
                    lens=gen_input_ids_lens,
                    insert_side="left",
                    pad_side=dataset.pad_side,
                    pad_id=pad_embeds,
                )
                gen_attention_mask = insert_based_on_sides(
                    data=gen_attention_mask,
                    to_insert=torch.full((bs, dataset.ntokens), 1, device=device, dtype=dtype),
                    lens=gen_input_ids_lens,
                    insert_side="left",
                    pad_side=dataset.pad_side,
                    pad_id=0,
                )
                gen_label_ids = insert_based_on_sides(
                    data=gen_label_ids,
                    to_insert=torch.full((bs, dataset.ntokens), -100, device=device, dtype=dtype),
                    lens=gen_input_ids_lens,
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
                        sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=device, dtype=dtype)
                        position_start = mask_for_kv.sum()
                        n_new_positions = mask_after_kv.sum()
                        new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=dtype)
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
                        sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=device, dtype=dtype)
                        n_new_positions = m.sum()
                        new_positions = torch.tensor(range(n_new_positions), device=device, dtype=dtype)
                        if dataset.pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)
                    model_kwargs["position_ids"] = position_ids

                # get losses
                model_out = model(**model_kwargs)
                assert model_out is not None
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
    output_list = gather_object(output_list)
    assert len(output_list) == len(dataset), (len(output_list), len(dataset))

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

    return score, output_list


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
    parser.add_argument("--tracker_project_name", type=str, default="metaicl")
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_len", type=int, default=-1)
    parser.add_argument("--debug_fixed_order", action="store_true")
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)
    parser.add_argument("--debug_no_resume", action='store_true')

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--loss_type", type=str, choices=['only_last', 'all', 'exclude_first'], default='only_last')
    parser.add_argument("--ar_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--token_weighted_loss", action="store_true")
    parser.add_argument("--weird_cast", action="store_true")
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
    parser.add_argument("--no_discrete_prior", action="store_true")
    parser.add_argument("--warmup_cookbook_only_epochs", type=int, default=0)

    # vae
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--subset_kl", action="store_true")
    parser.add_argument("--mlp_factor", type=int, default=4)

    # Training
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_program", type=float, default=1e-5)
    parser.add_argument("--lr_prior", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--samples_per_epoch", type=int, default=5000)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")
    parser.add_argument("--dry_train_run", action="store_true")
    parser.add_argument("--dry_eval_run", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--eval_pretrained", action="store_true")
    parser.add_argument("--full_attention_dropout", type=float, default=0.0)
    parser.add_argument("--demonstration_attention_dropout", type=float, default=0.0) # activate naive selfattn but oom
    parser.add_argument("--program_dropout", type=float, default=0.0)
    parser.add_argument("--program_noise_std", type=float, default=0.0)
    parser.add_argument("--short_context", action='store_true')

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
    parser.add_argument("--config_file", type=str, default="MetaICL/config/hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="MetaICL/data")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--max_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--eval_min_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--max_pair_len", type=int, default=2048)
    parser.add_argument('--eval_seeds', type=str, nargs="+", default=['100'])
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--kv_pad_side", type=str, choices=["left", "right"], default="right")

    # limit eval
    parser.add_argument('--eval_train_test_per_task', type=int, default=50)
    parser.add_argument('--eval_train_ratio', type=float, default=1.0)
    parser.add_argument('--eval_eval_test_per_task', type=int, default=10000000)
    parser.add_argument('--eval_eval_ratio', type=float, default=1.0)

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
        args.log_every = 1
        args.debug_no_resume = True
        args.eval_train_test_per_task = 1
        args.eval_eval_ratio = 0.01
        args.samples_per_epoch = 8

    # check args
    if args.warmup_cookbook_only_epochs:
        assert args.codebook_loss_offset_epochs == 0
        assert args.codebook_loss_linear_epochs == 0
    assert args.commitment_loss_offset_epochs >= args.warmup_cookbook_only_epochs
    if args.demonstration_attention_dropout:
        assert args.no_flash_attn

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
    tokenizer.pad_token = tokenizer.eos_token
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
        data_dir=args.data_dir,
        config_file=args.config_file,
        tokenizer=tokenizer,
        total_steps=args.samples_per_epoch,
        seed=args.seed,
        process_index=accelerator.process_index,
        debug_fixed_order=args.debug_fixed_order,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        pad_side=args.pad_side,
        min_num_pair=args.min_num_pair,
        max_num_pair=args.max_num_pair,
        debug_len=args.debug_len,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        max_pair_len=args.max_pair_len,
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

    # Build eval train dataset (NOTE: only a subset of tasks have options)
    eval_train_datasets = [
        EvalDataset(
            data_dir=args.data_dir,
            config_file=args.config_file,
            seed=args.seed,
            eval_seed=eval_seed,
            tokenizer=tokenizer,
            debug_random_pad=args.debug_random_pad,
            debug_pad_len=args.debug_pad_len,
            pad_side=args.pad_side,
            debug_len=args.debug_len,
            max_seq_len=args.max_seq_len,
            max_pair_len=args.max_pair_len,
            min_num_train_pair=args.eval_min_num_pair - 1,
            max_num_train_pair=args.max_num_pair - 1,
            ntokens=args.ntokens,
            eval_test_per_task=args.eval_train_test_per_task,
            eval_ratio=args.eval_train_ratio,
            split='train',
            debug_fixed_order=args.debug_fixed_order,
        )
        for eval_seed in args.eval_seeds
    ]

    # Build eval eval dataset
    eval_eval_datasets = [
        EvalDataset(
            data_dir=args.data_dir,
            config_file=args.config_file,
            seed=args.seed,
            eval_seed=eval_seed,
            tokenizer=tokenizer,
            debug_random_pad=args.debug_random_pad,
            debug_pad_len=args.debug_pad_len,
            pad_side=args.pad_side,
            debug_len=args.debug_len,
            max_seq_len=args.max_seq_len,
            max_pair_len=args.max_pair_len,
            min_num_train_pair=args.eval_min_num_pair - 1,
            max_num_train_pair=args.max_num_pair - 1,
            ntokens=args.ntokens,
            eval_test_per_task=args.eval_eval_test_per_task,
            eval_ratio=args.eval_eval_ratio,
            split='test',
            debug_fixed_order=args.debug_fixed_order,
        )
        for eval_seed in args.eval_seeds
    ]
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_train_datasets[0]) # only use tokenizer, padding info
    if args.debug_len > 0:
        eval_collate_fn = partial(collate_fn_eval_dummy, dataset=eval_train_datasets[0])

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
    epoch_to_total_score = {}

    # recovery
    logger.info(f"start/resume training from epoch {start_epoch} global_step {global_step} batch {resume_batch_idx}")
    if global_step > 0:
        progress_bar.update(global_step)

    if args.eval_pretrained and start_epoch == 0:
        start_epoch = -1

    # train!
    for epoch in range(start_epoch, args.num_epochs):
        if epoch > -1:
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

                train_codebook_only = (global_step < args.warmup_cookbook_only_epochs * steps_per_epoch)

                with accelerator.accumulate(model, prior_embeddings, program_embeddings, vae_projection, quantizer, program_norm):
                    with accelerator.autocast():
                        ce_loss, kl_loss, codebook_loss, commitment_loss, perplexity, program_loss, consistency_loss, invar_loss, total_loss = model_loss(
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
                            is_same=is_same,
                            num_pairs=num_pairs,
                            # others
                            ntokens=args.ntokens,
                            pad_side=args.pad_side,
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
                            loss_type=args.loss_type,
                            debug=args.debug,
                            contrastive_loss=contrastive_loss,
                            short_context=args.short_context,
                            attention_reduction_ratio=args.attention_reduction_ratio,
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
                                "train/program_loss_accum": program_loss_accum,
                                "train/consistency_loss_accum": consistency_loss_accum,
                                "train/invar_loss_accum": invar_loss_accum,
                                "train/total_loss": total_loss_accum,
                                "train/grad_norm": grad_norm_accum,
                                "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                                "train/lr_prior": lr_scheduler.get_last_lr()[1],
                                "train/lr_program": lr_scheduler.get_last_lr()[2],
                                "train/lr_other": lr_scheduler.get_last_lr()[3],
                            }, step=global_step)
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

            # Eval Train Datasets
            train_scores, train_all_output_list = [], None
            for dataset_i, dataset in enumerate(eval_train_datasets):
                score, output_list = evaluate(
                    model=model,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    vae_projection=vae_projection,
                    quantizer=quantizer,
                    program_norm=program_norm,
                    dataset=dataset,
                    accelerator=accelerator,
                    batch_size=args.eval_batch_size,
                    collate_fn=eval_collate_fn,
                    dry_eval_run=args.dry_eval_run,
                    no_residual=args.no_residual,
                    no_discrete_prior=args.no_discrete_prior,
                    no_codebook=no_codebook,
                    weird_cast=args.weird_cast,
                    short_context=args.short_context,
                    kv_pad_side=args.kv_pad_side,
                    attention_reduction_ratio=args.attention_reduction_ratio,
                    log_every=args.log_every,
                )
                if dataset_i == 0:
                    train_all_output_list = output_list
                train_scores.append(score)
            train_score = sum(train_scores) / len(train_scores)

            # Eval Eval Datasets
            eval_scores, eval_all_output_list = [], None
            for dataset_i, dataset in enumerate(eval_eval_datasets):
                score, output_list = evaluate(
                    model=model,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    vae_projection=vae_projection,
                    quantizer=quantizer,
                    program_norm=program_norm,
                    dataset=dataset,
                    accelerator=accelerator,
                    batch_size=args.eval_batch_size,
                    collate_fn=eval_collate_fn,
                    dry_eval_run=args.dry_eval_run,
                    no_residual=args.no_residual,
                    no_discrete_prior=args.no_discrete_prior,
                    no_codebook=no_codebook,
                    weird_cast=args.weird_cast,
                    short_context=args.short_context,
                    kv_pad_side=args.kv_pad_side,
                    attention_reduction_ratio=args.attention_reduction_ratio,
                    log_every=args.log_every,
                )
                if dataset_i == 0:
                    eval_all_output_list = output_list
                eval_scores.append(score)
            eval_score = sum(eval_scores) / len(eval_scores)

            torch.cuda.empty_cache()
            gc.collect()

            if accelerator.is_main_process:
                eval_metric_dict = {
                    "train/score": train_score,
                    "eval/score": eval_score,
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
                    json.dump(train_all_output_list, f)
                with open(save_eval_eval_pred_gt_path, 'w') as f:
                    json.dump(eval_all_output_list, f)
                logger.info(f"Saved eval train pred gt to {save_eval_train_pred_gt_path}")
                logger.info(f"Saved eval eval pred gt to {save_eval_eval_pred_gt_path}")

                # Save model
                do_save_model = args.save_all_models
                if not args.save_all_models:
                    if (not epoch_to_total_score) or eval_score >= max(epoch_to_total_score.values()):
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
                epoch_to_total_score[epoch] = eval_score

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
