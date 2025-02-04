from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import copy
import arclib # required
import numpy as np
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Dict, Optional, Set, Iterator
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
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
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
    load_tasks_from_data_dir,
    TrainDataset,
    EvalDataset,
    GSDataset,
    collate_fn_train,
    collate_fn_eval,
    collate_fn_gs,
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

import wandb
wandb.login(key='faf21d9ff65ee150697c7e96f070616f6b662134', relogin=True)

logger = get_logger(__name__, log_level="INFO")


MODEL_NAME_TO_PATH = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}
NBIT_TO_DTYPE = {
    16: torch.bfloat16,
    32: torch.float32,
}


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


class Prefix2PrefixProjection(nn.Module):
    def __init__(
            self,
            ntokens: int,
            encoder_model: nn.Module,
            decoder_model: nn.Module,
            identity_init: bool,
            projection_type: str,
            vae: bool,
            device: torch.device,
        ):
        super(Prefix2PrefixProjection, self).__init__()

        # prefixes are formatted as 16 of (2=2, BS=1, nhead=8, nvirtualtoken=1, tokendim / nhead=64)
        self.ntokens = ntokens
        self.vae = vae
        self.device = device
        self.projection_type = projection_type

        # model config
        self.enc_num_layers = encoder_model.config.num_hidden_layers
        self.enc_num_kv_heads = encoder_model.config.num_key_value_heads
        self.enc_embed_size_per_head = encoder_model.config.hidden_size // encoder_model.config.num_attention_heads
        self.enc_size = self.enc_num_layers * self.enc_num_kv_heads * self.enc_embed_size_per_head
        self.dec_num_layers = decoder_model.config.num_hidden_layers
        self.dec_num_kv_heads = decoder_model.config.num_key_value_heads
        self.dec_embed_size_per_head = decoder_model.config.hidden_size // decoder_model.config.num_attention_heads
        self.dec_size = self.dec_num_layers * self.dec_num_kv_heads * self.dec_embed_size_per_head

        # weights
        if projection_type == "none":
            pass
        elif not vae:
            self.k_weights, self.v_weights, self.k_biases, self.v_biases = self.get_projection(
                scheme="identity" if identity_init else "random",
                projection_type=projection_type,
            )
        else:
            self.mu_k_weights, self.mu_v_weights, self.mu_k_biases, self.mu_v_biases = self.get_projection(
                scheme="identity" if identity_init else "random",
                projection_type=projection_type,
            )
            self.logvar_k_weights, self.logvar_v_weights, self.logvar_k_biases, self.logvar_v_biases = self.get_projection(
                scheme="zero" if identity_init else "random",
                projection_type=projection_type,
            )

    def get_projection(self, scheme: str, projection_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if projection_type == "shared":
            k_projections = nn.Linear(self.enc_size, self.dec_size, device=self.device)
            v_projections = nn.Linear(self.enc_size, self.dec_size, device=self.device)
            if scheme == "identity":
                assert self.enc_size == self.dec_size
                with torch.no_grad():
                    k_projections.weight.data.copy_(torch.eye(self.encoder_hidden_size, dtype=k_projections.weight.dtype, device=self.device))
                    k_projections.bias.data.zero_()
                    v_projections.weight.data.copy_(torch.eye(self.encoder_hidden_size, dtype=v_projections.weight.dtype, device=self.device))
                    v_projections.bias.data.zero_()
            elif scheme == "zero":
                with torch.no_grad():
                    k_projections.weight.data.zero_()
                    k_projections.bias.data.zero_()
                    v_projections.weight.data.zero_()
                    v_projections.bias.data.zero_()

            k_weights = nn.Parameter(k_projections.weight)
            v_weights = nn.Parameter(v_projections.weight)
            k_biases = nn.Parameter(k_projections.bias)
            v_biases = nn.Parameter(v_projections.bias)

        elif projection_type == "full":
            k_projections = nn.ModuleList(
                nn.Linear(
                    self.enc_size, self.dec_size, device=self.device,
                ) for _ in range(self.ntokens)
            )
            v_projections = nn.ModuleList(
                nn.Linear(
                    self.enc_size, self.dec_size, device=self.device,
                ) for _ in range(self.ntokens)
            )
            if scheme == "identity":
                assert self.enc_size == self.dec_size
                with torch.no_grad():
                    for projection in k_projections:
                        projection.weight.data.copy_(torch.eye(self.enc_size, dtype=projection.weight.dtype, device=self.device))
                        projection.bias.data.zero_()
                    for projection in v_projections:
                        projection.weight.data.copy_(torch.eye(self.enc_size, dtype=projection.weight.dtype, device=self.device))
                        projection.bias.data.zero_()
            elif scheme == "zero":
                with torch.no_grad():
                    for projection in k_projections:
                        projection.weight.data.zero_()
                        projection.bias.data.zero_()
                    for projection in v_projections:
                        projection.weight.data.zero_()
                        projection.bias.data.zero_()

            k_weights = nn.Parameter(torch.stack([projection.weight for projection in k_projections], dim=0))
            v_weights = nn.Parameter(torch.stack([projection.weight for projection in v_projections], dim=0))
            k_biases = nn.Parameter(torch.stack([projection.bias for projection in k_projections], dim=0))
            v_biases = nn.Parameter(torch.stack([projection.bias for projection in k_projections], dim=0))

        else:
            raise NotImplementedError(f"{projection_type} not yet implemented")

        del k_projections, v_projections
        return k_weights, v_weights, k_biases, v_biases

    def forward(
            self,
            predicted_program: List[Tuple[torch.Tensor, torch.Tensor]],
            vae_no_sample: bool,
            vae_no_kl: bool,
        ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:

        kv_tensor = torch.stack([torch.stack(p) for p in predicted_program]) # (num_layer, 2, batch_size, nhead, nvirtualtoken, token_per_nhead)
        kv_tensor = kv_tensor.permute(1, 2, 4, 0, 3, 5) # (2, batch_size, nvirtualtoken, num_layer, nhead, token_per_nhead)
        kv_tensor = kv_tensor.flatten(start_dim=3) # (2, batch_size, nvirtualtoken, num_layer x nhead x token_per_nhead = enc_size)
        assert kv_tensor.shape[0] == 2
        k_tensor, v_tensor = kv_tensor[0], kv_tensor[1] # (batch_size, nvirtualtoken, num_layer x nhead x token_per_nhead = enc_size)

        kl = torch.tensor(0.0, device=self.device)
        if self.projection_type == "none":
            return predicted_program, kl
        elif not self.vae:
            # self.k_weights size (nvirtualtoken, enc_size, dec_size)
            if self.projection_type == "shared":
                k_tensor = torch.einsum("bnh,hd->bnd", k_tensor, self.k_weights.transpose(0, 1)) + self.k_biases
                v_tensor = torch.einsum("bnh,hd->bnd", v_tensor, self.v_weights.transpose(0, 1)) + self.v_biases
            elif self.projection_type == "full":
                k_tensor = torch.einsum("bnh,nhd->bnd", k_tensor, self.k_weights.transpose(1, 2)) + self.k_biases
                v_tensor = torch.einsum("bnh,nhd->bnd", v_tensor, self.v_weights.transpose(1, 2)) + self.v_biases
            else:
                raise NotImplementedError(f"{self.projection_type} not yet implemented")
            # format
            kv_tensor = torch.stack([k_tensor, v_tensor]) # (2, batch_size, nvirtualtoken, num_layer x nhead x token_per_nhead = dec_size)
            kv_tensor = kv_tensor.view(2, kv_tensor.shape[1], self.ntokens, self.dec_num_layers, self.dec_num_kv_heads, self.dec_embed_size_per_head)
            kv_tensor = kv_tensor.permute(3, 0, 1, 4, 2, 5) # (num_layer, 2, batch_size, nhead, nvirtualtoken, token_per_nhead)
            return [(kv[0], kv[1]) for kv in kv_tensor], kl

        else:
            if self.projection_type == "shared":
                k_mu = torch.einsum("bnh,hd->bnd", k_tensor, self.mu_k_weights.transpose(0, 1)) + self.mu_k_biases
                v_mu = torch.einsum("bnh,hd->bnd", v_tensor, self.mu_v_weights.transpose(0, 1)) + self.mu_v_biases
                k_logvar = torch.einsum("bnh,hd->bnd", k_tensor, self.logvar_k_weights.transpose(0, 1)) + self.logvar_k_biases
                v_logvar = torch.einsum("bnh,hd->bnd", v_tensor, self.logvar_v_weights.transpose(0, 1)) + self.logvar_v_biases
            elif self.projection_type == "full":
                k_mu = torch.einsum("bnh,nhd->bnd", k_tensor, self.mu_k_weights.transpose(1, 2)) + self.mu_k_biases
                v_mu = torch.einsum("bnh,nhd->bnd", v_tensor, self.mu_v_weights.transpose(1, 2)) + self.mu_v_biases
                k_logvar = torch.einsum("bnh,nhd->bnd", k_tensor, self.logvar_k_weights.transpose(1, 2)) + self.logvar_k_biases
                v_logvar = torch.einsum("bnh,nhd->bnd", v_tensor, self.logvar_v_weights.transpose(1, 2)) + self.logvar_v_biases
            else:
                raise NotImplementedError(f"{self.projection_type} not yet implemented")

            mu = torch.stack([k_mu, v_mu]) # (2, batch_size, nvirtualtoken, num_layer x nhead x token_per_nhead = dec_size)
            logvar = torch.stack([k_logvar, v_logvar]) # (2, batch_size, nvirtualtoken, num_layer x nhead x token_per_nhead = dec_size)
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std) if vae_no_sample else torch.randn_like(std)
            kv_tensor = mu + eps * std
            if not vae_no_kl:
                kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2.0) - logvar.exp())
            # format
            kv_tensor = kv_tensor.view(2, kv_tensor.shape[1], self.ntokens, self.dec_num_layers, self.dec_num_kv_heads, self.dec_embed_size_per_head)
            kv_tensor = kv_tensor.permute(3, 0, 1, 4, 2, 5) # (num_layer, 2, batch_size, nhead, nvirtualtoken, token_per_nhead)
            return [(kv[0], kv[1]) for kv in kv_tensor], kl

    def get_memory_footprint(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters()) + \
            sum(p.nelement() * p.element_size() for p in self.buffers())


class Hidden2PromptProjection(nn.Module):
    def __init__(
            self,
            ntokens: int,
            encoder_model: nn.Module,
            decoder_model: nn.Module,
            identity_init: bool,
            projection_type: str,
            vae: bool,
            device: torch.device,
        ):
        super(Hidden2PromptProjection, self).__init__()

        self.ntokens = ntokens
        self.vae = vae
        self.projection_type = projection_type
        self.device = device

        # model config
        self.encoder_hidden_size = encoder_model.config.hidden_size
        self.decoder_hidden_size = decoder_model.config.hidden_size

        # weights
        if projection_type == "none":
            pass
        elif not vae:
            self.weights, self.biases = self.get_projection(
                scheme="identity" if identity_init else "random",
                projection_type=projection_type,
            )
        else:
            self.mu_weights, self.mu_biases = self.get_projection(
                scheme="identity" if identity_init else "random",
                projection_type=projection_type,
            )
            self.logvar_weights, self.logvar_biases = self.get_projection(
                scheme="zero" if identity_init else "random",
                projection_type=projection_type,
            )

    def get_projection(self, scheme: str, projection_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if projection_type == "shared":
            projections = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, device=self.device)
            if scheme == "identity":
                assert self.encoder_hidden_size == self.decoder_hidden_size
                with torch.no_grad():
                    projections.weight.data.copy_(torch.eye(self.encoder_hidden_size, dtype=projections.weight.dtype, device=self.device))
                    projections.bias.data.zero_()
            elif scheme == "zero":
                with torch.no_grad():
                    projections.weight.data.zero_()
                    projections.bias.data.zero_()

            weights = nn.Parameter(projections.weight)
            biases = nn.Parameter(projections.bias)

        elif projection_type == "full":
            projections = nn.ModuleList([
                nn.Linear(
                    self.encoder_hidden_size,
                    self.decoder_hidden_size,
                    device=self.device,
                ) for _ in range(self.ntokens)
            ])
            if scheme == "identity":
                assert self.encoder_hidden_size == self.decoder_hidden_size
                with torch.no_grad():
                    for projection in projections:
                        projection.weight.data.copy_(torch.eye(self.encoder_hidden_size, dtype=projection.weight.dtype, device=self.device))
                        projection.bias.data.zero_()
            elif scheme == "zero":
                with torch.no_grad():
                    for projection in projections:
                        projection.weight.data.zero_()
                        projection.bias.data.zero_()

            weights = nn.Parameter(torch.stack([projection.weight for projection in projections], dim=0))
            biases = nn.Parameter(torch.stack([projection.bias for projection in projections], dim=0))

        else:
            raise NotImplementedError(f"{projection_type} not yet implemented")

        del projections
        return weights, biases

    def forward(
            self,
            enc_hidden_states: torch.Tensor,
            vae_no_sample: bool,
            vae_no_kl: bool
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # enc_hidden_states has shape (batch_size, ntokens, hidden_dim)
        assert enc_hidden_states.shape[1] == self.ntokens

        kl = torch.tensor(0.0, device=self.device)
        if self.projection_type == "none":
            return enc_hidden_states, kl
        elif not self.vae:
            if self.projection_type == "shared":
                enc_hidden_states = torch.einsum("bnh,hd->bnd", enc_hidden_states, self.weights.transpose(0, 1)) + self.biases
            elif self.projection_type == "full":
                enc_hidden_states = torch.einsum("bnh,nhd->bnd", enc_hidden_states, self.weights.transpose(1, 2)) + self.biases
            else:
                raise NotImplementedError(f"{self.projection_type} not yet implemented")
            return enc_hidden_states, kl

        else:
            if self.projection_type == "shared":
                mu = torch.einsum("bnh,hd->bnd", enc_hidden_states, self.mu_weights.transpose(0, 1)) + self.mu_biases
                logvar = torch.einsum("bnh,hd->bnd", enc_hidden_states, self.logvar_weights.transpose(0, 1)) + self.logvar_biases
            elif self.projection_type == "full":
                mu = torch.einsum("bnh,nhd->bnd", enc_hidden_states, self.mu_weights.transpose(1, 2)) + self.mu_biases
                logvar = torch.einsum("bnh,nhd->bnd", enc_hidden_states, self.logvar_weights.transpose(1, 2)) + self.logvar_biases
            else:
                raise NotImplementedError(f"{self.projection_type} not yet implemented")

            std = torch.exp(0.5 * logvar)
            eps = torch.zeros_like(std) if vae_no_sample else torch.randn_like(std)
            enc_hidden_states = mu + eps * std
            if not vae_no_kl:
                kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2.0) - logvar.exp())
            return enc_hidden_states, kl

    def get_memory_footprint(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters()) + \
            sum(p.nelement() * p.element_size() for p in self.buffers())


################################################
# A shared forward pass for training & evaluation
################################################
def encoder_decoder_loss(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    conditioning_method: str,
    conditioning_projection: Union[Prefix2PrefixProjection, Hidden2PromptProjection],
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    encoder_labels: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    decoder_labels: torch.Tensor,
    enc_ids_lens: List[int],
    dec_ids_lens: List[int],
    anti_invars: List[bool],
    ntokens: int,
    invar_loss_lambda: float,
    encoder_loss_lambda: float,
    no_lora: bool,
    decoder_ce_loss: bool,
    encoder_pad_side: str,
    decoder_pad_side: str,
    trainable_nbit: int,
    no_flash_attn: bool,
    debug_vae_no_sample: bool,
    debug_vae_no_kl: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
    # batch size is not necessarily true due to eval
    # Encoder forward and get loss
    enc_out = encoder_model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        labels=encoder_labels,
        output_hidden_states=True,
    )
    encoder_loss = enc_out.loss

    # get predicted program for decoder and enc_hidden_states for invar loss
    if conditioning_method == "prefix2prefix":
        # get prefix
        predicted_program = []
        if encoder_pad_side == "right":
            for x1, x2 in enc_out.past_key_values:
                predicted_program.append((
                    torch.stack([x[:, l-ntokens:l, :] for x, l in zip(x1, enc_ids_lens)]),
                    torch.stack([x[:, l-ntokens:l, :] for x, l in zip(x2, enc_ids_lens)]),
                ))
        else:
            for x1, x2 in enc_out.past_key_values:
                predicted_program.append((
                    torch.stack([x[:, -ntokens:, :] for x in x1]),
                    torch.stack([x[:, -ntokens:, :] for x in x2]),
                ))
        # prefix to either prefix or prompt
        predicted_program, kl_loss = conditioning_projection(
            predicted_program=predicted_program,
            vae_no_sample=debug_vae_no_sample,
            vae_no_kl=debug_vae_no_kl,
        )
        # hidden state will be used for invar loss, so make it a batch first tensor
        enc_hidden_states = torch.stack([torch.stack(x) for x in predicted_program]) # each (batch_size, num_kv_heads, ntokens, embed_size_per_head)
        enc_hidden_states = enc_hidden_states.permute(2, 0, 1, 3, 4, 5) # (batch_size, num_layer, 2, num_kv_heads, ntokens, embed_size_per_head)

    elif conditioning_method == "hidden2prompt":
        # get hidden states
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-ntokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-ntokens:] for x in enc_hidden_states])
        # predicted program is either just the hidden states, or projected
        predicted_program, kl_loss = conditioning_projection(
            enc_hidden_states=enc_hidden_states,
            vae_no_sample=debug_vae_no_sample,
            vae_no_kl=debug_vae_no_kl,
        )

    else:
        raise ValueError(f"invalid conditioning method {conditioning_method}")

    # decoder ce loss
    ce_loss = torch.tensor(-1.0, device=encoder_input_ids.device)
    if decoder_ce_loss:
        ce_loss = decoder_loss(
            decoder_model=decoder_model,
            conditioning_method=conditioning_method,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_labels=decoder_labels,
            dec_ids_lens=dec_ids_lens,
            ntokens=ntokens,
            no_lora=no_lora,
            decoder_pad_side=decoder_pad_side,
            trainable_nbit=trainable_nbit,
            no_flash_attn=no_flash_attn,
            predicted_program=predicted_program,
        )

    # invariance loss (batch only not 2x in evaluation)
    batch_size = encoder_input_ids.shape[0]
    assert enc_hidden_states.shape[0] == len(anti_invars) == batch_size
    invar_loss = torch.tensor(0.0, device=encoder_input_ids.device)
    if batch_size % 2 == 0:
        for batch_i in range(0, batch_size, 2):
            assert anti_invars[batch_i] == anti_invars[batch_i + 1]
            l = nn.functional.mse_loss(enc_hidden_states[batch_i], enc_hidden_states[batch_i + 1])
            invar_loss += -l if anti_invars[batch_i] else l
        invar_loss /= (batch_size // 2)

    total_loss = ce_loss + invar_loss_lambda * invar_loss + encoder_loss_lambda * encoder_loss + kl_loss
    return ce_loss, invar_loss, encoder_loss, kl_loss, total_loss, predicted_program


def decoder_loss(
    decoder_model: nn.Module,
    conditioning_method: str,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    decoder_labels: torch.Tensor,
    dec_ids_lens: List[int],
    ntokens: int,
    no_lora: bool,
    decoder_pad_side: str,
    trainable_nbit: int,
    no_flash_attn: bool,
    predicted_program: Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:

    # decoder attention mask must be extended
    prefix_attention_mask = torch.full(
        (decoder_attention_mask.shape[0], ntokens),
        1,
        device=decoder_attention_mask.device,
        dtype=decoder_attention_mask.dtype,
    )

    if conditioning_method == "prefix2prefix":
        assert isinstance(predicted_program, list)
        # pad decoder attention mask
        decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)
        if not no_flash_attn:
            predicted_program = [
                tuple([x[0].to(NBIT_TO_DTYPE[trainable_nbit]), x[1].to(NBIT_TO_DTYPE[trainable_nbit])])
            for x in predicted_program] # type: ignore
        # decoder forward
        dec_out = decoder_model(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=predicted_program,
            labels=decoder_labels,
        )

    elif conditioning_method == "hidden2prompt":
        assert isinstance(predicted_program, torch.Tensor)
        # pad decoder attention mask
        if decoder_pad_side == "right":
            decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)
        else:
            decoder_attention_mask_new = []
            for x, m, l in zip(decoder_attention_mask, prefix_attention_mask, dec_ids_lens):
                x = torch.cat([x[:-l], m, x[-l:]])
                decoder_attention_mask_new.append(x)
            decoder_attention_mask = torch.stack(decoder_attention_mask_new)
        # pad decoder inputs embeds
        decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
        if no_lora:
            decoder_inputs_embeds = decoder_module.model.embed_tokens(decoder_input_ids)
        else:
            decoder_inputs_embeds = decoder_module.model.model.embed_tokens(decoder_input_ids)
        if decoder_pad_side == "right":
            decoder_inputs_embeds = torch.cat([predicted_program, decoder_inputs_embeds], dim=1)
        else:
            decoder_inputs_embeds_new = []
            for x, p, l in zip(decoder_inputs_embeds, predicted_program, dec_ids_lens):
                x = torch.cat([x[:-l], p, x[-l:]])
                decoder_inputs_embeds_new.append(x)
            decoder_inputs_embeds = torch.stack(decoder_inputs_embeds_new)
        # pad label
        prefix_label_mask = torch.full(
            (decoder_labels.shape[0], ntokens),
            -100,
            device=decoder_labels.device,
            dtype=decoder_labels.dtype,
        )
        if decoder_pad_side == "right":
            decoder_labels = torch.cat([prefix_label_mask, decoder_labels], dim=1)
        else:
            decoder_labels_new = []
            for x, m, l in zip(decoder_labels, prefix_label_mask, dec_ids_lens):
                x = torch.cat([x[:-l], m, x[-l:]])
                decoder_labels_new.append(x)
            decoder_labels = torch.stack(decoder_labels_new)
        # decoder forward
        dec_out = decoder_model(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            labels=decoder_labels,
        )

    else:
        raise ValueError(f"invalid conditioning method {conditioning_method}")

    return dec_out.loss


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
        decoder_model: nn.Module,
        iters: int,
        predicted_program: Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
        conditioning_method: str,
        no_lora: bool,
        trainable_nbit: int,
        no_flash_attn: bool,
        max_grad_norm: float,
    ) -> Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    # note gradient checkpointing does not matter because we are freezing the model here
    # however, if we choose to tune the decoder as well, then gradient checkpointing might be desired

    assert len(batch_idxs) == 1
    eval_task = eval_dataset.eval_tasks[batch_idxs[0]]

    # gradient search dataset and dataloader
    gs_dataset = GSDataset(
        task=eval_task,
        encoder_tokenizer=eval_dataset.encoder_tokenizer,
        decoder_tokenizer=eval_dataset.decoder_tokenizer,
        max_seq_len=eval_dataset.max_seq_len,
        no_compact_grids=eval_dataset.no_compact_grids,
        ntokens=eval_dataset.ntokens,
        debug_random_pad=eval_dataset.debug_random_pad,
        debug_pad_len=eval_dataset.debug_pad_len,
        decoder_pad_side=eval_dataset.decoder_pad_side,
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
    if isinstance(predicted_program, torch.Tensor):
        program_params = [predicted_program]
    else:
        program_params = [item for sublist in predicted_program for item in sublist]
    assert all(not p.requires_grad for p in program_params)
    for p in program_params:
        p.requires_grad = True

    # expand to match predicted program with batch size
    if isinstance(predicted_program, torch.Tensor):
        assert predicted_program.shape[0] == 1
        predicted_program = predicted_program.expand(batch_size, *predicted_program.shape[1:])
    else:
        for layer_i, (layer_program_0, layer_program_1) in enumerate(predicted_program):
            assert layer_program_0.shape[0] == layer_program_1.shape[0] == 1
            predicted_program[layer_i] = (
                layer_program_0.expand(batch_size, *layer_program_0.shape[1:]),
                layer_program_1.expand(batch_size, *layer_program_1.shape[1:]),
            )

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
    optim, gs_loader = accelerator.prepare(optim, gs_loader)

    curr_iter = 0
    best_loss = float("inf")
    best_program = None
    decoder_model.train()

    # train!
    while curr_iter < iters:
        for gs_batch_data in gs_loader:
            dec_ids = gs_batch_data["decoder_input_ids"].to(accelerator.device)
            dec_mask = gs_batch_data["decoder_attention_mask"].to(accelerator.device)
            dec_labels = gs_batch_data["decoder_labels"].to(accelerator.device)
            dec_ids_lens = gs_batch_data["decoder_input_ids_lens"]
            with accelerator.autocast():
                gs_loss = decoder_loss(
                    decoder_model=decoder_model,
                    conditioning_method=conditioning_method,
                    decoder_input_ids=dec_ids,
                    decoder_attention_mask=dec_mask,
                    decoder_labels=dec_labels,
                    dec_ids_lens=dec_ids_lens,
                    ntokens=eval_dataset.ntokens,
                    no_lora=no_lora,
                    decoder_pad_side=eval_dataset.decoder_pad_side,
                    trainable_nbit=trainable_nbit,
                    no_flash_attn=no_flash_attn,
                    predicted_program=predicted_program,
                )
            accelerator.backward(gs_loss)
            accelerator.clip_grad_norm_(program_params, max_grad_norm)
            optim.step()
            scheduler.step()

            if take_best and gs_loss.item() < best_loss:
                best_loss = gs_loss.item()
                if isinstance(predicted_program, torch.Tensor):
                    best_program = predicted_program.detach().clone()
                else:
                    best_program = [
                        (layer_program_0.detach().clone(), layer_program_1.detach().clone())
                        for layer_program_0, layer_program_1 in predicted_program
                    ]

            curr_iter += 1
            if curr_iter >= iters:
                break

    # set decoder back to eval mode
    decoder_model.eval()

    for p in program_params:
        p.requires_grad = False

    if take_best:
        assert best_program is not None
        predicted_program = best_program

    # shrink to match predicted program with batch size 1
    if isinstance(predicted_program, torch.Tensor):
        predicted_program = predicted_program[:1]
    else:
        for layer_i, (layer_program_0, layer_program_1) in enumerate(predicted_program):
            predicted_program[layer_i] = (layer_program_0[:1], layer_program_1[:1])

    return predicted_program


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    task_to_ttt_model_paths: Optional[Dict[str, Tuple[str, Optional[str], str]]],
    encoder_ttt_param_names: Optional[Set[str]],
    decoder_ttt_param_names: Optional[Set[str]],
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    conditioning_method: str,
    conditioning_projection: Union[Prefix2PrefixProjection, Hidden2PromptProjection],
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    no_lora: bool,
    decoder_ce_loss: bool,
    trainable_nbit: int,
    no_flash_attn: bool,
    tie_models: bool,
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
    debug_vae_no_sample: bool,
    debug_vae_no_kl: bool,
):
    encoder_model.eval()
    if not tie_models:
        decoder_model.eval()
    conditioning_projection.eval()

    # get modules in case of DDP
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model

    # if ttt provided, same model weights for the missing ttt task weights
    cached_enc_weights_path = None
    cached_dec_weights_path = None
    cached_proj_weights_path = None
    curr_ttt_task_name = None
    if task_to_ttt_model_paths is not None: # run on both processes
        # save encoder
        cached_enc_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_encoder_cache.pt")
        assert isinstance(encoder_ttt_param_names, set)
        enc_name_to_params = set(name for name, _ in encoder_module.named_parameters())
        assert all(name in enc_name_to_params for name in encoder_ttt_param_names), f"process{accelerator.process_index} {encoder_ttt_param_names} {enc_name_to_params}"
        enc_weights = {name: param for name, param in encoder_module.named_parameters() if name in encoder_ttt_param_names}
        torch.save(enc_weights, cached_enc_weights_path)
        logger.info(f"ttt provided, cached {len(enc_weights)} encoder weights to {cached_enc_weights_path}")
        # save decoder
        if not tie_models:
            cached_dec_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_decoder_cache.pt")
            assert isinstance(decoder_ttt_param_names, set)
            dec_name_to_params = set(name for name, _ in decoder_module.named_parameters())
            assert all(name in dec_name_to_params for name in decoder_ttt_param_names), f"process{accelerator.process_index} {decoder_ttt_param_names}"
            dec_weights = {name: param for name, param in decoder_module.named_parameters() if name in decoder_ttt_param_names}
            torch.save(dec_weights, cached_dec_weights_path)
            logger.info(f"ttt provided, cached {len(dec_weights)} decoder weights to {cached_dec_weights_path}")
        # save projection
        cached_proj_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_conditioning_projection_cache.pt")
        torch.save(conditioning_projection, cached_proj_weights_path)
        logger.info(f"ttt provided, cached conditioning projection weights to {cached_proj_weights_path}")
        # save default to model paths and set current ttt weights to default
        task_to_ttt_model_paths["default"] = (cached_enc_weights_path, cached_dec_weights_path, cached_proj_weights_path)
        curr_ttt_task_name = "default"

    # setup terminators and suppress warning
    terminators = [
        dataset.decoder_tokenizer.eos_token_id,
    ]
    decoder_module.generation_config.pad_token_id = dataset.decoder_tokenizer.pad_token_id

    distributed_state = PartialState()
    task_id_and_text_list = []
    task_id_and_inverter_grids = []
    ce_loss_list = []
    encoder_loss_list = []
    kl_loss_list = []
    exact_acc_list = []
    valid_grid_list = []
    correct_grid_dim_list = []
    token_acc_list = []
    ttt_provided_list = []

    data_idxs = list(range(len(dataset)))
    assert len(data_idxs) >= accelerator.num_processes # avoid padding issue

    with distributed_state.split_between_processes(data_idxs) as process_data_idxs:
        assert isinstance(process_data_idxs, list)
        n_batches = math.ceil(len(process_data_idxs) / batch_size)

        # if ttt provided, make sure all batches are of the same task name
        if task_to_ttt_model_paths is not None:
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
            if task_to_ttt_model_paths is not None:
                task_names = [dataset.eval_tasks[idx].name.split('-')[0] for idx in batch_idxs]
                assert len(set(task_names)) == 1 # have to be the same task
                task_name = task_names[0]
                if task_name != curr_ttt_task_name:
                    if task_name not in task_to_ttt_model_paths:
                        task_name = "default"
                    else:
                        ttt_provided = [1] * bs
                    # get paths
                    enc_ttt_path, dec_ttt_path, proj_ttt_path = task_to_ttt_model_paths[task_name]
                    # load encoder
                    encoder_model_ttt_state_dict = torch.load(
                        enc_ttt_path,
                        weights_only=True,
                        map_location=accelerator.device
                    )
                    assert set(encoder_model_ttt_state_dict.keys()) == encoder_ttt_param_names
                    encoder_module.load_state_dict(encoder_model_ttt_state_dict, strict=False)
                    del encoder_model_ttt_state_dict
                    # load decoder
                    if dec_ttt_path is not None:
                        decoder_model_ttt_state_dict = torch.load(
                            dec_ttt_path,
                            weights_only=True,
                            map_location=accelerator.device
                        )
                        assert set(decoder_model_ttt_state_dict.keys()) == decoder_ttt_param_names
                        decoder_module.load_state_dict(decoder_model_ttt_state_dict, strict=False)
                        del decoder_model_ttt_state_dict
                    conditioning_projection = torch.load(
                        proj_ttt_path,
                        weights_only=False,
                        map_location=accelerator.device
                    )
                    # set current task name
                    curr_ttt_task_name = task_name

                    # another eval after loading weight just in case
                    encoder_model.eval()
                    if not tie_models:
                        decoder_model.eval()
                    conditioning_projection.eval()

            ttt_provided_list += ttt_provided

            # predict multiple programs from i/o permutations
            predicted_program_accum = None
            avail_accum = [0] * bs
            first_batch = None

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
                enc_ids = batch["encoder_input_ids"].to(accelerator.device)
                enc_mask = batch["encoder_attention_mask"].to(accelerator.device)
                enc_labels = batch["encoder_labels"].to(accelerator.device)
                dec_ids = batch["decoder_input_ids"].to(accelerator.device)
                dec_mask = batch["decoder_attention_mask"].to(accelerator.device)
                dec_gen_ids = batch["decoder_gen_input_ids"].to(accelerator.device)
                dec_gen_mask = batch["decoder_gen_attention_mask"].to(accelerator.device)
                dec_labels = batch["decoder_labels"].to(accelerator.device)
                label_texts = batch["decoder_label_texts"]
                out_token_length = batch["decoder_out_token_length"]
                enc_ids_lens = batch["encoder_input_ids_lens"]
                dec_ids_lens = batch["decoder_input_ids_lens"]
                dec_gen_ids_lens = batch["decoder_gen_input_ids_lens"]

                # save first batch with complete tasks, rest might be incomplete
                if batch_permute_i == 0:
                    first_batch = {
                        "task_ids": copy.deepcopy(task_ids),
                        "inverters": copy.deepcopy(inverters),
                        "decoder_input_ids": copy.deepcopy(dec_ids),
                        "decoder_attention_mask": copy.deepcopy(dec_mask),
                        "decoder_gen_input_ids": copy.deepcopy(dec_gen_ids),
                        "decoder_gen_attention_mask": copy.deepcopy(dec_gen_mask),
                        "decoder_labels": copy.deepcopy(dec_labels),
                        "decoder_label_texts": copy.deepcopy(label_texts),
                        "decoder_out_token_length": copy.deepcopy(out_token_length),
                        "decoder_input_ids_lens": copy.deepcopy(dec_ids_lens),
                        "decoder_gen_input_ids_lens": copy.deepcopy(dec_gen_ids_lens),
                    }

                # compute ce loss
                with accelerator.autocast():
                    # invarloss is always 0, lambdas and total loss are unnecessary
                    ce_loss, _, encoder_loss, kl_loss, _, predicted_program = encoder_decoder_loss(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        conditioning_method=conditioning_method,
                        conditioning_projection=conditioning_projection,
                        encoder_input_ids=enc_ids,
                        encoder_attention_mask=enc_mask,
                        encoder_labels=enc_labels,
                        decoder_input_ids=dec_ids,
                        decoder_attention_mask=dec_mask,
                        decoder_labels=dec_labels,
                        enc_ids_lens=enc_ids_lens,
                        dec_ids_lens=dec_ids_lens,
                        anti_invars=[True] * len(dec_ids_lens), # HARDCODE
                        ntokens=dataset.ntokens,
                        invar_loss_lambda=0.0, # HARDCODE
                        encoder_loss_lambda=0.0, # HARDCODE
                        no_lora=no_lora,
                        decoder_ce_loss=decoder_ce_loss,
                        encoder_pad_side=dataset.encoder_pad_side,
                        decoder_pad_side=dataset.decoder_pad_side,
                        trainable_nbit=trainable_nbit,
                        no_flash_attn=no_flash_attn,
                        debug_vae_no_sample=debug_vae_no_sample,
                        debug_vae_no_kl=debug_vae_no_kl,
                    )

                # ce loss should be from the original permutation, which is set to the first permuted batch
                if batch_permute_i == 0:
                    ce_loss_list += [ce_loss.item()] * bs
                    encoder_loss_list += [encoder_loss.item()] * bs
                    kl_loss_list += [kl_loss.item()] * bs

                # accumulate program
                if batch_permute_i == 0:
                    predicted_program_accum = predicted_program
                elif conditioning_method == "prefix2prefix":
                    assert isinstance(predicted_program, list)
                    assert len(predicted_program) == len(predicted_program_accum) # same number of layers # type: ignore
                    for layer_i, (layer_program_0, layer_program_1) in enumerate(predicted_program):
                        avail_count = 0
                        for batch_i, avail in enumerate(batch_avail):
                            if avail:
                                predicted_program_accum[layer_i][0][batch_i] += layer_program_0[avail_count] # type: ignore
                                predicted_program_accum[layer_i][1][batch_i] += layer_program_1[avail_count] # type: ignore
                                avail_count += 1
                else:
                    assert isinstance(predicted_program, torch.Tensor)
                    assert predicted_program.shape[1:] == predicted_program_accum.shape[1:] # type: ignore
                    for batch_i, avail in enumerate(batch_avail):
                        avail_count = 0
                        if avail:
                            predicted_program_accum[batch_i] += predicted_program[avail_count] # type: ignore
                            avail_count += 1

                # accumulate avail (count of permutations)
                assert len(batch_avail) == len(avail_accum) == bs
                for batch_i in range(bs):
                    avail_accum[batch_i] += batch_avail[batch_i]

            # average program
            assert all(avail > 0 for avail in avail_accum)
            if conditioning_method == "prefix2prefix":
                for layer_i, (layer_program_0, layer_program_1) in enumerate(predicted_program_accum): # type: ignore
                    assert len(layer_program_0) == len(layer_program_1) == len(avail_accum) == bs
                    for batch_i in range(bs):
                        layer_program_0[batch_i] /= avail_accum[batch_i]
                        layer_program_1[batch_i] /= avail_accum[batch_i]
                    predicted_program_accum[layer_i] = (layer_program_0, layer_program_1) # type: ignore
            else:
                assert len(predicted_program_accum) == len(avail_accum) == bs # type: ignore
                for i, avail in enumerate(avail_accum):
                    predicted_program_accum[i] /= avail # type: ignore
            predicted_program: Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]] = predicted_program_accum # type: ignore

            # recover data from first batch (e.g. task_ids might be missing tasks due to permute_iters)
            assert isinstance(first_batch, dict)
            task_ids = first_batch["task_ids"]
            inverters = first_batch["inverters"]
            dec_ids = first_batch["decoder_input_ids"].to(accelerator.device)
            dec_mask = first_batch["decoder_attention_mask"].to(accelerator.device)
            dec_gen_ids = first_batch["decoder_gen_input_ids"].to(accelerator.device)
            dec_gen_mask = first_batch["decoder_gen_attention_mask"].to(accelerator.device)
            dec_labels = first_batch["decoder_labels"].to(accelerator.device)
            label_texts = first_batch["decoder_label_texts"]
            out_token_length = first_batch["decoder_out_token_length"]
            dec_ids_lens = first_batch["decoder_input_ids_lens"]
            dec_gen_ids_lens = first_batch["decoder_gen_input_ids_lens"]

            if gs_iters > 0:
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
                    decoder_model=decoder_model,
                    iters=gs_iters,
                    predicted_program=predicted_program,
                    conditioning_method=conditioning_method,
                    no_lora=no_lora,
                    trainable_nbit=trainable_nbit,
                    no_flash_attn=no_flash_attn,
                    max_grad_norm=gs_max_grad_norm,
                )
                # need to compute new ce loss to be more representative
                with accelerator.autocast():
                    ce_loss = decoder_loss(
                        decoder_model=decoder_model,
                        conditioning_method=conditioning_method,
                        decoder_input_ids=dec_ids,
                        decoder_attention_mask=dec_mask,
                        decoder_labels=dec_labels,
                        dec_ids_lens=dec_ids_lens,
                        ntokens=dataset.ntokens,
                        no_lora=no_lora,
                        decoder_pad_side=dataset.decoder_pad_side,
                        trainable_nbit=trainable_nbit,
                        no_flash_attn=no_flash_attn,
                        predicted_program=predicted_program,
                    )
                    ce_loss_list[-1] = ce_loss.item()

            # recover data from first batch (e.g. task_ids might be missing tasks due to permute_iters)
            assert isinstance(first_batch, dict)
            task_ids = first_batch["task_ids"]
            inverters = first_batch["inverters"]
            dec_gen_ids = first_batch["decoder_gen_input_ids"].to(accelerator.device)
            dec_gen_mask = first_batch["decoder_gen_attention_mask"].to(accelerator.device)
            label_texts = first_batch["decoder_label_texts"]
            out_token_length = first_batch["decoder_out_token_length"]
            dec_gen_ids_lens = first_batch["decoder_gen_input_ids_lens"]

            # compute accuracy
            with accelerator.autocast():
                # padding at front because HF ignores it
                gen_texts = None
                if conditioning_method == "prefix2prefix":
                    assert isinstance(predicted_program, list)
                    dec_gen_ids = torch.cat([
                        torch.ones((bs, dataset.ntokens), device=dec_gen_ids.device, dtype=dec_gen_ids.dtype),
                        dec_gen_ids
                    ], dim=1) # the addition will be ignored, double checked
                    dec_gen_mask = torch.cat([
                        torch.ones((bs, dataset.ntokens), device=dec_gen_mask.device, dtype=dec_gen_mask.dtype),
                        dec_gen_mask
                    ], dim=1)
                    if not no_flash_attn:
                        predicted_program = [
                            ([x[0].to(NBIT_TO_DTYPE[trainable_nbit]), x[1].to(NBIT_TO_DTYPE[trainable_nbit])])
                        for x in predicted_program] # type: ignore
                    gen_tokens = decoder_module.generate(
                        input_ids=dec_gen_ids,
                        attention_mask=dec_gen_mask,
                        past_key_values=predicted_program,
                        max_new_tokens=max(out_token_length), # arbitrary increase
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=terminators,
                    )
                    gen_texts = dataset.decoder_tokenizer.batch_decode(gen_tokens[:, dec_gen_ids.shape[1]:], skip_special_tokens=True)
                elif conditioning_method == "hidden2prompt":
                    assert isinstance(predicted_program, torch.Tensor)
                    if no_lora:
                        decoder_inputs_embeds = decoder_module.model.embed_tokens(dec_gen_ids)
                    else:
                        decoder_inputs_embeds = decoder_module.model.model.embed_tokens(dec_gen_ids)
                    # pad decoder inputs embeds
                    if dataset.decoder_gen_pad_side == "right":
                        decoder_inputs_embeds = torch.cat([predicted_program, decoder_inputs_embeds], dim=1)
                    else:
                        decoder_inputs_embeds_new = []
                        for x, p, l in zip(decoder_inputs_embeds, predicted_program, dec_gen_ids_lens):
                            x = torch.cat([x[:-l], p, x[-l:]])
                            decoder_inputs_embeds_new.append(x)
                        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds_new)
                    if not no_flash_attn:
                        decoder_inputs_embeds = decoder_inputs_embeds.to(NBIT_TO_DTYPE[trainable_nbit])
                    # pad decoder attention masks
                    prefix_attention_mask = torch.full(
                        (dec_gen_mask.shape[0], dataset.ntokens),
                        1,
                        device=dec_gen_mask.device,
                        dtype=dec_gen_mask.dtype,
                    )
                    if dataset.decoder_gen_pad_side == "right":
                        dec_gen_mask = torch.cat([prefix_attention_mask, dec_gen_mask], dim=1)
                    else:
                        dec_gen_mask_new = []
                        for x, m, l in zip(dec_gen_mask, prefix_attention_mask, dec_gen_ids_lens):
                            x = torch.cat([x[:-l], m, x[-l:]])
                            dec_gen_mask_new.append(x)
                        dec_gen_mask = torch.stack(dec_gen_mask_new)
                    # generate
                    gen_tokens = decoder_module.generate(
                        inputs_embeds=decoder_inputs_embeds,
                        attention_mask=dec_gen_mask,
                        max_new_tokens=max(out_token_length), # arbitrary increase
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=terminators,
                    )
                    gen_texts = dataset.decoder_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                else:
                    raise ValueError(f"invalid conditioning method {conditioning_method}")

            # Compare each gen_text with label_texts
            assert len(task_ids) == len(inverters) == bs, (len(task_ids), len(inverters), bs)
            assert len(gen_texts) == len(label_texts) == bs, (len(gen_texts), len(label_texts), bs)
            for task_id, inverter, gen_text, label_text in zip(task_ids, inverters, gen_texts, label_texts):
                # save gen and gt text
                task_id_and_text_list.append((task_id, gen_text, label_text))
                # exact acc
                exact_acc_list.append(int(gen_text == label_text))
                # is valid grid
                gen_grid, gen_is_grid = text_to_2d_grid(gen_text, dataset.no_compact_grids)
                label_grid, label_is_grid = text_to_2d_grid(label_text, dataset.no_compact_grids)
                assert label_is_grid
                valid_grid_list.append(int(gen_is_grid))
                if not gen_is_grid:
                    correct_grid_dim_list.append(0)
                    token_acc_list.append(0)
                    continue
                assert isinstance(gen_grid, list)
                assert isinstance(label_grid, list)
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
    # losses
    ce_loss_list = gather_object(ce_loss_list)
    encoder_loss_list = gather_object(encoder_loss_list)
    kl_loss_list = gather_object(kl_loss_list)
    # accuracies
    exact_acc_list = gather_object(exact_acc_list)
    valid_grid_list = gather_object(valid_grid_list)
    correct_grid_dim_list = gather_object(correct_grid_dim_list)
    token_acc_list = gather_object(token_acc_list)
    ttt_provided_list = gather_object(ttt_provided_list)

    assert len(task_id_and_text_list) == len(dataset), (len(task_id_and_text_list), len(dataset))
    assert len(ce_loss_list) == len(dataset), (len(ce_loss_list), len(dataset))
    assert len(encoder_loss_list) == len(dataset), (len(encoder_loss_list), len(dataset))
    assert len(kl_loss_list) == len(dataset), (len(kl_loss_list), len(dataset))
    assert len(exact_acc_list) == len(dataset), (len(exact_acc_list), len(dataset))
    assert len(valid_grid_list) == len(dataset), (len(valid_grid_list), len(dataset))
    assert len(correct_grid_dim_list) == len(dataset), (len(correct_grid_dim_list), len(dataset))
    assert len(token_acc_list) == len(dataset), (len(token_acc_list), len(dataset))
    assert len(ttt_provided_list) == len(dataset), (len(ttt_provided_list), len(dataset))

    # average metrics
    # note these are all computed without accounting for skipped eval grids
    avg_ce_loss = sum(ce_loss_list) / len(dataset)
    avg_encoder_loss = sum(encoder_loss_list) / len(dataset)
    avg_kl_loss = sum(kl_loss_list) / len(dataset)
    exact_acc = sum(exact_acc_list) / len(dataset)
    valid_grid = sum(valid_grid_list) / len(dataset)
    correct_grid_dim = sum(correct_grid_dim_list) / len(dataset)
    token_acc = sum(token_acc_list) / len(dataset)
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

    return avg_ce_loss, avg_encoder_loss, avg_kl_loss, \
        exact_acc, valid_grid, correct_grid_dim, token_acc, task_id_to_texts, \
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


def text_to_2d_grid(text: str, no_compact_grids: bool) -> Tuple[Optional[List[List[int]]], bool]:
    try:
        grid_lines = text.split('\n')
        height, width = int(grid_lines[0]), int(grid_lines[1])
        assert height > 0 and width > 0
        grid = []
        row_lens = []
        for l in grid_lines[2:]:
            if not no_compact_grids:
                row = [int(x) for x in l]
            else:
                row = [int(x) for x in l.split(' ')]
            grid.append(row)
            row_lens.append(len(row))
            assert all(0 <= x and x < 10 for x in row)
        assert len(set(row_lens)) == 1
        assert len(grid) == height and len(grid[0]) == width
        return grid, True
    except:
        return None, False


class EMASpikeDetector:
    def __init__(self, spike_multiplier: float, momentum: float = 0.9):
        self.moving_average = None
        self.momentum = momentum
        self.spike_multiplier = spike_multiplier

    def update(self, new_val: float) -> bool:
        if self.moving_average == None:
            self.moving_average = new_val
            return False
        is_spike = (new_val > self.moving_average * self.spike_multiplier)
        self.moving_average = self.moving_average * self.momentum + new_val * (1.0 - self.momentum)
        return is_spike


def compute_grad_norm2(parameters) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2).item() ** 2
            total_norm += param_norm
    return total_norm ** 0.5


@torch.no_grad()
def save_train_model(
        encoder_model: nn.Module,
        decoder_model: nn.Module,
        conditioning_projection: Union[Prefix2PrefixProjection, Hidden2PromptProjection],
        output_dir: str,
        epoch: int,
        tie_models: bool,
    ) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str], str]:

    # encoder
    save_enc_path = os.path.join(output_dir, f"encoder_lora_epoch_{epoch+1}")
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    encoder_module.save_pretrained(save_enc_path, save_embedding_layers=False)
    logger.info(f"Saved encoder to {save_enc_path}")

    enc_lmhead_path = os.path.join(output_dir, f"encoder_lmhead_epoch_{epoch+1}.pt")
    enc_embeds_path = os.path.join(output_dir, f"encoder_embeds_epoch_{epoch+1}.pt")
    torch.save(encoder_module.lm_head.state_dict(), enc_lmhead_path)
    if hasattr(encoder_module.model, "embed_tokens"):
        embeds = encoder_module.model.embed_tokens
    else:
        embeds = encoder_module.model.model.embed_tokens
    torch.save(embeds.state_dict(), enc_embeds_path)
    logger.info(f"Saved encoder lmhead to {enc_lmhead_path}")
    logger.info(f"Saved encoder embeds to {enc_embeds_path}")

    # decoder
    save_dec_path = None
    dec_lmhead_path = None
    dec_embeds_path = None
    if not tie_models:
        save_dec_path = os.path.join(output_dir, f"decoder_lora_epoch_{epoch+1}")
        decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
        decoder_module.save_pretrained(save_dec_path, save_embedding_layers=False)
        logger.info(f"Saved decoder to {save_dec_path}")

        dec_lmhead_path = os.path.join(output_dir, f"decoder_lmhead_epoch_{epoch+1}.pt")
        dec_embeds_path = os.path.join(output_dir, f"decoder_embeds_epoch_{epoch+1}.pt")
        torch.save(decoder_module.lm_head.state_dict(), dec_lmhead_path)
        if hasattr(decoder_module.model, "embed_tokens"):
            embeds = decoder_module.model.embed_tokens
        else:
            embeds = decoder_module.model.model.embed_tokens
        torch.save(embeds.state_dict(), dec_embeds_path)
        logger.info(f"Saved decoder lmhead to {dec_lmhead_path}")
        logger.info(f"Saved decoder embeds to {dec_embeds_path}")

    # projection
    save_proj_path = os.path.join(output_dir, f"conditioning_projection_epoch_{epoch+1}.pt")
    conditioning_projection_module = conditioning_projection
    if isinstance(conditioning_projection, DistributedDataParallel):
        conditioning_projection_module = conditioning_projection.module
    torch.save(conditioning_projection_module, save_proj_path)
    logger.info(f"Saved conditioning projection to {save_proj_path}")

    return save_enc_path, enc_lmhead_path, enc_embeds_path, save_dec_path, dec_lmhead_path, dec_embeds_path, save_proj_path


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
    parser.add_argument("--tracker_project_name", type=str, default="arc")
    parser.add_argument("--no_log_spikes", action="store_true")
    parser.add_argument("--spike_multiplier", type=float, default=10.0)
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_enc_len", type=int, default=-1)
    parser.add_argument("--debug_dec_len", type=int, default=-1)
    parser.add_argument("--debug_fixed_train_order", action="store_true")
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")

    # Conditioning projection
    parser.add_argument("--conditioning_method", type=str, choices=["prefix2prefix", "hidden2prompt"], default="hidden2prompt")
    parser.add_argument("--projection_type", type=str, choices=["none", "shared", "full"], default="none")
    parser.add_argument("--identity_init", action="store_true")

    # vae
    # can try different projections, but just linear for now because llama already norms it
    parser.add_argument("--vae", action="store_true")
    parser.add_argument("--debug_vae_no_kl", action="store_true")
    parser.add_argument("--debug_vae_train_no_sample", action="store_true")
    parser.add_argument("--debug_vae_eval_no_sample", action="store_true")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=5120)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.0)
    parser.add_argument("--anti_invar_ratio", type=float, default=0.0)
    parser.add_argument("--encoder_loss_lambda", type=float, default=1.0)
    parser.add_argument("--encoder_loss_type", type=str, choices=["last", "rest", "all"], default="rest")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # both data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_compact_grids", action="store_true")
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

    # train data
    parser.add_argument("--train_data_dir", type=str, default="/scratch/yl11330/re-arc/train_data/tasks")
    parser.add_argument("--min_prefix", type=int, default=2)
    parser.add_argument("--max_prefix", type=int, default=7)
    parser.add_argument("--augment_ratio", type=float, default=0.0)
    parser.add_argument("--augment_single_grid", action="store_true")

    # eval train data
    parser.add_argument("--eval_train_dir", type=str, default="/scratch/yl11330/re-arc/arc_original/training")
    parser.add_argument("--eval_train_select_tasks_path", type=str, default=None)
    parser.add_argument("--eval_train_leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--eval_train_leave_ns_inc", action="store_true")
    parser.add_argument("--eval_train_permute_n", type=int, default=0)
    parser.add_argument("--eval_train_augment_n", type=int, default=0)

    # eval eval data (mirror eval train data)
    parser.add_argument("--eval_eval_dir", type=str, default="/scratch/yl11330/re-arc/arc_original/evaluation")
    parser.add_argument("--eval_eval_select_tasks_path", type=str, default=None)
    parser.add_argument("--eval_eval_leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--eval_eval_leave_ns_inc", action="store_true")
    parser.add_argument("--eval_eval_permute_n", type=int, default=0)
    parser.add_argument("--eval_eval_augment_n", type=int, default=0)

    # gradient search
    parser.add_argument("--gs_iters", type=int, default=0)
    parser.add_argument("--gs_lr", type=float, default=1.0)
    parser.add_argument("--gs_beta1", type=float, default=0.9)
    parser.add_argument("--gs_beta2", type=float, default=0.9)
    parser.add_argument("--gs_batch_size", type=int, default=2)
    parser.add_argument("--gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--gs_max_grad_norm", default=1e8, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_take_best", action="store_true")

    # Lora encoder
    parser.add_argument("--encoder_lora_rank", type=int, default=256)
    parser.add_argument("--encoder_lora_alpha", type=float, default=24.0)
    parser.add_argument("--encoder_lora_dropout", type=float, default=0.0)
    parser.add_argument('--encoder_lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'
    ])
    parser.add_argument("--encoder_no_rslora", action='store_true')

    # Lora decoder
    parser.add_argument("--decoder_lora_rank", type=int, default=256)
    parser.add_argument("--decoder_lora_alpha", type=float, default=24.0)
    parser.add_argument("--decoder_lora_dropout", type=float, default=0.0)
    parser.add_argument('--decoder_lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'
    ])
    parser.add_argument("--decoder_no_rslora", action='store_true')

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # override to debug stuff
    if args.debug:
        args.tag = 'test'
        args.wandb = False
        args.train_data_dir = '/scratch/yl11330/re-arc/train_data_debug/tasks'
        args.eval_train_dir = '/scratch/yl11330/re-arc/arc_original_debug/training'
        args.eval_eval_dir = '/scratch/yl11330/re-arc/arc_original_debug/evaluation'
        args.num_epochs = 3
        args.samples_per_epoch = 250
        args.eval_epochs = 1
        args.num_workers = 0

    # check args
    if args.conditioning_method == "prefix2prefix":
        assert not args.encoder_gradient_checkpointing
        assert not args.decoder_gradient_checkpointing
        assert args.decoder_pad_side == "right" # right for no middle padding
    if args.identity_init:
        assert args.encoder_name == args.decoder_name
    if args.projection_type == "none":
        assert args.encoder_name == args.decoder_name
    if args.vae:
        assert args.projection_type is not "none"
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
        assert args.encoder_gradient_checkpointing == args.decoder_gradient_checkpointing
    if args.debug_enc_len > -1 and args.debug_dec_len == -1:
        args.debug_dec_len = args.debug_enc_len // 2
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable
    if args.gs_iters > 0:
        assert args.eval_batch_size == 1

    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
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

    # setup spike detector
    gradnorm_spike_detector = None
    loss_spike_detector = None
    spike_gradnorm_samples_path = None
    spike_loss_samples_path = None
    if not args.no_log_spikes and accelerator.is_main_process:
        spike_gradnorm_samples_path = os.path.join(args.output_dir, "spike_gradnorm_samples.jsonl")
        spike_loss_samples_path = os.path.join(args.output_dir, "spike_loss_samples.jsonl")
        if os.path.exists(spike_gradnorm_samples_path): os.remove(spike_gradnorm_samples_path)
        if os.path.exists(spike_loss_samples_path): os.remove(spike_loss_samples_path)
        with open(spike_gradnorm_samples_path, 'w') as f: pass
        with open(spike_loss_samples_path, 'w') as f: pass
        gradnorm_spike_detector = EMASpikeDetector(spike_multiplier=args.spike_multiplier)
        loss_spike_detector = EMASpikeDetector(spike_multiplier=args.spike_multiplier)
        logger.info(f'logging gradnorm spike to {spike_gradnorm_samples_path}')
        logger.info(f'logging loss spike to {spike_loss_samples_path}')

    # Load tokenizers
    encoder_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.encoder_name], cache_dir='./encoder_decoder_cache')
    assert isinstance(encoder_tokenizer, PreTrainedTokenizerFast)
    if args.tie_models:
        decoder_tokenizer = encoder_tokenizer
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.decoder_name], cache_dir='./encoder_decoder_cache')
        assert isinstance(encoder_tokenizer, PreTrainedTokenizerFast)
    assert (encoder_tokenizer.pad_token is None) and (decoder_tokenizer.pad_token is None)
    assert isinstance(encoder_tokenizer.bos_token, str) and isinstance(decoder_tokenizer.bos_token, str)
    assert isinstance(encoder_tokenizer.eos_token, str) and isinstance(decoder_tokenizer.eos_token, str)
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
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"unrecognized untrainable_nbit {args.untrainable_nbit}")

    base_encoder = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.encoder_name],
        **from_pretrained_kwargs,
    )
    if args.tie_models:
        base_decoder = base_encoder
    else:
        base_decoder = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.decoder_name],
            **from_pretrained_kwargs,
        )

    if args.untrainable_nbit in ['4bit', '8bit']:
        base_encoder = prepare_model_for_kbit_training(base_encoder, use_gradient_checkpointing=args.encoder_gradient_checkpointing)
        if not args.tie_models:
            base_decoder = prepare_model_for_kbit_training(base_decoder, use_gradient_checkpointing=args.decoder_gradient_checkpointing)
    else:
        if args.encoder_gradient_checkpointing:
            base_encoder.gradient_checkpointing_enable()
        if args.decoder_gradient_checkpointing and not args.tie_models:
            base_decoder.gradient_checkpointing_enable()

    logger.info("Base models loaded.")

    # add new CLS tokens to encoder for program encoding
    cls_tokens = [f"<CLS{token_i}>" for token_i in range(args.ntokens)]
    encoder_tokenizer.add_tokens(cls_tokens) # type: ignore
    base_encoder.resize_token_embeddings(len(encoder_tokenizer))
    logger.info("CLS tokens added.")

    # only keep these tokens, resize model embedding (eos == pad)
    encoder_keep_tokens = [str(i) for i in range(10)] + \
        [encoder_tokenizer.bos_token, encoder_tokenizer.eos_token, "\n", ":\n", "input", "output"] + \
        cls_tokens
    decoder_keep_tokens = [str(i) for i in range(10)] + \
        [decoder_tokenizer.bos_token, decoder_tokenizer.eos_token, "\n", ":\n", "input", "output"]
    if args.tie_models:
        decoder_keep_tokens = encoder_keep_tokens
    assert len(set(encoder_keep_tokens)) == len(encoder_keep_tokens)
    assert len(set(decoder_keep_tokens)) == len(decoder_keep_tokens)

    # this breaks embedding tying, but whatever
    with torch.no_grad():
        encoder_keep_token_ids = []
        for token in encoder_keep_tokens:
            token_id = encoder_tokenizer(token)["input_ids"] # type: ignore
            assert isinstance(token_id, list) and len(token_id) == 2 # with start token
            encoder_keep_token_ids.append(token_id[1])
        decoder_keep_token_ids = []
        for token in decoder_keep_tokens:
            token_id = decoder_tokenizer(token)["input_ids"] # type: ignore
            assert isinstance(token_id, list) and len(token_id) == 2 # with start token
            decoder_keep_token_ids.append(token_id[1])
        assert len(set(encoder_keep_token_ids)) == len(encoder_keep_token_ids)
        assert len(set(decoder_keep_token_ids)) == len(decoder_keep_token_ids)

        # subset embeddings and lmheads
        assert base_encoder.model.embed_tokens.weight.shape == base_encoder.lm_head.weight.shape
        base_encoder.model.embed_tokens.weight = nn.Parameter(base_encoder.model.embed_tokens.weight[encoder_keep_token_ids])
        base_encoder.model.embed_tokens.num_embeddings = len(encoder_keep_token_ids)
        assert base_encoder.lm_head.bias is None
        base_encoder.lm_head.weight = nn.Parameter(base_encoder.lm_head.weight[encoder_keep_token_ids])
        base_encoder.lm_head.out_features = len(encoder_keep_token_ids)
        base_encoder.config.tie_word_embeddings = False

        # subset embeddings and lmheads
        if not args.tie_models:
            assert base_decoder.model.embed_tokens.weight.shape == base_decoder.lm_head.weight.shape
            base_decoder.model.embed_tokens.weight = nn.Parameter(base_decoder.model.embed_tokens.weight[decoder_keep_token_ids])
            base_decoder.model.embed_tokens.num_embeddings = len(decoder_keep_token_ids)
            assert base_decoder.lm_head.bias is None
            base_decoder.lm_head.weight = nn.Parameter(base_decoder.lm_head.weight[decoder_keep_token_ids])
            base_decoder.lm_head.out_features = len(decoder_keep_token_ids)
            base_decoder.config.tie_word_embeddings = False

    # update configs
    assert base_encoder.config.vocab_size and base_encoder.config.bos_token_id and base_encoder.config.eos_token_id
    base_encoder.config.vocab_size = len(encoder_keep_token_ids)
    base_encoder.config.bos_token_id = encoder_keep_tokens.index(encoder_tokenizer.bos_token)
    base_encoder.config.eos_token_id = encoder_keep_tokens.index(encoder_tokenizer.eos_token)
    if not args.tie_models:
        assert base_decoder.config.vocab_size and base_decoder.config.bos_token_id and base_decoder.config.eos_token_id
        base_decoder.config.vocab_size = len(decoder_keep_token_ids)
        base_decoder.config.bos_token_id = decoder_keep_tokens.index(decoder_tokenizer.bos_token)
        base_decoder.config.eos_token_id = decoder_keep_tokens.index(decoder_tokenizer.eos_token)

    # create custom tokenizer
    arc_encoder_tokenizer = ARCTokenizer(
        tokens=encoder_keep_tokens,
        bos_token=encoder_tokenizer.bos_token,
        eos_token=encoder_tokenizer.eos_token,
    )
    arc_decoder_tokenizer = arc_encoder_tokenizer
    if not args.tie_models:
        arc_decoder_tokenizer = ARCTokenizer(
            tokens=decoder_keep_tokens,
            bos_token=decoder_tokenizer.bos_token,
            eos_token=decoder_tokenizer.eos_token,
        )
    del encoder_tokenizer, decoder_tokenizer
    encoder_tokenizer, decoder_tokenizer = arc_encoder_tokenizer, arc_decoder_tokenizer

    # LoRA
    encoder_model = None
    if args.no_lora:
        encoder_model = base_encoder
    else:
        encoder_peft_config = LoraConfig(
            r=args.encoder_lora_rank,
            lora_alpha=args.encoder_lora_alpha,
            lora_dropout=args.encoder_lora_dropout,
            target_modules=args.encoder_lora_target_modules,
            use_rslora=not args.encoder_no_rslora,
            task_type=TaskType.CAUSAL_LM,
        )
        encoder_model = get_peft_model(base_encoder, encoder_peft_config)

    decoder_model = None
    if args.tie_models:
        decoder_model = encoder_model
    else:
        if args.no_lora:
            decoder_model = base_decoder
        else:
            decoder_peft_config = LoraConfig(
                r=args.decoder_lora_rank,
                lora_alpha=args.decoder_lora_alpha,
                lora_dropout=args.decoder_lora_dropout,
                target_modules=args.decoder_lora_target_modules,
                use_rslora=not args.decoder_no_rslora,
                task_type=TaskType.CAUSAL_LM,
            )
            decoder_model = get_peft_model(base_decoder, decoder_peft_config)

    logger.info("LoRA-wrapped models initialized (optional)")

    if args.conditioning_method == "prefix2prefix":
        conditioning_projection = Prefix2PrefixProjection(
            ntokens=args.ntokens,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            identity_init=args.identity_init,
            projection_type=args.projection_type,
            vae=args.vae,
            device=accelerator.device,
        )
    elif args.conditioning_method == "hidden2prompt":
        conditioning_projection = Hidden2PromptProjection(
            ntokens=args.ntokens,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            identity_init=args.identity_init,
            projection_type=args.projection_type,
            vae=args.vae,
            device=accelerator.device,
        )
    else:
        raise ValueError(f"invalid conditioning method {args.conditioning_method}")
    logger.info("conditioning projection initialized (optional)")

    # encoder tune embed and lmhead
    if args.no_lora:
        encoder_model.model.embed_tokens.requires_grad_(True)
    else:
        encoder_model.model.model.embed_tokens.requires_grad_(True)
    encoder_model.lm_head.requires_grad_(True)
    # decoder tune embed and lmhead
    if not args.tie_models:
        if args.no_lora:
            decoder_model.model.embed_tokens.requires_grad_(True)
        else:
            decoder_model.model.model.embed_tokens.requires_grad_(True)
        decoder_model.lm_head.requires_grad_(True)

    # convert model weights
    for name, param in encoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for name, param in conditioning_projection.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # number of parameters
    print_trainable_parameters(encoder_model)
    if not args.tie_models:
        print_trainable_parameters(decoder_model)
    proj_n_params = sum(p.numel() for p in conditioning_projection.parameters())
    logger.info(f'conditioning projection params {three_commas(proj_n_params)}')

    # model size
    logger.info(f'encoder size {round(encoder_model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    if not args.tie_models:
        logger.info(f'decoder size {round(decoder_model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    logger.info(f'conditioning projection size {round(conditioning_projection.get_memory_footprint() / 1024 ** 3, 2)}GB')

    # Build training dataset
    train_tasks_dict = load_tasks_from_data_dir(args.train_data_dir)
    train_dataset = TrainDataset(
        tasks_dict=train_tasks_dict,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        total_steps=args.samples_per_epoch,
        min_prefix=args.min_prefix,
        max_prefix=args.max_prefix,
        max_seq_len=args.max_seq_len,
        augment_ratio=args.augment_ratio,
        augment_single_grid=args.augment_single_grid,
        seed=args.seed,
        no_compact_grids=args.no_compact_grids,
        ntokens=args.ntokens,
        debug_fixed_train_order=args.debug_fixed_train_order,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side=args.decoder_pad_side,
        encoder_loss_type=args.encoder_loss_type,
        anti_invar_ratio=args.anti_invar_ratio,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
    if args.debug_enc_len > 0:
        train_collate_fn = partial(
            collate_fn_train_dummy,
            ntokens=args.ntokens,
            debug_enc_len=args.debug_enc_len,
            debug_dec_len=args.debug_dec_len,
        )
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
    for name, param in encoder_model.named_parameters():
        if param.requires_grad:
            if "embed" in name or "lm_head" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if param.requires_grad:
                if "embed" in name or "lm_head" in name:
                    embedding_params.append(param)
                else:
                    other_params.append(param)
    for param in conditioning_projection.parameters():
        other_params.append(param)

    optimizer_grouped_params = [
        {"params": embedding_params, "lr": args.lr_embedding},
        {"params": other_params, "lr": args.lr_other},
    ]
    all_params = embedding_params + other_params
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay) # type: ignore
    elif args.optimizer == 'adamw8bit':
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_params, weight_decay=args.weight_decay)
        # optimizer = bnb.optim.PagedAdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_params) # type: ignore
    logger.info(f"Optimizer with {len(embedding_params)} embed-params lr={args.lr_embedding}, {len(other_params)} other-params lr={args.lr_other}")

    # LR schedule
    steps_per_epoch = args.samples_per_epoch // (args.train_batch_size * args.grad_accum_steps * accelerator.num_processes)
    num_training_steps = steps_per_epoch * args.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps_per_epoch * args.grad_accum_steps,  # 1 epoch warmup
        num_training_steps=num_training_steps * args.grad_accum_steps
    )
    logger.info(f'lr scheduler with {steps_per_epoch} warmup steps')

    # Prepare with accelerator
    (
        encoder_model,
        decoder_model,
        conditioning_projection,
        optimizer,
        train_loader,
    ) = accelerator.prepare(
        encoder_model,
        decoder_model,
        conditioning_projection,
        optimizer,
        train_loader,
    )

    # sanity check that if weight tying is used, these should be same data (they are de-tied now)
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
    if encoder_module.config.tie_word_embeddings:
        assert torch.equal(encoder_module.model.model.embed_tokens.weight.data, encoder_module.lm_head.weight.data)
    if decoder_module.config.tie_word_embeddings:
        assert torch.equal(decoder_module.model.model.embed_tokens.weight.data, decoder_module.lm_head.weight.data)
    # sanity check that no more weight tying
    assert encoder_module.model.model.embed_tokens.weight.data_ptr() != encoder_module.lm_head.weight.data_ptr()
    assert decoder_module.model.model.embed_tokens.weight.data_ptr() != decoder_module.lm_head.weight.data_ptr()

    logger.info(f'======= TRAINING INFO START ======')
    logger.info(f'num_epochs={args.num_epochs}')
    logger.info(f'train_batch_size={args.train_batch_size}')
    logger.info(f'grad_accum_steps={args.grad_accum_steps}')
    logger.info(f'accelerator.num_processes={accelerator.num_processes}')
    logger.info(f'steps_per_epoch={steps_per_epoch}')
    logger.info(f'{three_commas(sum(p.numel() for p in all_params))} trainable params')
    logger.info(f'======= TRAINING INFO END ======\n')

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

    # when these conditions are met, DDP requires setting static graph due to reusing parameters
    if args.tie_models and args.encoder_gradient_checkpointing and accelerator.num_processes > 1:
        assert args.gs_iters == 0 # gradient search changes static graph? idk havent tried
        # set to train
        encoder_model.train()
        decoder_model.train()
        conditioning_projection.train()
        # set static graph
        encoder_model._set_static_graph()
        decoder_model._set_static_graph()
        conditioning_projection._set_static_graph()
        # get any data, single forward and backward pass
        batch_data = next(iter(train_loader))
        with accelerator.autocast():
            _, _, _, _, total_loss, _ = encoder_decoder_loss(
                encoder_model=encoder_model,
                decoder_model=decoder_model,
                conditioning_method=args.conditioning_method,
                conditioning_projection=conditioning_projection,
                encoder_input_ids=batch_data["encoder_input_ids"].to(accelerator.device),
                encoder_attention_mask=batch_data["encoder_attention_mask"].to(accelerator.device),
                encoder_labels=batch_data["encoder_labels"].to(accelerator.device),
                decoder_input_ids=batch_data["decoder_input_ids"].to(accelerator.device),
                decoder_attention_mask=batch_data["decoder_attention_mask"].to(accelerator.device),
                decoder_labels=batch_data["decoder_labels"].to(accelerator.device),
                enc_ids_lens=batch_data["encoder_input_ids_lens"],
                dec_ids_lens=batch_data["decoder_input_ids_lens"],
                anti_invars=batch_data["anti_invars"],
                ntokens=args.ntokens,
                encoder_loss_lambda=args.encoder_loss_lambda,
                invar_loss_lambda=args.invar_loss_lambda,
                no_lora=args.no_lora,
                decoder_ce_loss=True, # HARDCODE
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                debug_vae_no_sample=args.debug_vae_train_no_sample,
                debug_vae_no_kl=args.debug_vae_no_kl,
            )
        accelerator.backward(total_loss)
        optimizer.zero_grad()
        logger.info(f'To avoid DDP error, static graph is applied after doing one iteration of forward backward')

    # train!
    for epoch in range(args.num_epochs):
        encoder_model.train()
        decoder_model.train()
        conditioning_projection.train()

        ce_loss_accum = 0.0
        invar_loss_accum = 0.0
        encoder_loss_accum = 0.0
        kl_loss_accum = 0.0
        total_loss_accum = 0.0
        grad_norm_accum = 0.0

        for batch_data in train_loader:
            enc_ids = batch_data["encoder_input_ids"].to(accelerator.device)
            enc_mask = batch_data["encoder_attention_mask"].to(accelerator.device)
            enc_labels = batch_data["encoder_labels"].to(accelerator.device)
            dec_ids = batch_data["decoder_input_ids"].to(accelerator.device)
            dec_mask = batch_data["decoder_attention_mask"].to(accelerator.device)
            dec_labels = batch_data["decoder_labels"].to(accelerator.device)
            enc_ids_lens = batch_data["encoder_input_ids_lens"]
            dec_ids_lens = batch_data["decoder_input_ids_lens"]
            anti_invars = batch_data["anti_invars"]

            with accelerator.accumulate(encoder_model, decoder_model, conditioning_projection):
                with accelerator.autocast():
                    ce_loss, invar_loss, encoder_loss, kl_loss, total_loss, _ = encoder_decoder_loss(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        conditioning_method=args.conditioning_method,
                        conditioning_projection=conditioning_projection,
                        encoder_input_ids=enc_ids,
                        encoder_attention_mask=enc_mask,
                        encoder_labels=enc_labels,
                        decoder_input_ids=dec_ids,
                        decoder_attention_mask=dec_mask,
                        decoder_labels=dec_labels,
                        enc_ids_lens=enc_ids_lens,
                        dec_ids_lens=dec_ids_lens,
                        anti_invars=anti_invars,
                        ntokens=args.ntokens,
                        encoder_loss_lambda=args.encoder_loss_lambda,
                        invar_loss_lambda=args.invar_loss_lambda,
                        no_lora=args.no_lora,
                        decoder_ce_loss=True, # HARDCODE
                        encoder_pad_side=args.encoder_pad_side,
                        decoder_pad_side=args.decoder_pad_side,
                        trainable_nbit=args.trainable_nbit,
                        no_flash_attn=args.no_flash_attn,
                        debug_vae_no_sample=args.debug_vae_train_no_sample,
                        debug_vae_no_kl=args.debug_vae_no_kl,
                    )

                # just accumulate for logging
                avg_ce_loss = accelerator.gather(ce_loss.repeat(args.train_batch_size)).mean() # type: ignore
                avg_invar_loss = accelerator.gather(invar_loss.repeat(args.train_batch_size)).mean() # type: ignore
                avg_encoder_loss = accelerator.gather(encoder_loss.repeat(args.train_batch_size)).mean() # type: ignore
                avg_kl_loss = accelerator.gather(kl_loss.repeat(args.train_batch_size)).mean() # type: ignore
                avg_total_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean() # type: ignore
                ce_loss_accum += avg_ce_loss.item() / args.grad_accum_steps
                invar_loss_accum += avg_invar_loss.item() / args.grad_accum_steps
                encoder_loss_accum += avg_encoder_loss.item() / args.grad_accum_steps
                kl_loss_accum += avg_kl_loss.item() / args.grad_accum_steps
                total_loss_accum += avg_total_loss.item() / args.grad_accum_steps

                accelerator.backward(total_loss)

                # sanity check so
                if global_step == 0:
                    for name, param in encoder_model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            raise RuntimeError(f"Parameter '{name}' requires a gradient but received none!")
                    for name, param in decoder_model.named_parameters():
                        if param.requires_grad and param.grad is None:
                            raise RuntimeError(f"Parameter '{name}' requires a gradient but received none!")
                    for name, param in conditioning_projection.named_parameters():
                        if param.requires_grad and param.grad is None:
                            raise RuntimeError(f"Parameter '{name}' requires a gradient but received none!")

                if accelerator.sync_gradients:
                    grad_norm_accum += accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item() # type: ignore
                optimizer.step()
                lr_scheduler.step()

                # detect and log spike
                if not args.no_log_spikes and accelerator.is_main_process:
                    assert gradnorm_spike_detector is not None
                    assert loss_spike_detector is not None
                    assert isinstance(spike_gradnorm_samples_path, str)
                    assert isinstance(spike_loss_samples_path, str)
                    is_gradnorm_spike = gradnorm_spike_detector.update(compute_grad_norm2(all_params))
                    is_loss_spike = loss_spike_detector.update(total_loss.item())
                    enc_texts = encoder_tokenizer.batch_decode(enc_ids, skip_special_tokens=True)
                    dec_texts = decoder_tokenizer.batch_decode(dec_ids, skip_special_tokens=True)
                    data_string = json.dumps({"enc": enc_texts, "dec": dec_texts})
                    if is_gradnorm_spike:
                        with open(spike_gradnorm_samples_path, "a") as f:
                            f.write(f"{global_step} {data_string}\n")
                    if is_loss_spike:
                        with open(spike_loss_samples_path, "a") as f:
                            f.write(f"{global_step} {data_string}\n")

                optimizer.zero_grad()


            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    progress_bar.update(args.log_every)
                    try:
                        accelerator.log({
                            "train/ce_loss": ce_loss_accum,
                            "train/invar_loss": invar_loss_accum,
                            "train/encoder_loss": encoder_loss_accum,
                            "train/kl_loss": kl_loss_accum,
                            "train/total_loss": total_loss_accum,
                            "train/grad_norm_accum": grad_norm_accum,
                            "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                            "train/lr_other": lr_scheduler.get_last_lr()[1],
                        }, step=global_step)
                    except:
                        logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                ce_loss_accum = 0.0
                invar_loss_accum = 0.0
                encoder_loss_accum = 0.0
                kl_loss_accum = 0.0
                total_loss_accum = 0.0
                grad_norm_accum = 0.0

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
            # Build evaluation datasets
            eval_train_dataset = EvalDataset(
                args.eval_train_dir,
                select_tasks_path=args.eval_train_select_tasks_path,
                leave_ns=args.eval_train_leave_ns,
                leave_ns_inc=args.eval_train_leave_ns_inc,
                permute_n=args.eval_train_permute_n,
                augment_n=args.eval_train_augment_n,
                permute_iters=0, # HARDCODE
                seed=args.seed,
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                max_seq_len=args.max_seq_len,
                no_compact_grids=args.no_compact_grids,
                ntokens=args.ntokens,
                encoder_loss_type=args.encoder_loss_type,
                debug_random_pad=args.debug_random_pad,
                debug_pad_len=args.debug_pad_len,
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
            )
            eval_eval_dataset = EvalDataset(
                eval_dir=args.eval_eval_dir,
                select_tasks_path=args.eval_eval_select_tasks_path,
                leave_ns=args.eval_eval_leave_ns,
                leave_ns_inc=args.eval_eval_leave_ns_inc,
                permute_n=args.eval_eval_permute_n,
                augment_n=args.eval_eval_augment_n,
                permute_iters=0, # HARDCODE
                seed=args.seed,
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                max_seq_len=args.max_seq_len,
                no_compact_grids=args.no_compact_grids,
                ntokens=args.ntokens,
                encoder_loss_type=args.encoder_loss_type,
                debug_random_pad=args.debug_random_pad,
                debug_pad_len=args.debug_pad_len,
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
            )
            eval_collate_fn = partial(collate_fn_eval, dataset=eval_train_dataset) # only use tokenizer, padding info
            if args.debug_enc_len > 0:
                eval_collate_fn = partial(
                    collate_fn_eval_dummy,
                    ntokens=args.ntokens,
                    debug_enc_len=args.debug_enc_len,
                    debug_dec_len=args.debug_dec_len,
                )

            train_ce_loss, train_encoder_loss, train_kl_loss, \
                train_exact_acc, train_valid_grid, train_correct_grid_dim, train_token_acc, train_texts, \
                train_votes, train_competition_sub_acc, train_competition_all_acc, _ = evaluate(
                task_to_ttt_model_paths=None, # HARDCODE
                encoder_ttt_param_names=None, # HARDCODE
                decoder_ttt_param_names=None, # HARDCODE
                encoder_model=encoder_model,
                decoder_model=decoder_model,
                conditioning_method=args.conditioning_method,
                conditioning_projection=conditioning_projection,
                dataset=eval_train_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                no_lora=args.no_lora,
                decoder_ce_loss=True, # HARDCODE
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                tie_models=args.tie_models,
                output_dir=args.output_dir,
                gs_iters=args.gs_iters,
                gs_lr=args.gs_lr,
                gs_beta1=args.gs_beta1,
                gs_beta2=args.gs_beta2,
                gs_batch_size=args.gs_batch_size,
                gs_optimizer=args.gs_optimizer,
                gs_max_grad_norm=args.gs_max_grad_norm,
                gs_lr_scheduler=args.gs_lr_scheduler,
                gs_take_best=args.gs_take_best,
                debug_vae_no_kl=args.debug_vae_no_kl,
                debug_vae_no_sample=args.debug_vae_eval_no_sample,
            )
            eval_ce_loss, eval_encoder_loss, eval_kl_loss, \
                eval_exact_acc, eval_valid_grid, eval_correct_grid_dim, eval_token_acc, eval_texts, \
                eval_votes, eval_competition_sub_acc, eval_competition_all_acc, _ = evaluate(
                task_to_ttt_model_paths=None, # HARDCODE
                encoder_ttt_param_names=None, # HARDCODE
                decoder_ttt_param_names=None, # HARDCODE
                encoder_model=encoder_model,
                decoder_model=decoder_model,
                conditioning_method=args.conditioning_method,
                conditioning_projection=conditioning_projection,
                dataset=eval_eval_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                no_lora=args.no_lora,
                decoder_ce_loss=True, # HARDCODE
                trainable_nbit=args.trainable_nbit,
                no_flash_attn=args.no_flash_attn,
                tie_models=args.tie_models,
                output_dir=args.output_dir,
                gs_iters=args.gs_iters,
                gs_lr=args.gs_lr,
                gs_beta1=args.gs_beta1,
                gs_beta2=args.gs_beta2,
                gs_batch_size=args.gs_batch_size,
                gs_optimizer=args.gs_optimizer,
                gs_max_grad_norm=args.gs_max_grad_norm,
                gs_lr_scheduler=args.gs_lr_scheduler,
                gs_take_best=args.gs_take_best,
                debug_vae_no_kl=args.debug_vae_no_kl,
                debug_vae_no_sample=args.debug_vae_eval_no_sample,
            )

            if accelerator.is_main_process:
                eval_metric_dict = {
                    "eval/train_ce_loss": train_ce_loss,
                    "eval/train_encoder_loss": train_encoder_loss,
                    "eval/train_kl_loss": train_kl_loss,
                    "eval/train_exact_acc": train_exact_acc,
                    "eval/train_valid_grid": train_valid_grid,
                    "eval/train_correct_grid_dim": train_correct_grid_dim,
                    "eval/train_token_acc": train_token_acc,
                    "eval/train_competition_all_acc": train_competition_all_acc,
                    "eval/train_competition_sub_acc": train_competition_sub_acc,
                    "eval/eval_ce_loss": eval_ce_loss,
                    "eval/eval_encoder_loss": eval_encoder_loss,
                    "eval/eval_kl_loss": eval_kl_loss,
                    "eval/eval_exact_acc": eval_exact_acc,
                    "eval/eval_valid_grid": eval_valid_grid,
                    "eval/eval_correct_grid_dim": eval_correct_grid_dim,
                    "eval/eval_token_acc": eval_token_acc,
                    "eval/eval_competition_all_acc": eval_competition_all_acc,
                    "eval/eval_competition_sub_acc": eval_competition_sub_acc,
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
                        save_enc_path, enc_lmhead_path, enc_embeds_path, save_dec_path, dec_lmhead_path, dec_embeds_path, save_proj_path = last_save_model_path
                        rm_cmd = f"rm -rf {save_enc_path} {enc_lmhead_path} {enc_embeds_path}"
                        if save_dec_path is not None:
                            rm_cmd += f" {save_dec_path} {dec_lmhead_path} {dec_embeds_path}"
                        if save_proj_path is not None:
                            rm_cmd += f" {save_proj_path}"
                        os.system(rm_cmd)
                    last_save_model_path = save_train_model(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        conditioning_projection=conditioning_projection,
                        output_dir=args.output_dir,
                        epoch=epoch,
                        tie_models=args.tie_models,
                    )
                epoch_to_eval_exact_acc[epoch] = eval_exact_acc

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
