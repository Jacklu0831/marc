from datetime import timedelta
from collections import Counter
import copy
import arclib # required
import numpy as np
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Dict, Optional, Set
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
)
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import logging
import datasets
import transformers
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

from data_utils import (
    load_tasks_from_data_dir,
    TrainDataset,
    EvalDataset,
    collate_fn_train,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # weird tokenizer issue
os.environ["NCCL_TIMEOUT"] = "14400" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "14400000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"

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


class Hidden2PrefixProjection(nn.Module):
    def __init__(
            self,
            num_virtual_tokens: int,
            encoder_model: nn.Module,
            decoder_model: nn.Module,
            conditioning_method: str,
        ):
        super(Hidden2PrefixProjection, self).__init__()
        # prefixes are formatted as 16 of (2=2, BS=1, nhead=8, nvirtualtoken=1, tokendim / nhead=64)
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = decoder_model.config.num_hidden_layers
        self.num_kv_heads = decoder_model.config.num_key_value_heads
        self.embed_size_per_head = decoder_model.config.hidden_size // decoder_model.config.num_attention_heads
        self.conditioning_method = conditioning_method
        # weights
        if self.conditioning_method == "hidden2prefix_shared":
            projections = nn.Linear(
                encoder_model.config.hidden_size,
                self.num_layers * 2 * self.num_kv_heads * self.embed_size_per_head
            )
            self.weights = nn.Parameter(projections.weight)
            self.biases = nn.Parameter(projections.bias)
        elif self.conditioning_method == "hidden2prefix_full":
            projections = nn.ModuleList([
                nn.Linear(
                    encoder_model.config.hidden_size,
                    self.num_layers * 2 * self.num_kv_heads * self.embed_size_per_head
                ) for _ in range(self.num_virtual_tokens)
            ])
            self.weights = nn.Parameter(torch.stack([projection.weight for projection in projections], dim=0))
            self.biases = nn.Parameter(torch.stack([projection.bias for projection in projections], dim=0))
        else:
            raise ValueError(f'unrecognized conditioning method {self.conditioning_method}')
        del projections

    def forward(self, enc_hidden_states: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # enc_hidden_states has shape (batch_size, num_virtual_tokens, hidden_dim)
        assert enc_hidden_states.shape[1] == self.num_virtual_tokens
        if self.conditioning_method == "hidden2prefix_shared":
            outs = torch.einsum("bnh,hd->bnd", enc_hidden_states, self.weights.transpose(0, 1)) + self.biases
        else:
            outs = torch.einsum("bnh,nhd->bnd", enc_hidden_states, self.weights.transpose(1, 2)) + self.biases # (batch_size, num_virtual_tokens, out_dim)
        outs = outs.view(outs.size(0), outs.size(1), self.num_layers, 2, self.num_kv_heads, self.embed_size_per_head)
        outs = outs.permute(2, 3, 0, 4, 1, 5)
        past_key_values = [(x[0], x[1]) for x in outs] # each (batch_size, num_kv_heads, num_virtual_tokens, embed_size_per_head)
        return past_key_values

    def get_memory_footprint(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters()) + \
            sum(p.nelement() * p.element_size() for p in self.buffers())


class Hidden2PromptProjection(nn.Module):
    def __init__(
            self,
            num_virtual_tokens: int,
            encoder_model: nn.Module,
            decoder_model: nn.Module,
            conditioning_method: str,
        ):
        super(Hidden2PromptProjection, self).__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.conditioning_method = conditioning_method
        self.encoder_hidden_size = encoder_model.config.hidden_size
        self.decoder_hidden_size = decoder_model.config.hidden_size
        # weights
        if "hidden2prompt_shared" in self.conditioning_method:
            projections = nn.Linear(encoder_model.config.hidden_size, decoder_model.config.hidden_size)
            if "identity" in self.conditioning_method:
                assert self.encoder_hidden_size == self.decoder_hidden_size
                with torch.no_grad():
                    projections.weight.data.copy_(torch.eye(encoder_model.config.hidden_size, dtype=projections.weight.dtype))
                    projections.bias.data.zero_()
            self.weights = nn.Parameter(projections.weight)
            self.biases = nn.Parameter(projections.bias)
        elif "hidden2prompt_full" in conditioning_method:
            projections = nn.ModuleList([
                nn.Linear(
                    encoder_model.config.hidden_size,
                    decoder_model.config.hidden_size,
                ) for _ in range(self.num_virtual_tokens)
            ])
            if "identity" in self.conditioning_method:
                assert self.encoder_hidden_size == self.decoder_hidden_size
                with torch.no_grad():
                    for projection in projections:
                        projection.weight.data.copy_(torch.eye(encoder_model.config.hidden_size, dtype=projection.weight.dtype))
                        projection.bias.data.zero_()
            self.weights = nn.Parameter(torch.stack([projection.weight for projection in projections], dim=0))
            self.biases = nn.Parameter(torch.stack([projection.bias for projection in projections], dim=0))
        else:
            raise ValueError(f'unrecognized conditioning method {self.conditioning_method}')
        del projections

    def forward(self, enc_hidden_states: torch.Tensor) -> torch.Tensor:
        # enc_hidden_states has shape (batch_size, num_virtual_tokens, hidden_dim)
        assert enc_hidden_states.shape[1] == self.num_virtual_tokens
        # outs (batch_size, num_virtual_tokens, out_dim)
        if self.conditioning_method in ["hidden2prompt_shared", "hidden2prompt_shared_identity"]:
            return torch.einsum("bnh,hd->bnd", enc_hidden_states, self.weights.transpose(0, 1)) + self.biases
        else:
            return torch.einsum("bnh,nhd->bnd", enc_hidden_states, self.weights.transpose(1, 2)) + self.biases

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
    conditioning_projection: nn.Module,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    encoder_labels: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    decoder_labels: torch.Tensor,
    enc_ids_lens: List[int],
    dec_ids_lens: List[int],
    num_virtual_tokens: int,
    invar_loss_lambda: float,
    encoder_loss_lambda: float,
    no_lora: bool,
    decoder_ce_loss: bool,
    encoder_pad_side: str,
    decoder_pad_side: str,
    trainable_nbit: int,
    flash_attn: bool,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
           Union[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
    # Encoder forward and get loss
    enc_out = encoder_model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        labels=encoder_labels,
        output_hidden_states=True,
    )
    encoder_loss = enc_out.loss

    # get predicted program for decoder and enc_hidden_states for invar loss
    enc_hidden_states = None
    predicted_program = None
    if conditioning_method == "prefix2prefix":
        predicted_program = []
        if encoder_pad_side == "right":
            for x1, x2 in enc_out.past_key_values:
                predicted_program.append((
                    torch.stack([x[:, l-num_virtual_tokens:l, :] for x, l in zip(x1, enc_ids_lens)]),
                    torch.stack([x[:, l-num_virtual_tokens:l, :] for x, l in zip(x2, enc_ids_lens)]),
                ))
        else:
            for x1, x2 in enc_out.past_key_values:
                predicted_program.append((
                    torch.stack([x[:, -num_virtual_tokens:, :] for x in x1]),
                    torch.stack([x[:, -num_virtual_tokens:, :] for x in x2]),
                ))
        enc_hidden_states = torch.stack([torch.stack(x) for x in predicted_program]) # each (batch_size, num_kv_heads, num_virtual_tokens, embed_size_per_head)
        enc_hidden_states = enc_hidden_states.permute(2, 0, 1, 3, 4, 5)
    elif conditioning_method in ["hidden2prefix_shared", "hidden2prefix_full"]:
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-num_virtual_tokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-num_virtual_tokens:] for x in enc_hidden_states])
        predicted_program = conditioning_projection(enc_hidden_states=enc_hidden_states)
    elif conditioning_method == "hidden2prompt":
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-num_virtual_tokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-num_virtual_tokens:] for x in enc_hidden_states])
        predicted_program = enc_hidden_states
    elif conditioning_method in ["hidden2prompt_shared", "hidden2prompt_full", "hidden2prompt_shared_identity", "hidden2prompt_full_identity"]:
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-num_virtual_tokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-num_virtual_tokens:] for x in enc_hidden_states])
        predicted_program = conditioning_projection(enc_hidden_states)
    else:
        raise ValueError(f"invalid conditioning method {conditioning_method}")

    if not decoder_ce_loss:
        ce_loss = torch.tensor(-1.0, device=encoder_input_ids.device)
        invar_loss = torch.tensor(0.0, device=encoder_input_ids.device)
        if enc_hidden_states.shape[0] % 2 == 0:
            invar_loss = nn.functional.mse_loss(enc_hidden_states[::2], enc_hidden_states[1::2])
        total_loss = ce_loss + invar_loss_lambda * invar_loss + encoder_loss_lambda * encoder_loss
        return ce_loss, invar_loss, encoder_loss, total_loss, predicted_program

    # decoder attention mask must be extended
    prefix_attention_mask = torch.full(
        (decoder_attention_mask.shape[0], num_virtual_tokens),
        1,
        device=decoder_attention_mask.device,
        dtype=decoder_attention_mask.dtype,
    )

    if conditioning_method in ["prefix2prefix", "hidden2prefix_shared", "hidden2prefix_full"]:
        # pad decoder attention mask
        decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)
        if flash_attn:
            predicted_program = tuple([
                tuple([x[0].to(NBIT_TO_DTYPE[trainable_nbit]), x[1].to(NBIT_TO_DTYPE[trainable_nbit])])
            for x in predicted_program])
        # decoder forward
        dec_out = decoder_model(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=predicted_program,
            labels=decoder_labels,
        )
    else:
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
            (decoder_labels.shape[0], num_virtual_tokens),
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

    # cross-entropy loss
    ce_loss = dec_out.loss

    # invariance loss (batch only not 2x in evaluation)
    invar_loss = torch.tensor(0.0, device=encoder_input_ids.device)
    if enc_hidden_states.shape[0] % 2 == 0:
        invar_loss = nn.functional.mse_loss(enc_hidden_states[::2], enc_hidden_states[1::2])

    total_loss = ce_loss + invar_loss_lambda * invar_loss + encoder_loss_lambda * encoder_loss
    return ce_loss, invar_loss, encoder_loss, total_loss, predicted_program


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunks_uniform_batch(task_ids, data_idxs, n):
    assert len(task_ids) == len(data_idxs)
    # group by first item in tuple (task_id)
    task_id_to_data_idx = defaultdict(list)
    for task_id, data_idx in zip(task_ids, data_idxs):
        task_id_to_data_idx[task_id].append(data_idx)
    # for each task_id, yield chunks of data idxs
    for task_id, data_idxs in task_id_to_data_idx.items():
        yield from chunks(data_idxs, n)


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    task_to_ttt_model_paths: Optional[Dict[str, Tuple[str, str, str]]],
    encoder_ttt_param_names: Optional[Set[str]],
    decoder_ttt_param_names: Optional[Set[str]],
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    conditioning_method: str,
    conditioning_projection: Union[Hidden2PrefixProjection, Hidden2PromptProjection],
    dataset: EvalDataset,
    accelerator: Accelerator,
    num_virtual_tokens: int,
    decoder_tokenizer,
    batch_size: int,
    collate_fn: Callable,
    compact_grids: bool,
    no_lora: bool,
    decoder_ce_loss: bool,
    trainable_nbit: int,
    flash_attn: bool,
    tie_models: bool,
    output_dir: str,
    encoder_pad_side: str,
    decoder_pad_side: str,
    decoder_gen_pad_side: str,
    search_iters: int,
):
    encoder_model.eval()
    if not tie_models:
        decoder_model.eval()
    if conditioning_projection is not None:
        conditioning_projection.eval()

    # get modules in case of DDP
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model

    # if ttt provided, same model weights for the missing ttt task weights
    cached_enc_weights_path = None
    cached_dec_weights_path = None
    cached_proj_weights_path = None
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
        if conditioning_projection is not None:
            cached_proj_weights_path = os.path.join(output_dir, f"process{accelerator.process_index}_conditioning_projection_cache.pt")
            torch.save(conditioning_projection, cached_proj_weights_path)
            logger.info(f"ttt provided, cached conditioning projection weights to {cached_proj_weights_path}")
        # save default to model paths and set current ttt weights to default
        task_to_ttt_model_paths["default"] = (cached_enc_weights_path, cached_dec_weights_path, cached_proj_weights_path)
        curr_ttt_task_name = "default"

    # setup terminators and suppress warning
    terminators = [
        decoder_tokenizer.eos_token_id,
        decoder_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    decoder_module.generation_config.pad_token_id = decoder_tokenizer.pad_token_id

    distributed_state = PartialState()
    task_id_and_text_list = []
    task_id_and_inverter_grids = []
    loss_list = []
    exact_acc_list = []
    valid_grid_list = []
    correct_grid_dim_list = []
    token_acc_list = []
    ttt_provided_list = []

    data_idxs = list(range(len(dataset)))
    with distributed_state.split_between_processes(data_idxs) as process_data_idxs:
        n_batches = math.ceil(len(process_data_idxs) / batch_size)

        # if ttt provided, make sure all batches are of the same task name
        if task_to_ttt_model_paths is not None:
            # tackle tasks in orderly fashion
            task_names = [dataset.eval_tasks[idx].name for idx in process_data_idxs]
            task_ids = [task_name.split('-')[0] for task_name in task_names]
            n_batches = len(list(chunks_uniform_batch(task_ids, process_data_idxs, batch_size)))
            data_idx_iterator = tqdm(chunks_uniform_batch(task_ids, process_data_idxs, batch_size), total=n_batches)
        else:
            data_idx_iterator = tqdm(chunks(process_data_idxs, batch_size), total=n_batches)

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
                    encoder_model_ttt_state_dict = torch.load(enc_ttt_path, weights_only=True, map_location=accelerator.device)
                    assert set(encoder_model_ttt_state_dict.keys()) == encoder_ttt_param_names
                    encoder_module.load_state_dict(encoder_model_ttt_state_dict, strict=False)
                    del encoder_model_ttt_state_dict
                    # load decoder
                    if dec_ttt_path is not None:
                        decoder_model_ttt_state_dict = torch.load(dec_ttt_path, weights_only=True, map_location=accelerator.device)
                        assert set(decoder_model_ttt_state_dict.keys()) == decoder_ttt_param_names
                        decoder_module.load_state_dict(decoder_model_ttt_state_dict, strict=False)
                        del decoder_model_ttt_state_dict
                    if proj_ttt_path is not None:
                        conditioning_projection = torch.load(proj_ttt_path, weights_only=False, map_location=accelerator.device)
                    # set current task name
                    curr_ttt_task_name = task_name

                    # another eval after loading weight just in case
                    encoder_model.eval()
                    if not tie_models:
                        decoder_model.eval()
                    if conditioning_projection is not None:
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
                        "decoder_gen_input_ids": copy.deepcopy(dec_gen_ids),
                        "decoder_gen_attention_mask": copy.deepcopy(dec_gen_mask),
                        "decoder_label_texts": copy.deepcopy(label_texts),
                        "decoder_out_token_length": copy.deepcopy(out_token_length),
                        "decoder_gen_input_ids_lens": copy.deepcopy(dec_gen_ids_lens),
                    }

                # compute ce loss
                with accelerator.autocast():
                    ce_loss, _, _, _, predicted_program = encoder_decoder_loss(
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
                        num_virtual_tokens=num_virtual_tokens,
                        invar_loss_lambda=0.0,
                        encoder_loss_lambda=0.0,
                        no_lora=no_lora,
                        decoder_ce_loss=decoder_ce_loss,
                        encoder_pad_side=encoder_pad_side,
                        decoder_pad_side=decoder_pad_side,
                        trainable_nbit=trainable_nbit,
                        flash_attn=flash_attn,
                    )

                # ce loss should be from the original permutation, which is set to the first permuted batch
                if batch_permute_i == 0:
                    loss_list += [ce_loss.item()] * bs

                # accumulate program
                if batch_permute_i == 0:
                    predicted_program_accum = predicted_program
                elif "2prefix" in conditioning_method:
                    assert isinstance(predicted_program, list)
                    assert len(predicted_program) == len(predicted_program_accum) # same number of layers
                    for layer_i, (layer_program_0, layer_program_1) in enumerate(predicted_program):
                        avail_count = 0
                        for batch_i, avail in enumerate(batch_avail):
                            if avail:
                                predicted_program_accum[layer_i][0][batch_i] += layer_program_0[avail_count]
                                predicted_program_accum[layer_i][1][batch_i] += layer_program_1[avail_count]
                                avail_count += 1
                else:
                    assert isinstance(predicted_program, torch.Tensor)
                    assert predicted_program.shape[1:] == predicted_program_accum.shape[1:]
                    for batch_i, avail in enumerate(batch_avail):
                        avail_count = 0
                        if avail:
                            predicted_program_accum[batch_i] += predicted_program[avail_count]
                            avail_count += 1

                # accumulate avail (count of permutations)
                assert len(batch_avail) == len(avail_accum) == bs
                for batch_i in range(bs):
                    avail_accum[batch_i] += batch_avail[batch_i]

            # average program
            assert all(avail > 0 for avail in avail_accum)
            if "2prefix" in conditioning_method:
                for layer_i, (layer_program_0, layer_program_1) in enumerate(predicted_program_accum):
                    assert len(layer_program_0) == len(layer_program_1) == len(avail_accum) == bs
                    for batch_i in range(bs):
                        layer_program_0[batch_i] /= avail_accum[batch_i]
                        layer_program_1[batch_i] /= avail_accum[batch_i]
                    predicted_program_accum[layer_i] = (layer_program_0, layer_program_1)
            else:
                assert len(predicted_program_accum) == len(avail_accum) == bs
                for i, avail in enumerate(avail_accum):
                    predicted_program_accum[i] /= avail
            predicted_program = predicted_program_accum

            # recover data from first batch (e.g. task_ids might be missing tasks due to permute_iters)
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
                if "hidden2prompt" in conditioning_method:
                    if no_lora:
                        decoder_inputs_embeds = decoder_module.model.embed_tokens(dec_gen_ids)
                    else:
                        decoder_inputs_embeds = decoder_module.model.model.embed_tokens(dec_gen_ids)
                    # pad decoder inputs embeds
                    if decoder_gen_pad_side == "right":
                        decoder_inputs_embeds = torch.cat([predicted_program, decoder_inputs_embeds], dim=1)
                    else:
                        decoder_inputs_embeds_new = []
                        for x, p, l in zip(decoder_inputs_embeds, predicted_program, dec_gen_ids_lens):
                            x = torch.cat([x[:-l], p, x[-l:]])
                            decoder_inputs_embeds_new.append(x)
                        decoder_inputs_embeds = torch.stack(decoder_inputs_embeds_new)
                    if flash_attn:
                        decoder_inputs_embeds = decoder_inputs_embeds.to(NBIT_TO_DTYPE[trainable_nbit])
                    # pad decoder attention masks
                    prefix_attention_mask = torch.full(
                        (dec_gen_mask.shape[0], num_virtual_tokens),
                        1,
                        device=dec_gen_mask.device,
                        dtype=dec_gen_mask.dtype,
                    )
                    if decoder_gen_pad_side == "right":
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
                    gen_texts = decoder_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
                else:
                    dec_gen_ids = torch.cat([
                        torch.ones((bs, num_virtual_tokens), device=dec_gen_ids.device, dtype=dec_gen_ids.dtype),
                        dec_gen_ids
                    ], dim=1) # the addition will be ignored, double checked
                    dec_gen_mask = torch.cat([
                        torch.ones((bs, num_virtual_tokens), device=dec_gen_mask.device, dtype=dec_gen_mask.dtype),
                        dec_gen_mask
                    ], dim=1)
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
                    gen_texts = decoder_tokenizer.batch_decode(gen_tokens[:, dec_gen_ids.shape[1]:], skip_special_tokens=True)

            # Compare each gen_text with label_texts
            assert len(task_ids) == len(inverters) == bs, (len(task_ids), len(inverters), bs)
            assert len(gen_texts) == len(label_texts) == bs, (len(gen_texts), len(label_texts), bs)
            for task_id, inverter, gen_text, label_text in zip(task_ids, inverters, gen_texts, label_texts):
                # save gen and gt text
                task_id_and_text_list.append((task_id, gen_text, label_text))
                # exact acc
                exact_acc_list.append(int(gen_text == label_text))
                # is valid grid
                gen_grid, gen_is_grid = text_to_2d_grid(gen_text, compact_grids)
                label_grid, label_is_grid = text_to_2d_grid(label_text, compact_grids)
                assert label_is_grid
                valid_grid_list.append(int(gen_is_grid))
                if not gen_is_grid:
                    correct_grid_dim_list.append(0)
                    token_acc_list.append(0)
                    continue
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
    task_id_and_text_list = gather_object(task_id_and_text_list)
    task_id_and_inverter_grids = gather_object(task_id_and_inverter_grids) # likely diff len from dataset
    loss_list = gather_object(loss_list)
    exact_acc_list = gather_object(exact_acc_list)
    valid_grid_list = gather_object(valid_grid_list)
    correct_grid_dim_list = gather_object(correct_grid_dim_list)
    token_acc_list = gather_object(token_acc_list)
    ttt_provided_list = gather_object(ttt_provided_list)
    assert len(task_id_and_text_list) == len(dataset), (len(task_id_and_text_list), len(dataset))
    assert len(loss_list) == len(dataset), (len(loss_list), len(dataset))
    assert len(exact_acc_list) == len(dataset), (len(exact_acc_list), len(dataset))
    assert len(valid_grid_list) == len(dataset), (len(valid_grid_list), len(dataset))
    assert len(correct_grid_dim_list) == len(dataset), (len(correct_grid_dim_list), len(dataset))
    assert len(token_acc_list) == len(dataset), (len(token_acc_list), len(dataset))
    assert len(ttt_provided_list) == len(dataset), (len(ttt_provided_list), len(dataset))

    # average metrics
    # note these are all computed without accounting for skipped eval grids
    avg_ce = sum(loss_list) / len(dataset)
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

    return avg_ce, exact_acc, valid_grid, correct_grid_dim, token_acc, task_id_to_texts, \
        votes, competition_sub_acc, competition_all_acc, ttt_provided


def list2d_to_tuple(l: List[List[int]]) -> Tuple[Tuple[int]]:
    return tuple(tuple(row) for row in l)


def row_base_majority_voting(
        grids: Tuple[Tuple[int]],
        transpose: bool = False
    ) -> Tuple[Tuple[int]]:
    # transpose if needed
    if transpose:
        grids = [list2d_to_tuple((np.array(grid).T).tolist()) for grid in grids]
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


def get_three_votes(grids: Tuple[Tuple[int]]) -> List[Tuple[Tuple[int]]]:
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


def text_to_2d_grid(text: str, compact_grids: bool):
    try:
        grid_lines = text.split('\n')
        height, width = int(grid_lines[0]), int(grid_lines[1])
        assert height > 0 and width > 0
        grid = []
        row_lens = []
        for l in grid_lines[2:]:
            if compact_grids:
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
    def __init__(self, spike_multiplier, momentum=0.9):
        self.moving_average = None
        self.momentum = momentum
        self.spike_multiplier = spike_multiplier

    def update(self, new_val):
        if self.moving_average == None:
            self.moving_average = new_val
            return False
        is_spike = (new_val > self.moving_average * self.spike_multiplier)
        self.moving_average = self.moving_average * self.momentum + new_val * (1.0 - self.momentum)
        return is_spike


def compute_grad_norm2(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2).item() ** 2
            total_norm += param_norm
    return total_norm ** 0.5


def save_train_model(encoder_model, decoder_model, conditioning_projection, output_dir, epoch, tie_models):
    # encoder
    save_enc_path = os.path.join(output_dir, f"encoder_lora_epoch_{epoch+1}")
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    encoder_module.save_pretrained(save_enc_path, save_embedding_layers=True)
    logger.info(f"Saved encoder to {save_enc_path}")
    # decoder
    save_dec_path = None
    if not tie_models:
        save_dec_path = os.path.join(output_dir, f"decoder_lora_epoch_{epoch+1}")
        decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
        decoder_module.save_pretrained(save_dec_path, save_embedding_layers=True)
        logger.info(f"Saved decoder to {save_dec_path}")
    # projection
    save_proj_path = None
    if conditioning_projection is not None:
        save_proj_path = os.path.join(output_dir, f"conditioning_projection_epoch_{epoch+1}.pt")
        conditioning_projection_module = conditioning_projection.module if isinstance(conditioning_projection, DistributedDataParallel) else conditioning_projection
        torch.save(conditioning_projection_module, save_proj_path)
        logger.info(f"Saved conditioning projection to {save_proj_path}")
    return save_enc_path, save_dec_path, save_proj_path


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
    parser.add_argument("--debug_batch_size_1", action="store_true")

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")

    # Conditioning projection
    parser.add_argument("--conditioning_method",
                        type=str,
                        choices=[
                            "prefix2prefix",
                            "hidden2prefix_shared",
                            "hidden2prefix_full",
                            "hidden2prompt",
                            "hidden2prompt_shared",
                            "hidden2prompt_shared_identity",
                            "hidden2prompt_full",
                            "hidden2prompt_full_identity",
                        ],
                        default="hidden2prompt")

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
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.03)
    parser.add_argument("--encoder_loss_lambda", type=float, default=1.0)
    parser.add_argument("--encoder_loss_type", type=str, choices=["last", "rest", "all"], default="rest")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # both data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compact_grids", action="store_true")
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

    # train data
    parser.add_argument("--train_data_dir", type=str, default="/scratch/yl11330/re-arc/train_data/tasks")
    parser.add_argument("--min_prefix", type=int, default=2)
    parser.add_argument("--max_prefix", type=int, default=7)
    parser.add_argument("--augment_ratio", type=float, default=0.0)

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

    # Lora encoder
    parser.add_argument("--encoder_lora_rank", type=int, default=256)
    parser.add_argument("--encoder_lora_alpha", type=float, default=24.0)
    parser.add_argument("--encoder_lora_dropout", type=float, default=0.0)
    parser.add_argument('--encoder_lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens'
    ])
    parser.add_argument("--encoder_no_rslora", action='store_true')

    # Lora decoder
    parser.add_argument("--decoder_lora_rank", type=int, default=256)
    parser.add_argument("--decoder_lora_alpha", type=float, default=24.0)
    parser.add_argument("--decoder_lora_dropout", type=float, default=0.0)
    parser.add_argument('--decoder_lora_target_modules', type=str, nargs="+", default=[
        'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens'
    ])
    parser.add_argument("--decoder_no_rslora", action='store_true')

    # Virtual tokens approach
    parser.add_argument("--num_virtual_tokens", type=int, default=8)
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
    if "prefix2" in args.conditioning_method:
        assert not args.encoder_gradient_checkpointing
    if "2prefix" in args.conditioning_method:
        assert not args.decoder_gradient_checkpointing
        assert args.decoder_pad_side == "right" # right for no middle padding
    if args.encoder_name == args.decoder_name:
        assert args.encoder_gradient_checkpointing == args.decoder_gradient_checkpointing
    if args.conditioning_method in ["prefix2prefix", "hidden2prompt"] or "identity" in args.conditioning_method:
        assert args.encoder_name == args.decoder_name
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable
    if args.debug_enc_len > -1 and args.debug_dec_len == -1:
        args.debug_dec_len = args.debug_enc_len // 2

    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=14400)
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
    if args.tie_models:
        decoder_tokenizer = encoder_tokenizer
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.decoder_name], cache_dir='./encoder_decoder_cache')
    if not encoder_tokenizer.pad_token:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
    if not decoder_tokenizer.pad_token:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    logger.info("Tokenizers loaded and pad tokens handled.")

    # Build base models
    from_pretrained_kwargs = {
        "cache_dir": "./encoder_decoder_cache",
        "low_cpu_mem_usage": True
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
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"unrecognized untrainable_nbit {args.untrainable_nbit}")

    base_encoder = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.encoder_name],
        **from_pretrained_kwargs
    )
    if args.tie_models:
        base_decoder = base_encoder
    else:
        base_decoder = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.decoder_name],
            **from_pretrained_kwargs
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

    # add new CLS tokens for program encoding
    cls_tokens = [f"<CLS{token_i}>" for token_i in range(args.num_virtual_tokens)]
    encoder_tokenizer.add_tokens(cls_tokens)
    base_encoder.resize_token_embeddings(len(encoder_tokenizer))
    logger.info("Base models loaded.")

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

    conditioning_projection = None
    if "hidden2prefix" in args.conditioning_method:
        conditioning_projection = Hidden2PrefixProjection(
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            conditioning_method=args.conditioning_method,
        )
    elif ("hidden2prompt" in args.conditioning_method) and (args.conditioning_method != "hidden2prompt"):
        conditioning_projection = Hidden2PromptProjection(
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            conditioning_method=args.conditioning_method,
        )
    logger.info("conditioning projection initialized (optional)")

    # if only encoder prefix is needed, the non-kv-projection weights of encoder model last layer are not used
    if (args.conditioning_method == "prefix2prefix") and not args.tie_models:
        encoder_num_layer = len(encoder_model.model.layers) if args.no_lora else len(encoder_model.model.model.layers)
        for name, param in encoder_model.named_parameters():
            if f'layers.{encoder_num_layer-1}' in name and param.requires_grad:
                if 'mlp' in name or 'o_proj' in name or 'q_proj' in name:
                    param.requires_grad = False
        logger.info(f'Set last layer of encoder to not require grad')

    # convert model weights
    for name, param in encoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if conditioning_projection is not None:
        for name, param in conditioning_projection.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # number of parameters
    print_trainable_parameters(encoder_model)
    if not args.tie_models:
        print_trainable_parameters(decoder_model)
    if conditioning_projection is not None:
        proj_n_params = sum(p.numel() for p in conditioning_projection.parameters())
        logger.info(f'conditioning projection params {three_commas(proj_n_params)}')

    # model size
    logger.info(f'encoder size {round(encoder_model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    if not args.tie_models:
        logger.info(f'decoder size {round(decoder_model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    if conditioning_projection is not None:
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
        seed=args.seed,
        compact_grids=args.compact_grids,
        num_virtual_tokens=args.num_virtual_tokens,
        debug_fixed_train_order=args.debug_fixed_train_order,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        debug_batch_size_1=args.debug_batch_size_1,
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side=args.decoder_pad_side,
        encoder_loss_type=args.encoder_loss_type,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
    if args.debug_enc_len > 0:
        train_collate_fn = partial(
            collate_fn_train_dummy,
            debug_enc_len=args.debug_enc_len,
            debug_dec_len=args.debug_dec_len,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
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
            if "embed" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if param.requires_grad:
                if "embed" in name:
                    embedding_params.append(param)
                else:
                    other_params.append(param)
    if conditioning_projection is not None:
        for param in conditioning_projection.parameters():
            other_params.append(param)

    optimizer_grouped_params = [
        {"params": embedding_params, "lr": args.lr_embedding},
        {"params": other_params, "lr": args.lr_other},
    ]
    all_params = embedding_params + other_params
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw8bit':
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_params, weight_decay=args.weight_decay)
        # optimizer = bnb.optim.PagedAdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(optimizer_grouped_params)
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

    # for name, param in encoder_model.named_parameters(): print(name, param.dtype)
    # for name, param in conditioning_projection.named_parameters(): print(name, param.dtype)
    # for name, param in decoder_model.named_parameters(): print(name, param.dtype)
    # breakpoint()

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
    if args.tie_models and args.encoder_gradient_checkpointing and accelerator.num_processes > 0:
        # set to train
        encoder_model.train()
        decoder_model.train()
        if conditioning_projection is not None:
            conditioning_projection.train()
        # set static graph
        encoder_model._set_static_graph()
        decoder_model._set_static_graph()
        if conditioning_projection is not None:
            conditioning_projection._set_static_graph()
        # get any data, single forward and backward pass
        batch_data = next(iter(train_loader))
        with accelerator.autocast():
            _, _, _, total_loss, _ = encoder_decoder_loss(
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
                num_virtual_tokens=args.num_virtual_tokens,
                encoder_loss_lambda=args.encoder_loss_lambda,
                invar_loss_lambda=args.invar_loss_lambda,
                no_lora=args.no_lora,
                decoder_ce_loss=True, # HARDCODE
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                trainable_nbit=args.trainable_nbit,
                flash_attn=args.flash_attn,
            )
        accelerator.backward(total_loss)
        optimizer.zero_grad()
        logger.info(f'To avoid DDP error, static graph is applied after doing one iteration of forward backward')

    # train!
    for epoch in range(args.num_epochs):
        encoder_model.train()
        decoder_model.train()
        if conditioning_projection is not None:
            conditioning_projection.train()

        ce_loss_accum = 0.0
        invar_loss_accum = 0.0
        total_loss_accum = 0.0
        encoder_loss_accum = 0.0
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

            with accelerator.accumulate(encoder_model, decoder_model, conditioning_projection):
                with accelerator.autocast():
                    ce_loss, invar_loss, encoder_loss, total_loss, _ = encoder_decoder_loss(
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
                        num_virtual_tokens=args.num_virtual_tokens,
                        encoder_loss_lambda=args.encoder_loss_lambda,
                        invar_loss_lambda=args.invar_loss_lambda,
                        no_lora=args.no_lora,
                        decoder_ce_loss=True, # HARDCODE
                        encoder_pad_side=args.encoder_pad_side,
                        decoder_pad_side=args.decoder_pad_side,
                        trainable_nbit=args.trainable_nbit,
                        flash_attn=args.flash_attn,
                    )

                # just accumulate for logging
                avg_ce_loss = accelerator.gather(ce_loss.repeat(args.train_batch_size)).mean()
                avg_invar_loss = accelerator.gather(invar_loss.repeat(args.train_batch_size)).mean()
                avg_encoder_loss = accelerator.gather(encoder_loss.repeat(args.train_batch_size)).mean()
                avg_total_loss = accelerator.gather(total_loss.repeat(args.train_batch_size)).mean()
                ce_loss_accum += avg_ce_loss.item() / args.grad_accum_steps
                invar_loss_accum += avg_invar_loss.item() / args.grad_accum_steps
                encoder_loss_accum += avg_encoder_loss.item() / args.grad_accum_steps
                total_loss_accum += avg_total_loss.item() / args.grad_accum_steps

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm_accum += accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item()
                optimizer.step()
                lr_scheduler.step()

                # detect and log spike
                if not args.no_log_spikes and accelerator.is_main_process:
                    is_gradnorm_spike = gradnorm_spike_detector.update(compute_grad_norm2(all_params))
                    is_loss_spike = loss_spike_detector.update(total_loss.item())
                    if is_gradnorm_spike or is_loss_spike:
                        enc_texts = encoder_tokenizer.batch_decode(enc_ids, skip_special_tokens=True)
                        dec_texts = decoder_tokenizer.batch_decode(dec_ids, skip_special_tokens=True)
                        data_string = json.dumps({'enc': enc_texts, 'dec': dec_texts})
                    if is_gradnorm_spike:
                        with open(spike_gradnorm_samples_path, 'a') as f:
                            f.write(f"{global_step} {data_string}\n")
                    if is_loss_spike:
                        with open(spike_loss_samples_path, 'a') as f:
                            f.write(f"{global_step} {data_string}\n")

                optimizer.zero_grad()


            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    progress_bar.update(args.log_every)
                    accelerator.log({
                        "train/ce_loss": ce_loss_accum,
                        "train/invar_loss": invar_loss_accum,
                        "train/encoder_loss": encoder_loss_accum,
                        "train/total_loss": total_loss_accum,
                        "train/grad_norm_accum": grad_norm_accum,
                        "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                        "train/lr_other": lr_scheduler.get_last_lr()[1],
                    }, step=global_step)

                ce_loss_accum = 0.0
                invar_loss_accum = 0.0
                encoder_loss_accum = 0.0
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
                compact_grids=args.compact_grids,
                num_virtual_tokens=args.num_virtual_tokens,
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
                compact_grids=args.compact_grids,
                num_virtual_tokens=args.num_virtual_tokens,
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
                    debug_enc_len=args.debug_enc_len,
                    debug_dec_len=args.debug_dec_len,
                )

            train_ce, train_exact_acc, train_valid_grid, train_correct_grid_dim, train_token_acc, train_texts, \
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
                num_virtual_tokens=args.num_virtual_tokens,
                decoder_tokenizer=decoder_tokenizer,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                compact_grids=args.compact_grids,
                no_lora=args.no_lora,
                decoder_ce_loss=True, # HARDCODE
                trainable_nbit=args.trainable_nbit,
                flash_attn=args.flash_attn,
                tie_models=args.tie_models,
                output_dir=args.output_dir,
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
                search_iters=0, # HARDCODE
            )
            eval_ce, eval_exact_acc, eval_valid_grid, eval_correct_grid_dim, eval_token_acc, eval_texts, \
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
                num_virtual_tokens=args.num_virtual_tokens,
                decoder_tokenizer=decoder_tokenizer,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                compact_grids=args.compact_grids,
                no_lora=args.no_lora,
                decoder_ce_loss=True, # HARDCODE
                trainable_nbit=args.trainable_nbit,
                flash_attn=args.flash_attn,
                tie_models=args.tie_models,
                output_dir=args.output_dir,
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
                search_iters=0, # HARDCODE
            )

            if accelerator.is_main_process:
                eval_metric_dict = {
                    "eval/train_ce_loss": train_ce,
                    "eval/train_exact_acc": train_exact_acc,
                    "eval/train_valid_grid": train_valid_grid,
                    "eval/train_correct_grid_dim": train_correct_grid_dim,
                    "eval/train_token_acc": train_token_acc,
                    "eval/train_competition_all_acc": train_competition_all_acc,
                    "eval/train_competition_sub_acc": train_competition_sub_acc,
                    "eval/eval_ce_loss": eval_ce,
                    "eval/eval_exact_acc": eval_exact_acc,
                    "eval/eval_valid_grid": eval_valid_grid,
                    "eval/eval_correct_grid_dim": eval_correct_grid_dim,
                    "eval/eval_token_acc": eval_token_acc,
                    "eval/eval_competition_all_acc": eval_competition_all_acc,
                    "eval/eval_competition_sub_acc": eval_competition_sub_acc,
                }
                logger.info(f'Evaluation results:\n{pprint.pformat(eval_metric_dict, indent=4)}')
                accelerator.log(eval_metric_dict, step=global_step)

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
                        save_enc_path, save_dec_path, save_proj_path = last_save_model_path
                        rm_cmd = f"rm -rf {save_enc_path}"
                        if save_dec_path is not None:
                            rm_cmd += f" {save_dec_path}"
                        if save_proj_path is not None:
                            rm_cmd += f" {save_proj_path}"
                        os.system(rm_cmd)
                    save_enc_path, save_dec_path, save_proj_path = save_train_model(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        conditioning_projection=conditioning_projection,
                        output_dir=args.output_dir,
                        epoch=epoch,
                        tie_models=args.tie_models,
                    )
                    last_save_model_path = (save_enc_path, save_dec_path, save_proj_path)
                epoch_to_eval_exact_acc[epoch] = eval_exact_acc

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
