# train_script.py

import pprint
import math
import json
from tqdm import tqdm
from functools import partial
import argparse
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from accelerate import PartialState
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


class ProjectKV(nn.Module):
    def __init__(self, num_virtual_tokens, encoder_model, decoder_model, conditioning_method):
        super(ProjectKV, self).__init__()
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

    def forward(self, enc_hidden_states):
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


class ProjectPrompt(nn.Module):
    def __init__(self, num_virtual_tokens, encoder_model, decoder_model, conditioning_method):
        super(ProjectPrompt, self).__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.conditioning_method = conditioning_method
        self.encoder_hidden_size = encoder_model.config.hidden_size
        self.decoder_hidden_size = decoder_model.config.hidden_size
        # weights
        if self.conditioning_method == "hidden2prompt_shared":
            projections = nn.Linear(encoder_model.config.hidden_size, decoder_model.config.hidden_size)
            self.weights = nn.Parameter(projections.weight)
            self.biases = nn.Parameter(projections.bias)
        elif conditioning_method == "hidden2prompt_full":
            projections = nn.ModuleList([
                nn.Linear(
                    encoder_model.config.hidden_size,
                    decoder_model.config.hidden_size,
                ) for _ in range(self.num_virtual_tokens)
            ])
            self.weights = nn.Parameter(torch.stack([projection.weight for projection in projections], dim=0))
            self.biases = nn.Parameter(torch.stack([projection.bias for projection in projections], dim=0))
        else:
            raise ValueError(f'unrecognized conditioning method {self.conditioning_method}')
        del projections

    def forward(self, enc_hidden_states):
        # enc_hidden_states has shape (batch_size, num_virtual_tokens, hidden_dim)
        assert enc_hidden_states.shape[1] == self.num_virtual_tokens
        # outs (batch_size, num_virtual_tokens, out_dim)
        if self.conditioning_method == "hidden2prompt_shared":
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
    enc_ids_lens: list,
    dec_ids_lens: list,
    num_virtual_tokens: int,
    invar_loss_lambda: float,
    encoder_loss_lambda: float,
    encoder_pad_side: str,
    decoder_pad_side: str,
):
    """
    This function shows how we:
      1) Pass input_ids thru the encoder -> final hidden states
      2) Extract the CLS token (the last token in the seq)
      3) Project it => prefix
      4) Pass prefix + decoder_input_ids => decoder with labels => cross-entropy
    Returns (ce_loss, hidden_cls).
    """
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
    if conditioning_method in ["hidden2prefix_shared", "hidden2prefix_full"]:
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-num_virtual_tokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-num_virtual_tokens:] for x in enc_hidden_states])
        predicted_program = conditioning_projection(enc_hidden_states=enc_hidden_states)
    elif conditioning_method == "prefix2prefix":
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
    elif conditioning_method in ["hidden2prompt_shared", "hidden2prompt_full"]:
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-num_virtual_tokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-num_virtual_tokens:] for x in enc_hidden_states])
        predicted_program = conditioning_projection(enc_hidden_states)
    elif conditioning_method == "hidden2prompt":
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        if encoder_pad_side == "right":
            enc_hidden_states = torch.stack([x[l-num_virtual_tokens:l] for x, l in zip(enc_hidden_states, enc_ids_lens)])
        else:
            enc_hidden_states = torch.stack([x[-num_virtual_tokens:] for x in enc_hidden_states])
        predicted_program = enc_hidden_states
    else:
        raise ValueError(f"invalid conditioning method {conditioning_method}")

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


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    encoder_model,
    decoder_model,
    conditioning_method,
    conditioning_projection,
    dataset,
    accelerator: Accelerator,
    num_virtual_tokens: int,
    decoder_tokenizer,
    batch_size: int,
    collate_fn,
    compact_grids: bool,
    encoder_pad_side: str,
    decoder_pad_side: str,
    decoder_gen_pad_side: str,
):
    """
    For each task in dataset, compute:
      - cross-entropy
      - generate => exact match vs decoder_label_texts

    Returns (avg_ce, accuracy, total_samples).
    We also log how many total items are valid => sum of 'num_valid' from collate.
    """
    encoder_model.eval()
    decoder_model.eval()
    if conditioning_projection != None:
        conditioning_projection.eval()

    decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model

    # setup terminators and suppress warning
    terminators = [
        decoder_tokenizer.eos_token_id,
        decoder_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    decoder_module.generation_config.pad_token_id = decoder_tokenizer.pad_token_id

    distributed_state = PartialState()
    task_id_and_text_list = []
    loss_list = []
    exact_acc_list = []
    valid_grid_list = []
    correct_grid_dim_list = []
    token_acc_list = []

    with distributed_state.split_between_processes(dataset.parsed_data) as data:
        n_batches = math.ceil(len(data) / batch_size)
        for batch in tqdm(chunks(data, batch_size), total=n_batches):
            batch = collate_fn(batch)
            bs = batch["encoder_input_ids"].size(0)

            task_ids = batch["task_ids"]
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
                    encoder_pad_side=encoder_pad_side,
                    decoder_pad_side=decoder_pad_side,
                )

            loss_list += [ce_loss.item()] * bs

            # compute accuracy
            with accelerator.autocast():
                # padding at front because HF ignores it
                gen_texts = None
                if conditioning_method in ["hidden2prompt", "hidden2prompt_shared", "hidden2prompt_full"]:
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
            for task_id, gen_text, label_text in zip(task_ids, gen_texts, label_texts):
                # save gen and gt
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
    loss_list = gather_object(loss_list)
    exact_acc_list = gather_object(exact_acc_list)
    valid_grid_list = gather_object(valid_grid_list)
    correct_grid_dim_list = gather_object(correct_grid_dim_list)
    token_acc_list = gather_object(token_acc_list)
    assert len(task_id_and_text_list) == len(loss_list) == len(exact_acc_list) == len(dataset.parsed_data)
    assert len(valid_grid_list) == len(correct_grid_dim_list) == len(token_acc_list) == len(dataset.parsed_data)

    task_id_to_texts = {x[0]: (x[1], x[2]) for x in task_id_and_text_list}
    avg_ce = sum(loss_list) / len(dataset.parsed_data)
    exact_acc = sum(exact_acc_list) / len(dataset.parsed_data)
    valid_grid = sum(valid_grid_list) / len(dataset.parsed_data)
    correct_grid_dim = sum(correct_grid_dim_list) / len(dataset.parsed_data)
    token_acc = sum(token_acc_list) / len(dataset.parsed_data)
    return avg_ce, exact_acc, valid_grid, correct_grid_dim, token_acc, task_id_to_texts


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


def save_model(encoder_model, decoder_model, conditioning_projection, output_dir, epoch):
    # encoder decoder
    save_enc_path = os.path.join(output_dir, f"encoder_lora_epoch_{epoch+1}")
    save_dec_path = os.path.join(output_dir, f"decoder_lora_epoch_{epoch+1}")
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
    encoder_module.save_pretrained(save_enc_path, save_embedding_layers=True)
    decoder_module.save_pretrained(save_dec_path, save_embedding_layers=True)
    logger.info(f"Saved encoder to {save_enc_path}")
    logger.info(f"Saved decoder to {save_dec_path}")
    # projection
    save_proj_path = None
    if conditioning_projection is not None:
        save_proj_path = os.path.join(output_dir, f"conditioning_projection_epoch_{epoch+1}.pt")
        torch.save(conditioning_projection, save_proj_path)
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
    parser.add_argument("--spike_multiplier", type=float, default=3.0)
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
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=32)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=32)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")

    # Conditioning projection
    parser.add_argument("--conditioning_method",
                        choices=[
                            "prefix2prefix",
                            "hidden2prefix_shared",
                            "hidden2prefix_full",
                            "hidden2prompt",
                            "hidden2prompt_shared",
                            "hidden2prompt_full",
                        ],
                        default="prefix2prefix")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--samples_per_epoch", type=int, default=500)
    parser.add_argument("--eval_epochs", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.1)
    parser.add_argument("--encoder_loss_lambda", type=float, default=0.0)
    parser.add_argument("--encoder_demonstration_loss", action="store_true")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # Data
    parser.add_argument("--train_data_dir", type=str, default="/scratch/yl11330/re-arc/train_data/tasks")
    parser.add_argument("--eval_train_dir", type=str, default="/scratch/yl11330/re-arc/arc_original/training")
    parser.add_argument("--eval_eval_dir", type=str, default="/scratch/yl11330/re-arc/arc_original/evaluation")
    parser.add_argument("--eval_train_ratio", type=float, default=1.0)
    parser.add_argument("--eval_eval_ratio", type=float, default=1.0)
    parser.add_argument("--min_prefix", type=int, default=2)
    parser.add_argument("--max_prefix", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--augment_ratio", type=float, default=0.3)
    parser.add_argument("--compact_grids", action="store_true")
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

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
    parser.add_argument("--decoder_lm_head", action='store_true')
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
    if args.debug_enc_len > -1 and args.debug_dec_len == -1:
        args.debug_dec_len = args.debug_enc_len // 2
    if args.conditioning_method == "prefix2prefix":
        assert not args.encoder_gradient_checkpointing
    if args.conditioning_method in ["prefix2prefix", "hidden2prefix_shared", "hidden2prefix_full"]:
        assert not args.decoder_gradient_checkpointing
        assert args.decoder_pad_side == "right" # right for no middle padding
    if args.conditioning_method in ["prefix2prefix", "hidden2prompt"]:
        assert args.encoder_name == args.decoder_name
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable

    if args.decoder_lm_head and 'lm_head' not in args.decoder_lora_target_modules:
        args.decoder_lora_target_modules.append('lm_head')

    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",
        project_config=project_config,
        log_with="wandb" if args.wandb else None,
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
        base_decoder = prepare_model_for_kbit_training(base_decoder, use_gradient_checkpointing=args.decoder_gradient_checkpointing)
    else:
        if args.encoder_gradient_checkpointing:
            base_encoder.gradient_checkpointing_enable()
        if args.decoder_gradient_checkpointing:
            base_decoder.gradient_checkpointing_enable()

    # add [CLS] is not in model tokenizer
    if not encoder_tokenizer.cls_token:
        encoder_tokenizer.add_special_tokens({"cls_token": "[CLS]"})
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

    logger.info("LoRA-wrapped models initialized.")

    conditioning_projection = None
    if args.conditioning_method in ["hidden2prefix_shared", "hidden2prefix_full"]:
        conditioning_projection = ProjectKV(args.num_virtual_tokens, encoder_model, decoder_model, conditioning_method=args.conditioning_method)
    elif args.conditioning_method in ["hidden2prompt_shared", "hidden2prompt_full"]:
        conditioning_projection = ProjectPrompt(args.num_virtual_tokens, encoder_model, decoder_model, conditioning_method=args.conditioning_method)
    logger.info("conditioning projection initialized (optional).")

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
    logger.info(f'converted model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # number of parameters
    print_trainable_parameters(encoder_model)
    if not args.tie_models:
        print_trainable_parameters(decoder_model)
    if conditioning_projection is not None:
        logger.info(f'conditioning projection params {sum(p.numel() for p in conditioning_projection.parameters())}')

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
        encoder_demonstration_loss=args.encoder_demonstration_loss,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
    if args.debug_enc_len > 0:
        train_collate_fn = partial(
            collate_fn_train_dummy,
            debug_enc_len=args.debug_enc_len,
            debug_dec_len=args.debug_dec_len
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
            if "lora_embedding" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if param.requires_grad:
                if "lora_embedding" in name:
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
        train_loader
    ) = accelerator.prepare(
        encoder_model,
        decoder_model,
        conditioning_projection,
        optimizer,
        train_loader
    )

    # for name, param in encoder_model.named_parameters(): print(name, param.dtype)
    # for name, param in conditioning_projection.named_parameters(): print(name, param.dtype)
    # for name, param in decoder_model.named_parameters(): print(name, param.dtype)
    # breakpoint()

    logger.info(f'\n======= TRAINING INFO START ======')
    logger.info(f'num_epochs={args.num_epochs}')
    logger.info(f'train_batch_size={args.train_batch_size}')
    logger.info(f'grad_accum_steps={args.grad_accum_steps}')
    logger.info(f'accelerator.num_processes={accelerator.num_processes}')
    logger.info(f'steps_per_epoch={steps_per_epoch}')
    logger.info(f'{sum(p.numel() for p in all_params)} trainable params')
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
                        encoder_pad_side=args.encoder_pad_side,
                        decoder_pad_side=args.decoder_pad_side,
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
                        "train/lr_other": lr_scheduler.get_last_lr()[1]
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
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                max_seq_len=args.max_seq_len,
                keep_ratio=args.eval_train_ratio,
                compact_grids=args.compact_grids,
                num_virtual_tokens=args.num_virtual_tokens,
                encoder_demonstration_loss=args.encoder_demonstration_loss,
                debug_random_pad=args.debug_random_pad,
                debug_pad_len=args.debug_pad_len,
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
            )
            eval_eval_dataset = EvalDataset(
                args.eval_eval_dir,
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                max_seq_len=args.max_seq_len,
                keep_ratio=args.eval_eval_ratio,
                compact_grids=args.compact_grids,
                num_virtual_tokens=args.num_virtual_tokens,
                encoder_demonstration_loss=args.encoder_demonstration_loss,
                debug_random_pad=args.debug_random_pad,
                debug_pad_len=args.debug_pad_len,
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
            )
            eval_collate_fn = partial(collate_fn_eval, dataset=eval_train_dataset) # only use tokenizer, debug_random_pad
            if args.debug_enc_len > 0:
                eval_collate_fn = partial(
                    collate_fn_eval_dummy,
                    debug_enc_len=args.debug_enc_len,
                    debug_dec_len=args.debug_dec_len
                )
            logger.info(f"len(eval_train_dataset) = {len(eval_train_dataset)}")
            logger.info(f"len(eval_eval_dataset) = {len(eval_eval_dataset)}")

            train_ce, train_exact_acc, train_valid_grid, train_correct_grid_dim, train_token_acc, train_texts = evaluate(
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
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
            )
            eval_ce, eval_exact_acc, eval_valid_grid, eval_correct_grid_dim, eval_token_acc, eval_texts = evaluate(
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
                encoder_pad_side=args.encoder_pad_side,
                decoder_pad_side=args.decoder_pad_side,
                decoder_gen_pad_side=args.decoder_gen_pad_side,
            )

            if accelerator.is_main_process:
                eval_metric_dict = {
                    "eval/train_ce_loss": train_ce,
                    "eval/train_exact_acc": train_exact_acc,
                    "eval/train_valid_grid": train_valid_grid,
                    "eval/train_correct_grid_dim": train_correct_grid_dim,
                    "eval/train_token_acc": train_token_acc,
                    "eval/eval_ce_loss": eval_ce,
                    "eval/eval_exact_acc": eval_exact_acc,
                    "eval/eval_valid_grid": eval_valid_grid,
                    "eval/eval_correct_grid_dim": eval_correct_grid_dim,
                    "eval/eval_token_acc": eval_token_acc,
                }
                logger.info(f'Evaluation results:\n{pprint.pformat(eval_metric_dict, indent=4)}')
                accelerator.log(eval_metric_dict, step=global_step)

                # Save outputs
                save_eval_train_path = os.path.join(args.output_dir, f"eval_train_{epoch+1}.json")
                save_eval_eval_path = os.path.join(args.output_dir, f"eval_eval_{epoch+1}.json")
                with open(save_eval_train_path, 'w') as f:
                    json.dump(train_texts, f)
                with open(save_eval_eval_path, 'w') as f:
                    json.dump(eval_texts, f)
                logger.info(f"Saved eval train generated text to {save_eval_train_path}")
                logger.info(f"Saved eval eval generated text to {save_eval_eval_path}")

                # Save model
                do_save_model = args.save_all_models
                if not args.save_all_models:
                    if (not epoch_to_eval_exact_acc) or eval_exact_acc >= max(epoch_to_eval_exact_acc.values()):
                        do_save_model = True

                if do_save_model:
                    if (not args.save_all_models) and (last_save_model_path is not None):
                        save_enc_path, save_dec_path, save_proj_path = last_save_model_path
                        if save_proj_path is not None:
                            os.system(f"rm -rf {save_enc_path} {save_dec_path} {save_proj_path}")
                        else:
                            os.system(f"rm -rf {save_enc_path} {save_dec_path}")
                    save_enc_path, save_dec_path, save_proj_path = save_model(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        conditioning_projection=conditioning_projection,
                        output_dir=args.output_dir,
                        epoch=epoch,
                    )
                    last_save_model_path = (save_enc_path, save_dec_path, save_proj_path)
                epoch_to_eval_exact_acc[epoch] = eval_exact_acc

    logger.info("All done training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
