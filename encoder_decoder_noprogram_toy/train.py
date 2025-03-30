import gc
from custom_llama import MyLlamaModel
from datetime import timedelta
from typing import Union, Callable, List, Tuple, Optional, Iterator
import pprint
import math
from tqdm import tqdm
from functools import partial
import argparse
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    LlamaConfig,
)
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, gather_object

import logging
import datasets
import transformers
import bitsandbytes as bnb

from data_utils import (
    TrainDataset,
    EvalDataset,
    GSDataset,
    collate_fn_train,
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


class ProgramEmbeddings(nn.Module):
    def __init__(self, embedding: torch.Tensor):
        super(ProgramEmbeddings, self).__init__()
        self.embedding = nn.Parameter(embedding)

    def forward(self, program_i: int) -> torch.Tensor:
        del program_i
        return self.embedding


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


def chunks(lst: List[int], n: int) -> Iterator[List[int]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_memory_footprint(module: nn.Module):
    return sum(p.nelement() * p.element_size() for p in module.parameters()) + \
        sum(p.nelement() * p.element_size() for p in module.buffers())


@torch.enable_grad()
def gradient_search(
        batch_idx: int,
        eval_dataset: EvalDataset,
        accelerator: Accelerator,
        model: Union[nn.Module, DistributedDataParallel],
        # inputs
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
        past_key_values_attention_mask: torch.Tensor,
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
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # NOTE: demonstration interval

    # didnt use grad accum, dont think needed

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
    program_params = []
    for layer_k, layer_v in past_key_values:
        program_params.append(layer_k)
        program_params.append(layer_v)

    # set requires grad
    assert all(not p.requires_grad for p in program_params)
    for p in program_params:
        p.requires_grad = True

    # expand to match predicted program with batch size
    past_key_values = tuple(
        (
            layer_k.expand(batch_size, *layer_k.shape[1:]),
            layer_v.expand(batch_size, *layer_v.shape[1:]),
        )
        for layer_k, layer_v in past_key_values
    ) # type: ignore
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
    best_past_key_values = past_key_values
    model.train()

    # train!
    while curr_iter < iters:
        for batch in gs_loader:
            inputs_embeds = batch["inputs_embeds"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            device = inputs_embeds.device

            with accelerator.autocast():
                attention_mask = torch.cat([past_key_values_attention_mask, attention_mask], dim=1)

                # build position ids (does NOT depend on dropout)
                attention_mask_just_for_kv = attention_mask[:, :past_key_values[0][0].shape[2]]
                attention_mask_after_kv = attention_mask[:, past_key_values[0][0].shape[2]:]
                position_ids = []
                for mask_for_kv, mask_after_kv in zip(attention_mask_just_for_kv, attention_mask_after_kv):
                    sequence_position_ids = torch.zeros(inputs_embeds.shape[1], device=device, dtype=torch.int64)
                    position_start = mask_for_kv.sum()
                    n_new_positions = mask_after_kv.sum()
                    new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=device, dtype=torch.int64)
                    if gs_dataset.pad_side == "right":
                        sequence_position_ids[:n_new_positions] = new_positions
                    else:
                        sequence_position_ids[-n_new_positions:] = new_positions
                    position_ids.append(sequence_position_ids)
                position_ids = torch.stack(position_ids)

                model_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "use_cache": True,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }

                # get ce loss
                predictions = model(**model_kwargs).predictions.squeeze(-1) # (bs, 2)
                if gs_dataset.pad_side == 'left':
                    loss = ((predictions[:, -2] - labels) ** 2.0).mean()
                else:
                    loss = ((predictions[:, 0] - labels) ** 2.0).mean()

                # print(loss.item())
                # breakpoint()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(program_params, max_grad_norm)
            optim.step()
            scheduler.step()
            optim.zero_grad()

            if take_best and loss.item() < best_loss:
                best_loss = loss.item()
                best_past_key_values = tuple(
                    (layer_k.detach().clone(), layer_v.detach().clone())
                    for layer_k, layer_v in past_key_values
                )

            curr_iter += 1
            if curr_iter >= iters:
                break

    model.eval()

    if take_best:
        past_key_values = best_past_key_values  # type: ignore

    # shrink to match predicted program with batch size 1
    if batch_size > 1:
        past_key_values = tuple(
            (layer_k[:1], layer_v[:1])
            for layer_k, layer_v in past_key_values
        ) # type: ignore

    return past_key_values


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    gs_iters: int,
    gs_batch_size: int,
    gs_lr: float,
    gs_beta1: float,
    gs_beta2: float,
    gs_optimizer: str,
    gs_max_grad_norm: float,
    gs_lr_scheduler: str,
    gs_take_best: bool,
    attention_cutoff: bool,
    attend_prev_programs: bool,
):
    model.eval()
    if prior_embeddings is not None:
        prior_embeddings.eval()
    if program_embeddings is not None:
        program_embeddings.eval()

    distributed_state = PartialState()
    loss_list = []

    data_idxs = list(range(len(dataset)))
    assert len(data_idxs) >= accelerator.num_processes # avoid padding issue

    with distributed_state.split_between_processes(data_idxs) as process_data_idxs:
        assert isinstance(process_data_idxs, list)
        n_batches = math.ceil(len(process_data_idxs) / batch_size)
        data_idx_iterator = tqdm(chunks(process_data_idxs, batch_size), total=n_batches)  # type: ignore

        for batch_idxs in data_idx_iterator:
            bs = len(batch_idxs)
            batch_data = [dataset[i] for i in batch_idxs]
            batch = collate_fn(batch_data)

            # get tensors
            inputs_embeds = batch["inputs_embeds"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            inputs_embeds_lens = batch["inputs_embeds_lens"]
            # for gs (no ntoken)
            demon_inputs_embeds = batch["demon_inputs_embeds"].to(accelerator.device)
            demon_attention_mask = batch["demon_attention_mask"].to(accelerator.device)
            gen_inputs_embeds = batch["gen_inputs_embeds"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_labels = batch["gen_labels"].to(accelerator.device)

            # compute loss
            if program_embeddings is None and gs_iters > 0:
                with accelerator.autocast():
                    # perform 2-step noprogram generation with gs
                    assert prior_embeddings is None
                    device = demon_inputs_embeds.device

                    # necessary for padside left and does not change when padside is right, idky
                    position_ids = []
                    for m in demon_attention_mask:
                        sequence_position_ids = torch.zeros(demon_inputs_embeds.shape[1], device=device, dtype=torch.int64)
                        n_new_positions = m.sum()
                        new_positions = torch.tensor(range(n_new_positions), device=device, dtype=torch.int64)
                        if dataset.pad_side == "right":
                            sequence_position_ids[:n_new_positions] = new_positions
                        else:
                            sequence_position_ids[-n_new_positions:] = new_positions
                        position_ids.append(sequence_position_ids)
                    position_ids = torch.stack(position_ids)

                    past_key_values = model(
                        inputs_embeds=demon_inputs_embeds,
                        attention_mask=demon_attention_mask,
                        output_hidden_states=True,
                        position_ids=position_ids,
                    ).past_key_values
                    past_key_values_attention_mask = demon_attention_mask # rename for convenience

                # construct new programs and kv to be filled
                new_past_key_values = tuple([[], []] for _ in past_key_values)
                assert past_key_values[0][0].shape[0] == bs

                # gradient search is done individually for simplicity
                with accelerator.no_sync(model):
                    for batch_i, batch_idx in enumerate(batch_idxs):

                        # extract the batchsize1 task inputs
                        assert past_key_values is not None and past_key_values_attention_mask is not None
                        task_past_key_values = tuple(
                            (layer_k[batch_i: batch_i+1].detach().clone(), layer_v[batch_i: batch_i+1].detach().clone())
                            for layer_k, layer_v in past_key_values
                        )
                        task_past_key_values_attention_mask = past_key_values_attention_mask[batch_i: batch_i+1].detach().clone()

                        # search!
                        task_past_key_values = gradient_search(
                            batch_idx=batch_idx,
                            eval_dataset=dataset,
                            accelerator=accelerator,
                            model=model,
                            # inputs
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
                        )

                        # fill new kv, no need to update kv attention mask
                        for layer_i, (layer_k, layer_v) in enumerate(task_past_key_values):
                            new_past_key_values[layer_i][0].append(layer_k)
                            new_past_key_values[layer_i][1].append(layer_v)

                    # finally, tuple-lize kv and rename
                    past_key_values = tuple(
                        (torch.cat(layer_k), torch.cat(layer_v))
                        for layer_k, layer_v in new_past_key_values
                    )

                with accelerator.autocast():
                    # add past key values portion to attention mask
                    gen_attention_mask = torch.cat([past_key_values_attention_mask, gen_attention_mask], dim=1)

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

                    predictions = model(
                        inputs_embeds=gen_inputs_embeds,
                        attention_mask=gen_attention_mask,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                    ).predictions.squeeze(-1) # (bs, 2)

                    losses = compute_loss(predictions, gen_labels, individual_loss=True)

            else:
                with accelerator.autocast():
                    losses = model_loss(
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        inputs_embeds_lens=inputs_embeds_lens,
                        ntokens=dataset.ntokens,
                        pad_side=dataset.pad_side,
                        attention_cutoff=attention_cutoff,
                        attend_prev_programs=attend_prev_programs,
                        individual_loss=True,
                    )

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
        prior_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
        program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
        output_dir: str,
        epoch: int,
    ) -> Tuple[str, Optional[str], Optional[str]]:

    # model
    save_model_path = os.path.join(output_dir, f"epoch_{epoch+1}")
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.save_pretrained(save_model_path, save_embedding_layers=True)
    logger.info(f"Saved model to {save_model_path}")

    # prior embeddings
    save_prior_embeddings_path = None
    if prior_embeddings is not None:
        save_prior_embeddings_path = os.path.join(output_dir, f"prior_embeddings_epoch_{epoch+1}.pt")
        prior_embeddings_module = prior_embeddings
        if isinstance(prior_embeddings, DistributedDataParallel):
            prior_embeddings_module = prior_embeddings.module
        torch.save(prior_embeddings_module, save_prior_embeddings_path)
        logger.info(f"Saved prior embeddings to {save_prior_embeddings_path}")

    # program embeddings
    save_program_embeddings_path = None
    if program_embeddings is not None:
        save_program_embeddings_path = os.path.join(output_dir, f"program_embeddings_epoch_{epoch+1}.pt")
        program_embeddings_module = program_embeddings
        if isinstance(program_embeddings, DistributedDataParallel):
            program_embeddings_module = program_embeddings.module
        torch.save(program_embeddings_module, save_program_embeddings_path)
        logger.info(f"Saved program embeddings to {save_program_embeddings_path}")

    return save_model_path, save_prior_embeddings_path, save_program_embeddings_path


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
    model: Union[nn.Module, DistributedDataParallel],
    prior_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    # data
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    inputs_embeds_lens: List[int],
    ntokens: int,
    # others
    pad_side: str,
    attention_cutoff: bool,
    attend_prev_programs: bool,
    individual_loss: bool,
) -> Union[torch.Tensor, List[torch.Tensor]]:

    # original baseline
    if program_embeddings is None:
        assert prior_embeddings is None
        predictions = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).predictions.squeeze(-1) # (bs, num_pair * 2)

        loss = compute_loss(predictions, labels, individual_loss)
        # print([l.item() for l in loss] if individual_loss else loss.item())
        # breakpoint()
        return loss

    # NOTE: using ntokens is very inefficient, might revise if needed
    # on rtx8000, ntokens0 is slower than ntokens-1 by 10 times or so

    assert prior_embeddings is not None
    device = inputs_embeds.device

    # insert program to inputs embeds
    batch_size, seq_len, net_input_dim = inputs_embeds.shape
    pad_lens = [seq_len - l for l in inputs_embeds_lens]
    pad_lens_with_programs = [pad_len + pad_len // 2 * ntokens for pad_len in pad_lens]
    groups = inputs_embeds.view(batch_size, -1, 2, net_input_dim)
    insert_expanded = program_embeddings('dummy')[None, ...].unsqueeze(1).expand(batch_size, groups.size(1), ntokens, net_input_dim)
    combined = torch.cat([insert_expanded, groups], dim=2)
    inputs_embeds_with_programs = combined.view(batch_size, -1, net_input_dim)

    # for inputs embeds, we need to make sure the first program is prior
    if pad_side == 'left':
        for batch_i, pad_len in enumerate(pad_lens_with_programs):
            inputs_embeds_with_programs[batch_i, pad_len: pad_len + ntokens] = prior_embeddings('dummy')
    else:
        inputs_embeds_with_programs[:, :ntokens, :] = prior_embeddings('dummy')[None, ...]

    # insert program to attention masks
    attention_mask_pad = torch.full((ntokens,), 1, device=device, dtype=torch.int64)
    groups = attention_mask.view(batch_size, -1, 2)
    insert_expanded = attention_mask_pad[None, ...].unsqueeze(1).expand(batch_size, groups.size(1), ntokens)
    combined = torch.cat([insert_expanded, groups], dim=2)
    attention_mask_with_programs = combined.view(batch_size, -1)

    # for attention mask, we need to make sure what's padded should be padded
    for batch_i, pad_len in enumerate(pad_lens_with_programs):
        if pad_len > 0:
            if pad_side == 'left':
                attention_mask_with_programs[batch_i, :pad_len] = 0
            else:
                attention_mask_with_programs[batch_i, -pad_len:] = 0

    # insert program in labels
    labels_pad = torch.full((ntokens,), -100, device=device, dtype=torch.int64)
    groups = labels.view(batch_size, -1, 2)
    insert_expanded = labels_pad[None, ...].unsqueeze(1).expand(batch_size, groups.size(1), ntokens)
    combined = torch.cat([insert_expanded, groups], dim=2)
    labels_with_programs = combined.view(batch_size, -1)

    assert inputs_embeds_with_programs.shape[:2] == attention_mask_with_programs.shape == labels_with_programs.shape

    # construct program intervals
    program_intervals = []
    for pad_len, l in zip(pad_lens_with_programs, inputs_embeds_lens):
        starts = torch.tensor(range(0, l, 2)) + torch.tensor(range(l // 2)) * ntokens
        if pad_side == 'left':
            starts += pad_len
        intervals = [(s, s + ntokens) for s in starts.tolist()]
        program_intervals.append(intervals)

    if not attention_cutoff:
        predictions = model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
        ).predictions.squeeze(-1) # (bs, num_pair * 2)
    else:
        predictions = model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            program_intervals=program_intervals,
            attend_prev_programs=attend_prev_programs,
        ).predictions.squeeze(-1) # (bs, num_pair * 2)

    loss = compute_loss(predictions, labels_with_programs, individual_loss)
    # print([l.item() for l in loss] if individual_loss else loss.item())
    # breakpoint()

    return loss


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

    # Gist/thinking tokens
    parser.add_argument("--ntokens", type=int, default=-1)
    parser.add_argument("--attention_cutoff", action="store_true")
    parser.add_argument("--attend_prev_programs", action="store_true")

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

    # data
    parser.add_argument("--net_input_dim", type=int, default=20)
    parser.add_argument("--net_hidden_dim", type=int, default=100)
    parser.add_argument("--num_train_net", type=int, default=-1)
    # train
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_train_num_pair", type=int, default=101) # includes test pair
    parser.add_argument("--max_train_num_pair", type=int, default=101) # includes test pair
    # eval
    parser.add_argument("--num_eval_net", type=int, default=10000) # includes test pair
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="left")
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

    # model architecture
    parser.add_argument("--model_hidden_size", type=int, default=256) # 64, 128, 256
    parser.add_argument("--num_hidden_layers", type=int, default=12) # 3, 6, 12
    parser.add_argument("--num_attention_heads", type=int, default=8) # 2, 4, 8
    parser.add_argument("--intermediate_size", type=int, default=1024) # 2, 4, 8

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # override to debug stuff
    if args.debug:
        args.tag = 'test'
        args.wandb = False
        args.samples_per_epoch = 1024
        args.log_every = 1

    # check args
    assert min(args.min_train_num_pair, args.min_eval_num_pair) >= 3
    assert args.num_eval_net > 0

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
    model = MyLlamaModel(model_config)

    # for n, p in model.named_parameters(): print(n, p.numel())
    # print_trainable_parameters(model)
    # breakpoint()

    logger.info("Base models loaded.")

    # initialize program embeddings
    prior_embeddings = None
    program_embeddings = None
    if args.ntokens != -1:
        prior_embeddings = ProgramEmbeddings(
            embedding=torch.randn((args.ntokens, args.net_input_dim), device=accelerator.device, dtype=torch.float32)
        )
        program_embeddings = ProgramEmbeddings(
            embedding=torch.randn((args.ntokens, args.net_input_dim), device=accelerator.device, dtype=torch.float32)
        )
        logger.info("Prior & Program embeddings initialized.")

    # ensure require grad
    for param in model.parameters():
        assert param.requires_grad
    if prior_embeddings is not None:
        for param in prior_embeddings.parameters():
            param.requires_grad = True
    if program_embeddings is not None:
        for param in program_embeddings.parameters():
            param.requires_grad = True

    # number of parameters
    print_trainable_parameters(model)
    if prior_embeddings is not None:
        prior_embeddings_n_params = sum(p.numel() for p in prior_embeddings.parameters())
        logger.info(f'prior embedding params {three_commas(prior_embeddings_n_params)}')
    if program_embeddings is not None:
        program_embeddings_n_params = sum(p.numel() for p in program_embeddings.parameters())
        logger.info(f'program embedding params {three_commas(program_embeddings_n_params)}')

    # model size
    logger.info(f'model size {round(model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    if prior_embeddings is not None:
        logger.info(f'prior embeddings size {round(get_memory_footprint(prior_embeddings) / 1024 ** 3, 2)}GB')
    if program_embeddings is not None:
        logger.info(f'program embeddings size {round(get_memory_footprint(program_embeddings) / 1024 ** 3, 2)}GB')

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
        process_index=accelerator.process_index,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        min_num_pair=args.min_train_num_pair,
        max_num_pair=args.max_train_num_pair,
        debug_len=args.debug_len,
        num_workers=args.num_workers,
        net_input_dim=args.net_input_dim,
        net_hidden_dim=args.net_hidden_dim,
        groundtruth_nets=train_groundtruth_nets,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
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

    # Param groups
    other_params = [p for p in model.parameters()]
    prior_params = [param for param in prior_embeddings.parameters()] if prior_embeddings is not None else []
    program_params = [param for param in program_embeddings.parameters()] if program_embeddings is not None else []

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

    # Prepare with accelerator
    (
        model,
        prior_embeddings,
        program_embeddings,
        optimizer,
        train_loader,
    ) = accelerator.prepare(
        model,
        prior_embeddings,
        program_embeddings,
        optimizer,
        train_loader,
    )

    start_epoch = 0
    global_step = 0

    assert isinstance(model, (nn.Module, DistributedDataParallel))
    if program_embeddings is not None:
        assert isinstance(program_embeddings, (ProgramEmbeddings, DistributedDataParallel))
    if prior_embeddings is not None:
        assert isinstance(prior_embeddings, (ProgramEmbeddings, DistributedDataParallel))

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
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Total Train Steps")

    # model saving
    last_save_model_path = None
    epoch_to_eval_loss = {}

    # train!
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        if prior_embeddings is not None:
            prior_embeddings.train()
        if program_embeddings is not None:
            program_embeddings.train()

        loss_accum = 0.0
        grad_norm_accum = 0.0

        train_dataset.set_rngs(epoch)
        for batch_data in train_loader:
            inputs_embeds = batch_data["inputs_embeds"].to(accelerator.device)
            attention_mask = batch_data["attention_mask"].to(accelerator.device)
            labels = batch_data["labels"]
            inputs_embeds_lens = batch_data["inputs_embeds_lens"]

            with accelerator.accumulate(model, prior_embeddings, program_embeddings):
                with accelerator.autocast():
                    loss = model_loss(
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        inputs_embeds_lens=inputs_embeds_lens,
                        ntokens=args.ntokens,
                        pad_side=args.pad_side,
                        attention_cutoff=args.attention_cutoff,
                        attend_prev_programs=args.attend_prev_programs,
                        individual_loss=False,
                    )
                    assert isinstance(loss, torch.Tensor)

                loss_accum += loss.item() / args.grad_accum_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm_accum += accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item() / args.grad_accum_steps # type: ignore
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    progress_bar.update(args.log_every)
                    try:
                        accelerator.log({
                            "train/loss": loss_accum,
                            "train/grad_norm": grad_norm_accum,
                            "train/lr_prior": lr_scheduler.get_last_lr()[0],
                            "train/lr_program": lr_scheduler.get_last_lr()[1],
                            "train/lr_other": lr_scheduler.get_last_lr()[2],
                        }, step=global_step)
                    except:
                        logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                loss_accum = 0.0
                grad_norm_accum = 0.0

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
            torch.cuda.empty_cache()
            gc.collect()

            train_loss = None
            if eval_train_dataset is not None:
                train_loss = evaluate(
                    model=model,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    dataset=eval_train_dataset,
                    accelerator=accelerator,
                    batch_size=args.eval_batch_size,
                    collate_fn=eval_collate_fn,
                    gs_iters=args.train_gs_iters,
                    gs_lr=args.train_gs_lr,
                    gs_beta1=args.train_gs_beta1,
                    gs_beta2=args.train_gs_beta2,
                    gs_batch_size=args.train_gs_batch_size,
                    gs_optimizer=args.train_gs_optimizer,
                    gs_max_grad_norm=args.train_gs_max_grad_norm,
                    gs_lr_scheduler=args.train_gs_lr_scheduler,
                    gs_take_best=args.train_gs_take_best,
                    attention_cutoff=args.attention_cutoff,
                    attend_prev_programs=args.attend_prev_programs,
                )
            eval_loss = evaluate(
                model=model,
                prior_embeddings=prior_embeddings,
                program_embeddings=program_embeddings,
                dataset=eval_eval_dataset,
                accelerator=accelerator,
                batch_size=args.eval_batch_size,
                collate_fn=eval_collate_fn,
                gs_iters=args.eval_gs_iters,
                gs_lr=args.eval_gs_lr,
                gs_beta1=args.eval_gs_beta1,
                gs_beta2=args.eval_gs_beta2,
                gs_batch_size=args.eval_gs_batch_size,
                gs_optimizer=args.eval_gs_optimizer,
                gs_max_grad_norm=args.eval_gs_max_grad_norm,
                gs_lr_scheduler=args.eval_gs_lr_scheduler,
                gs_take_best=args.eval_gs_take_best,
                attention_cutoff=args.attention_cutoff,
                attend_prev_programs=args.attend_prev_programs,
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
                        save_model_path, save_prior_embeddings_path, save_program_embeddings_path = last_save_model_path
                        rm_cmd = f"rm -rf {save_model_path}"
                        if save_prior_embeddings_path is not None:
                            rm_cmd += f" {save_prior_embeddings_path}"
                        if save_program_embeddings_path is not None:
                            rm_cmd += f" {save_program_embeddings_path}"
                        os.system(rm_cmd)
                    last_save_model_path = save_train_model(
                        model=model,
                        prior_embeddings=prior_embeddings,
                        program_embeddings=program_embeddings,
                        output_dir=args.output_dir,
                        epoch=epoch,
                    )
                epoch_to_eval_loss[epoch] = eval_loss


    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
