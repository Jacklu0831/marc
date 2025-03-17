import numpy as np
import shutil
import wandb
import gc
import matplotlib.pyplot as plt
from custom_llama import MyLlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
from collections import defaultdict
from typing import Union, Callable, List, Tuple, Optional, Iterator
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
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    AutoModelForCausalLM,
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
    TrainDataset,
    EvalDataset,
    collate_fn_train,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
    pad_sequence_with_side,
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
    "gpt2": "openai-community/gpt2-large",
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
    dry_eval_run: bool,
    attention_cutoff: bool,
    attend_prev_programs: bool,
    log_every: int,
) -> Tuple[float, List]:

    model.eval()
    if prior_embeddings is not None:
        prior_embeddings.eval()
    if program_embeddings is not None:
        program_embeddings.eval()

    # get modules in case of DDP
    module = model.module if isinstance(model, DistributedDataParallel) else model

    # setup terminators and suppress warning
    module.generation_config.pad_token_id = dataset.tokenizer.pad_token_id # type: ignore

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
            bs = len(batch_data)
            batch = collate_fn(batch_data)

            if dry_eval_run:
                continue

            # get tensors
            task = batch['task']
            test_idx = batch['test_idx']
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            label_ids = batch["label_ids"].to(accelerator.device)
            input_ids_lens = batch["input_ids_lens"]
            pair_start_idxs = batch["pair_start_idxs"]
            option = batch['option']
            correct_option = batch['correct_option']

            with accelerator.autocast():
                losses = model_loss(
                    model=module,
                    tokenizer=dataset.tokenizer,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label_ids=label_ids,
                    input_ids_lens=input_ids_lens,
                    pair_start_idxs=pair_start_idxs,
                    ntokens=dataset.ntokens,
                    pad_side=dataset.pad_side,
                    program_loss_lambda_scheduler=None,
                    global_step=0,
                    program_type='none',
                    attention_cutoff=attention_cutoff,
                    attend_prev_programs=attend_prev_programs,
                    debug=False,
                    individual_loss=True,
                )
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
        prior_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
        program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
        output_dir: str,
        epoch: int,
    ) -> Tuple[str, Optional[str], Optional[str]]:

    # model
    save_model_path = os.path.join(output_dir, f"lora_epoch_{epoch+1}")
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


def model_loss(
    # model
    model: Union[nn.Module, DistributedDataParallel],
    tokenizer: PreTrainedTokenizerFast,
    prior_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    program_embeddings: Optional[Union[ProgramEmbeddings, DistributedDataParallel]],
    # data
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,
    input_ids_lens: List[int],
    pair_start_idxs: List[List[int]],
    ntokens: int,
    # others
    pad_side: str,
    program_loss_lambda_scheduler: Optional[LambdaScheduler],
    global_step: int,
    program_type: str,
    attention_cutoff: bool,
    attend_prev_programs: bool,
    debug: bool,
    individual_loss: bool,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

    # original baseline
    if program_embeddings is None:
        assert prior_embeddings is None
        model_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
        )
        if individual_loss:
            return get_individual_loss(lm_logits=model_out.logits.half(), label_ids=label_ids)
        else:
            ce_loss = model_out.loss
            # print(ce_loss.item())
            # breakpoint()
            return ce_loss, torch.tensor(0.0, device=input_ids.device), ce_loss

    assert prior_embeddings is not None
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens
    batch_size = input_ids.shape[0]
    device, dtype = input_ids.device, input_ids.dtype
    max_num_pairs = max(len(start_idxs) for start_idxs in pair_start_idxs)

    # get embeddings
    pad_embeds = embed_tokens(torch.tensor(tokenizer.pad_token_id, device=device)) # (hidden_dim,)
    inputs_embeds = embed_tokens(input_ids)

    # loss lambdas
    program_loss_lambda = 0.0
    if program_loss_lambda_scheduler is not None:
        program_loss_lambda = program_loss_lambda_scheduler.get_lambda(step=global_step)

    # insert program embeddings
    inputs_embeds_with_programs = []
    attention_mask_with_programs = []
    label_ids_with_programs = []
    program_intervals = []

    # store pairwise inputs_embeds, attention_mask, and label_ids for program loss
    pair_inputs_embeds = []
    pair_attention_mask = []
    pair_label_ids = []

    for task_input_ids, task_inputs_embeds, task_attention_mask, task_label_ids, input_ids_len, start_idxs in zip(input_ids, inputs_embeds, attention_mask, label_ids, input_ids_lens, pair_start_idxs):
        assert start_idxs[0] == 0
        # start_idxs are offset if padding side is left
        if pad_side == "left":
            start_idxs = [s + task_input_ids.shape[0] - input_ids_len for s in start_idxs]

        # insert program token before every pair
        task_inputs_embeds_with_programs = [task_inputs_embeds[:start_idxs[0]]] if start_idxs[0] > 0 else []
        task_attention_mask_with_programs = [task_attention_mask[:start_idxs[0]]] if start_idxs[0] > 0 else []
        task_label_ids_with_programs = [task_label_ids[:start_idxs[0]]] if start_idxs[0] > 0 else []
        task_program_intervals = []

        # store pairwise inputs_embeds, attention_mask, and label_ids for program loss
        task_pair_inputs_embeds = []
        task_pair_attention_mask = []
        task_pair_label_ids = []

        for i, start_idx in enumerate(start_idxs):
            end_idx = start_idxs[i+1] if i < len(start_idxs) - 1 else len(task_inputs_embeds)
            # program intervals
            program_start = sum(len(x) for x in task_inputs_embeds_with_programs)
            task_program_intervals.append((program_start, program_start + ntokens))
            # prior or program
            embedding = prior_embeddings('dummy') if i == 0 else program_embeddings('dummy')
            # insert program embedding into inputs_embeds
            task_inputs_embeds_with_programs.append(embedding)
            task_inputs_embeds_with_programs.append(task_inputs_embeds[start_idx: end_idx])
            # insert full attention for programs
            task_attention_mask_with_programs.append(torch.full((ntokens,), 1, device=device, dtype=dtype))
            task_attention_mask_with_programs.append(task_attention_mask[start_idx: end_idx])
            # insert no label supervision for programs
            task_label_ids_with_programs.append(torch.full((ntokens,), -100, device=device, dtype=dtype))
            task_label_ids_with_programs.append(task_label_ids[start_idx: end_idx])

            task_pair_inputs_embeds.append(task_inputs_embeds[start_idx: end_idx])
            task_pair_attention_mask.append(task_attention_mask[start_idx: end_idx])
            task_pair_label_ids.append(task_label_ids[start_idx: end_idx])

        pair_inputs_embeds.append(task_pair_inputs_embeds)
        pair_attention_mask.append(task_pair_attention_mask)
        pair_label_ids.append(task_pair_label_ids)

        # full attention at this point for all beside the ends
        if pad_side == 'left':
            for m in task_attention_mask_with_programs[1:]: assert m.sum() == m.numel()
        else:
            for m in task_attention_mask_with_programs[:-1]: assert m.sum() == m.numel()

        # some programs in batch have fewer pairs, pad manually, so hacky
        pad_length = (max_num_pairs - len(start_idxs)) * ntokens
        task_pad_inputs_embeds = pad_embeds[None, ...].expand(pad_length, -1)
        task_pad_attention_mask = torch.full((pad_length,), 0, device=device, dtype=dtype)
        task_pad_label_ids = torch.full((pad_length,), -100, device=device, dtype=dtype)
        if pad_side == 'left':
            task_inputs_embeds_with_programs.insert(0, task_pad_inputs_embeds)
            task_attention_mask_with_programs.insert(0, task_pad_attention_mask)
            task_label_ids_with_programs.insert(0, task_pad_label_ids)
            task_program_intervals = [(x[0] + pad_length, x[1] + pad_length) for x in task_program_intervals]
        else:
            task_inputs_embeds_with_programs.append(task_pad_inputs_embeds)
            task_attention_mask_with_programs.append(task_pad_attention_mask)
            task_label_ids_with_programs.append(task_pad_label_ids)

        # concat all
        inputs_embeds_with_programs.append(torch.cat(task_inputs_embeds_with_programs))
        attention_mask_with_programs.append(torch.cat(task_attention_mask_with_programs))
        label_ids_with_programs.append(torch.cat(task_label_ids_with_programs))
        program_intervals.append(task_program_intervals)

    # stack and check, now we have the three inputs with programs
    inputs_embeds_with_programs = torch.stack(inputs_embeds_with_programs)
    attention_mask_with_programs = torch.stack(attention_mask_with_programs)
    label_ids_with_programs = torch.stack(label_ids_with_programs)
    assert inputs_embeds_with_programs.shape[1] == input_ids.shape[1] + max_num_pairs * ntokens
    assert inputs_embeds_with_programs.shape[:2] == attention_mask_with_programs.shape[:2] == label_ids_with_programs.shape[:2]

    # debug: assert programs are programs based on stored intervals
    for embs, attn, lab, intervals in zip(inputs_embeds_with_programs, attention_mask_with_programs, label_ids_with_programs, program_intervals):
        for a, b in intervals:
            assert torch.equal(embs[a:b], program_embeddings('dummy'))
            assert attn[a:b].sum() == attn[a:b].numel()
            assert (lab[a:b] == -100).sum() == lab[a:b].numel()

    # debug: assert no middle padding when we don't cut off attention
    if not attention_cutoff:
        assert set(torch.unique(attention_mask_with_programs).tolist()).issubset({0, 1})
        if pad_side == 'left':
            assert torch.all(attention_mask_with_programs[:, :-1] <= attention_mask_with_programs[:, 1:])
        else:
            assert torch.all(attention_mask_with_programs[:, :-1] >= attention_mask_with_programs[:, 1:])

    if not attention_cutoff:
        model_out = model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            labels=label_ids_with_programs,
            output_hidden_states=(program_type != 'none') or debug,
        )
    else:
        model_out = model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            labels=label_ids_with_programs,
            output_hidden_states=(program_type != 'none') or debug,
            program_intervals=program_intervals,
            attend_prev_programs=attend_prev_programs,
        )
    ce_loss = model_out.loss

    program_loss = torch.tensor(0.0, device=device)
    if program_type != 'none':
        hidden_states = model_out.hidden_states[-1] # (batch_size, seq_len, hidden_dim) # type: ignore
        # e.g., p0 x0 y0 p1 x1 y1 p2 x2 y2 (max_num_pairs=3)
        # get all programs and format
        programs = [] # batch_size x (num_pair, ntokens, hidden_dim)
        for task_program_intervals, task_hidden_states in zip(program_intervals, hidden_states):
            task_programs = torch.stack([task_hidden_states[start: end] for start, end in task_program_intervals])
            programs.append(task_programs)
        programs = torch.stack(programs).permute(1, 0, 2, 3) # (num_pair, batch_size, ntokens, hidden_dim)
        # select program and pair
        if program_type == 'concat':
            # concatenate all programs and a random pair
            select_program = torch.cat([x for x in programs], dim=1)
            select_idx = int(torch.randint(low=0, high=max_num_pairs, size=(1,)).item())
        else:
            # select random program and a random pair AFTER it
            program_idx = int(torch.randint(low=2, high=max_num_pairs, size=(1,)).item()) # do not select the first two
            select_program = programs[program_idx]
            select_idx = int(torch.randint(low=program_idx, high=max_num_pairs, size=(1,)).item())
        # format pair data
        select_pair_inputs_embeds = [x[select_idx] for x in pair_inputs_embeds]
        select_pair_attention_mask = [x[select_idx] for x in pair_attention_mask]
        select_pair_label_ids = [x[select_idx] for x in pair_label_ids]
        pair_input_ids_lens = [x.shape[0] for x in select_pair_inputs_embeds]
        max_pair_len = max(pair_input_ids_lens)
        # pad pair data
        select_pair_inputs_embeds = torch.stack([
            (
                torch.cat([pad_embeds[None, ...].expand(max_pair_len - x.shape[0], -1), x], dim=0)
                if pad_side == "left"
                else torch.cat([x, pad_embeds[None, ...].expand(max_pair_len - x.shape[0], -1)], dim=0)
            )
            for x in select_pair_inputs_embeds
        ])
        select_pair_attention_mask = pad_sequence_with_side(select_pair_attention_mask, padding_value=0, side=pad_side)
        select_pair_label_ids = pad_sequence_with_side(select_pair_label_ids, padding_value=-100, side=pad_side)
        # time to insert
        program_len = select_program.shape[1]
        select_pair_inputs_embeds = insert_based_on_sides(
            data=select_pair_inputs_embeds,
            to_insert=select_program,
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=pad_embeds,
        )
        select_pair_attention_mask = insert_based_on_sides(
            data=select_pair_attention_mask,
            to_insert=torch.full((batch_size, program_len), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=0,
        )
        select_pair_label_ids = insert_based_on_sides(
            data=select_pair_label_ids,
            to_insert=torch.full((batch_size, program_len), -100, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_side=pad_side,
            pad_id=-100,
        )
        # debug: no attention middle padding
        assert set(torch.unique(select_pair_attention_mask).tolist()).issubset({0, 1})
        if pad_side == 'left':
            assert torch.all(select_pair_attention_mask[:, :-1] <= select_pair_attention_mask[:, 1:])
        else:
            assert torch.all(select_pair_attention_mask[:, :-1] >= select_pair_attention_mask[:, 1:])
        assert select_pair_inputs_embeds.shape[:2] == select_pair_attention_mask.shape == select_pair_label_ids.shape
        # forward
        model_out = model(
            inputs_embeds=select_pair_inputs_embeds,
            attention_mask=select_pair_attention_mask,
            labels=select_pair_label_ids,
        )
        program_loss = model_out.loss
        program_loss /= max_num_pairs # normalize based on num pairs to not dominate

    total_loss = ce_loss + program_loss_lambda * program_loss

    return ce_loss, program_loss, total_loss


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2000)
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
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--loss_type", type=str, choices=['only_last', 'all', 'exclude_first'], default='only_last')

    # program loss
    parser.add_argument("--program_type", type=str, choices=["none", "random", "concat"], default="none")

    # Gist/thinking tokens
    parser.add_argument("--ntokens", type=int, default=-1)
    parser.add_argument("--attention_cutoff", action="store_true")
    parser.add_argument("--attend_prev_programs", action="store_true")

    # Training
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16) # 32 is fine, but dont want to risk
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

    # scheduled extra losses
    parser.add_argument("--program_loss_lambda", type=float, default=1.0)
    parser.add_argument("--program_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--program_loss_linear_epochs", type=int, default=0)

    # both data
    parser.add_argument("--config_file", type=str, default="MetaICL/config/hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="MetaICL/data")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--min_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--max_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--eval_min_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--allow_truncate", action='store_true')
    parser.add_argument("--max_pair_len", type=int, default=256)
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument('--eval_seeds', type=str, nargs="+", default=['100'])

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
                assert set(recovery_state_file.keys()) == {"run_id", "global_step", "batch_idx", "epoch"}
            else:
                assert set(recovery_state_file.keys()) == {"global_step", "batch_idx", "epoch"}
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

    if 'llama' in args.model_name:
        base_model = MyLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
            **from_pretrained_kwargs,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
            **from_pretrained_kwargs,
        )

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
    prior_embeddings = None
    program_embeddings = None
    if args.ntokens != -1:
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

    # ensure require grad
    if prior_embeddings is not None:
        for param in prior_embeddings.parameters():
            param.requires_grad = True
    if program_embeddings is not None:
        for param in program_embeddings.parameters():
            param.requires_grad = True

    # convert model weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if prior_embeddings is not None:
        for name, param in prior_embeddings.named_parameters():
            assert param.requires_grad
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if program_embeddings is not None:
        for name, param in program_embeddings.named_parameters():
            assert param.requires_grad
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

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
        loss_type=args.loss_type,
        debug_len=args.debug_len,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        max_pair_len=args.max_pair_len,
        allow_truncate=args.allow_truncate,
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

    # Param groups for LoRA
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "embed" in name or "lm_head" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    prior_params = [param for param in prior_embeddings.parameters()] if prior_embeddings is not None else []
    program_params = [param for param in program_embeddings.parameters()] if program_embeddings is not None else []

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
    program_loss_lambda_scheduler = LambdaScheduler(
        loss_lambda=args.program_loss_lambda,
        start_epoch=args.program_loss_offset_epochs,
        linear_epochs=args.program_loss_linear_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    # program_loss_lambda_scheduler.visualize(num_training_steps, 'program.jpg')

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
    if program_embeddings is not None:
        assert isinstance(program_embeddings, (ProgramEmbeddings, DistributedDataParallel))
    if prior_embeddings is not None:
        assert isinstance(prior_embeddings, (ProgramEmbeddings, DistributedDataParallel))

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
            allow_truncate=args.allow_truncate,
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
            allow_truncate=args.allow_truncate,
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
            if prior_embeddings is not None:
                prior_embeddings.train()
            if program_embeddings is not None:
                program_embeddings.train()

            ce_loss_accum = 0.0
            program_loss_accum = 0.0
            total_loss_accum = 0.0
            grad_norm_accum = 0.0

            train_dataset.set_rngs(epoch)
            for batch_idx, batch_data in enumerate(train_loader):
                # skip batch idx if recovered run already encountered it
                if batch_idx < resume_batch_idx:
                    continue

                input_ids = batch_data["input_ids"].to(accelerator.device)
                attention_mask = batch_data["attention_mask"].to(accelerator.device)
                label_ids = batch_data["label_ids"].to(accelerator.device)
                input_ids_lens = batch_data["input_ids_lens"]
                pair_start_idxs = batch_data["pair_start_idxs"]

                with accelerator.accumulate(model, prior_embeddings, program_embeddings):
                    with accelerator.autocast():
                        ce_loss, program_loss, total_loss = model_loss(
                            model=model,
                            tokenizer=tokenizer,
                            prior_embeddings=prior_embeddings,
                            program_embeddings=program_embeddings,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            label_ids=label_ids,
                            input_ids_lens=input_ids_lens,
                            pair_start_idxs=pair_start_idxs,
                            ntokens=args.ntokens,
                            pad_side=args.pad_side,
                            program_loss_lambda_scheduler=program_loss_lambda_scheduler,
                            global_step=global_step,
                            program_type=args.program_type,
                            attention_cutoff=args.attention_cutoff,
                            attend_prev_programs=args.attend_prev_programs,
                            debug=args.debug,
                            individual_loss=False,
                        )

                    ce_loss_accum += ce_loss.item() / args.grad_accum_steps
                    program_loss_accum += program_loss.item() / args.grad_accum_steps
                    total_loss_accum += total_loss.item() / args.grad_accum_steps

                    accelerator.backward(total_loss)
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
                                "train/ce_loss": ce_loss_accum,
                                "train/program_loss": program_loss_accum,
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
                    program_loss_accum = 0.0
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

            # Eval Train Datasets
            train_scores, train_all_output_list = [], None
            for dataset_i, dataset in enumerate(eval_train_datasets):
                score, output_list = evaluate(
                    model=model,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    dataset=dataset,
                    accelerator=accelerator,
                    batch_size=args.eval_batch_size,
                    collate_fn=eval_collate_fn,
                    dry_eval_run=args.dry_eval_run,
                    attention_cutoff=args.attention_cutoff,
                    attend_prev_programs=args.attend_prev_programs,
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
                    dataset=dataset,
                    accelerator=accelerator,
                    batch_size=args.eval_batch_size,
                    collate_fn=eval_collate_fn,
                    dry_eval_run=args.dry_eval_run,
                    attention_cutoff=args.attention_cutoff,
                    attend_prev_programs=args.attend_prev_programs,
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
                epoch_to_total_score[epoch] = eval_score

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
