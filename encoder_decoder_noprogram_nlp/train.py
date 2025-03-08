import shutil
import wandb
from collections import Counter
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
    pad_sequence_left,
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

HR_TASKS = set(["piqa", "hate_speech_offensive", "google_wellformed_query", "social_i_qa", "circa", "quoref",
                    "glue-sst2", "scitail", "emo", "cosmos_qa", "freebase_qa", "ag_news", "art", "paws", "kilt_ay2",
                    "glue-qnli", "quail", "ade_corpus_v2-classification", "sciq", "hatexplain", "emotion", "glue-qqp",
                    "kilt_fever", "kilt_nq", "dbpedia_14", "kilt_zsre", "hellaswag", "squad-with_context", "hotpot_qa",
                    "glue-mnli", "ropes", "squad-no_context", "kilt_hotpotqa", "discovery", "superglue-record",
                    "race-middle", "race-high", "lama-trex", "swag", "gigaword", "amazon_polarity", "biomrc", "tab_fact",
                    "tweet_eval-emoji", "tweet_eval-offensive", "tweet_eval-sentiment", "tweet_qa", "imdb",
                    "lama-conceptnet", "liar", "anli", "wiki_qa", "kilt_trex", "wikisql", "wino_grande", "wiqa",
                    "search_qa", "xsum", "yahoo_answers_topics", "yelp_polarity", "yelp_review_full"])
LR_TASKS = set(["quarel", "financial_phrasebank", "openbookqa", "codah", "qasc", "glue-mrpc", "dream", "sick",
                "commonsense_qa", "medical_questions_pairs", "quartz-with_knowledge", "poem_sentiment",
                "quartz-no_knowledge", "glue-wnli", "climate_fever", "ethos-national_origin", "ethos-race",
                "ethos-religion", "ai2_arc", "hate_speech18", "glue-rte", "superglue-cb", "superglue-copa",
                "tweet_eval-hate", "tweet_eval-stance_atheism", "tweet_eval-stance_feminist"])
CLASSIFICATION_TASKS = set(["superglue-rte", "tweet_eval-sentiment", "discovery", "glue-rte", "superglue-wsc",
                            "glue-mrpc", "tweet_eval-stance_hillary", "tweet_eval-offensive", "emotion", "hatexplain",
                            "glue-cola", "sick", "paws", "ethos-sexual_orientation", "glue-qqp", "tweet_eval-emotion",
                            "sms_spam", "health_fact", "glue-mnli", "imdb", "ethos-disability", "glue-wnli", "scitail",
                            "trec", "yahoo_answers_topics", "liar", "glue-sst2", "tweet_eval-stance_abortion", "circa",
                            "tweet_eval-stance_climate", "glue-qnli", "tweet_eval-emoji", "ethos-directed_vs_generalized",
                            "ade_corpus_v2-classification", "hate_speech_offensive", "superglue-wic",
                            "google_wellformed_query", "tweet_eval-irony", "ethos-gender", "onestop_english", "trec",
                            "rotten_tomatoes", "kilt_fever"])


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


def macro_f1_score(generated: torch.Tensor, groundtruth: torch.Tensor) -> float:
    gen_counter = Counter(generated.tolist())
    gt_counter = Counter(groundtruth.tolist())
    all_tokens = set(gen_counter.keys()).union(set(gt_counter.keys()))

    f1_scores = []
    for token in all_tokens:
        tp = min(gen_counter.get(token, 0), gt_counter.get(token, 0)) # true positive
        fp = gen_counter.get(token, 0) - tp # false positive
        fn = gt_counter.get(token, 0) - tp # false negative
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return macro_f1


def get_memory_footprint(module: nn.Module):
    return sum(p.nelement() * p.element_size() for p in module.parameters()) + \
        sum(p.nelement() * p.element_size() for p in module.buffers())


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
    ntokens: int,
    attention_cutoff: bool,
    attend_prev_programs: bool,
):
    model.eval()
    if prior_embeddings is not None:
        prior_embeddings.eval()
    if program_embeddings is not None:
        program_embeddings.eval()

    # get modules in case of DDP
    module = model.module if isinstance(model, DistributedDataParallel) else model
    embed_tokens = module.model.embed_tokens if hasattr(module.model, "embed_tokens") else module.model.model.embed_tokens

    # setup terminators and suppress warning
    module.generation_config.pad_token_id = dataset.tokenizer.pad_token_id

    distributed_state = PartialState()
    output_list = []
    macro_f1_list = []
    accuracy_list = []

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

            if dry_eval_run:
                continue

            # get tensors
            task_ids = batch["task_ids"]
            gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
            gen_output_ids = [x.to(accelerator.device) for x in batch["gen_output_ids"]]
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_input_ids_lens = batch["gen_input_ids_lens"]
            out_token_length = batch["out_token_length"]
            pair_start_idxs = batch["pair_start_idxs"]

            # compute accuracy
            with accelerator.autocast():
                arbitrary_increase = 5
                if program_embeddings is None:
                    assert prior_embeddings is None
                    # generate with no program
                    gen_tokens_padded = module.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max(out_token_length) + arbitrary_increase, # arbitrary increase
                        num_return_sequences=1,
                        temperature=1.0,
                        top_p=1.0,
                        do_sample=False,
                        eos_token_id=[dataset.tokenizer.eos_token_id],
                    )
                    gen_tokens_padded = gen_tokens_padded[:, gen_input_ids.shape[1]:]
                else:
                    assert prior_embeddings is not None
                    # generate with no program
                    device, dtype = gen_input_ids.device, gen_input_ids.dtype
                    max_num_pairs = max(len(start_idxs) for start_idxs in pair_start_idxs)
                    pad_embeds = embed_tokens(torch.tensor(dataset.tokenizer.pad_token_id, device=device)) # (hidden_dim,)
                    gen_inputs_embeds = embed_tokens(gen_input_ids)

                    # insert program embeddings
                    inputs_embeds_with_programs = []
                    attention_mask_with_programs = []
                    program_intervals = []

                    for task_input_ids, task_inputs_embeds, task_attention_mask, input_ids_len, start_idxs in zip(gen_input_ids, gen_inputs_embeds, gen_attention_mask, gen_input_ids_lens, pair_start_idxs):
                        assert start_idxs[0] == 1 # first pair starts after bos
                        # start_idxs are offset if padding side is left
                        start_idxs = [s + task_input_ids.shape[0] - input_ids_len for s in start_idxs]

                        # insert program token before every pair
                        task_inputs_embeds_with_programs = [task_inputs_embeds[:start_idxs[0]]]
                        task_attention_mask_with_programs = [task_attention_mask[:start_idxs[0]]]
                        task_program_intervals = []

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

                        # some programs in batch have fewer pairs, pad manually, so hacky
                        pad_length = (max_num_pairs - len(start_idxs)) * ntokens
                        task_pad_inputs_embeds = pad_embeds[None, ...].expand(pad_length, -1)
                        task_pad_attention_mask = torch.full((pad_length,), 0, device=device, dtype=dtype)
                        task_inputs_embeds_with_programs.insert(0, task_pad_inputs_embeds)
                        task_attention_mask_with_programs.insert(0, task_pad_attention_mask)
                        task_program_intervals = [(x[0] + pad_length, x[1] + pad_length) for x in task_program_intervals]

                        # concat all
                        inputs_embeds_with_programs.append(torch.cat(task_inputs_embeds_with_programs))
                        attention_mask_with_programs.append(torch.cat(task_attention_mask_with_programs))
                        program_intervals.append(task_program_intervals)

                    # stack and check, now we have the three inputs with programs
                    inputs_embeds_with_programs = torch.stack(inputs_embeds_with_programs)
                    attention_mask_with_programs = torch.stack(attention_mask_with_programs)
                    assert inputs_embeds_with_programs.shape[1] == gen_input_ids.shape[1] + max_num_pairs * ntokens
                    assert inputs_embeds_with_programs.shape[:2] == attention_mask_with_programs.shape[:2]

                    # debug: assert programs are programs based on stored intervals
                    for embs, attn, intervals in zip(inputs_embeds_with_programs, attention_mask_with_programs, program_intervals):
                        for a, b in intervals:
                            assert torch.equal(embs[a:b], program_embeddings('dummy'))
                            assert attn[a:b].sum() == attn[a:b].numel()

                    # debug: assert no middle padding
                    assert set(torch.unique(attention_mask_with_programs).tolist()).issubset({0, 1})
                    assert torch.all(attention_mask_with_programs[:, :-1] <= attention_mask_with_programs[:, 1:])

                    # generate
                    if not attention_cutoff:
                        gen_tokens_padded = module.generate(
                            inputs_embeds=inputs_embeds_with_programs,
                            attention_mask=attention_mask_with_programs,
                            max_new_tokens=max(out_token_length) + arbitrary_increase, # arbitrary increase
                            num_return_sequences=1,
                            temperature=1.0,
                            top_p=1.0,
                            do_sample=False,
                            eos_token_id=[dataset.tokenizer.eos_token_id],
                        )
                    else:
                        gen_tokens_padded = module.generate(
                            inputs_embeds=inputs_embeds_with_programs,
                            attention_mask=attention_mask_with_programs,
                            max_new_tokens=max(out_token_length) + arbitrary_increase, # arbitrary increase
                            num_return_sequences=1,
                            temperature=1.0,
                            top_p=1.0,
                            do_sample=False,
                            eos_token_id=[dataset.tokenizer.eos_token_id],
                            program_intervals=program_intervals,
                            attend_prev_programs=attend_prev_programs,
                        )

                # remove pads
                gen_tokens = []
                for t, l in zip(gen_tokens_padded, out_token_length):
                    gen_tokens.append(t[:l + arbitrary_increase])

            # compute metrics
            assert len(gen_tokens) == len(gen_output_ids) == len(task_ids)
            for task, gen, gt in zip(task_ids, gen_tokens, gen_output_ids):
                # metric
                if task in CLASSIFICATION_TASKS:
                    accuracy = int(gen.tolist() == gt[:-1].tolist()) # gt includes eos
                    macro_f1_list.append(-1.0)
                    accuracy_list.append(accuracy)
                else:
                    macro_f1 = macro_f1_score(gen, gt[:-1]) # gt includes eos
                    assert 0 <= macro_f1 <= 1
                    macro_f1_list.append(macro_f1)
                    accuracy_list.append(-1.0)

            # log output
            gen_texts = dataset.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            gt_texts = dataset.tokenizer.batch_decode(gen_output_ids, skip_special_tokens=True)
            assert len(gen_texts) == len(gt_texts) == len(task_ids)
            for gen_text, gt_text, task_id in zip(gen_texts, gt_texts, task_ids):
                output_list.append({
                    'task_id': task_id,
                    'gen_text': gen_text,
                    'gt_text': gt_text,
                })

    distributed_state.wait_for_everyone()
    # results
    output_list = gather_object(output_list)
    macro_f1_list = gather_object(macro_f1_list)
    accuracy_list = gather_object(accuracy_list)

    assert len(output_list) == len(dataset), (len(output_list), len(dataset))
    assert len(macro_f1_list) == len(dataset), (len(macro_f1_list), len(dataset))
    assert len(accuracy_list) == len(dataset), (len(accuracy_list), len(dataset))

    # average metrics
    macro_f1_list = [x for x in macro_f1_list if x != -1.0]
    accuracy_list = [x for x in accuracy_list if x != -1.0]
    assert len(macro_f1_list) + len(accuracy_list) == len(dataset)
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list) if len(macro_f1_list) > 0 else 0
    accuracy = sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else 0

    # grab all results
    task_id_to_texts = defaultdict(list)
    for output in output_list:
        task_id_to_texts[output['task_id']].append((output['gen_text'], output['gt_text']))

    return macro_f1, accuracy, task_id_to_texts


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
    program_loss_lambda_scheduler: LambdaScheduler,
    global_step: int,
    program_type: str,
    attention_cutoff: bool,
    attend_prev_programs: bool,
    debug_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # original baseline
    if program_embeddings is None:
        assert prior_embeddings is None
        ce_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
        ).loss
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
        assert start_idxs[0] == 1 # first pair starts after bos
        # start_idxs are offset if padding side is left
        start_idxs = [s + task_input_ids.shape[0] - input_ids_len for s in start_idxs]

        # insert program token before every pair
        task_inputs_embeds_with_programs = [task_inputs_embeds[:start_idxs[0]]]
        task_attention_mask_with_programs = [task_attention_mask[:start_idxs[0]]]
        task_label_ids_with_programs = [task_label_ids[:start_idxs[0]]]
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
        for m in task_attention_mask_with_programs[1:]: assert m.sum() == m.numel()

        # some programs in batch have fewer pairs, pad manually, so hacky
        pad_length = (max_num_pairs - len(start_idxs)) * ntokens
        task_pad_inputs_embeds = pad_embeds[None, ...].expand(pad_length, -1)
        task_pad_attention_mask = torch.full((pad_length,), 0, device=device, dtype=dtype)
        task_pad_label_ids = torch.full((pad_length,), -100, device=device, dtype=dtype)
        task_inputs_embeds_with_programs.insert(0, task_pad_inputs_embeds)
        task_attention_mask_with_programs.insert(0, task_pad_attention_mask)
        task_label_ids_with_programs.insert(0, task_pad_label_ids)
        task_program_intervals = [(x[0] + pad_length, x[1] + pad_length) for x in task_program_intervals]

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
        assert torch.all(attention_mask_with_programs[:, :-1] <= attention_mask_with_programs[:, 1:])

    if not attention_cutoff:
        model_out = model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            labels=label_ids_with_programs,
            output_hidden_states=(program_type != 'none'),
        )
    else:
        model_out = model(
            inputs_embeds=inputs_embeds_with_programs,
            attention_mask=attention_mask_with_programs,
            labels=label_ids_with_programs,
            output_hidden_states=(program_type != 'none'),
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
            torch.cat([pad_embeds[None, ...].expand(max_pair_len - x.shape[0], -1), x], dim=0)
            for x in select_pair_inputs_embeds
        ])
        select_pair_attention_mask = pad_sequence_left(select_pair_attention_mask, padding_value=0)
        select_pair_label_ids = pad_sequence_left(select_pair_label_ids, padding_value=-100)
        # time to insert
        program_len = select_program.shape[1]
        select_pair_inputs_embeds = insert_based_on_sides(
            data=select_pair_inputs_embeds,
            to_insert=select_program,
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_id=pad_embeds,
        )
        select_pair_attention_mask = insert_based_on_sides(
            data=select_pair_attention_mask,
            to_insert=torch.full((batch_size, program_len), 1, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_id=0,
        )
        select_pair_label_ids = insert_based_on_sides(
            data=select_pair_label_ids,
            to_insert=torch.full((batch_size, program_len), -100, device=device, dtype=dtype),
            lens=pair_input_ids_lens,
            insert_side="left",
            pad_id=-100,
        )
        # debug: no attention middle padding
        assert set(torch.unique(select_pair_attention_mask).tolist()).issubset({0, 1})
        assert torch.all(select_pair_attention_mask[:, :-1] <= select_pair_attention_mask[:, 1:])
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
        pad_id: Union[int, torch.Tensor],
    ) -> torch.Tensor:

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
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--tracker_project_name", type=str, default="metaicl")
    parser.add_argument("--save_all_models", action="store_true") # otherwise save only best

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_len", type=int, default=-1)
    parser.add_argument("--debug_random_pad", action="store_true")
    parser.add_argument("--debug_pad_len", type=int, default=-1)
    parser.add_argument("--debug_no_resume", action='store_true')

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--loss_type", type=str, choices=['only_last', 'all', 'exclude_first'], default='exclude_first')

    # program loss
    parser.add_argument("--program_type", type=str, choices=["none", "random", "concat"], default="none")

    # Gist/thinking tokens
    parser.add_argument("--ntokens", type=int, default=-1)
    parser.add_argument("--attention_cutoff", action="store_true")
    parser.add_argument("--attend_prev_programs", action="store_true")

    # Training
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr_embedding", type=float, default=1e-6)
    parser.add_argument("--lr_program", type=float, default=1e-5)
    parser.add_argument("--lr_prior", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=4)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")
    parser.add_argument("--dry_train_run", action="store_true")
    parser.add_argument("--dry_eval_run", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=0)

    # scheduled extra losses
    parser.add_argument("--program_loss_lambda", type=float, default=1.0)
    parser.add_argument("--program_loss_offset_epochs", type=int, default=0)
    parser.add_argument("--program_loss_linear_epochs", type=int, default=0)

    # both data
    parser.add_argument("--delimiter", type=str, default=' ')
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_pair", type=int, default=4) # includes test pair
    parser.add_argument("--eval_n_sample_per_task", type=int, default=250)
    parser.add_argument("--max_seq_len", type=int, default=512)

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
        args.samples_per_epoch = 32
        args.log_every = 1

    # check args
    if not args.lora:
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
    tokenizer.padding_side = 'left'
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

    base_model = MyLlamaForCausalLM.from_pretrained(
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
    if args.ntokens > 0:
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
    if not args.lora:
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

    # load dataset and figure out splits
    dataset = datasets.load_dataset("bigheiniuJ/EvalMetaICLAll")
    train_split: Dataset = dataset['meta_train'] # type: ignore
    test_split: Dataset = dataset['meta_eval_100shot'] # type: ignore
    assert not set(HR_TASKS).intersection(set(LR_TASKS))
    # filter and remove useless columns
    train_split = train_split.filter(lambda example: example["task"] in HR_TASKS)
    test_split = test_split.filter(lambda example: example["task"] in LR_TASKS)
    train_split = train_split.remove_columns(["seed", "split"])
    test_split = test_split.remove_columns(["split"])
    # # print counts
    # logger.info('train task sample count')
    # logger.info(f"{pprint.pformat(Counter(train_split['task']), indent=4)}")
    # logger.info('')
    # logger.info('test task sample count')
    # logger.info(f"{pprint.pformat(Counter(test_split['task']), indent=4)}")
    # logger.info('')

    # Build training dataset
    train_dataset = TrainDataset(
        data=train_split, # type: ignore
        tokenizer=tokenizer,
        total_steps=args.samples_per_epoch,
        seed=args.seed,
        process_index=accelerator.process_index,
        ntokens=args.ntokens,
        num_pair=args.num_pair,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        delimiter=args.delimiter,
        debug_random_pad=args.debug_random_pad,
        debug_len=args.debug_len,
        debug_pad_len=args.debug_pad_len,
        loss_type=args.loss_type,
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
        logger.info("Loading checkpoint from", recovery_checkpoint_dir)
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

    # Build evaluation datasets
    seeds = sorted(set(test_split['seed']))
    seed_to_test_split = [(seed, test_split.filter(lambda example: example['seed'] == seed).remove_columns('seed'))
                          for seed in seeds]
    eval_datasets = [
        EvalDataset(
            data=data, # type: ignore
            seed=args.seed,
            split_name=split_name,
            tokenizer=tokenizer,
            ntokens=args.ntokens,
            num_pair=args.num_pair,
            max_seq_len=args.max_seq_len,
            delimiter=args.delimiter,
            eval_n_sample_per_task=args.eval_n_sample_per_task,
            debug_len=args.debug_len,
            debug_random_pad=args.debug_random_pad,
            debug_pad_len=args.debug_pad_len,
        )
        for split_name, data in seed_to_test_split
    ]
    assert len(set(tuple(sorted(data.task_to_indices.keys())) for data in eval_datasets)) == 1 # all eval splits same tasks
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_datasets[0]) # only use tokenizer, padding info
    if args.debug_len > 0:
        eval_collate_fn = partial(collate_fn_eval_dummy, dataset=eval_datasets[0])

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
        # Only show the progress bar once on each machine.
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

    # train!
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        if prior_embeddings is not None:
            prior_embeddings.train()
        if program_embeddings is not None:
            program_embeddings.train()

        ce_loss_accum = 0.0
        program_loss_accum = 0.0
        total_loss_accum = 0.0
        grad_norm_accum = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # skip batch idx if recovered run already encountered it
            if batch_idx < resume_batch_idx:
                continue

            input_ids = batch_data["input_ids"].to(accelerator.device)
            attention_mask = batch_data["attention_mask"].to(accelerator.device)
            label_ids = batch_data["label_ids"].to(accelerator.device)
            input_ids_lens = batch_data["input_ids_lens"]
            pair_start_idxs = batch_data["pair_start_idxs"]
            assert len(input_ids) == args.train_batch_size
            assert input_ids.shape == attention_mask.shape == label_ids.shape

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
                        program_loss_lambda_scheduler=program_loss_lambda_scheduler,
                        global_step=global_step,
                        program_type=args.program_type,
                        attention_cutoff=args.attention_cutoff,
                        attend_prev_programs=args.attend_prev_programs,
                        debug_len=args.debug_len,
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

            macro_f1s, accuracys, total_scores = [], [], []
            all_task_id_to_texts = defaultdict(list)
            for eval_dataset in eval_datasets:
                macro_f1, accuracy, task_id_to_texts = evaluate(
                    model=model,
                    prior_embeddings=prior_embeddings,
                    program_embeddings=program_embeddings,
                    dataset=eval_dataset,
                    accelerator=accelerator,
                    batch_size=args.eval_batch_size,
                    collate_fn=eval_collate_fn,
                    dry_eval_run=args.dry_eval_run,
                    ntokens=args.ntokens,
                    attention_cutoff=args.attention_cutoff,
                    attend_prev_programs=args.attend_prev_programs,
                )
                total_score = (macro_f1 + accuracy) / 2.0
                # aggregate
                macro_f1s.append(macro_f1)
                accuracys.append(accuracy)
                total_scores.append(total_score)
                for task_id, texts in task_id_to_texts.items():
                    texts = [(eval_dataset.split_name, t[0], t[1]) for t in texts]
                    all_task_id_to_texts[task_id] += texts

                torch.cuda.empty_cache()
                gc.collect()

            # print for debugging
            logger.info(f"macro_f1s: {macro_f1s}")
            logger.info(f"accuracys: {accuracys}")
            logger.info(f"total_scores: {total_scores}")

            macro_f1 = sum(macro_f1s) / len(macro_f1s)
            accuracy = sum(accuracys) / len(accuracys)
            total_score = sum(total_scores) / len(total_scores)

            if accelerator.is_main_process:
                eval_metric_dict = {
                    "eval/macro_f1": macro_f1,
                    "eval/accuracy": accuracy,
                    "eval/total_score": total_score,
                }
                logger.info(f'Evaluation results:\n{pprint.pformat(eval_metric_dict, indent=4)}')
                try:
                    accelerator.log(eval_metric_dict, step=global_step)
                except:
                    logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                # merge and save outputs
                output_dict_path = os.path.join(args.output_dir, f"{epoch+1}_output_list.json")
                with open(output_dict_path, 'w') as f:
                    json.dump(all_task_id_to_texts, f)
                logger.info(f"Saved output list to {output_dict_path}")

                # Save model
                do_save_model = args.save_all_models
                if not args.save_all_models:
                    if (not epoch_to_total_score) or total_score >= max(epoch_to_total_score.values()):
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
                epoch_to_total_score[epoch] = total_score

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
