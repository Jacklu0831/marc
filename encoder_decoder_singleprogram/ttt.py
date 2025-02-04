from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import gc
from multiprocessing import Pool
from pathlib import Path
import csv
from peft import PeftModel # type: ignore
from typing import Union, Optional, Tuple, Dict, List
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
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training  # type: ignore

from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

from data_utils import TTTDataset, collate_fn_ttt, collate_fn_ttt_dummy, ARCTokenizer
from train import (
    three_commas,
    encoder_decoder_loss,
    set_up_main_process_logger,
    LambdaScheduler,
)
from train import Prefix2PrefixProjection, Hidden2PromptProjection


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
    "llama3b_uncensored": "chuanli11/Llama-3.2-3B-Instruct-uncensored",
    "nemo8b": "nvidia/Mistral-NeMo-Minitron-8B-Base",
}
NBIT_TO_DTYPE = {
    16: torch.bfloat16,
    32: torch.float32,
}


def save_model_ttt(
        encoder_model: nn.Module,
        decoder_model: nn.Module,
        conditioning_projection: Union[Prefix2PrefixProjection, Hidden2PromptProjection],
        output_dir: str,
        task_id: str,
        epoch: int,
        tie_models: bool,
        lora_target_modules: List[str],
        trainable_nbit: int,
    ) -> Tuple[str, Optional[str], Optional[str]]:

    # save to output_dir/task_id and only save lora
    os.makedirs(os.path.join(output_dir, task_id), exist_ok=True)

    # encoder
    save_enc_path = os.path.join(output_dir, task_id, f"encoder_lora_epoch_{epoch+1}.pt")
    enc_lora_weights = get_lora_data(encoder_model, lora_target_modules, trainable_nbit)
    torch.save(enc_lora_weights, save_enc_path)
    logger.info(f"Saved encoder to {save_enc_path}")

    # decoder
    save_dec_path = None
    if not tie_models:
        save_dec_path = os.path.join(output_dir, task_id, f"decoder_lora_epoch_{epoch+1}.pt")
        dec_lora_weights = get_lora_data(decoder_model, lora_target_modules, trainable_nbit)
        torch.save(dec_lora_weights, save_dec_path)
        logger.info(f"Saved decoder to {save_dec_path}")

    # projection
    save_proj_path = os.path.join(output_dir, task_id, f"conditioning_projection_epoch_{epoch+1}.pt")
    conditioning_projection_module = conditioning_projection
    if isinstance(conditioning_projection, DistributedDataParallel):
        conditioning_projection_module = conditioning_projection.module
    torch.save(conditioning_projection_module, save_proj_path)
    logger.info(f"Saved conditioning projection to {save_proj_path}")

    return save_enc_path, save_dec_path, save_proj_path


def get_lora_data(model: nn.Module, lora_target_modules: List[str], trainable_nbit: int) -> Dict[str, torch.Tensor]:
    model = model.module if isinstance(model, DistributedDataParallel) else model
    name_to_weight = {}
    for name, param in model.named_parameters():
        if "lora" in name and any(t in name for t in lora_target_modules):
            assert param.data.dtype == NBIT_TO_DTYPE[trainable_nbit]
            name_to_weight[name] = param.data
    return name_to_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs_ttt")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--tracker_project_name", type=str, default="arc")

    # Debug
    parser.add_argument("--debug_no_aug", action="store_true")
    parser.add_argument("--debug_enc_len", type=int, default=-1)
    parser.add_argument("--debug_dec_len", type=int, default=-1)

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, default="test_evaluation")
    parser.add_argument("--weight_epoch", type=int, default=1)

    # Conditioning projection
    parser.add_argument("--conditioning_method", type=str, choices=["prefix2prefix", "hidden2prompt"], default="hidden2prompt")

    # Training
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--save_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_samples_per_task", type=int, default=250)
    parser.add_argument("--max_seq_len", type=int, default=5120)
    parser.add_argument("--encoder_loss_type", type=str, choices=["last", "rest", "all"], default="rest")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # both data
    parser.add_argument("--data_dir", type=str, default="/scratch/yl11330/re-arc/train_data/tasks")
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--permute_n", type=int, default=1)

    # scheduled extra losses
    parser.add_argument("--encoder_loss_lambda", type=float, default=1.0)
    parser.add_argument("--linear_encoder", action="store_true")
    parser.add_argument("--kl_loss_lambda", type=float, default=1.0)
    parser.add_argument("--linear_kl", action="store_true")

    # lora
    parser.add_argument("--full_lora", action="store_true")

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # check args
    if args.conditioning_method == "prefix2prefix":
        assert not args.encoder_gradient_checkpointing
        assert not args.decoder_gradient_checkpointing
        assert args.decoder_pad_side == "right" # right for no middle padding
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
        assert args.encoder_gradient_checkpointing == args.decoder_gradient_checkpointing
    if args.debug_enc_len > -1 and args.debug_dec_len == -1:
        args.debug_dec_len = args.debug_enc_len // 2
    assert (args.encoder_name == "nemo8b") == (args.decoder_name == "nemo8b")
    if args.encoder_name == "nemo8b":
        assert args.encoder_pad_side == args.decoder_pad_side == args.decoder_gen_pad_side == "left"

    assert args.trainable_nbit == 16 # TODO: test 32
    assert not args.wandb # TODO: support this without crashing wandb

    args.tag = f"ttt_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    lora_target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    if args.full_lora:
        lora_target_modules += ["k_proj", "o_proj"]

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
        if args.wandb:
            wandb.define_metric('Steps')
            wandb.define_metric("*", step_metric="Steps")
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info("Accelerator and seed set up.")

    # log args
    logger.info("#### BEGIN ALL ARGUMENTS ####")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("#### END ALL ARGUMENTS ####\n")

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
        "low_cpu_mem_usage": True
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
        base_encoder = prepare_model_for_kbit_training(
            base_encoder,
            use_gradient_checkpointing=args.encoder_gradient_checkpointing
        )
        if not args.tie_models:
            base_decoder = prepare_model_for_kbit_training(
                base_decoder,
                use_gradient_checkpointing=args.decoder_gradient_checkpointing
            )
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

    # only keep these tokens, resize model embedding
    decoder_keep_tokens = [str(i) for i in range(10)] + \
        [decoder_tokenizer.bos_token, decoder_tokenizer.eos_token, "\n", "input", "output"] # eos == pad
    encoder_keep_tokens = decoder_keep_tokens + cls_tokens
    assert len(set(encoder_keep_tokens)) == len(encoder_keep_tokens)
    assert len(set(decoder_keep_tokens)) == len(decoder_keep_tokens)
    if args.tie_models:
        decoder_keep_tokens = encoder_keep_tokens

    # this breaks embedding tying, but whatever
    with torch.no_grad():
        encoder_keep_token_ids = []
        for token in encoder_keep_tokens:
            token_id = encoder_tokenizer(token)["input_ids"] # type: ignore
            assert isinstance(token_id, list)
            assert len(token_id) == 2 # with start token
            encoder_keep_token_ids.append(token_id[1])
        decoder_keep_token_ids = []
        for token in decoder_keep_tokens:
            token_id = decoder_tokenizer(token)["input_ids"] # type: ignore
            assert isinstance(token_id, list)
            assert len(token_id) == 2 # with start token
            decoder_keep_token_ids.append(token_id[1])
        assert len(set(encoder_keep_token_ids)) == len(encoder_keep_token_ids)
        assert len(set(decoder_keep_token_ids)) == len(decoder_keep_token_ids)

        # subset embeddings and lmheads
        base_encoder.model.embed_tokens.weight = nn.Parameter(base_encoder.model.embed_tokens.weight[encoder_keep_token_ids])
        base_encoder.model.embed_tokens.num_embeddings = len(encoder_keep_token_ids)
        assert base_encoder.lm_head.bias is None
        base_encoder.lm_head.weight = nn.Parameter(base_encoder.lm_head.weight[encoder_keep_token_ids])
        base_encoder.lm_head.out_features = len(encoder_keep_token_ids)
        base_encoder.config.tie_word_embeddings = False

        # subset embeddings and lmheads
        if not args.tie_models:
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
        eos_token=decoder_tokenizer.eos_token,
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

    # load encoder decoder weights, projection later
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    enc_weight_path = os.path.join(weight_dir, f"encoder_lora_epoch_{args.weight_epoch}")
    enc_lmhead_path = os.path.join(weight_dir, f"encoder_lmhead_epoch_{args.weight_epoch}.pt")
    enc_embeds_path = os.path.join(weight_dir, f"encoder_embeds_epoch_{args.weight_epoch}.pt")
    dec_weight_path = os.path.join(weight_dir, f"decoder_lora_epoch_{args.weight_epoch}")
    dec_lmhead_path = os.path.join(weight_dir, f"decoder_lmhead_epoch_{args.weight_epoch}.pt")
    dec_embeds_path = os.path.join(weight_dir, f"decoder_embeds_epoch_{args.weight_epoch}.pt")
    proj_weight_path = os.path.join(weight_dir, f"conditioning_projection_epoch_{args.weight_epoch}.pt")

    # load encoder decoder FT lora weights
    encoder_model = PeftModel.from_pretrained(base_encoder, enc_weight_path)
    encoder_model.lm_head.load_state_dict(torch.load(enc_lmhead_path, weights_only=True))
    if hasattr(encoder_model.model, "embed_tokens"):
        encoder_model.model.embed_tokens.load_state_dict(torch.load(enc_embeds_path, weights_only=True))
    else:
        encoder_model.model.model.embed_tokens.load_state_dict(torch.load(enc_embeds_path, weights_only=True))
    # decoder
    decoder_model = encoder_model
    if not args.tie_models:
        decoder_model = PeftModel.from_pretrained(base_decoder, dec_weight_path)
        decoder_model.lm_head.load_state_dict(torch.load(dec_lmhead_path, weights_only=True))
        if hasattr(decoder_model.model, "embed_tokens"):
            decoder_model.model.embed_tokens.load_state_dict(torch.load(dec_embeds_path, weights_only=True))
        else:
            decoder_model.model.model.embed_tokens.load_state_dict(torch.load(dec_embeds_path, weights_only=True))
    logger.info("loaded encoder and decoder model weights (not for nolora runs)")

    # set requires grad for model weight conversion
    for name, param in encoder_model.named_parameters():
        param.requires_grad = ("lora" in name) or ("lm_head" in name) or ("embed" in name)
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            param.requires_grad = ("lora" in name) or ("lm_head" in name) or ("embed" in name)

    # convert model weights
    for _, param in encoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if not args.tie_models:
        for _, param in decoder_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')
    # we do not log model weight and model memory size because they are inaccurate

    # set encoder decoder require grad (either partial lora or all lora)
    for name, param in encoder_model.named_parameters():
        param.requires_grad = ("lora" in name) and any(t in name for t in lora_target_modules)
    for name, param in decoder_model.named_parameters():
        param.requires_grad = ("lora" in name) and any(t in name for t in lora_target_modules)

    # convert nbits
    for name, param in encoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted trainable model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # save encoder decoder original lora weights
    # when not training full lora, this saves some redundant weights
    enc_lora_weights_path = os.path.join(args.output_dir, "ft_lora_encoder_cache.pt")
    dec_lora_weights_path = os.path.join(args.output_dir, "ft_lora_decoder_cache.pt")
    enc_lora_weights = get_lora_data(encoder_model, lora_target_modules, args.trainable_nbit)
    torch.save(enc_lora_weights, enc_lora_weights_path)
    logger.info(f"cached {len(enc_lora_weights)} lora encoder weights to {enc_lora_weights_path}")
    del enc_lora_weights
    if not args.tie_models:
        dec_lora_weights = get_lora_data(decoder_model, lora_target_modules, args.trainable_nbit)
        torch.save(dec_lora_weights, dec_lora_weights_path)
        logger.info(f"cached {len(dec_lora_weights)} lora decoder weights to {dec_lora_weights_path}")
        del dec_lora_weights

    # get file_paths
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Eval directory '{args.data_dir}' not found.")
    file_paths = []
    for filename in os.listdir(args.data_dir):
        if filename.endswith(".json"):
            file_paths.append(os.path.join(args.data_dir, filename))
    file_paths.sort()
    logger.info(f"found {len(file_paths)} files")

    # filter based on select tasks file
    if args.select_tasks_path is not None:
        with open(args.select_tasks_path, mode='r') as file:
            csv_reader = csv.reader(file)
            data_as_tuples = [tuple(row) for row in csv_reader]
            data_as_tuples = data_as_tuples[1:] # first row contains col names
            select_task_ids = [d[0] for d in data_as_tuples]
            assert len(select_task_ids) == len(set(select_task_ids))
            select_task_ids = set(select_task_ids)
        # filter tasks
        file_paths = [p for p in file_paths if Path(p).stem in select_task_ids]
        assert len(file_paths) == len(select_task_ids), (len(file_paths), len(select_task_ids))
        logger.info(f"filtered to {len(file_paths)} files from {args.select_tasks_path}")

    # each task has one ttt dataset
    ttt_dataset_maker = partial(
        TTTDataset,
        max_samples_per_task=args.max_samples_per_task,
        permute_n=args.permute_n,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        ntokens=args.ntokens,
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side=args.decoder_pad_side,
        encoder_loss_type=args.encoder_loss_type,
        debug_no_aug=args.debug_no_aug,
    )

    # save memory by making datasets on the fly
    with Pool(args.num_workers) as p:
        ttt_datasets = p.map(ttt_dataset_maker, file_paths)
    for dataset in ttt_datasets:
        logger.info(f"task {dataset.task_id} augmented to {len(dataset)} ttt data")
    ttt_datasets = [dataset for dataset in ttt_datasets if len(dataset) > 0]

    # Prepare with accelerator
    encoder_model, decoder_model = accelerator.prepare(encoder_model, decoder_model)

    # train!
    while len(ttt_datasets) > 0:
        ttt_dataset = ttt_datasets[0]
        task_id = ttt_dataset.task_id

        logger.info(f'=====================')
        logger.info(f"Training {task_id}")
        logger.info(f'=====================')

        # set up ttt dataloader
        ttt_collate_fn = partial(collate_fn_ttt, dataset=ttt_dataset)
        if args.debug_enc_len > 0:
            ttt_collate_fn = partial(
                collate_fn_ttt_dummy,
                ntokens=args.ntokens,
                debug_enc_len=args.debug_enc_len,
                debug_dec_len=args.debug_dec_len,
            )
        ttt_dataloader = DataLoader(
            ttt_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=ttt_collate_fn,
            drop_last=True,
            num_workers=args.num_workers,
        )
        logger.info(f"len(train_dataset) = {len(ttt_dataset)}")
        logger.info(f"len(ttt_dataloader) = {len(ttt_dataloader)}")

        # reset encoder and decoder
        encoder_model.load_state_dict(
            torch.load(enc_lora_weights_path, weights_only=True, map_location=accelerator.device),
            strict=False
        )
        if not args.tie_models:
            decoder_model.load_state_dict(
                torch.load(dec_lora_weights_path, weights_only=True, map_location=accelerator.device),
                strict=False
            )

        # load and set conditioning projection grads and nbit
        conditioning_projection: Union[Hidden2PromptProjection, Prefix2PrefixProjection] = torch.load(proj_weight_path, weights_only=False, map_location=accelerator.device)
        for param in conditioning_projection.parameters():
            param.requires_grad = True
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])

        # Param groups for LoRA
        all_params = [p for p in encoder_model.parameters() if p.requires_grad]
        if not args.tie_models:
            all_params += [p for p in decoder_model.parameters() if p.requires_grad]
        logger.info(f"Optimizer with {len(all_params)} params (should be no emb tuning)")

        # optimizer
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay) # type: ignore
        elif args.optimizer == 'adamw8bit':
            optimizer = bnb.optim.Adam8bit(all_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(all_params, lr=args.lr) # type: ignore

        # LR schedule
        steps_per_epoch = len(ttt_dataset) // (args.batch_size * args.grad_accum_steps * accelerator.num_processes)
        num_training_steps = steps_per_epoch * args.num_epochs
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps,  # 1 epoch warmup
            num_training_steps=num_training_steps * args.grad_accum_steps
        )
        logger.info(f'lr scheduler with {num_training_steps} warmup steps')

        # lambda schedulers
        encoder_loss_lambda_scheduler = LambdaScheduler(
            loss_lambda=args.encoder_loss_lambda,
            linear=args.linear_encoder,
            total_steps=num_training_steps,
        )
        kl_loss_lambda_scheduler = LambdaScheduler(
            loss_lambda=args.kl_loss_lambda,
            linear=args.linear_kl,
            total_steps=num_training_steps,
        )
        invar_loss_lambda_scheduler = LambdaScheduler(loss_lambda=0.0, linear=False, total_steps=0) # HARDCODE

        # Prepare with accelerator
        (
            conditioning_projection,
            optimizer,
            ttt_dataloader,
        ) = accelerator.prepare(
            conditioning_projection,
            optimizer,
            ttt_dataloader,
        )

        logger.info(f'\n======= TRAINING INFO START ======')
        logger.info(f'num_epochs={args.num_epochs}')
        logger.info(f'train_batch_size={args.batch_size}')
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

        encoder_model.train()
        decoder_model.train()
        conditioning_projection.train()

        # when these conditions are met, DDP requires setting static graph due to reusing parameters
        if args.tie_models and args.encoder_gradient_checkpointing and accelerator.num_processes > 1 and args.full_lora:
            # set static graph
            encoder_model._set_static_graph()
            decoder_model._set_static_graph()
            conditioning_projection._set_static_graph()
            # get any data, single forward and backward pass
            batch_data = next(iter(ttt_dataloader))
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
                    decoder_ce_loss=True, # HARDCODE
                    encoder_pad_side=args.encoder_pad_side,
                    decoder_pad_side=args.decoder_pad_side,
                    trainable_nbit=args.trainable_nbit,
                    no_flash_attn=args.no_flash_attn,
                    vae_no_sample=False, # HARDCODE
                    encoder_loss_lambda_scheduler=encoder_loss_lambda_scheduler,
                    invar_loss_lambda_scheduler=invar_loss_lambda_scheduler,
                    kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
                    global_step=global_step,
                )
            accelerator.backward(total_loss)
            optimizer.zero_grad()
            logger.info(f'To avoid DDP error, static graph is applied after doing one iteration of forward backward')

        # start training
        for epoch in range(args.num_epochs):
            ce_loss_accum = 0.0
            invar_loss_accum = 0.0
            encoder_loss_accum = 0.0
            kl_loss_accum = 0.0
            total_loss_accum = 0.0
            grad_norm_accum = 0.0

            for batch_data in ttt_dataloader:
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
                            decoder_ce_loss=True, # HARDCODE
                            encoder_pad_side=args.encoder_pad_side,
                            decoder_pad_side=args.decoder_pad_side,
                            trainable_nbit=args.trainable_nbit,
                            no_flash_attn=args.no_flash_attn,
                            vae_no_sample=False, # HARDCODE
                            encoder_loss_lambda_scheduler=encoder_loss_lambda_scheduler,
                            invar_loss_lambda_scheduler=invar_loss_lambda_scheduler,
                            kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
                            global_step=global_step,
                        )

                    # just accumulate for logging
                    avg_ce_loss = accelerator.gather(ce_loss.repeat(args.batch_size)).mean() # type: ignore
                    avg_invar_loss = accelerator.gather(invar_loss.repeat(args.batch_size)).mean() # type: ignore
                    avg_encoder_loss = accelerator.gather(encoder_loss.repeat(args.batch_size)).mean() # type: ignore
                    avg_kl_loss = accelerator.gather(kl_loss.repeat(args.batch_size)).mean() # type: ignore
                    avg_total_loss = accelerator.gather(total_loss.repeat(args.batch_size)).mean() # type: ignore
                    ce_loss_accum += avg_ce_loss.item() / args.grad_accum_steps
                    invar_loss_accum += avg_invar_loss.item() / args.grad_accum_steps
                    encoder_loss_accum += avg_encoder_loss.item() / args.grad_accum_steps
                    kl_loss_accum += avg_kl_loss.item() / args.grad_accum_steps
                    total_loss_accum += avg_total_loss.item() / args.grad_accum_steps

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
                                f"train/{task_id}_ce_loss": ce_loss_accum,
                                f"train/{task_id}_invar_loss": invar_loss_accum,
                                f"train/{task_id}_encoder_loss": encoder_loss_accum,
                                f"train/{task_id}_kl_loss": kl_loss_accum,
                                f"train/{task_id}_total_loss": total_loss_accum,
                                f"train/{task_id}_grad_norm_accum": grad_norm_accum,
                                f"train/{task_id}_lr_embedding": lr_scheduler.get_last_lr()[0],
                                f"train/{task_id}_lr_other": lr_scheduler.get_last_lr()[1],
                                'Steps': global_step,
                            })
                        except:
                            logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

                    ce_loss_accum = 0.0
                    invar_loss_accum = 0.0
                    encoder_loss_accum = 0.0
                    kl_loss_accum = 0.0
                    total_loss_accum = 0.0
                    grad_norm_accum = 0.0

            if accelerator.is_main_process and (epoch + 1) % args.save_epochs == 0:
                # done training for task, save model for evaluation
                _, _, _ = save_model_ttt(
                    encoder_model=encoder_model,
                    decoder_model=decoder_model,
                    conditioning_projection=conditioning_projection,
                    output_dir=args.output_dir,
                    task_id=task_id,
                    epoch=epoch,
                    tie_models=args.tie_models,
                    lora_target_modules=lora_target_modules,
                    trainable_nbit=args.trainable_nbit,
                )

        # zero grads
        optimizer.state.clear()
        encoder_model.zero_grad(set_to_none=True)
        if not args.tie_models:
            decoder_model.zero_grad(set_to_none=True)
        conditioning_projection.zero_grad(set_to_none=True)

        # delete stuff
        del ttt_datasets[0], ttt_collate_fn, ttt_dataloader
        del optimizer, lr_scheduler, progress_bar
        del conditioning_projection

        # more cleaning
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        os.remove(enc_lora_weights_path)
        if not args.tie_models:
            os.remove(dec_lora_weights_path)

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
