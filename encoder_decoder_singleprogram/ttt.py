import time
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import gc
from multiprocessing import Pool
from pathlib import Path
import csv
from peft import PeftModel # type: ignore
from typing import Union, Dict, List
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

from data_utils import TTTDataset, collate_fn_ttt, ARCTokenizer
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
    ) -> None:

    # save to output_dir/task_id and only save lora
    os.makedirs(os.path.join(output_dir, task_id), exist_ok=True)

    # encoder
    save_encoder_path = os.path.join(output_dir, task_id, f"encoder_lora_epoch_{epoch+1}.pt")
    encoder_old_lora_weights = get_lora_data(encoder_model, lora_target_modules)
    torch.save(encoder_old_lora_weights, save_encoder_path)
    logger.info(f"Saved encoder to {save_encoder_path}")

    # decoder
    save_decoder_path = None
    if not tie_models:
        save_decoder_path = os.path.join(output_dir, task_id, f"decoder_lora_epoch_{epoch+1}.pt")
        decoder_lora_weights = get_lora_data(decoder_model, lora_target_modules)
        torch.save(decoder_lora_weights, save_decoder_path)
        logger.info(f"Saved decoder to {save_decoder_path}")

    # projection
    save_projection_path = os.path.join(output_dir, task_id, f"conditioning_projection_epoch_{epoch+1}.pt")
    conditioning_projection_module = conditioning_projection
    if isinstance(conditioning_projection, DistributedDataParallel):
        conditioning_projection_module = conditioning_projection.module
    torch.save(conditioning_projection_module, save_projection_path)
    logger.info(f"Saved conditioning projection to {save_projection_path}")


def get_lora_data(model: nn.Module, lora_target_modules: List[str]) -> Dict[str, torch.Tensor]:
    model = model.module if isinstance(model, DistributedDataParallel) else model
    name_to_weight = {}
    for name, param in model.named_parameters():
        if "lora" in name and any(t in name for t in lora_target_modules):
            name_to_weight[name] = param.data
    return name_to_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs_ttt")
    parser.add_argument("--log_every", type=int, default=10)

    # Debug
    parser.add_argument("--debug_no_aug", action="store_true")

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--full_lora", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_epoch", type=int, required=True)

    # Conditioning projection
    parser.add_argument("--conditioning_method", type=str, choices=["prefix2prefix", "hidden2prompt"], default="hidden2prompt")

    # vae
    parser.add_argument("--no_sample", action="store_true")

    # Training
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--save_epochs", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--encoder_loss_type", type=str, choices=["last", "rest", "all"], default="rest")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # both data
    parser.add_argument("--aug_type", type=str, choices=['none', 'd8', 'extra', 'both'], default='extra')
    parser.add_argument("--max_samples_per_task", type=int, default=250)
    parser.add_argument("--permute_n", type=int, default=1)
<<<<<<< HEAD
    parser.add_argument("--data_dir", type=str, default="/scratch/zy3101/re-arc/arc_original/evaluation")
=======
    parser.add_argument("--data_dir", type=str, default="./data/re-arc/arc_original/evaluation")
>>>>>>> origin/main
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")

    # scheduled extra losses
    parser.add_argument("--encoder_loss_lambda", type=float, default=1.0)
    parser.add_argument("--linear_encoder", action="store_true")
    parser.add_argument("--kl_loss_lambda", type=float, default=1.0)
    parser.add_argument("--linear_kl", action="store_true")

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=64)
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
    assert (args.encoder_name == "nemo8b") == (args.decoder_name == "nemo8b")
    if args.encoder_name == "nemo8b":
        assert args.encoder_pad_side == args.decoder_pad_side == "left"

    assert args.trainable_nbit == 16 # TODO, test otherwise

    # default to saving the last epoch
    if args.save_epochs == -1:
        args.save_epochs = args.num_epochs

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
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_up_main_process_logger(accelerator, logger)
    set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
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

    if args.untrainable_nbit in [4, 8]:
        base_encoder = prepare_model_for_kbit_training(
            base_encoder,
            use_gradient_checkpointing=args.encoder_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        if not args.tie_models:
            base_decoder = prepare_model_for_kbit_training(
                base_decoder,
                use_gradient_checkpointing=args.decoder_gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
    else:
        if args.encoder_gradient_checkpointing:
            base_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.decoder_gradient_checkpointing and not args.tie_models:
            base_decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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

    # load encoder decoder projection weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    enc_weight_path = os.path.join(weight_dir, f"encoder_lora_epoch_{args.weight_epoch}")
    dec_weight_path = os.path.join(weight_dir, f"decoder_lora_epoch_{args.weight_epoch}")
    proj_weight_path = os.path.join(weight_dir, f"conditioning_projection_epoch_{args.weight_epoch}.pt")

    # load weights
    encoder_model = PeftModel.from_pretrained(base_encoder, enc_weight_path)
    decoder_model = encoder_model
    if not args.tie_models:
        decoder_model = PeftModel.from_pretrained(base_decoder, dec_weight_path)
    logger.info("loaded model weights")

    # convert lora weights to trainable nbit
    for name, param in encoder_model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            if "lora" in name:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # set requires_grad
    for name, param in encoder_model.named_parameters():
        param.requires_grad = ("lora" in name) and any(t in name for t in lora_target_modules)
    if not args.tie_models:
        for name, param in decoder_model.named_parameters():
            param.requires_grad = ("lora" in name) and any(t in name for t in lora_target_modules)
    logger.info(f'set grad')

    # save encoder decoder original lora weights
    # when not training full lora, this saves some redundant weights
    encoder_old_lora_weights_path = os.path.join(args.output_dir, "encoder_lora_cache.pt")
    encoder_old_lora_weights = get_lora_data(encoder_model, lora_target_modules)
    torch.save(encoder_old_lora_weights, encoder_old_lora_weights_path)
    logger.info(f"cached {len(encoder_old_lora_weights)} lora encoder weights to {encoder_old_lora_weights_path}")
    del encoder_old_lora_weights

    decoder_old_lora_weights_path = os.path.join(args.output_dir, "decoder_lora_cache.pt")
    if not args.tie_models:
        decoder_lora_weights = get_lora_data(decoder_model, lora_target_modules)
        torch.save(decoder_lora_weights, decoder_old_lora_weights_path)
        logger.info(f"cached {len(decoder_lora_weights)} lora decoder weights to {decoder_old_lora_weights_path}")
        del decoder_lora_weights

    # get file_paths
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
        aug_type=args.aug_type,
    )

    # save memory by making datasets on the fly
    with Pool(args.num_workers) as p:
        ttt_datasets = p.map(ttt_dataset_maker, file_paths)
    for dataset in ttt_datasets:
        logger.info(f"task {dataset.task_id} augmented to {len(dataset)} ttt data")
    ttt_datasets = [dataset for dataset in ttt_datasets if len(dataset) > 0]

    # Prepare with accelerator
    encoder_model, decoder_model = accelerator.prepare(encoder_model, decoder_model)

    start_time = time.time()
    num_task_done = 0
    num_task = len(ttt_datasets)

    # train!
    while len(ttt_datasets) > 0:
        ttt_dataset = ttt_datasets[0]
        task_id = ttt_dataset.task_id

        logger.info(f'=====================')
        logger.info(f"Training {task_id}")
        logger.info(f'=====================')

        # set up ttt dataloader
        ttt_dataloader = DataLoader(
            ttt_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn_ttt, dataset=ttt_dataset),
            drop_last=True,
            num_workers=args.num_workers,
        )
        logger.info(f"len(train_dataset) = {len(ttt_dataset)}")
        logger.info(f"len(ttt_dataloader) = {len(ttt_dataloader)}")

        # reset encoder and decoder
        encoder_model.load_state_dict(
            torch.load(encoder_old_lora_weights_path, weights_only=True, map_location=accelerator.device),
            strict=False,
        )
        if not args.tie_models:
            decoder_model.load_state_dict(
                torch.load(decoder_old_lora_weights_path, weights_only=True, map_location=accelerator.device),
                strict=False,
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
        all_params += [p for p in conditioning_projection.parameters() if p.requires_grad]
        logger.info(f"Optimizer with {len(all_params)} params")

        # optimizer
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay) # type: ignore
        elif args.optimizer == 'adamw8bit':
            optimizer = bnb.optim.Adam8bit(all_params, lr=args.lr, weight_decay=args.weight_decay)
            # optimizer = bnb.optim.PagedAdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(all_params, lr=args.lr) # type: ignore

        # LR schedule
        steps_per_epoch = len(ttt_dataset) // (args.batch_size * args.grad_accum_steps * accelerator.num_processes)
        num_training_steps = steps_per_epoch * args.num_epochs
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps * args.warmup_epochs,
            num_training_steps=num_training_steps * args.grad_accum_steps * args.warmup_epochs
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
        logger.info(f'batch_size={args.batch_size}')
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

        # start training
        for epoch in range(args.num_epochs):
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
                        _, _, _, _, total_loss, _ = encoder_decoder_loss(
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
                            encoder_pad_side=args.encoder_pad_side,
                            decoder_pad_side=args.decoder_pad_side,
                            trainable_nbit=args.trainable_nbit,
                            no_flash_attn=args.no_flash_attn,
                            vae_no_sample=args.no_sample,
                            encoder_loss_lambda_scheduler=encoder_loss_lambda_scheduler,
                            invar_loss_lambda_scheduler=invar_loss_lambda_scheduler,
                            kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
                            global_step=global_step,
                            anti_invar_margin=0,
                        )

                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(all_params, args.max_grad_norm).item() # type: ignore
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % args.log_every == 0:
                        progress_bar.update(args.log_every)

            if accelerator.is_main_process and (epoch + 1) % args.save_epochs == 0:
                # done training for task, save model for evaluation
                save_model_ttt(
                    encoder_model=encoder_model,
                    decoder_model=decoder_model,
                    conditioning_projection=conditioning_projection,
                    output_dir=args.output_dir,
                    task_id=task_id,
                    epoch=epoch,
                    tie_models=args.tie_models,
                    lora_target_modules=lora_target_modules,
                )

        # zero grads
        optimizer.state.clear()
        encoder_model.zero_grad(set_to_none=True)
        if not args.tie_models:
            decoder_model.zero_grad(set_to_none=True)
        conditioning_projection.zero_grad(set_to_none=True)

        # delete stuff
        del ttt_datasets[0], ttt_dataloader
        del optimizer, lr_scheduler, progress_bar
        del conditioning_projection

        # more cleaning
        gc.collect()
        torch.cuda.empty_cache()

        # log time
        num_task_done += 1
        elapsed_time = (time.time() - start_time) / 3600
        estimated_total_time = elapsed_time / num_task_done * num_task
        print(f'estimated total time {round(elapsed_time, 1)}/{round(estimated_total_time, 1)}hr')

    if accelerator.is_main_process:
        os.remove(encoder_old_lora_weights_path)
        if not args.tie_models:
            os.remove(decoder_old_lora_weights_path)

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
