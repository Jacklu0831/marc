import json
import copy
from multiprocessing import Pool
from pathlib import Path
import csv
from peft import PeftModel
from typing import Union
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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training

from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

from data_utils import TTTDataset, collate_fn_ttt
from train import (
    three_commas,
    encoder_decoder_loss,
    set_up_main_process_logger,
    Hidden2PrefixProjection,
    Hidden2PromptProjection,
)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # weird tokenizer issue
os.environ["NCCL_TIMEOUT"] = "14400" # 4hr for evaluation time variance across gpus

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


def save_model(
        encoder_model: nn.Module,
        decoder_model: nn.Module,
        conditioning_projection: Union[Hidden2PrefixProjection, Hidden2PromptProjection, None],
        output_dir: str,
        task_id: str,
        epoch: int,
        tie_models: bool,
        no_lora: bool,
    ):
    os.makedirs(os.path.join(output_dir, task_id), exist_ok=True)
    # encoder
    save_enc_path = None
    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
    if no_lora:
        save_enc_path = os.path.join(output_dir, task_id, f"encoder_lora_epoch_{epoch+1}")
        encoder_module.save_pretrained(save_enc_path, save_embedding_layers=True, adapter_name="ttt")
    else:
        save_enc_path = os.path.join(output_dir, task_id, f"encoder_lora_epoch_{epoch+1}.pt")
        encoder_lora_dict = {name: param.data for name, param in encoder_model.state_dict().items() if 'lora' in name}
        torch.save(encoder_lora_dict, save_enc_path)
    logger.info(f"Saved encoder to {save_enc_path}")
    # decoder
    save_dec_path = None
    if not tie_models:
        decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
        if no_lora:
            save_dec_path = os.path.join(output_dir, task_id, f"decoder_lora_epoch_{epoch+1}")
            decoder_module.save_pretrained(save_dec_path, save_embedding_layers=True, adapter_name="ttt")
        else:
            save_dec_path = os.path.join(output_dir, task_id, f"decoder_lora_epoch_{epoch+1}.pt")
            decoder_lora_dict = {name: param.data for name, param in decoder_model.state_dict().items() if 'lora' in name}
            torch.save(decoder_lora_dict, save_dec_path)
        logger.info(f"Saved decoder to {save_dec_path}")
    # projection
    save_proj_path = None
    if conditioning_projection is not None:
        save_proj_path = os.path.join(output_dir, task_id, f"conditioning_projection_epoch_{epoch+1}.pt")
        torch.save(conditioning_projection, save_proj_path)
        logger.info(f"Saved conditioning projection to {save_proj_path}")
    return save_enc_path, save_dec_path, save_proj_path


def print_trainable_parameters(model: nn.Module):
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
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs_ttt")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--tracker_project_name", type=str, default="arc")

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=32)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=32)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true") # original model

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, default="test_evaluation")
    parser.add_argument("--epoch", type=int, default=1)

    # Conditioning projection
    parser.add_argument("--conditioning_method",
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
                        default="prefix2prefix")

    # Training
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_samples_per_task", type=int, default=250)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.1)
    parser.add_argument("--encoder_loss_lambda", type=float, default=0.0)
    parser.add_argument("--no_encoder_demonstration_loss", action="store_true")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # both data
    parser.add_argument("--data_dir", type=str, default="/scratch/yl11330/re-arc/train_data/tasks")
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compact_grids", action="store_true")
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

    # train data
    parser.add_argument("--train_permute_n", type=int, default=1)

    # Virtual tokens approach
    parser.add_argument("--num_virtual_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # check args
    if "prefix2" in args.conditioning_method:
        assert not args.encoder_gradient_checkpointing
    if "2prefix" in args.conditioning_method:
        assert not args.decoder_gradient_checkpointing
        assert args.decoder_pad_side == "right" # right for no middle padding
    if args.conditioning_method in ["prefix2prefix", "hidden2prompt"] or "identity" in args.conditioning_method:
        assert args.encoder_name == args.decoder_name
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable

    args.tag = f"ttt_{args.tag}_{args.weight_dir}"
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

    # load encoder decoder weights, projection later
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    enc_weight_path = os.path.join(weight_dir, f"encoder_lora_epoch_{args.epoch}")
    dec_weight_path = os.path.join(weight_dir, f"decoder_lora_epoch_{args.epoch}")
    proj_weight_path = os.path.join(weight_dir, f"conditioning_projection_epoch_{args.epoch}.pt")

    encoder_model, encoder_saved_weights = None, None
    decoder_model, decoder_saved_weights = None, None
    if not args.no_lora:
        encoder_model = PeftModel.from_pretrained(base_encoder, enc_weight_path)
        decoder_model = PeftModel.from_pretrained(base_decoder, dec_weight_path) if not args.tie_models else encoder_model
        # saved original lora weights (move to cpu if heavy)
        encoder_saved_weights = {name: copy.deepcopy(param.data) for name, param in encoder_model.state_dict().items() if 'lora' in name}
        logger.info(f"cached {len(encoder_saved_weights)} lora encoder weights")
        if not args.tie_models:
            decoder_saved_weights = {name: copy.deepcopy(param.data) for name, param in decoder_model.state_dict().items() if 'lora' in name}
            logger.info(f"cached {len(decoder_saved_weights)} lora decoder weights")
        else:
            decoder_saved_weights = encoder_saved_weights
    logger.info("loaded encoder and decoder model weights (not for nolora runs)")

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
        permute_n=args.train_permute_n,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        compact_grids=args.compact_grids,
        num_virtual_tokens=args.num_virtual_tokens,
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side=args.decoder_pad_side,
        no_encoder_demonstration_loss=args.no_encoder_demonstration_loss,
    )

    # save memory by making datasets on the fly
    with Pool(args.num_workers) as p:
        ttt_datasets = p.map(ttt_dataset_maker, file_paths)
    for dataset in ttt_datasets:
        logger.info(f"task {dataset.task_id} has {len(dataset.tasks)} tests, augmented to {len(dataset)} ttt data")
    ttt_datasets = [dataset for dataset in ttt_datasets if len(dataset) > 0]
    ttt_collate_fn = partial(collate_fn_ttt, dataset=ttt_datasets[0])

    # train!
    while len(ttt_datasets) > 0:
        ttt_dataset = ttt_datasets[0]
        task_id = ttt_dataset.task_id
        logger.info(f"task {task_id} has {len(ttt_dataset.tasks)} tests, turned to {len(ttt_dataset.data)} ttt data")

        logger.info(f'=====================')
        logger.info(f"Training {task_id}")
        logger.info(f'=====================')

        # set up ttt dataloader
        ttt_collate_fn = partial(collate_fn_ttt, dataset=ttt_dataset)
        ttt_dataloader = DataLoader(
            ttt_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=ttt_collate_fn,
            drop_last=True,
            num_workers=args.num_workers,
        )
        logger.info(f"len(train_dataset) = {len(ttt_dataset)}")
        logger.info(f"len(ttt_dataloader) = {len(ttt_dataloader)}")

        # reset encoder and decoder
        if args.no_lora:
            encoder_model = base_encoder.from_pretrained(enc_weight_path)
            decoder_model = base_decoder.from_pretrained(dec_weight_path) if not args.tie_models else encoder_model
        else:
            encoder_model_state_dict = encoder_model.state_dict()
            decoder_model_state_dict = decoder_model.state_dict()
            for name, param in encoder_saved_weights.items():
                assert name in encoder_model_state_dict
                encoder_model_state_dict[name].data.copy_(copy.deepcopy(encoder_saved_weights[name]))
            if not args.tie_models:
                for name, param in decoder_saved_weights.items():
                    assert name in decoder_model_state_dict
                    decoder_model_state_dict[name].data.copy_(copy.deepcopy(decoder_saved_weights[name]))

        # load original conditioning projection
        conditioning_projection = None
        if args.conditioning_method not in ["prefix2prefix", "hidden2prompt"]:
            conditioning_projection = torch.load(proj_weight_path, weights_only=False)
        logger.info("model reinitialized")

        # set require grad
        if args.no_lora:
            for param in encoder_model.parameters():
                param.requires_grad = True
            if not args.tie_models:
                for param in decoder_model.parameters():
                    param.requires_grad = True
        else:
            for name, param in encoder_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            for name, param in decoder_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if conditioning_projection:
            for param in conditioning_projection.parameters():
                param.requires_grad = True

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
        logger.info(f"Optimizer with {len(embedding_params)} embed-params lr={args.lr_embedding}, {len(other_params)} other-params lr={args.lr_other}")

        # optimizer
        if args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw8bit':
            optimizer = bnb.optim.Adam8bit(optimizer_grouped_params, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(optimizer_grouped_params)

        # LR schedule
        steps_per_epoch = len(ttt_dataset) // (args.train_batch_size * args.grad_accum_steps * accelerator.num_processes)
        num_training_steps = steps_per_epoch * args.num_epochs
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch * args.grad_accum_steps,  # 1 epoch warmup
            num_training_steps=num_training_steps * args.grad_accum_steps
        )
        logger.info(f'lr scheduler with {num_training_steps} warmup steps')

        # Prepare with accelerator
        (
            encoder_model,
            decoder_model,
            conditioning_projection,
            optimizer,
            ttt_dataloader,
        ) = accelerator.prepare(
            encoder_model,
            decoder_model,
            conditioning_projection,
            optimizer,
            ttt_dataloader,
        )

        logger.info(f'\n======= TRAINING INFO START ======')
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

        # start training
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

            for batch_data in ttt_dataloader:
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

                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % args.log_every == 0:
                        progress_bar.update(args.log_every)
                        accelerator.log({
                            f"train/{task_id}_ce_loss": ce_loss_accum,
                            f"train/{task_id}_invar_loss": invar_loss_accum,
                            f"train/{task_id}_encoder_loss": encoder_loss_accum,
                            f"train/{task_id}_total_loss": total_loss_accum,
                            f"train/{task_id}_grad_norm_accum": grad_norm_accum,
                            f"train/{task_id}_lr_embedding": lr_scheduler.get_last_lr()[0],
                            f"train/{task_id}_lr_other": lr_scheduler.get_last_lr()[1],
                            'Steps': global_step,
                        })

                    ce_loss_accum = 0.0
                    invar_loss_accum = 0.0
                    encoder_loss_accum = 0.0
                    total_loss_accum = 0.0
                    grad_norm_accum = 0.0

            if accelerator.is_main_process:
                # done training for task, save model for evaluation
                _, _, _ = save_model(
                    encoder_model=encoder_model,
                    decoder_model=decoder_model,
                    conditioning_projection=conditioning_projection,
                    output_dir=args.output_dir,
                    task_id=task_id,
                    epoch=epoch,
                    tie_models=args.tie_models,
                    no_lora=args.no_lora,
                )

        del ttt_datasets[0], ttt_collate_fn, ttt_dataloader
        del optimizer, lr_scheduler, conditioning_projection, progress_bar
        if args.no_lora:
            del encoder_model
            del decoder_model
        else:
            encoder_model.zero_grad(set_to_none=True)
            decoder_model.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()

    logger.info("All done training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
