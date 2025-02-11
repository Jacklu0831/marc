import time
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import gc
from multiprocessing import Pool
from pathlib import Path
import csv
from peft import PeftModel # type: ignore
from typing import Dict, List
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
    set_up_main_process_logger,
)

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
        model: nn.Module,
        output_dir: str,
        task_id: str,
        epoch: int,
        lora_target_modules: List[str],
    ) -> None:

    # save to output_dir/task_id and only save lora
    os.makedirs(os.path.join(output_dir, task_id), exist_ok=True)

    # encoder
    save_path = os.path.join(output_dir, task_id, f"lora_epoch_{epoch+1}.pt")
    lora_weights = get_lora_data(model, lora_target_modules)
    torch.save(lora_weights, save_path)
    logger.info(f"Saved model to {save_path}")


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
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--full_lora", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_epoch", type=int, required=True)

    # Training
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--save_epochs", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")

    # data
    parser.add_argument("--aug_type", type=str, choices=['none', 'd8', 'extra', 'both'], default='extra')
    parser.add_argument("--max_samples_per_task", type=int, default=250)
    parser.add_argument("--permute_n", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="/scratch/yl11330/re-arc/arc_original/evaluation")
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="right")

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # check args
    if args.model_name == "nemo8b":
        assert args.pad_side == "left"

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.model_name], cache_dir='./encoder_decoder_cache')
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.pad_token is None
    assert isinstance(tokenizer.bos_token, str)
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

    # only keep these tokens, resize model embedding (eos == pad)
    # we do not include program tokens here, those are added later during training and inference
    keep_tokens = [str(i) for i in range(31)] + \
        [tokenizer.bos_token, tokenizer.eos_token, "\n", "input", "output", "pad"]
    assert len(set(keep_tokens)) == len(keep_tokens)

    with torch.no_grad():
        keep_token_ids = []
        for token in keep_tokens:
            token_id = tokenizer(token)["input_ids"] # type: ignore
            assert isinstance(token_id, list) and len(token_id) == 2 # with start token
            keep_token_ids.append(token_id[1])
        assert len(set(keep_token_ids)) == len(keep_token_ids)

        # subset embeddings and lmheads
        assert base_model.model.embed_tokens.weight.shape == base_model.lm_head.weight.shape
        base_model.model.embed_tokens.weight = nn.Parameter(base_model.model.embed_tokens.weight[keep_token_ids])
        base_model.model.embed_tokens.num_embeddings = len(keep_token_ids)
        assert base_model.lm_head.bias is None
        base_model.lm_head.weight = nn.Parameter(base_model.lm_head.weight[keep_token_ids])
        base_model.lm_head.out_features = len(keep_token_ids)
        base_model.config.tie_word_embeddings = False

    # update configs
    assert base_model.config.vocab_size and base_model.config.bos_token_id and base_model.config.eos_token_id
    base_model.config.vocab_size = len(keep_token_ids)
    base_model.config.bos_token_id = keep_tokens.index(tokenizer.bos_token)
    base_model.config.eos_token_id = keep_tokens.index(tokenizer.eos_token)

    # create custom tokenizer
    arc_tokenizer = ARCTokenizer(
        tokens=keep_tokens, # type: ignore
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token, # type: ignore
        pad_token="pad",
    )
    del tokenizer
    tokenizer = arc_tokenizer

    # load weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    model_weight_path = os.path.join(weight_dir, f"lora_epoch_{args.weight_epoch}")
    model = PeftModel.from_pretrained(base_model, model_weight_path)
    logger.info("loaded model weights")

    # convert lora weights to trainable nbit
    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # set requires_grad
    for name, param in model.named_parameters():
        param.requires_grad = ("lora" in name) and any(t in name for t in lora_target_modules)
    logger.info(f'set grad')

    # save original lora weights
    # when not training full lora, this saves some redundant weights
    old_lora_weights_path = os.path.join(args.output_dir, "lora_cache.pt")
    old_lora_weights = get_lora_data(model, lora_target_modules)
    torch.save(old_lora_weights, old_lora_weights_path)
    logger.info(f"cached {len(old_lora_weights)} lora weights to {old_lora_weights_path}")
    del old_lora_weights

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
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        pad_side=args.pad_side,
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
    model = accelerator.prepare(model)

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
        model.load_state_dict(
            torch.load(old_lora_weights_path, weights_only=True, map_location=accelerator.device),
            strict=False,
        )

        # Param groups for LoRA
        all_params = [p for p in model.parameters() if p.requires_grad]
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

        # Prepare with accelerator
        optimizer, ttt_dataloader = accelerator.prepare(optimizer, ttt_dataloader)

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

        model.train()

        # start training
        for epoch in range(args.num_epochs):
            for batch_data in ttt_dataloader:
                input_ids = batch_data["input_ids"].to(accelerator.device)
                attention_mask = batch_data["attention_mask"].to(accelerator.device)
                label_ids = batch_data["label_ids"].to(accelerator.device)

                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        ce_loss = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=label_ids,
                        ).loss

                    accelerator.backward(ce_loss)
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
                    model=model,
                    output_dir=args.output_dir,
                    task_id=task_id,
                    epoch=epoch,
                    lora_target_modules=lora_target_modules,
                )

        # zero grads
        optimizer.state.clear()
        model.zero_grad(set_to_none=True)

        # delete stuff
        del ttt_datasets[0], ttt_dataloader
        del optimizer, lr_scheduler, progress_bar

        # more cleaning
        gc.collect()
        torch.cuda.empty_cache()

        # log time
        num_task_done += 1
        elapsed_time = (time.time() - start_time) / 3600
        estimated_total_time = elapsed_time / num_task_done * num_task
        print(f'estimated total time {round(elapsed_time, 1)}/{round(estimated_total_time, 1)}hr')

    if accelerator.is_main_process:
        os.remove(old_lora_weights_path)

    accelerator.end_training()
    logger.info("All done training.")


if __name__ == "__main__":
    main()
