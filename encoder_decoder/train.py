# train_script.py

import pprint
import json
from tqdm import tqdm
from functools import partial
import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, TaskType, get_peft_model

import logging
import datasets
import transformers

from data_utils import (
    load_tasks_from_data_dir,
    TrainDataset,
    EvalDataset,
    collate_fn_train,
    collate_fn_eval,
    collate_fn_train_dummy,
    collate_fn_eval_dummy,
)

logger = get_logger(__name__, log_level="INFO")


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


################################################
# A shared forward pass for training & evaluation
################################################
def encoder_decoder_loss(
    encoder_model,
    decoder_model,
    project_kv: nn.Module,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    decoder_labels: torch.Tensor,
    # extra shape info
    num_layers: int,
    num_kv_heads: int,
    num_virtual_tokens: int,
    embed_size_per_head: int,
    # whether to compute invariance loss
    invar_loss_lambda: float = 0.0,
):
    """
    This function shows how we:
      1) Pass input_ids thru the encoder -> final hidden states
      2) Extract the CLS token (the last token in the seq)
      3) Project it => prefix
      4) Pass prefix + decoder_input_ids => decoder with labels => cross-entropy
    Returns (ce_loss, hidden_cls).
    """
    # 1) Encoder forward
    enc_out = encoder_model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        output_hidden_states=True
    )
    enc_hidden = enc_out.hidden_states[-1] # last layer [B, seq_len, hidden_dim]
    hidden_cls = enc_hidden[:, -1, :]  # last token [B, hidden_dim]
    past_key_values = project_kv(hidden_cls)
    past_key_values = past_key_values.view(past_key_values.size(0), num_layers, 2, num_kv_heads, num_virtual_tokens, embed_size_per_head)
    past_key_values = past_key_values.permute(1, 2, 0, 3, 4, 5)
    past_key_values = [(x[0], x[1]) for x in past_key_values]

    prefix_attention_mask = torch.full(
        (decoder_attention_mask.shape[0], num_virtual_tokens),
        1,
        device=decoder_attention_mask.device
    )
    decoder_attention_mask = torch.cat([prefix_attention_mask, decoder_attention_mask], dim=1)

    # 5) decoder forward => cross-entropy
    dec_out = decoder_model(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        past_key_values=past_key_values,
        labels=decoder_labels,
    )
    ce_loss = dec_out.loss

    # invariance loss
    invar_loss = torch.tensor(0.0, device=encoder_input_ids.device)
    if invar_loss_lambda != 0.0:
        invar_loss = nn.MSELoss()(enc_hidden[::2], enc_hidden[1::2])

    total_loss = ce_loss + invar_loss_lambda * invar_loss
    return ce_loss, invar_loss, total_loss, past_key_values


################################################
# Evaluate with cross-entropy + exact-match
################################################
@torch.no_grad()
def evaluate(
    encoder_model,
    decoder_model,
    project_kv,
    eval_loader,
    accelerator: Accelerator,
    num_layers: int,
    num_kv_heads: int,
    num_virtual_tokens: int,
    embed_size_per_head: int,
    encoder_tokenizer,
    decoder_tokenizer,
):
    """
    For each batch in eval_loader, compute:
      - cross-entropy using `encoder_decoder_loss`
      - generate => exact match vs decoder_label_texts

    Returns (avg_ce, accuracy, total_samples).
    We also log how many total items are valid => sum of 'num_valid' from collate.
    """
    encoder_model.eval()
    decoder_model.eval()
    project_kv.eval()

    task_id_to_texts = {}
    total_loss = 0.0
    total_exact = 0
    total_valid_samples = 0

    terminators = [
        decoder_tokenizer.eos_token_id,
        decoder_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for batch in tqdm(eval_loader):
        if len(batch) == 0:
            # means we had no valid samples in this batch
            continue

        bs = batch["encoder_input_ids"].size(0)
        total_valid_samples += bs

        task_ids = batch["task_ids"]
        enc_ids = batch["encoder_input_ids"].to(accelerator.device)
        enc_mask = batch["encoder_attention_mask"].to(accelerator.device)
        dec_ids = batch["decoder_input_ids"].to(accelerator.device)
        dec_mask = batch["decoder_attention_mask"].to(accelerator.device)
        dec_gen_ids = batch["decoder_gen_input_ids"].to(accelerator.device)
        dec_gen_mask = batch["decoder_gen_attention_mask"].to(accelerator.device)
        labels = batch["decoder_labels"].to(accelerator.device)
        label_texts = batch["decoder_label_texts"]
        out_token_length = batch["decoder_out_token_length"]

        # print(encoder_tokenizer.batch_decode(enc_ids)[0])
        # print(decoder_tokenizer.batch_decode(dec_ids)[0])
        # print(decoder_tokenizer.batch_decode(dec_gen_ids)[0])
        # assert (enc_mask == 0).sum() == 0
        # assert (dec_mask == 0).sum() == 0
        # assert (dec_gen_mask == 0).sum() == 0

        # compute ce loss
        with accelerator.autocast():
            ce_loss, _, _, past_key_values = encoder_decoder_loss(
                encoder_model=encoder_model,
                decoder_model=decoder_model,
                project_kv=project_kv,
                encoder_input_ids=enc_ids,
                encoder_attention_mask=enc_mask,
                decoder_input_ids=dec_ids,
                decoder_attention_mask=dec_mask,
                decoder_labels=labels,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                num_virtual_tokens=num_virtual_tokens,
                embed_size_per_head=embed_size_per_head,
            )
            # print('eval loss', ce_loss.item())

        total_loss += ce_loss.item() * bs

        # compute accuracy
        with accelerator.autocast():
            # padding at front because HF ignores it
            dec_gen_ids = torch.cat([torch.ones((1, num_virtual_tokens), device='cuda').to(torch.int64), dec_gen_ids], dim=1)
            dec_gen_mask = torch.cat([torch.ones((1, num_virtual_tokens), device='cuda').to(torch.int64), dec_gen_mask], dim=1)
            gen_tokens = decoder_model.generate(
                input_ids=dec_gen_ids,
                attention_mask=dec_gen_mask,
                past_key_values=past_key_values,
                max_new_tokens=max(out_token_length) + 50, # arbitrary increase
                num_return_sequences=1,
                do_sample=False,
                eos_token_id=terminators,
            )
        gen_texts = decoder_tokenizer.batch_decode(gen_tokens[:, dec_gen_ids.shape[1]:], skip_special_tokens=True)
        # print(decoder_tokenizer.batch_decode(dec_gen_ids)[0])
        # print(decoder_tokenizer.batch_decode(gen_tokens)[0])
        # print(decoder_tokenizer.batch_decode(dec_ids)[0])

        # Compare each gen_text with label_texts
        for gen_text, label_text in zip(gen_texts, label_texts):
            if gen_text.strip() == label_text.strip():
                total_exact += 1

        # Save generated texts
        for task_id, gen_text, label_text in zip(task_ids, gen_texts, label_texts):
            task_id_to_texts[task_id] = (gen_text, label_text)

    avg_ce = total_loss / max(1, total_valid_samples)
    accuracy = total_exact / max(1, total_valid_samples)
    return avg_ce, accuracy, total_valid_samples, task_id_to_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--tracker_project_name", type=str, default="arc")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--dummy_seq_enc_len", type=int, default=-1)
    parser.add_argument("--dummy_seq_dec_len", type=int, default=-1)
    parser.add_argument("--debug_fixed_train_order", action="store_true")

    # Model
    parser.add_argument("--encoder_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--decoder_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--no_gradient_checkpointing", action="store_true") # note decoder cannot have this due to caching

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--samples_per_epoch", type=int, default=500)
    parser.add_argument("--eval_epochs", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--invar_loss_lambda", type=float, default=0.1)

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
    parser.add_argument("--num_virtual_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # override to debug stuff
    if args.debug:
        args.tag = 'test'
        args.wandb = False
        args.train_data_dir = '/scratch/yl11330/re-arc/train_data_debug/tasks'
        args.eval_train_dir = '/scratch/yl11330/re-arc/arc_original_debug/training'
        args.eval_eval_dir = '/scratch/yl11330/re-arc/arc_original_debug/evaluation'
        args.num_epochs = 1
        args.samples_per_epoch = 500
        args.eval_epochs = 1
        args.num_workers = 0
        # args.dummy_seq_enc_len = 8192
        # args.dummy_seq_dec_len = 4096

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

    # Load tokenizers
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, padding_side='left')
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_name, padding_side='left')

    if not encoder_tokenizer.pad_token:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
    if not decoder_tokenizer.pad_token:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    logger.info("Tokenizers loaded and pad tokens handled.")

    # Build base models
    base_encoder = AutoModelForCausalLM.from_pretrained(args.encoder_name)
    base_decoder = AutoModelForCausalLM.from_pretrained(args.decoder_name)

    if not args.no_gradient_checkpointing:
        base_encoder.gradient_checkpointing_enable()

    # add [CLS] is not in model tokenizer
    if not encoder_tokenizer.cls_token:
        encoder_tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        base_encoder.resize_token_embeddings(len(encoder_tokenizer))
    logger.info("Base models loaded.")

    # LoRA config
    encoder_peft_config = LoraConfig(
        r=args.encoder_lora_rank,
        lora_alpha=args.encoder_lora_alpha,
        lora_dropout=args.encoder_lora_dropout,
        target_modules=args.encoder_lora_target_modules,
        use_rslora=not args.encoder_no_rslora,
        task_type=TaskType.CAUSAL_LM,
    )
    decoder_peft_config = LoraConfig(
        r=args.decoder_lora_rank,
        lora_alpha=args.decoder_lora_alpha,
        lora_dropout=args.decoder_lora_dropout,
        target_modules=args.decoder_lora_target_modules,
        use_rslora=not args.decoder_no_rslora,
        task_type=TaskType.CAUSAL_LM,
    )
    encoder_model = get_peft_model(base_encoder, encoder_peft_config)
    decoder_model = get_peft_model(base_decoder, decoder_peft_config)
    logger.info("LoRA-wrapped models initialized.")

    # prefixes are formatted as 16 of (2=2, BS=1, nhead=8, nvirtualtoken=1, tokendim / nhead=64)
    num_layers = decoder_model.config.num_hidden_layers
    num_kv_heads = decoder_model.config.num_key_value_heads
    embed_size_per_head = decoder_model.config.hidden_size // decoder_model.config.num_attention_heads
    project_kv = nn.Linear(
        encoder_model.config.hidden_size,
        num_layers * 2 * num_kv_heads * args.num_virtual_tokens * embed_size_per_head
    )

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
    )
    collate_fn = partial(collate_fn_train_dummy, dummy_seq_enc_len=args.dummy_seq_enc_len, dummy_seq_dec_len=args.dummy_seq_dec_len) \
                 if args.dummy_seq_enc_len > 0 else partial(collate_fn_train, dataset=train_dataset, debug_fixed_train_order=args.debug_fixed_train_order)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )
    logger.info(f"len(train_dataset) = {len(train_dataset)}")
    logger.info(f"len(train_loader) = {len(train_loader)}")

    # Build evaluation datasets
    eval_train_dataset = EvalDataset(args.eval_train_dir, encoder_tokenizer, decoder_tokenizer, args.max_seq_len, args.eval_train_ratio)
    eval_eval_dataset = EvalDataset(args.eval_eval_dir, encoder_tokenizer, decoder_tokenizer, args.max_seq_len, args.eval_eval_ratio)
    collate_fn = partial(collate_fn_eval_dummy, dummy_seq_enc_len=args.dummy_seq_enc_len, dummy_seq_dec_len=args.dummy_seq_dec_len) \
                 if args.dummy_seq_enc_len > 0 else partial(collate_fn_eval, dataset=eval_train_dataset)
    eval_train_loader = DataLoader(
        eval_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
    )
    collate_fn = partial(collate_fn_eval_dummy, dummy_seq_enc_len=args.dummy_seq_enc_len, dummy_seq_dec_len=args.dummy_seq_dec_len) \
                 if args.dummy_seq_enc_len > 0 else partial(collate_fn_eval, dataset=eval_eval_dataset)
    eval_eval_loader = DataLoader(
        eval_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=args.num_workers,
    )
    logger.info(f"len(eval_train_dataset) = {len(eval_train_dataset)}")
    logger.info(f"len(eval_train_loader) = {len(eval_train_loader)}")
    logger.info(f"len(eval_eval_dataset) = {len(eval_eval_dataset)}")
    logger.info(f"len(eval_eval_loader) = {len(eval_eval_loader)}")

    # Param groups for LoRA
    embedding_params = []
    other_lora_params = []
    for name, param in encoder_model.named_parameters():
        assert ('lora' in name) == param.requires_grad, name
        if param.requires_grad:
            if "lora_embedding" in name:
                embedding_params.append(param)
            else:
                other_lora_params.append(param)
    for name, param in decoder_model.named_parameters():
        assert ('lora' in name) == param.requires_grad, name
        if param.requires_grad:
            if "lora_embedding" in name:
                embedding_params.append(param)
            else:
                other_lora_params.append(param)
    for param in project_kv.parameters():
        other_lora_params.append(param)

    optimizer_grouped_params = [
        {"params": embedding_params, "lr": args.lr_embedding},
        {"params": other_lora_params, "lr": args.lr_other},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    logger.info(f"Optimizer with {len(embedding_params)} embed-params lr={args.lr_embedding}, {len(other_lora_params)} other-params lr={args.lr_other}")

    # LR schedule
    steps_per_epoch = args.samples_per_epoch // (args.batch_size * args.grad_accum_steps * accelerator.num_processes)
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
        project_kv,
        optimizer,
        train_loader,
        eval_train_loader,
        eval_eval_loader
    ) = accelerator.prepare(
        encoder_model,
        decoder_model,
        project_kv,
        optimizer,
        train_loader,
        eval_train_loader,
        eval_eval_loader
    )

    logger.info(f'\n======= TRAINING INFO START ======')
    logger.info(f'num_epochs={args.num_epochs}')
    logger.info(f'batch_size={args.batch_size}')
    logger.info(f'grad_accum_steps={args.grad_accum_steps}')
    logger.info(f'accelerator.num_processes={accelerator.num_processes}')
    logger.info(f'steps_per_epoch={steps_per_epoch}')
    logger.info(f'{sum(p.numel() for p in embedding_params + other_lora_params)} trainable params')
    logger.info(f'======= TRAINING INFO END ======\n')

    global_step = 0
    progress_bar = tqdm(
        range(num_training_steps),
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Total Train Steps")

    for epoch in range(args.num_epochs):
        encoder_model.train()
        decoder_model.train()
        project_kv.train()

        ce_loss_accum = 0.0
        invar_loss_accum = 0.0
        total_loss_accum = 0.0

        for batch_data in train_loader:
            enc_ids = batch_data["encoder_input_ids"].to(accelerator.device)
            enc_mask = batch_data["encoder_attention_mask"].to(accelerator.device)
            dec_ids = batch_data["decoder_input_ids"].to(accelerator.device)
            dec_mask = batch_data["decoder_attention_mask"].to(accelerator.device)
            labels = batch_data["decoder_labels"].to(accelerator.device)
            # logger.info(f'enc_ids {enc_ids.shape}')
            # logger.info(f'dec_ids {dec_ids.shape}')

            with accelerator.accumulate(encoder_model, decoder_model, project_kv):
                with accelerator.autocast():
                    # assert torch.equal(enc_ids[0], enc_ids[1])
                    # assert torch.equal(enc_mask[0], enc_mask[1])
                    # assert torch.equal(dec_ids[0], dec_ids[1])
                    # assert torch.equal(dec_mask[0], dec_mask[1])
                    # assert torch.equal(labels[0], labels[1])
                    # assert (enc_mask == 0).sum() == 0
                    # assert (dec_mask == 0).sum() == 0
                    # assert labels.shape == dec_ids.shape

                    # logger.info('training:')
                    # logger.info(f'enc_ids:\n{encoder_tokenizer.batch_decode(enc_ids)[0]}')
                    # logger.info(f'dec_ids:\n{decoder_tokenizer.batch_decode(dec_ids)[0]}')
                    # for x, y in zip(labels[0], dec_ids[0]):
                    #     assert x in [-100, y], (x, y)

                    ce_loss, invar_loss, total_loss, _ = encoder_decoder_loss(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        project_kv=project_kv,
                        encoder_input_ids=enc_ids,
                        encoder_attention_mask=enc_mask,
                        decoder_input_ids=dec_ids,
                        decoder_attention_mask=dec_mask,
                        decoder_labels=labels,
                        num_layers=num_layers,
                        num_kv_heads=num_kv_heads,
                        num_virtual_tokens=args.num_virtual_tokens,
                        embed_size_per_head=embed_size_per_head,
                        invar_loss_lambda=args.invar_loss_lambda,
                    )
                    # logger.info(f'train loss {ce_loss.item()}')

                # just accumulate for logging
                avg_ce_loss = accelerator.gather(ce_loss.repeat(args.batch_size)).mean()
                avg_invar_loss = accelerator.gather(invar_loss.repeat(args.batch_size)).mean()
                avg_total_loss = accelerator.gather(total_loss.repeat(args.batch_size)).mean()
                ce_loss_accum += avg_ce_loss.item() / args.grad_accum_steps
                invar_loss_accum += avg_invar_loss.item() / args.grad_accum_steps
                total_loss_accum += avg_total_loss.item() / args.grad_accum_steps

                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({
                        "train/ce_loss": ce_loss_accum,
                        "train/invar_loss": invar_loss_accum,
                        "train/total_loss": total_loss_accum,
                        "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                        "train/lr_other": lr_scheduler.get_last_lr()[1]
                    }, step=global_step)
                    ce_loss_accum = 0.0
                    invar_loss_accum = 0.0
                    total_loss_accum = 0.0

        # Evaluate every N epochs
        if (epoch + 1) % args.eval_epochs == 0:
            train_ce, train_acc, train_num_valid_samples, train_texts = evaluate(
                encoder_model,
                decoder_model,
                project_kv,
                eval_train_loader,
                accelerator,
                num_layers,
                num_kv_heads,
                args.num_virtual_tokens,
                embed_size_per_head,
                encoder_tokenizer,
                decoder_tokenizer,
            )
            eval_ce, eval_acc, eval_num_valid_samples, eval_texts = evaluate(
                encoder_model,
                decoder_model,
                project_kv,
                eval_eval_loader,
                accelerator,
                num_layers,
                num_kv_heads,
                args.num_virtual_tokens,
                embed_size_per_head,
                encoder_tokenizer,
                decoder_tokenizer,
            )

            eval_metric_dict = {
                "eval/train_ce_loss": train_ce,
                "eval/train_accuracy": train_acc,
                "eval/train_count": train_num_valid_samples,
                "eval/eval_ce_loss": eval_ce,
                "eval/eval_accuracy": eval_acc,
                "eval/eval_count": eval_num_valid_samples
            }
            accelerator.log(eval_metric_dict, step=global_step)
            logger.info(f'Evaluation results:\n{pprint.pformat(eval_metric_dict, indent=4)}')

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
            save_enc_path = os.path.join(args.output_dir, f"encoder_lora_epoch_{epoch+1}")
            save_dec_path = os.path.join(args.output_dir, f"decoder_lora_epoch_{epoch+1}")
            encoder_model.save_pretrained(save_enc_path)
            decoder_model.save_pretrained(save_dec_path)
            logger.info(f"Saved encoder to {save_enc_path}")
            logger.info(f"Saved decoder to {save_dec_path}")

    logger.info("All done training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
