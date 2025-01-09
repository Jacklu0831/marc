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
    def __init__(self, num_virtual_tokens, encoder_model, decoder_model):
        super(ProjectKV, self).__init__()
        # prefixes are formatted as 16 of (2=2, BS=1, nhead=8, nvirtualtoken=1, tokendim / nhead=64)
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = decoder_model.config.num_hidden_layers
        self.num_kv_heads = decoder_model.config.num_key_value_heads
        self.embed_size_per_head = decoder_model.config.hidden_size // decoder_model.config.num_attention_heads
        # weights
        projections = nn.ModuleList([
            nn.Linear(
                encoder_model.config.hidden_size,
                self.num_layers * 2 * self.num_kv_heads * self.embed_size_per_head
            ) for _ in range(self.num_virtual_tokens)
        ])
        self.weights = nn.Parameter(torch.stack([layer.weight for layer in projections], dim=0))
        self.biases = nn.Parameter(torch.stack([layer.bias for layer in projections], dim=0))
        del projections

    def forward(self, enc_hidden_states):
        # enc_hidden_states has shape (batch_size, num_virtual_tokens, hidden_dim)
        assert enc_hidden_states.shape[1] == self.num_virtual_tokens
        outs = torch.einsum("bnh,nhd->bnd", enc_hidden_states, self.weights.transpose(1, 2)) + self.biases # (batch_size, num_virtual_tokens, out_dim)
        outs = outs.view(outs.size(0), outs.size(1), self.num_layers, 2, self.num_kv_heads, self.embed_size_per_head)
        outs = outs.permute(2, 3, 0, 4, 1, 5)
        past_key_values = [(x[0], x[1]) for x in outs] # each (batch_size, num_kv_heads, num_virtual_tokens, embed_size_per_head)
        return past_key_values


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
    num_virtual_tokens: int,
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
        output_hidden_states=True,
    )

    # get enc_hidden_states (for invar loss) and past_key_values
    if project_kv != None:
        enc_hidden_states = enc_out.hidden_states[-1] # [B, seq_len, hidden_dim]
        enc_hidden_states = enc_hidden_states[:, -num_virtual_tokens:, :]
        past_key_values = project_kv(enc_hidden_states=enc_hidden_states)
    else:
        past_key_values = [(x[0][:, :, -num_virtual_tokens:, :], x[1][:, :, -num_virtual_tokens:, :]) for x in enc_out.past_key_values]
        enc_hidden_states = torch.stack([torch.stack(x) for x in past_key_values]) # each (batch_size, num_kv_heads, num_virtual_tokens, embed_size_per_head)
        enc_hidden_states = enc_hidden_states.permute(2, 0, 1, 3, 4, 5)

    # decoder attention mask must be extended
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

    # invariance loss (batch only not 2x in evaluation)
    invar_loss = torch.tensor(0.0).to(encoder_input_ids.device)
    if enc_hidden_states.shape[0] % 2 == 0:
        invar_loss = nn.functional.mse_loss(enc_hidden_states[::2], enc_hidden_states[1::2])

    total_loss = ce_loss + invar_loss_lambda * invar_loss
    return ce_loss, invar_loss, total_loss, past_key_values


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
    project_kv,
    dataset,
    accelerator: Accelerator,
    num_virtual_tokens: int,
    decoder_tokenizer,
    batch_size: int,
    eval_collate_fn,
    compact_grids: bool,
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
    if project_kv != None:
        project_kv.eval()

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
            batch = eval_collate_fn(batch)
            bs = batch["encoder_input_ids"].size(0)

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
                    num_virtual_tokens=num_virtual_tokens,
                )

            loss_list += [ce_loss.item()] * bs

            # compute accuracy
            with accelerator.autocast():
                # padding at front because HF ignores it
                dec_gen_ids = torch.cat([torch.ones((bs, num_virtual_tokens), device='cuda').to(torch.int64), dec_gen_ids], dim=1)
                dec_gen_mask = torch.cat([torch.ones((bs, num_virtual_tokens), device='cuda').to(torch.int64), dec_gen_mask], dim=1)
                gen_tokens = decoder_module.generate(
                    input_ids=dec_gen_ids,
                    attention_mask=dec_gen_mask,
                    past_key_values=past_key_values,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--tracker_project_name", type=str, default="arc")
    parser.add_argument("--save_model", action="store_true")

    # Debug
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--dummy_seq_enc_len", type=int, default=-1)
    parser.add_argument("--dummy_seq_dec_len", type=int, default=-1)
    parser.add_argument("--debug_fixed_train_order", action="store_true")

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--no_gradient_checkpointing", action="store_true") # note decoder cannot have this due to caching
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--no_project_kv", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=32)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=32)

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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

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
        # args.dummy_seq_enc_len = 8192
        # args.dummy_seq_dec_len = 4096

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

    logger.info("#### BEGIN ALL ARGUMENTS ####")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("#### END ALL ARGUMENTS ####\n")

    # Load tokenizers
    encoder_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.encoder_name], padding_side='left', cache_dir='./encoder_decoder_cache')
    decoder_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_PATH[args.decoder_name], padding_side='left', cache_dir='./encoder_decoder_cache')
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
    base_encoder = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.encoder_name], **from_pretrained_kwargs)
    base_decoder = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.decoder_name], **from_pretrained_kwargs)

    if args.untrainable_nbit in ['4bit', '8bit']:
        base_encoder = prepare_model_for_kbit_training(base_encoder, use_gradient_checkpointing=not args.no_gradient_checkpointing)
        base_decoder = prepare_model_for_kbit_training(base_decoder, use_gradient_checkpointing=False)
    elif not args.no_gradient_checkpointing:
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

    project_kv = None if args.no_project_kv else ProjectKV(args.num_virtual_tokens, encoder_model, decoder_model)

    # convert model weights
    for name, param in encoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for name, param in decoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if project_kv != None:
        for name, param in project_kv.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])

    encoder_model.print_trainable_parameters()
    decoder_model.print_trainable_parameters()
    if project_kv != None:
        logger.info(f'project_kv params {sum(p.numel() for p in project_kv.parameters())}')
    logger.info(f'encoder size {round(encoder_model.get_memory_footprint() / 1024 ** 3, 2)}GB')
    logger.info(f'decoder size {round(decoder_model.get_memory_footprint() / 1024 ** 3, 2)}GB')

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
    )
    train_collate_fn = partial(collate_fn_train,
                               dataset=train_dataset, debug_fixed_train_order=args.debug_fixed_train_order)
    if args.dummy_seq_enc_len > 0:
        train_collate_fn = partial(collate_fn_train_dummy,
                                   dummy_seq_enc_len=args.dummy_seq_enc_len, dummy_seq_dec_len=args.dummy_seq_dec_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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
        assert ('lora' in name) == param.requires_grad, name
        if param.requires_grad:
            if "lora_embedding" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    for name, param in decoder_model.named_parameters():
        assert ('lora' in name) == param.requires_grad, name
        if param.requires_grad:
            if "lora_embedding" in name:
                embedding_params.append(param)
            else:
                other_params.append(param)
    if project_kv != None:
        for param in project_kv.parameters():
            other_params.append(param)

    optimizer_grouped_params = [
        {"params": embedding_params, "lr": args.lr_embedding},
        {"params": other_params, "lr": args.lr_other},
    ]
    all_params = embedding_params + other_params
    optimizer = torch.optim.AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    logger.info(f"Optimizer with {len(embedding_params)} embed-params lr={args.lr_embedding}, {len(other_params)} other-params lr={args.lr_other}")

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
        train_loader
    ) = accelerator.prepare(
        encoder_model,
        decoder_model,
        project_kv,
        optimizer,
        train_loader
    )

    # breakpoint()
    # for name, param in encoder_model.named_parameters(): print(name, param.dtype)
    # breakpoint()
    # for name, param in project_kv.named_parameters(): print(name, param.dtype)
    # breakpoint()
    # for name, param in decoder_model.named_parameters(): print(name, param.dtype)
    # breakpoint()

    logger.info(f'\n======= TRAINING INFO START ======')
    logger.info(f'num_epochs={args.num_epochs}')
    logger.info(f'batch_size={args.batch_size}')
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

    for epoch in range(args.num_epochs):
        encoder_model.train()
        decoder_model.train()
        if project_kv != None:
            project_kv.train()

        ce_loss_accum = 0.0
        invar_loss_accum = 0.0
        total_loss_accum = 0.0
        grad_norm_accum = 0.0

        for batch_data in train_loader:
            enc_ids = batch_data["encoder_input_ids"].to(accelerator.device)
            enc_mask = batch_data["encoder_attention_mask"].to(accelerator.device)
            dec_ids = batch_data["decoder_input_ids"].to(accelerator.device)
            dec_mask = batch_data["decoder_attention_mask"].to(accelerator.device)
            labels = batch_data["decoder_labels"].to(accelerator.device)

            with accelerator.accumulate(encoder_model, decoder_model, project_kv):
                with accelerator.autocast():
                    ce_loss, invar_loss, total_loss, _ = encoder_decoder_loss(
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        project_kv=project_kv,
                        encoder_input_ids=enc_ids,
                        encoder_attention_mask=enc_mask,
                        decoder_input_ids=dec_ids,
                        decoder_attention_mask=dec_mask,
                        decoder_labels=labels,
                        num_virtual_tokens=args.num_virtual_tokens,
                        invar_loss_lambda=args.invar_loss_lambda,
                    )

                # just accumulate for logging
                avg_ce_loss = accelerator.gather(ce_loss.repeat(args.batch_size)).mean()
                avg_invar_loss = accelerator.gather(invar_loss.repeat(args.batch_size)).mean()
                avg_total_loss = accelerator.gather(total_loss.repeat(args.batch_size)).mean()
                ce_loss_accum += avg_ce_loss.item() / args.grad_accum_steps
                invar_loss_accum += avg_invar_loss.item() / args.grad_accum_steps
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
                        "train/ce_loss": ce_loss_accum,
                        "train/invar_loss": invar_loss_accum,
                        "train/total_loss": total_loss_accum,
                        "train/grad_norm_accum": grad_norm_accum,
                        "train/lr_embedding": lr_scheduler.get_last_lr()[0],
                        "train/lr_other": lr_scheduler.get_last_lr()[1]
                    }, step=global_step)

                ce_loss_accum = 0.0
                invar_loss_accum = 0.0
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
            )
            eval_eval_dataset = EvalDataset(
                args.eval_eval_dir,
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                max_seq_len=args.max_seq_len,
                keep_ratio=args.eval_eval_ratio,
                compact_grids=args.compact_grids,
                num_virtual_tokens=args.num_virtual_tokens,
            )
            eval_collate_fn = partial(
                collate_fn_eval,
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
            )
            if args.dummy_seq_enc_len > 0:
                eval_collate_fn = partial(
                    collate_fn_eval_dummy,
                    dummy_seq_enc_len=args.dummy_seq_enc_len,
                    dummy_seq_dec_len=args.dummy_seq_dec_len
                )
            logger.info(f"len(eval_train_dataset) = {len(eval_train_dataset)}")
            logger.info(f"len(eval_eval_dataset) = {len(eval_eval_dataset)}")

            train_ce, train_exact_acc, train_valid_grid, train_correct_grid_dim, train_token_acc, train_texts = evaluate(
                encoder_model,
                decoder_model,
                project_kv,
                eval_train_dataset,
                accelerator,
                args.num_virtual_tokens,
                decoder_tokenizer,
                args.batch_size,
                eval_collate_fn,
                args.compact_grids,
            )
            eval_ce, eval_exact_acc, eval_valid_grid, eval_correct_grid_dim, eval_token_acc, eval_texts = evaluate(
                encoder_model,
                decoder_model,
                project_kv,
                eval_eval_dataset,
                accelerator,
                args.num_virtual_tokens,
                decoder_tokenizer,
                args.batch_size,
                eval_collate_fn,
                args.compact_grids,
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
                if args.save_model:
                    save_enc_path = os.path.join(args.output_dir, f"encoder_lora_epoch_{epoch+1}")
                    save_dec_path = os.path.join(args.output_dir, f"decoder_lora_epoch_{epoch+1}")
                    encoder_module = encoder_model.module if isinstance(encoder_model, DistributedDataParallel) else encoder_model
                    decoder_module = decoder_model.module if isinstance(decoder_model, DistributedDataParallel) else decoder_model
                    encoder_module.save_pretrained(save_enc_path, save_embedding_layers=True)
                    decoder_module.save_pretrained(save_dec_path, save_embedding_layers=True)
                    logger.info(f"Saved encoder to {save_enc_path}")
                    logger.info(f"Saved decoder to {save_dec_path}")

    logger.info("All done training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
