from typing import Union
from torch import nn
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
from pathlib import Path
import glob
import pprint
import json
from functools import partial
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training # type: ignore

from transformers import BitsAndBytesConfig
from peft import PeftModel # type: ignore

from data_utils import EvalDataset, collate_fn_eval, ARCTokenizer
from train import (
    set_up_main_process_logger,
    evaluate,
    Hidden2PromptProjection,
    Prefix2PrefixProjection,
    LambdaScheduler,
)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "28800" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "28800000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

import wandb

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs_eval")
    parser.add_argument("--tracker_project_name", type=str, default="arc")

    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--no_lora", action="store_true")

    # vae
    parser.add_argument("--vae_no_sample", action="store_true") # applies to inference only

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, default="test_evaluation")
    parser.add_argument("--weight_epoch", type=int, default=1)
    parser.add_argument("--ttt_weight_root_dir", type=str, default="./encoder_decoder/outputs_ttt")
    parser.add_argument("--ttt_weight_dir", type=str, default=None)
    parser.add_argument("--ttt_weight_epoch", type=int, default=-1)

    # Conditioning projection
    parser.add_argument("--conditioning_method", type=str, choices=["prefix2prefix", "hidden2prompt"], default="hidden2prompt")
    parser.add_argument("--projection_type", type=str, choices=["none", "shared", "full"], default="shared")
    parser.add_argument("--identity_init", action="store_true")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=5120)
    parser.add_argument("--decoder_ce_loss", action="store_true")
    parser.add_argument("--encoder_loss_type", type=str, choices=["last", "rest", "all"], default="rest")

    # data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

    # eval data
    parser.add_argument("--data_dir", type=str, default="/scratch/zy3101/re-arc/arc_original/training")
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--leave_ns_inc", action="store_true")
    parser.add_argument("--permute_n", type=int, default=0)
    parser.add_argument("--augment_n", type=int, default=0)
    parser.add_argument("--permute_iters", type=int, default=0)

    # gradient search
    parser.add_argument("--gs_iters", type=int, default=0)
    parser.add_argument("--gs_lr", type=float, default=1.0)
    parser.add_argument("--gs_beta1", type=float, default=0.9)
    parser.add_argument("--gs_beta2", type=float, default=0.9)
    parser.add_argument("--gs_batch_size", type=int, default=2)
    parser.add_argument("--gs_optimizer", type=str, choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--gs_lr_scheduler", type=str, choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--gs_max_grad_norm", default=1e8, type=float, help="Max gradient norm.")
    parser.add_argument("--gs_take_best", action="store_true")

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert not args.no_lora # TODO, implement no lora
    assert args.trainable_nbit == 16 # TODO, test this

    # check args
    if args.conditioning_method == "prefix2prefix":
        assert not args.encoder_gradient_checkpointing
        assert not args.decoder_gradient_checkpointing
    if args.identity_init:
        assert args.encoder_name == args.decoder_name
    if args.projection_type == "none":
        assert args.encoder_name == args.decoder_name
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable
    if args.gs_iters > 0:
        assert args.batch_size == 1
    assert (args.encoder_name == "nemo8b") == (args.decoder_name == "nemo8b")
    if args.encoder_name == "nemo8b":
        assert args.encoder_pad_side == args.decoder_gen_pad_side == "left"

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    os.environ["WANDB_API_KEY"]="8f75801720d1540f03dd3a73e52f11d8ec74f395"
    
    # os.environ["WANDB_API_KEY"]="faf21d9ff65ee150697c7e96f070616f6b662134"
    accelerator = Accelerator(
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
        base_encoder = prepare_model_for_kbit_training(base_encoder, use_gradient_checkpointing=False)
        base_decoder = prepare_model_for_kbit_training(base_decoder, use_gradient_checkpointing=False)

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
    # TODO: check if nemo uses these
    assert base_encoder.config.vocab_size and base_encoder.config.bos_token_id and base_encoder.config.eos_token_id
    base_encoder.config.vocab_size = len(encoder_keep_token_ids)
    base_encoder.config.bos_token_id = encoder_keep_tokens.index(encoder_tokenizer.bos_token)
    base_encoder.config.eos_token_id = encoder_keep_tokens.index(encoder_tokenizer.eos_token)
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

    # load encoder decoder projection weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    enc_weight_path = os.path.join(weight_dir, f"encoder_lora_epoch_{args.weight_epoch}")
    enc_lmhead_path = os.path.join(weight_dir, f"encoder_lmhead_epoch_{args.weight_epoch}.pt")
    enc_embeds_path = os.path.join(weight_dir, f"encoder_embeds_epoch_{args.weight_epoch}.pt")
    dec_weight_path = os.path.join(weight_dir, f"decoder_lora_epoch_{args.weight_epoch}")
    dec_lmhead_path = os.path.join(weight_dir, f"decoder_lmhead_epoch_{args.weight_epoch}.pt")
    dec_embeds_path = os.path.join(weight_dir, f"decoder_embeds_epoch_{args.weight_epoch}.pt")
    proj_weight_path = os.path.join(weight_dir, f"conditioning_projection_epoch_{args.weight_epoch}.pt")

    if args.no_lora:
        encoder_model = base_encoder.from_pretrained(enc_weight_path)
        decoder_model = base_decoder.from_pretrained(dec_weight_path) if not args.tie_models else encoder_model
    else:
        # encoder
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
    logger.info("loaded encoder and decoder model weights")

    conditioning_projection: Union[Hidden2PromptProjection, Prefix2PrefixProjection] = torch.load(proj_weight_path, weights_only=False, map_location=accelerator.device)
    logger.info("loaded conditioning projection weights")

    # set requires grad for model weight conversion
    if args.no_lora:
        # TODO: fix this, not all model params are trainable
        for name, param in encoder_model.named_parameters():
            param.requires_grad = True
    else:
        for name, param in encoder_model.named_parameters():
            param.requires_grad = ("lora" in name) or ("lm_head" in name) or ("embed" in name)
        if not args.tie_models:
            for name, param in decoder_model.named_parameters():
                param.requires_grad = ("lora" in name) or ("lm_head" in name) or ("embed" in name)
        for param in conditioning_projection.parameters():
            param.requires_grad = True

    # convert model weights
    for _, param in encoder_model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if not args.tie_models:
        for _, param in decoder_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for _, param in conditioning_projection.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    logger.info(f'converted model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')
    # we do not log model weight and model memory size because they are inaccurate

    # Prepare with accelerator
    (
        encoder_model,
        decoder_model,
        conditioning_projection,
    ) = accelerator.prepare(
        encoder_model,
        decoder_model,
        conditioning_projection,
    )

    # get ttt model paths
    task_to_ttt_model_paths = None
    encoder_ttt_param_names, decoder_ttt_param_names = None, None
    if args.ttt_weight_dir != None and args.ttt_weight_epoch > -1:
        ttt_weight_dir = os.path.join(args.ttt_weight_root_dir, args.ttt_weight_dir)
        task_to_ttt_model_paths = {}
        for task_weight_dir in glob.glob(f"{ttt_weight_dir}/*"):
            task_name = Path(task_weight_dir).stem
            if os.path.isdir(task_weight_dir) and len(task_name) == 8:
                enc_ttt_path = os.path.join(task_weight_dir, f"encoder_lora_epoch_{args.ttt_weight_epoch}.pt")
                assert os.path.exists(enc_ttt_path), enc_ttt_path
                dec_ttt_path = None
                proj_ttt_path = None
                if not args.tie_models:
                    dec_ttt_path = os.path.join(task_weight_dir, f"decoder_lora_epoch_{args.ttt_weight_epoch}.pt")
                    assert os.path.exists(dec_ttt_path), dec_ttt_path
                proj_ttt_path = os.path.join(task_weight_dir, f"conditioning_projection_epoch_{args.ttt_weight_epoch}.pt")
                assert os.path.exists(proj_ttt_path), proj_ttt_path
                task_to_ttt_model_paths[task_name] = (enc_ttt_path, dec_ttt_path, proj_ttt_path)
        logger.info(f"found {len(task_to_ttt_model_paths)} ttt task loras")
        assert len(task_to_ttt_model_paths) > 0, ttt_weight_dir

        # hacky way to get param names
        enc_ttt_path, dec_ttt_path, proj_ttt_path = list(task_to_ttt_model_paths.values())[0]
        encoder_ttt_param_names = set(torch.load(enc_ttt_path, weights_only=True, map_location=accelerator.device).keys())
        logger.info(f"found {len(encoder_ttt_param_names)} encoder ttt params")
        if not args.tie_models:
            decoder_ttt_param_names = set(torch.load(dec_ttt_path, weights_only=True, map_location=accelerator.device).keys())
            logger.info(f"found {len(decoder_ttt_param_names)} decoder ttt params")

    # Build evaluation dataset
    eval_dataset = EvalDataset(
        args.data_dir,
        select_tasks_path=args.select_tasks_path,
        leave_ns=args.leave_ns,
        leave_ns_inc=args.leave_ns_inc,
        permute_n=args.permute_n,
        augment_n=args.augment_n,
        permute_iters=args.permute_iters,
        seed=args.seed,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_seq_len=args.max_seq_len,
        ntokens=args.ntokens,
        encoder_loss_type=args.encoder_loss_type,
        debug_random_pad=False, # HARDCODE
        debug_pad_len=-1, # HARDCODE
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side=args.decoder_pad_side,
        decoder_gen_pad_side=args.decoder_gen_pad_side,
    )
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_dataset) # only use tokenizer, debug_random_pad

    # lambda schedulers
    encoder_loss_lambda_scheduler = LambdaScheduler(loss_lambda=0.0, linear=False, total_steps=0) # HARDCODE
    invar_loss_lambda_scheduler = LambdaScheduler(loss_lambda=0.0, linear=False, total_steps=0) # HARDCODE
    kl_loss_lambda_scheduler = LambdaScheduler(loss_lambda=0.0, linear=False, total_steps=0) # HARDCODE

    # evaluate
    ce_loss, encoder_loss, kl_loss, \
        exact_acc, valid_grid, correct_grid_dim, token_acc, texts, \
        votes, competition_sub_acc, competition_all_acc, ttt_provided = evaluate(
        task_to_ttt_model_paths=task_to_ttt_model_paths,
        encoder_ttt_param_names=encoder_ttt_param_names,
        decoder_ttt_param_names=decoder_ttt_param_names,
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        conditioning_method=args.conditioning_method,
        conditioning_projection=conditioning_projection,
        dataset=eval_dataset,
        accelerator=accelerator,
        batch_size=args.batch_size,
        collate_fn=eval_collate_fn,
        decoder_ce_loss=args.decoder_ce_loss,
        trainable_nbit=args.trainable_nbit,
        no_flash_attn=args.no_flash_attn,
        tie_models=args.tie_models,
        output_dir=args.output_dir,
        gs_iters=args.gs_iters,
        gs_lr=args.gs_lr,
        gs_beta1=args.gs_beta1,
        gs_beta2=args.gs_beta2,
        gs_batch_size=args.gs_batch_size,
        gs_optimizer=args.gs_optimizer,
        gs_max_grad_norm=args.gs_max_grad_norm,
        gs_lr_scheduler=args.gs_lr_scheduler,
        gs_take_best=args.gs_take_best,
        vae_no_sample=args.vae_no_sample,
        encoder_loss_lambda_scheduler=encoder_loss_lambda_scheduler,
        invar_loss_lambda_scheduler=invar_loss_lambda_scheduler,
        kl_loss_lambda_scheduler=kl_loss_lambda_scheduler,
        global_step=0, # HARDCODE
    )

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {
            "eval/ce_loss": ce_loss,
            "eval/encoder_loss": encoder_loss,
            "eval/kl_loss": kl_loss,
            "eval/exact_acc": exact_acc,
            "eval/valid_grid": valid_grid,
            "eval/correct_grid_dim": correct_grid_dim,
            "eval/token_acc": token_acc,
            "eval/competition_all_acc": competition_all_acc,
            "eval/competition_sub_acc": competition_sub_acc,
            "eval/ttt_provided": ttt_provided,
        }
        logger.info(f'Evaluation results:\n{pprint.pformat(metric_dict, indent=4)}')

        try:
            accelerator.log(metric_dict, step=1)
        except:
            logger.info(f"wandb failed on process {accelerator.process_index}, skipping the error")

        # Save outputs
        save_pred_gt_path = os.path.join(args.output_dir, f"eval_pred_gt.json")
        with open(save_pred_gt_path, 'w') as f:
            json.dump(texts, f)
        logger.info(f"Saved eval generated text to {save_pred_gt_path}")

        # save votes
        save_vote_path = os.path.join(args.output_dir, f"eval_vote.json")
        with open(save_vote_path, 'w') as f:
            json.dump(votes, f)
        logger.info(f"Saved vote to {save_vote_path}")

    logger.info("All done evaluating.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
