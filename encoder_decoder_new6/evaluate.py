from datetime import timedelta
from pathlib import Path
import glob
from typing import Union, Optional
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

from data_utils import EvalDataset, collate_fn_eval
from train import set_up_main_process_logger, evaluate
from train import Hidden2PrefixProjection, Hidden2PromptProjection


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "14400" # 4hr for evaluation time variance across gpus
os.environ["NCCL_TIMEOUT_MS"] = "14400000"
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
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--no_lora", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, default="test_evaluation")
    parser.add_argument("--weight_epoch", type=int, default=1)
    parser.add_argument("--ttt_weight_root_dir", type=str, default="./encoder_decoder/outputs_ttt")
    parser.add_argument("--ttt_weight_dir", type=str, default=None)
    parser.add_argument("--ttt_weight_epoch", type=int, default=-1)

    # Conditioning projection
    parser.add_argument("--conditioning_method",
                        type=str,
                        choices=[
                            "prefix2prefix",
                            "hidden2prefix_shared",
                            "hidden2prefix_full",
                            "hidden2prompt",
                            "hidden2prompt_shared",
                            "hidden2prompt_shared_identity",
                            "hidden2prompt_full",
                            "hidden2prompt_full_identity"
                        ],
                        default="hidden2prompt")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--decoder_ce_loss", action="store_true")

    # data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compact_grids", action="store_true")
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

    # eval data
    parser.add_argument("--data_dir", type=str, default="/scratch/yl11330/re-arc/arc_original/training")
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
    parser.add_argument("--num_virtual_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert not args.no_lora # TODO, implement no lora
    assert args.trainable_nbit == 16 # TODO, test this

    # check args
    if args.conditioning_method in ["prefix2prefix", "hidden2prompt"] or "identity" in args.conditioning_method:
        assert args.encoder_name == args.decoder_name
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable
    if args.gs_iters > 0:
        assert args.batch_size == 1

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=14400)
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
        base_encoder = prepare_model_for_kbit_training(base_encoder, use_gradient_checkpointing=False)
        base_decoder = prepare_model_for_kbit_training(base_decoder, use_gradient_checkpointing=False)

    # add new CLS tokens for program encoding
    cls_tokens = [f"<CLS{token_i}>" for token_i in range(args.num_virtual_tokens)]
    encoder_tokenizer.add_tokens(cls_tokens) # type: ignore
    base_encoder.resize_token_embeddings(len(encoder_tokenizer))
    logger.info("Base models loaded.")

    # load encoder decoder projection weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    enc_weight_path = os.path.join(weight_dir, f"encoder_lora_epoch_{args.weight_epoch}")
    dec_weight_path = os.path.join(weight_dir, f"decoder_lora_epoch_{args.weight_epoch}")
    proj_weight_path = os.path.join(weight_dir, f"conditioning_projection_epoch_{args.weight_epoch}.pt")

    if args.no_lora:
        encoder_model = base_encoder.from_pretrained(enc_weight_path)
        decoder_model = base_decoder.from_pretrained(dec_weight_path) if not args.tie_models else encoder_model
    else:
        encoder_model = PeftModel.from_pretrained(base_encoder, enc_weight_path)
        decoder_model = PeftModel.from_pretrained(base_decoder, dec_weight_path) if not args.tie_models else encoder_model
    logger.info("loaded encoder and decoder model weights")

    conditioning_projection: Optional[Union[Hidden2PrefixProjection, Hidden2PromptProjection]] = None
    if args.conditioning_method not in ["prefix2prefix", "hidden2prompt"]:
        conditioning_projection = torch.load(proj_weight_path, weights_only=False, map_location=accelerator.device)
        logger.info("loaded conditioning projection weights")

    # set requires grad for model weight conversion
    if args.no_lora:
        # TODO: fix this, not all model params are trainable
        for name, param in encoder_model.named_parameters():
            param.requires_grad = True
    else:
        for name, param in encoder_model.named_parameters():
            param.requires_grad = ("lora" in name)
        if not args.tie_models:
            for name, param in decoder_model.named_parameters():
                param.requires_grad = ("lora" in name)
        if conditioning_projection is not None:
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
    if conditioning_projection is not None:
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
                if conditioning_projection is not None:
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
        compact_grids=args.compact_grids,
        num_virtual_tokens=args.num_virtual_tokens,
        encoder_loss_type="rest", # HARDCODE
        debug_random_pad=False, # HARDCODE
        debug_pad_len=-1, # HARDCODE
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side="right", # HARDCODE
        decoder_gen_pad_side=args.decoder_gen_pad_side,
    )
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_dataset) # only use tokenizer, debug_random_pad

    # evaluate
    ce, exact_acc, valid_grid, correct_grid_dim, token_acc, texts, \
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
        no_lora=args.no_lora,
        decoder_ce_loss=args.decoder_ce_loss,
        trainable_nbit=args.trainable_nbit,
        flash_attn=args.flash_attn,
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
    )

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {
            "eval/ce_loss": ce,
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
            print(f"wandb failed on process {accelerator.process_index}, skipping the error")

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
