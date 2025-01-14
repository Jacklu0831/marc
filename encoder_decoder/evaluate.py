import pprint
import json
from functools import partial
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training

from transformers import BitsAndBytesConfig
from peft import PeftModel

from data_utils import EvalDataset, collate_fn_eval
from train import set_up_main_process_logger, evaluate


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
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=32)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=32)
    parser.add_argument("--no_lora", action="store_true")

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
                            "hidden2prompt_full",
                        ],
                        default="prefix2prefix")

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=8192)

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

    # Virtual tokens approach
    parser.add_argument("--num_virtual_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # check args
    if args.conditioning_method in ["prefix2prefix", "hidden2prompt"]:
        assert args.encoder_name == args.decoder_name
    if args.tie_models:
        assert args.encoder_name == args.decoder_name
    if args.no_lora:
        args.untrainable_nbit = args.trainable_nbit # untrainable become trainable

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
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

    # add [CLS] is not in model tokenizer
    if not encoder_tokenizer.cls_token:
        encoder_tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        base_encoder.resize_token_embeddings(len(encoder_tokenizer))
    logger.info("Base models loaded.")

    # load encoder decoder weights
    weight_dir = os.path.join(args.weight_root_dir, args.weight_dir)
    enc_path = os.path.join(weight_dir, f"encoder_lora_epoch_{args.epoch}")
    dec_path = os.path.join(weight_dir, f"decoder_lora_epoch_{args.epoch}")
    if args.no_lora:
        encoder_model = base_encoder.from_pretrained(enc_path)
        decoder_model = base_decoder.from_pretrained(dec_path) if not args.tie_models else encoder_model
    else:
        encoder_model = PeftModel.from_pretrained(base_encoder, enc_path)
        decoder_model = PeftModel.from_pretrained(base_decoder, dec_path) if not args.tie_models else encoder_model
    logger.info("loaded encoder and decoder model weights")
    # load conditioning projection weights
    conditioning_projection = None
    if args.conditioning_method not in ["prefix2prefix", "hidden2prompt"]:
        proj_path = os.path.join(weight_dir, f"conditioning_projection_epoch_{args.epoch}.pt")
        conditioning_projection = torch.load(proj_path, weights_only=False)
        logger.info("loaded conditioning projection weights")

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

    # Build evaluation dataset
    eval_dataset = EvalDataset(
        args.data_dir,
        select_tasks_path=args.select_tasks_path,
        leave_ns=args.leave_ns,
        leave_ns_inc=args.leave_ns_inc,
        permute_n=args.permute_n,
        augment_n=args.augment_n,
        seed=args.seed,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_seq_len=args.max_seq_len,
        compact_grids=args.compact_grids,
        num_virtual_tokens=args.num_virtual_tokens,
        no_encoder_demonstration_loss=False, # HARDCODE
        debug_random_pad=False, # HARDCODE
        debug_pad_len=-1, # HARDCODE
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side="right", # not used
        decoder_gen_pad_side=args.decoder_gen_pad_side,
    )
    eval_collate_fn = partial(collate_fn_eval, dataset=eval_dataset) # only use tokenizer, debug_random_pad

    # evaluate
    ce, exact_acc, valid_grid, correct_grid_dim, token_acc, texts, \
        votes, competition_sub_acc, competition_all_acc = evaluate(
        ttt_model_paths=None,
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        conditioning_method=args.conditioning_method,
        conditioning_projection=conditioning_projection,
        dataset=eval_dataset,
        accelerator=accelerator,
        num_virtual_tokens=args.num_virtual_tokens,
        decoder_tokenizer=decoder_tokenizer,
        batch_size=args.batch_size,
        collate_fn=eval_collate_fn,
        compact_grids=args.compact_grids,
        no_lora=args.no_lora,
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side="right", # HARDCODE
        decoder_gen_pad_side=args.decoder_gen_pad_side,
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
        }
        logger.info(f'Evaluation results:\n{pprint.pformat(metric_dict, indent=4)}')
        accelerator.log(metric_dict, step=1)

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
