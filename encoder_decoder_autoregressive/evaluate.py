from typing import Optional
import glob
from pathlib import Path
from torch import nn
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import pprint
import json
from functools import partial
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training # type: ignore

from transformers import BitsAndBytesConfig
from peft import PeftModel # type: ignore

from data_utils import EvalDataset, collate_fn_eval, ARCTokenizer
from train import (
    ProgramEmbeddings,
    Quantizer,
    VaeProjection,
    set_up_main_process_logger,
    evaluate,
    initialize_program_embeddings,
    ProgramEmbeddings,
)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./encoder_decoder/outputs_eval")

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--concat_programs", action="store_true")
    parser.add_argument("--weird_cast", action="store_true")

    # vqvae
    parser.add_argument("--codebook_size", type=int, default=-1)
    parser.add_argument("--fsq_L", metavar='N', type=int, nargs='+', default=[])
    parser.add_argument("--no_discrete_prior", action="store_true")

    # vae
    parser.add_argument("--vae", action="store_true")

    # Weights
    parser.add_argument("--weight_root_dir", type=str, default="./encoder_decoder/outputs")
    parser.add_argument("--weight_dir", type=str, required=True)
    parser.add_argument("--weight_epoch", type=int, required=True)
    parser.add_argument("--ttt_weight_root_dir", type=str, default="./encoder_decoder/outputs_ttt")
    parser.add_argument("--ttt_weight_dir", type=str, default=None)
    parser.add_argument("--ttt_weight_epoch", type=int, default=-1)

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--max_num_pair", type=int, default=8) # includes test pair
    parser.add_argument("--extra_inference_pairs", type=int, default=0)
    parser.add_argument("--limit_inference_pairs", action='store_true')
    parser.add_argument("--limit_inference_pairs_strict", action='store_true') # overrides limit_inference_pairs
    parser.add_argument("--long_context", action="store_true")
    parser.add_argument("--long_context_repeat_demonstration", action="store_true")

    # data
    parser.add_argument("--train_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--gen_pad_side", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--no_dim", action='store_true')
    parser.add_argument("--no_separate_color_tokens", action='store_true')

    # eval data
    parser.add_argument("--data_dir", type=str, default="./data/re-arc/arc_original/evaluation")
    parser.add_argument("--select_tasks_path", type=str, default=None)
    parser.add_argument("--leave_ns", type=int, nargs="+", default=[0])
    parser.add_argument("--leave_ns_inc", action="store_true")
    parser.add_argument("--permute_n", type=int, default=0)
    parser.add_argument("--augment_n", type=int, default=0)
    parser.add_argument("--permute_iters", type=int, default=0)

    # gradient search eval
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
    parser.add_argument("--ntokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert args.trainable_nbit == 16 # TODO, test otherwise

    # check args
    if args.model_name == "nemo8b":
        assert args.train_pad_side == args.gen_pad_side == "left"
    if args.gs_iters > 0:
        assert args.batch_size == 1

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    accelerator = Accelerator(
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

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        **from_pretrained_kwargs,
    )

    if args.untrainable_nbit in [4, 8]:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False,
        )

    logger.info("Base models loaded.")

    # only keep these tokens, resize model embedding (eos == pad)
    # we do not include program tokens here, those are added later during training and inference
    if not args.no_separate_color_tokens:
        keep_tokens = [str(i) for i in range(31)]
        if args.no_dim:
            keep_tokens = []
    else:
        keep_tokens = [str(i) for i in range(31)]
        if args.no_dim:
            keep_tokens = [str(i) for i in range(10)]
    keep_tokens += [tokenizer.bos_token, tokenizer.eos_token, "\n", "input", "output", "pad"]
    assert len(set(keep_tokens)) == len(keep_tokens)

    keep_token_ids = []
    for token in keep_tokens:
        token_id = tokenizer(token)["input_ids"] # type: ignore
        assert isinstance(token_id, list) and len(token_id) == 2 # with start token
        keep_token_ids.append(token_id[1])
    assert len(set(keep_token_ids)) == len(keep_token_ids)

    color_embeddings = None
    if not args.no_separate_color_tokens:
        color_embeddings = initialize_program_embeddings(
            base_model.model.embed_tokens.weight.data.detach().clone(),
            accelerator,
            ntokens=10,
            cov_scale=1.0,
        )

    # this breaks embedding tying, but whatever
    with torch.no_grad():
        # subset embeddings and lmheads
        assert base_model.model.embed_tokens.weight.shape == base_model.lm_head.weight.shape
        base_model.model.embed_tokens.weight = nn.Parameter(base_model.model.embed_tokens.weight[keep_token_ids])
        base_model.model.embed_tokens.num_embeddings = len(keep_token_ids)
        assert base_model.lm_head.bias is None
        base_model.lm_head.weight = nn.Parameter(base_model.lm_head.weight[keep_token_ids])
        base_model.lm_head.out_features = len(keep_token_ids)
        base_model.config.tie_word_embeddings = False

        if not args.no_separate_color_tokens:
            assert isinstance(color_embeddings, torch.Tensor)
            base_model.model.embed_tokens.weight = nn.Parameter(torch.cat([color_embeddings, base_model.model.embed_tokens.weight]))
            base_model.model.embed_tokens.num_embeddings += 10
            base_model.lm_head.weight = nn.Parameter(torch.cat([color_embeddings, base_model.lm_head.weight]))
            base_model.lm_head.out_features += 10

    if not args.no_separate_color_tokens:
        keep_tokens = [f"c{c}" for c in range(10)] + keep_tokens

    # update configs
    assert base_model.config.vocab_size and base_model.config.bos_token_id and base_model.config.eos_token_id
    base_model.config.vocab_size = len(keep_token_ids) + (0 if args.no_separate_color_tokens else 10)
    base_model.config.bos_token_id = keep_tokens.index(tokenizer.bos_token) # type: ignore
    base_model.config.eos_token_id = keep_tokens.index(tokenizer.eos_token) # type: ignore

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
    prior_embeddings_weight_path = os.path.join(weight_dir, f"prior_embeddings_epoch_{args.weight_epoch}.pt")
    program_embeddings_weight_path = os.path.join(weight_dir, f"program_embeddings_epoch_{args.weight_epoch}.pt")
    vae_projection_weight_path = os.path.join(weight_dir, f"vae_projection_epoch_{args.weight_epoch}.pt")
    quantizer_weight_path = os.path.join(weight_dir, f"quantizer_epoch_{args.weight_epoch}.pt")
    program_norm_weight_path = os.path.join(weight_dir, f"program_norm_epoch_{args.weight_epoch}.pt")

    model = PeftModel.from_pretrained(base_model, model_weight_path)
    prior_embeddings: ProgramEmbeddings = torch.load(
        prior_embeddings_weight_path,
        weights_only=False,
        map_location=accelerator.device
    )
    program_embeddings: ProgramEmbeddings = torch.load(
        program_embeddings_weight_path,
        weights_only=False,
        map_location=accelerator.device
    )
    vae_projection: Optional[VaeProjection] = None
    if args.vae:
        vae_projection = torch.load(
            vae_projection_weight_path,
            weights_only=False,
            map_location=accelerator.device
        )
    quantizer: Optional[Quantizer] = None
    if args.codebook_size > 0 or args.fsq_L != []:
        quantizer = torch.load(
            quantizer_weight_path,
            weights_only=False,
            map_location=accelerator.device
        )
    program_norm: Optional[LlamaRMSNorm] = None
    if not args.no_normalize:
        program_norm = torch.load(
            program_norm_weight_path,
            weights_only=False,
            map_location=accelerator.device
        )
    logger.info("loaded model weights")

    # convert lora weights to trainable nbit
    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for param in prior_embeddings.parameters():
        param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    for param in program_embeddings.parameters():
        param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if vae_projection is not None:
        for param in vae_projection.parameters():
            param.data = param.data.to(torch.float32)
    if quantizer is not None:
        for param in quantizer.parameters():
            param.data = param.data.to(NBIT_TO_DTYPE[args.trainable_nbit])
    if program_norm is not None:
        for param in program_norm.parameters():
            param.data = param.data.to(torch.float32)
    logger.info(f'converted most model weights to {NBIT_TO_DTYPE[args.trainable_nbit]}')

    # get ttt model paths
    task_to_ttt_path = None
    ttt_param_names = None
    prior_embeddings_ttt_path = None
    program_embeddings_ttt_path = None
    vae_projection_ttt_path = None
    quantizer_ttt_path = None
    program_norm_ttt_path = None
    if args.ttt_weight_dir != None:
        ttt_weight_dir = os.path.join(args.ttt_weight_root_dir, args.ttt_weight_dir)
        task_to_ttt_path = {}
        for task_weight_dir in glob.glob(f"{ttt_weight_dir}/*"):
            task_name = Path(task_weight_dir).stem
            if os.path.isdir(task_weight_dir) and len(task_name) == 8:
                model_ttt_path = os.path.join(task_weight_dir, f"lora_epoch_{args.ttt_weight_epoch}.pt")
                assert os.path.exists(model_ttt_path), model_ttt_path
                prior_embeddings_ttt_path = os.path.join(task_weight_dir, f"prior_embeddings_epoch_{args.ttt_weight_epoch}.pt")
                assert os.path.exists(prior_embeddings_ttt_path), prior_embeddings_ttt_path
                program_embeddings_ttt_path = os.path.join(task_weight_dir, f"program_embeddings_epoch_{args.ttt_weight_epoch}.pt")
                assert os.path.exists(program_embeddings_ttt_path), program_embeddings_ttt_path
                if args.vae:
                    vae_projection_ttt_path = os.path.join(task_weight_dir, f"vae_projection_epoch_{args.ttt_weight_epoch}.pt")
                    assert os.path.exists(vae_projection_ttt_path), vae_projection_ttt_path
                if args.codebook_size > 0 or args.fsq_L != []:
                    quantizer_ttt_path = os.path.join(task_weight_dir, f"quantizer_epoch_{args.ttt_weight_epoch}.pt")
                    assert os.path.exists(quantizer_ttt_path), quantizer_ttt_path
                if not args.no_normalize:
                    program_norm_ttt_path = os.path.join(task_weight_dir, f"program_norm_epoch_{args.ttt_weight_epoch}.pt")
                    assert os.path.exists(program_norm_ttt_path), program_norm_ttt_path
                task_to_ttt_path[task_name] = (
                    model_ttt_path,
                    prior_embeddings_ttt_path,
                    program_embeddings_ttt_path,
                    vae_projection_ttt_path,
                    quantizer_ttt_path,
                    program_norm_ttt_path,
                )
        logger.info(f"found {len(task_to_ttt_path)} ttt task loras")
        assert len(task_to_ttt_path) > 0, ttt_weight_dir

        # hacky way to get param names
        model_ttt_path, _, _, _, _, _ = list(task_to_ttt_path.values())[0]
        ttt_param_names = set(torch.load(model_ttt_path, weights_only=True, map_location=accelerator.device).keys())
        logger.info(f"found {len(ttt_param_names)} ttt params")

    # Prepare with accelerator
    (
        model,
        prior_embeddings,
        program_embeddings,
        vae_projection,
        quantizer,
        program_norm,
    ) = accelerator.prepare(
        model,
        prior_embeddings,
        program_embeddings,
        vae_projection,
        quantizer,
        program_norm,
    )

    # Build evaluation dataset
    dataset = EvalDataset(
        eval_dir=args.data_dir,
        select_tasks_path=args.select_tasks_path,
        leave_ns=args.leave_ns,
        leave_ns_inc=args.leave_ns_inc,
        permute_n=args.permute_n,
        augment_n=args.augment_n,
        permute_iters=args.permute_iters,
        seed=args.seed,
        tokenizer=tokenizer,
        ntokens=args.ntokens,
        debug_random_pad=False,
        debug_pad_len=-1,
        train_pad_side=args.train_pad_side,
        gen_pad_side=args.gen_pad_side,
        debug_len=-1,
        no_dim=args.no_dim,
        no_separate_color_tokens=args.no_separate_color_tokens,
        extra_inference_pairs=args.extra_inference_pairs,
        limit_inference_pairs=args.limit_inference_pairs,
        limit_inference_pairs_strict=args.limit_inference_pairs_strict,
        max_num_train_pair=args.max_num_pair - 1,
        long_context=args.long_context,
        long_context_repeat_demonstration=args.long_context_repeat_demonstration,
        max_seq_len=args.max_seq_len,
    )
    collate_fn = partial(collate_fn_eval, dataset=dataset)

    # evaluate
    exact_acc, valid_grid, correct_grid_dim, token_acc, relaxed_token_acc, texts, \
        votes, competition_sub_acc, competition_all_acc, ttt_provided = evaluate(
        desc="eval",
        task_to_ttt_path=task_to_ttt_path,
        ttt_param_names=ttt_param_names,
        model=model,
        prior_embeddings=prior_embeddings,
        program_embeddings=program_embeddings,
        vae_projection=vae_projection,
        quantizer=quantizer,
        program_norm=program_norm,
        tokenizer=tokenizer,
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        trainable_nbit=args.trainable_nbit,
        no_flash_attn=args.no_flash_attn,
        dry_eval_run=False,
        gs_iters=args.gs_iters,
        gs_lr=args.gs_lr,
        gs_beta1=args.gs_beta1,
        gs_beta2=args.gs_beta2,
        gs_batch_size=args.gs_batch_size,
        gs_optimizer=args.gs_optimizer,
        gs_max_grad_norm=args.gs_max_grad_norm,
        gs_lr_scheduler=args.gs_lr_scheduler,
        gs_take_best=args.gs_take_best,
        no_residual=args.no_residual,
        no_discrete_prior=args.no_discrete_prior,
        output_dir=args.output_dir,
        concat_programs=args.concat_programs,
        no_codebook=False,
        weird_cast=args.weird_cast,
    )

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {
            "eval/exact_acc": exact_acc,
            "eval/valid_grid": valid_grid,
            "eval/correct_grid_dim": correct_grid_dim,
            "eval/token_acc": token_acc,
            "eval/relaxed_token_acc": relaxed_token_acc,
            "eval/competition_sub_acc": competition_sub_acc,
            "eval/competition_all_acc": competition_all_acc,
            "eval/ttt_provided": ttt_provided,
        }
        logger.info(f'Evaluation results:\n{pprint.pformat(metric_dict, indent=4)}')

        # save outputs
        save_pred_gt_path = os.path.join(args.output_dir, f"eval_pred_gt.json")
        with open(save_pred_gt_path, 'w') as f:
            json.dump(texts, f)
        logger.info(f"Saved eval train pred gt to {save_pred_gt_path}")

        # save votes
        save_vote_path = os.path.join(args.output_dir, f"eval_vote.json")
        with open(save_vote_path, 'w') as f:
            json.dump(votes, f)
        logger.info(f"Saved vote to {save_vote_path}")

    logger.info("All done evaluating.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
