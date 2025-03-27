from custom_llama import MyLlamaForCausalLM
from typing import Optional
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datetime import timedelta
import pprint
import json
from functools import partial
import argparse
import torch
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import prepare_model_for_kbit_training # type: ignore

from transformers import BitsAndBytesConfig, LlamaConfig
from peft import PeftModel # type: ignore

from data_utils import EvalDataset, collate_fn_eval
from train import (
    ProgramEmbeddings,
    Quantizer,
    VaeProjection,
    set_up_main_process_logger,
    evaluate,
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
    parser.add_argument("--log_every", type=int, default=10)

    # Model
    parser.add_argument("--model_name", type=str, default="llama1b")
    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=float, choices=[16, 32], default=16)
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--weird_cast", action="store_true")
    parser.add_argument("--attention_reduction_ratio", type=float, default=1.0)
    parser.add_argument("--short_context", action='store_true')

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

    # eval data
    parser.add_argument("--config_file", type=str, default="MetaICL/config/hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="MetaICL/data")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--min_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--max_num_pair", type=int, default=17) # includes test pair
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--max_pair_len", type=int, default=2048)
    parser.add_argument('--eval_seeds', type=str, nargs="+", default=['100'])
    parser.add_argument("--pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--kv_pad_side", type=str, choices=["left", "right"], default="right")

    # limit eval
    parser.add_argument('--eval_test_per_task', type=int, default=10000000)
    parser.add_argument('--eval_ratio', type=float, default=1.0)

    # Virtual tokens approach
    parser.add_argument("--ntokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # check args
    if args.model_name == "nemo8b":
        assert args.train_pad_side == args.gen_pad_side == "left"

    args.tag = f"eval_{args.tag}_{args.weight_dir}"
    args.output_dir = os.path.join(args.output_dir, args.tag)

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_up_main_process_logger(accelerator, logger)
    set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_tf32:
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
    tokenizer.pad_token = tokenizer.eos_token
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

    # load config here to set attention dropout params
    config = LlamaConfig.from_pretrained(MODEL_NAME_TO_PATH[args.model_name])
    config.demonstration_attention_dropout = 0.0

    base_model = MyLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_TO_PATH[args.model_name],
        config=config,
        **from_pretrained_kwargs,
    )

    if args.untrainable_nbit in [4, 8]:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=False,
        )

    logger.info("Base models loaded.")

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
    datasets = [
        EvalDataset(
            data_dir=args.data_dir,
            config_file=args.config_file,
            seed=args.seed,
            eval_seed=eval_seed,
            tokenizer=tokenizer,
            debug_random_pad=False,
            debug_pad_len=-1,
            pad_side=args.pad_side,
            debug_len=-1,
            max_seq_len=args.max_seq_len,
            max_pair_len=args.max_pair_len,
            min_num_train_pair=args.min_num_pair - 1,
            max_num_train_pair=args.max_num_pair - 1,
            ntokens=args.ntokens,
            eval_test_per_task=args.eval_test_per_task,
            eval_ratio=args.eval_ratio,
            split='train',
        )
        for eval_seed in args.eval_seeds
    ]
    collate_fn = partial(collate_fn_eval, dataset=datasets[0])

    # # debug: check if train eval load the same exact model
    # input_ids = torch.tensor([list(range(20)), list(range(20))], device=accelerator.device, dtype=torch.int64)
    # attention_mask = torch.full(input_ids.shape, 1, device=accelerator.device, dtype=torch.int64)
    # ce_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
    # print(ce_loss.item())
    # breakpoint()

    # Eval Datasets
    scores, all_output_list = [], None
    for dataset_i, dataset in enumerate(datasets):
        score, output_list = evaluate(
            model=model,
            prior_embeddings=prior_embeddings,
            program_embeddings=program_embeddings,
            vae_projection=vae_projection,
            quantizer=quantizer,
            program_norm=program_norm,
            dataset=dataset,
            accelerator=accelerator,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            dry_eval_run=False,
            no_residual=args.no_residual,
            no_discrete_prior=args.no_discrete_prior,
            no_codebook=False,
            weird_cast=args.weird_cast,
            short_context=args.short_context,
            kv_pad_side=args.kv_pad_side,
            attention_reduction_ratio=args.attention_reduction_ratio,
            log_every=args.log_every,
        )
        if dataset_i == 0:
            all_output_list = output_list
        scores.append(score)
    score = sum(scores) / len(scores)

    if accelerator.is_main_process:
        # log metrics
        metric_dict = {"eval/score": score,}
        logger.info(f'Evaluation results:\n{pprint.pformat(metric_dict, indent=4)}')

        # Save outputs
        save_pred_gt_path = os.path.join(args.output_dir, f"eval_pred_gt.json")
        with open(save_pred_gt_path, 'w') as f:
            json.dump(all_output_list, f)
        logger.info(f"Saved eval pred gt to {save_pred_gt_path}")

    logger.info("All done evaluating.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
