# custom_llama.py

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig

class GridEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.origin_embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        if config.scaled_position_embedding:
            self.row_embedding = nn.Embedding(1, config.hidden_size)
            self.col_embedding = nn.Embedding(1, config.hidden_size)
        else:
            self.row_embedding = nn.Embedding(config.max_rows, config.hidden_size)
            self.col_embedding = nn.Embedding(config.max_cols, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _grid_embed(self, grid_size):
        # example data
        # /n2 2 2/n2 2 2/n2 2 2
        if config.scaled_position_embeddings:
            pos_row_embed = self.pos_row_embed(torch.zeros(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.zeros(config.max_cols, dtype=torch.long, device=device))
            pos_row_embeds = (torch.arange(1, config.max_rows + 1, device=device)[:, None] * pos_row_embed)
            pos_col_embeds = (torch.arange(1, config.max_cols + 1, device=device)[:, None] * pos_col_embed)
            pos_embed = pos_row_embeds[:, None, None, :] + pos_col_embeds[None, :, None, :]
        else:
            pos_row_embed = self.pos_row_embed(torch.arange(config.max_rows, dtype=torch.long, device=device))
            pos_col_embed = self.pos_col_embed(torch.arange(config.max_cols, dtype=torch.long, device=device))
            pos_embed = pos_row_embed[:, None, None, :] + pos_col_embed[None, :, None, :]

    def grid_embedding(self, input_ids, prefix_counts):
        # prefix counts = encode grid counts


    def forward(self, input_ids):
        
        embed_tokens = self.origin_embedding(input_ids) + self.grid_embedding

        return embed_tokens
   
 # encoder input
    # # input_ids=encoder_input_ids,
    #     attention_mask=encoder_attention_mask,
    #     labels=encoder_labels,
    #     output_hidden_states=True,
    #     prefix_counts = prefix_counts,     
class CustomLlamaModel(LlamaModel):
    def __init__(self, config, extra_size):
        super().__init__(config)
        self.set_input_embeddings = GridEmbedding(config.vocab_size, config.hidden_size, extra_size)

    def forward(self, input_ids, attention_mask=None, labels=encoder_labels, output_hidden_states=True,prefix_counts = prefix_counts):
       
        inputs_embeds = self.embed_tokens(input_ids, prefix_counts)  # 传递额外参数
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=encoder_labels, output_hidden_states=True)

if __name__ == '__main__':
    # 请确保你的 LlamaConfig 中包含 max_rows 和 max_cols 参数
    # Model
    parser.add_argument("--encoder_name", type=str, default="llama1b")
    parser.add_argument("--decoder_name", type=str, default="llama1b")
    parser.add_argument("--tie_models", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--untrainable_nbit", type=float, choices=[3.6, 4, 8, 16, 32], default=16)
    parser.add_argument("--trainable_nbit", type=int, choices=[16, 32], default=16)
    parser.add_argument("--encoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--decoder_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_lora", action="store_true")

    # Training
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_embedding", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--samples_per_epoch", type=int, default=20000)
    parser.add_argument("--eval_epochs", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=5120)
    parser.add_argument("--anti_invar_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--optimizer", type=str, choices=["adamw8bit", "adamw", "sgd"], default="adamw")
    parser.add_argument("--encoder_loss_type", type=str, choices=["last", "rest", "all"], default="rest")
    parser.add_argument("--dry_train_run", action="store_true")
    parser.add_argument("--dry_eval_run", action="store_true")

    # both data
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--encoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_pad_side", type=str, choices=["left", "right"], default="right")
    parser.add_argument("--decoder_gen_pad_side", type=str, choices=["left", "right"], default="left")

    # train data
    parser.add_argument("--train_data_dir", type=str, default="/scratch/zy3101/re-arc/train_data/tasks")
    parser.add_argument("--verifier_file", type=str, default="/scratch/zy3101/re-arc/verifiers.py") # for re-arc and train-original invar loss
    parser.add_argument("--min_prefix", type=int, default=2)
    parser.add_argument("--max_prefix", type=int, default=7)
    parser.add_argument("--augment_ratio", type=float, default=0.0)
    parser.add_argument("--augment_single_grid", action="store_true")

    
    
     Load tokenizers
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

     # Build training dataset
    train_dataset = TrainDataset(
        train_data_dir=args.train_data_dir,
        eval_train_dir=args.eval_train_dir,
        verifier_file=args.verifier_file,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        total_steps=args.samples_per_epoch,
        min_prefix=args.min_prefix,
        max_prefix=args.max_prefix,
        max_seq_len=args.max_seq_len,
        augment_ratio=args.augment_ratio,
        augment_single_grid=args.augment_single_grid,
        seed=args.seed,
        process_index=accelerator.process_index,
        ntokens=args.ntokens,
        debug_fixed_train_order=args.debug_fixed_train_order,
        debug_random_pad=args.debug_random_pad,
        debug_pad_len=args.debug_pad_len,
        encoder_pad_side=args.encoder_pad_side,
        decoder_pad_side=args.decoder_pad_side,
        encoder_loss_type=args.encoder_loss_type,
        anti_invar_ratio=args.anti_invar_ratio,
        num_workers=args.num_workers,
    )
    train_collate_fn = partial(collate_fn_train, dataset=train_dataset)
    if args.debug_enc_len > 0:
        train_collate_fn = partial(
            collate_fn_train_dummy,
            ntokens=args.ntokens,
            debug_enc_len=args.debug_enc_len,
            debug_dec_len=args.debug_dec_len,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False, # this doesn't matter, collate does all the work
        collate_fn=train_collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )

    for batch_data in train_loader:
            enc_ids = batch_data["encoder_input_ids"].to(accelerator.device)
            enc_mask = batch_data["encoder_attention_mask"].to(accelerator.device)
            enc_labels = batch_data["encoder_labels"].to(accelerator.device)
            dec_ids = batch_data["decoder_input_ids"].to(accelerator.device)
            dec_mask = batch_data["decoder_attention_mask"].to(accelerator.device)
            dec_labels = batch_data["decoder_labels"].to(accelerator.device)
            enc_ids_lens = batch_data["encoder_input_ids_lens"]
            dec_ids_lens = batch_data["decoder_input_ids_lens"]
            anti_invars = batch_data["anti_invars"]
            prefix_counts = batch_data["prefix_counts "]

            enc_out = encoder_model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                labels=encoder_labels,
                output_hidden_states=True,
                prefix_counts = prefix_counts,
            )
            encoder_loss = enc_out.loss

