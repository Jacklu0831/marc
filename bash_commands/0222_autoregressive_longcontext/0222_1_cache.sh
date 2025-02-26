# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0222_autoregressive_longcontext/0222_0_base.sh

# arlong
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag test \
    --eval_batch_size 8 \
    --ar_gradient_checkpointing \
    --wandb



        # batch_size = 2
        # hidden_dim = 2048

        # seq_len_prev = 4
        # inputs_embeds_prev = torch.randn((batch_size, seq_len_prev, hidden_dim), dtype=torch.float32, device=device)
        # attention_mask_prev = torch.ones((batch_size, seq_len_prev), dtype=torch.int64, device=device)
        # label_ids_prev = torch.randint(1, 30, (batch_size, seq_len_prev), dtype=torch.int64, device=device)

        # seq_len_curr = 64
        # inputs_embeds_curr = torch.randn((batch_size, seq_len_curr, hidden_dim), dtype=torch.float32, device=device)
        # attention_mask_curr = torch.ones((batch_size, seq_len_curr), dtype=torch.int64, device=device)
        # label_ids_curr = torch.randint(1, 30, (batch_size, seq_len_curr), dtype=torch.int64, device=device)

        # model_out1 = model(
        #     inputs_embeds=inputs_embeds_prev,
        #     attention_mask=attention_mask_prev,
        #     labels=label_ids_prev,
        #     output_hidden_states=True,
        # )
        # past_key_values = model_out1.past_key_values
        # model_out2 = model(
        #     inputs_embeds=inputs_embeds_curr,
        #     attention_mask=attention_mask_curr,
        #     labels=label_ids_curr,
        #     past_key_values=past_key_values,
        #     output_hidden_states=True,
        # )
        # breakpoint()