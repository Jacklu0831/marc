# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0118_multigpu/0118_2_vae.sh

# prefix2prefix vae
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_prefix2prefix_vae \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method prefix2prefix \
    --vae \
    --wandb

# prefix2prefix vae identityinit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_prefix2prefix_vae_identityinit \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method prefix2prefix \
    --vae \
    --vae_identity_init \
    --wandb

# hidden2prompt vae
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_vae \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --vae \
    --wandb

# hidden2prompt vae identityinit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_vae_identityinit \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --vae \
    --vae_identity_init \
    --wandb
