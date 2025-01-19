# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_8_lastencoderlossrun_repro.sh
# all 3 with encoder loss and demonstration loss
# repro has float16 flashattn tiemodels
# original and original_old_script shouild get same performance, they are float32 noflashattn notiemodels

# encoder prefix2prefix repro
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag 0113_prefix2prefix_encoderloss_repro \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method prefix2prefix \
    --wandb

# encoder prefix2prefix original
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new/train.py \
    --tag 0113_prefix2prefix_encoderloss_original \
    --compact_grids \
    --max_seq_len 5120 \
    --num_epochs 20 \
    --conditioning_method prefix2prefix \
    --invar_loss_lambda 0.0 \
    --trainable_nbit 32 \
    --wandb

# encoder prefix2prefix original old script
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_prefix2prefix_encoderloss_original_old_script \
    --compact_grids \
    --max_seq_len 5120 \
    --num_epochs 20 \
    --conditioning_method prefix2prefix \
    --invar_loss_lambda 0.0 \
    --trainable_nbit 32 \
    --wandb
