# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0127_1_other.sh

# ar novae
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_novae \
    --train_no_sample \
    --eval_no_sample \
    --wandb

# ar nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-4_nolora \
    --kl_loss_lambda 1e-4 \
    --no_lora \
    --wandb

# ar novae_nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_novae_nolora \
    --train_no_sample \
    --eval_no_sample \
    --no_lora \
    --wandb

# ar novae_voting
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0202_ar_novae_voting \
    --train_no_sample \
    --eval_no_sample \
    --wandb

# ar kl1e-5_evalnosample
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-5_evalnosample \
    --kl_loss_lambda 1e-5 \
    --eval_no_sample \
    --wandb

# Submitted batch job 56923713
# Submitted batch job 56923714
# Submitted batch job 56923715
# Submitted batch job 56923716
# Submitted batch job 56923717