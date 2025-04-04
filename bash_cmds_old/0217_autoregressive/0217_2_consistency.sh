# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0217_autoregressive/0217_2_consistency.sh

# ar consistency0.1all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency0.1all \
    --consistency_type all \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.5all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency0.5all \
    --consistency_type all \
    --consistency_loss_lambda 0.5 \
    --wandb

# ar consistency1.0all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency1.0all \
    --consistency_type all \
    --consistency_loss_lambda 1.0 \
    --wandb




# ar consistency0.1excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency0.1excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.5excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency0.5excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 0.5 \
    --wandb

# ar consistency1.0excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency1.0excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 1.0 \
    --wandb




# ar consistency0.1onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency0.1onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.5onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency0.5onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 0.5 \
    --wandb

# ar consistency1.0onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_consistency1.0onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 1.0 \
    --wandb

# Submitted batch job 57353503
# Submitted batch job 57338612
# Submitted batch job 57338613
# Submitted batch job 57338614
# Submitted batch job 57338615
# Submitted batch job 57338616
# Submitted batch job 57338617
# Submitted batch job 57338618
# Submitted batch job 57338619