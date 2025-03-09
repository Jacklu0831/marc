# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0306_noprogram_nlp/0308_0_base.sh

# noprogram nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0308_noprogram_nlp_excludefirst \
    --loss_type exclude_first \
    --wandb

# noprogram nlp onlylast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0308_noprogram_nlp_onlylast \
    --loss_type only_last \
    --wandb



# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
#     --lr_scheduler constant \
#     --tag test \
#     --train_batch_size 1 \
#     --grad_accum_steps 1 \
#     --max_grad_norm 1e8 \
#     --optimizer sgd \
#     --num_workers 0 \
#     --eval_per_task 1 \
#     --samples_per_epoch 100 \
#     --num_epochs 100 \
#     --eval_epochs 1 \
#     --lr_embedding 1e-4 \
#     --lr_program 1e-4 \
#     --lr_prior 1e-4 \
#     --lr_other 1e-3 \
#     --debug_overfit_clf 10 \
#     --debug_overfit_noclf 10 \
#     --flash_attn \
#     --debug_no_resume



# Submitted batch job 58103466
# Submitted batch job 58103467