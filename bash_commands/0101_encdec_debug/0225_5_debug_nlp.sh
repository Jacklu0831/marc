# time to truly debug noprogram and arlongcache for nlp

# AR
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --tag test \
    --train_batch_size 1 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --num_workers 0 \
    --eval_per_task 1 \
    --samples_per_epoch 100 \
    --num_epochs 100 \
    --eval_epochs 1 \
    --lr_embedding 1e-4 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-3 \
    --debug_overfit_clf 10 \
    --debug_overfit_noclf 10 \
    --no_flash_attn \
    --debug_no_resume \
    --no_bos

# noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag test \
    --train_batch_size 1 \
    --grad_accum_steps 1 \
    --max_grad_norm 1e8 \
    --optimizer sgd \
    --num_workers 0 \
    --eval_per_task 1 \
    --samples_per_epoch 100 \
    --num_epochs 100 \
    --eval_epochs 1 \
    --lr_embedding 1e-4 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-3 \
    --debug_overfit_clf 10 \
    --debug_overfit_noclf 10 \
    --no_flash_attn \
    --debug_no_resume \
    --no_bos