# run locally

# nlp model ckpt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/train.py \
    --lr_scheduler constant \
    --tag nlp_pretrained \
    --eval_pretrained \
    --num_epochs 0 \
    --eval_train_test_per_task 1 \
    --eval_eval_ratio 0.01 \
    --eval_seeds 100

# nlp gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --eval_seeds 100

# ran locally, score: 0.36049255515499967


# nlp all seeds!
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test \
    --weight_dir nlp_pretrained \
    --weight_epoch 0

# 0.3567223353101746