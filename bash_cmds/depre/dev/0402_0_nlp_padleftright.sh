# debug pad left and right again
accelerate launch --main_process_port $MASTER_PORT inference_nlp_0404/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_flash_attn \
    --no_tf32 \
    --lr_scheduler constant \
    --tag test \
    --debug_no_resume \
    --eval_train_test_per_task 1 \
    --eval_eval_ratio 0.01 \
    --eval_seeds 100 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --dropout 0.0 \
    --eval_pretrained \
    --pad_side left

# next, make sure test_time_evaluate gets same eval/score with padside left and right
accelerate launch --main_process_port $MASTER_PORT inference_nlp_0404/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --pad_side left

# {   'eval/num_data': 0.0,
#     'eval/num_params': 0.0,
#     'eval/score': 0.4956160241874529,
#     'eval/time': 0.0}

# gradient search (made sure padleft == padright)
accelerate launch --main_process_port $MASTER_PORT inference_nlp_0404/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 5 \
    --pad_side left

# why score not increasing????
# {   'eval/num_data': 15.80952380952381,
#     'eval/num_params': 41103360.0,
#     'eval/score': 0.46598639455782326,
#     'eval/time': 0.3721050989060175}

# ttt (made sure padleft == padright)
accelerate launch --main_process_port $MASTER_PORT inference_nlp_0404/test_time_evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --ttt_iters 5 \
    --pad_side left
