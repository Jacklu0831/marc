# save a model
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/train.py \
    --lr_scheduler constant \
    --tag test \
    --debug_no_resume \
    --num_epochs 0 \
    --eval_train_test_per_task 1 \
    --eval_eval_ratio 0.01 \
    --eval_seeds 100 \
    --dropout 0.1 \
    --eval_pretrained
# {'eval/score': 0.44799697656840526, 'train/score': 0.42857142857142855}

# eval the model
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100
# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/score': 0.44799697656840526,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}




# NOW lets do gs
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 10 \
    --gs_lr 1e-2
# lr1e-2
# {   'eval/gs_num_data': 15.80952380952381,
#     'eval/gs_num_params': 41103360.0,
#     'eval/gs_time': 0.9022290025438581,
#     'eval/score': 0.5822349608063894,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}

# NOW lets do gs with lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test1 \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_lora \
    --gs_lora_lr 1e-3
# gsloralr1e-2 0.41783309283309283
# gsloralr1e-3 0.4463489883132741
# gsloralr1e-4 0.5663619449333736
# {   'eval/gs_num_data': 15.80952380952381,
#     'eval/gs_num_params': 135475200.0,
#     'eval/gs_time': 2.191129661741711,
#     'eval/score': 0.5663619449333736,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}

# NOW for some reason, gs with lora is bad, lets only do gs lora -> yah it does improve, just not much
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test2 \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 10 \
    --gs_no_key \
    --gs_no_value \
    --gs_lora \
    --gs_lora_lr 1e-2
# pretty shit

# NOW try gs with just key (does not retain most performance)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test3 \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_no_value
# {   'eval/gs_num_data': 15.80952380952381,
#     'eval/gs_num_params': 20551680.0,
#     'eval/gs_time': 6.411153373264131,
#     'eval/score': 0.5004854933426364,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}

# NOW try gs with just value (retains almost all performance!)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test4 \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_no_key
# {   'eval/gs_num_data': 15.80952380952381,
#     'eval/gs_num_params': 20551680.0,
#     'eval/gs_time': 0.9199259962354388,
#     'eval/score': 0.5710826210826211,
#     'eval/ttt_num_data': 0.0,
#     'eval/ttt_num_params': 0.0,
#     'eval/ttt_time': 0.0}

# NOW lets do ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test5 \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --ttt_iters 10 \
    --ttt_lr 1e-2
# {   'eval/gs_num_data': 0.0,
#     'eval/gs_num_params': 0.0,
#     'eval/gs_time': 0.0,
#     'eval/score': 0.5157407407407407,
#     'eval/ttt_num_data': 40.0,
#     'eval/ttt_num_params': 94371840.0,
#     'eval/ttt_time': 2.440523386001587}

# NOW lets do ttt then gs
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag test6 \
    --weight_dir test \
    --weight_epoch 0 \
    --eval_ratio 0.01 \
    --eval_seeds 100 \
    --gs_batch_size 16 \
    --gs_iters 10 \
    --gs_lr 1e-2 \
    --gs_lora \
    --gs_lora_lr 1e-2 \
    --ttt_iters 10 \
    --ttt_lr 1e-2 \
    --debug_max_len
# {   'eval/gs_num_data': 15.80952380952381,
#     'eval/gs_num_params': 135475200.0,
#     'eval/gs_time': 1.992700906026931,
#     'eval/score': 0.31812169312169314,
#     'eval/ttt_num_data': 40.0,
#     'eval/ttt_num_params': 94371840.0,
#     'eval/ttt_time': 2.4248283817654563}

# TODO: gslora gets bad performance, why
# TODO: test memory for rtx8000



# there is also a baseline experiment where we compare to ttt