# each one is 75min

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed0 \
    --eval_ratio 1.0 \
    --seed 0

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed1 \
    --eval_ratio 1.0 \
    --seed 1

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed2 \
    --eval_ratio 1.0 \
    --seed 2

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed3 \
    --eval_ratio 1.0 \
    --seed 3

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed4 \
    --eval_ratio 1.0 \
    --seed 4

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed42 \
    --eval_ratio 1.0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed43 \
    --eval_ratio 1.0 \
    --seed 43

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed44 \
    --eval_ratio 1.0 \
    --seed 44

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed45 \
    --eval_ratio 1.0 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag icl_seed46 \
    --eval_ratio 1.0 \
    --seed 46

# seed0: 42.44611511250709
# seed1: 42.54920574944201
# seed2: 42.24552872821339
# seed3: 41.321308286152984
# seed4: 42.55363491945567
# avg: 42.223158559154

# seed42: 42.407092094880696
# seed43: 42.52421262191587
# seed44: 42.3955912537786
# seed45: 42.19202730406371
# seed46: 42.284299662075014
# avg: 42.360644587343