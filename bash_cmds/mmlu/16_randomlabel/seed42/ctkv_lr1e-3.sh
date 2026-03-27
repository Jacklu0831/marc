# python make_sbatch.py --ngpu 1 --time 15 --single --bash_files bash_cmds/mmlu/16_randomlabel/seed42/ctkv_lr1e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mmlu gs15 lr1e-3 tokendrop0.1 wronglabel0.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_wronglabel0.0_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.0 \
    --seed 42

# mmlu gs20 lr1e-3 tokendrop0.1 wronglabel0.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_wronglabel0.0_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.0 \
    --seed 42

# mmlu gs25 lr1e-3 tokendrop0.1 wronglabel0.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_wronglabel0.0_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.0 \
    --seed 42

# mmlu gs30 lr1e-3 tokendrop0.1 wronglabel0.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_wronglabel0.0_seed42 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.0 \
    --seed 42







# mmlu gs15 lr1e-3 tokendrop0.1 wronglabel0.25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_wronglabel0.25_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.25 \
    --seed 42

# mmlu gs20 lr1e-3 tokendrop0.1 wronglabel0.25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_wronglabel0.25_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.25 \
    --seed 42

# mmlu gs25 lr1e-3 tokendrop0.1 wronglabel0.25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_wronglabel0.25_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.25 \
    --seed 42

# mmlu gs30 lr1e-3 tokendrop0.1 wronglabel0.25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_wronglabel0.25_seed42 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.25 \
    --seed 42








# mmlu gs15 lr1e-3 tokendrop0.1 wronglabel0.5 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_wronglabel0.5_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.5 \
    --seed 42

# mmlu gs20 lr1e-3 tokendrop0.1 wronglabel0.5 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_wronglabel0.5_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.5 \
    --seed 42

# mmlu gs25 lr1e-3 tokendrop0.1 wronglabel0.5 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_wronglabel0.5_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.5 \
    --seed 42

# mmlu gs30 lr1e-3 tokendrop0.1 wronglabel0.5 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_wronglabel0.5_seed42 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.5 \
    --seed 42






# mmlu gs15 lr1e-3 tokendrop0.1 wronglabel0.75 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_wronglabel0.75_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.75 \
    --seed 42

# mmlu gs20 lr1e-3 tokendrop0.1 wronglabel0.75 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_wronglabel0.75_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.75 \
    --seed 42

# mmlu gs25 lr1e-3 tokendrop0.1 wronglabel0.75 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_wronglabel0.75_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.75 \
    --seed 42

# mmlu gs30 lr1e-3 tokendrop0.1 wronglabel0.75 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_wronglabel0.75_seed42 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 0.75 \
    --seed 42







# mmlu gs15 lr1e-3 tokendrop0.1 wronglabel1.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0.1_wronglabel1.0_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 1.0 \
    --seed 42

# mmlu gs20 lr1e-3 tokendrop0.1 wronglabel1.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0.1_wronglabel1.0_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 1.0 \
    --seed 42

# mmlu gs25 lr1e-3 tokendrop0.1 wronglabel1.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0.1_wronglabel1.0_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 1.0 \
    --seed 42

# mmlu gs30 lr1e-3 tokendrop0.1 wronglabel1.0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs30_lr1e-3_tokendrop0.1_wronglabel1.0_seed42 \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --wrong_label 1.0 \
    --seed 42