# python make_sbatch.py --ngpu 1 --time 10 --single --bash_files bash_cmds/0401_mmlu/0401_probefulleval/1.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# mmlu llama8b gs10 lr1e-3 droptrain tokendrop0.05 lambda0 fulleval
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_droptrain_tokendrop0.05_lambda0_fulleval \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --eval_ratio 1.0

# mmlu llama8b gs20 lr1e-3 droptrain tokendrop0.05 lambda0 fulleval
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_droptrain_tokendrop0.05_lambda0_fulleval \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_lambda_param_sqr 0.0 \
    --eval_ratio 1.0

# Submitted batch job 59819883