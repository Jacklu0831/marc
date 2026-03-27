# python make_sbatch.py --ngpu 1 --time 12 --gb 128 --single --bash_files bash_cmds/bbh/15_bigllm/deepseek32b.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# # deepseek 32b zeroshot seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
#     --tag bbh_deepseek32b_zeroshot_seed42 \
#     --model_name deepseek32b \
#     --untrainable_nbit 4 \
#     --batch_size 4 \
#     --zero_shot \
#     --seed 42

# 27.49223602484472

# # deepseek 32b icl seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
#     --tag bbh_deepseek32b_icl_seed42 \
#     --model_name deepseek32b \
#     --untrainable_nbit 4 \
#     --batch_size 4 \
#     --seed 42

# 63.48084886128365








# deepseek 32b ctkv15 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek32b_ctkv15_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 15 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# deepseek 32b ctkv20 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek32b_ctkv20_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 20 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# deepseek 32b ctkv25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek32b_ctkv25_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 25 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 42

# deepseek 32b ctkv30 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
    --tag bbh_deepseek32b_ctkv30_lr3e-3_seed42 \
    --model_name deepseek32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 30 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --seed 42






# # test memory
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_bigllm/test_time_evaluate.py \
#     --tag test \
#     --model_name deepseek32b \
#     --untrainable_nbit 4 \
#     --batch_size 4 \
#     --gs_epochs 2 \
#     --gs_batch_size 1 \
#     --seed 42 \
#     --debug
