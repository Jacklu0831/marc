# python make_sbatch.py --ngpu 1 --time 12 --gb 128 --single --bash_files bash_cmds/mmlu/15_bigllm/seed42/qwen32b.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# # qwen 32b zeroshot seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
#     --tag mmlu_qwen32b_zeroshot_seed42 \
#     --model_name qwen32b \
#     --untrainable_nbit 4 \
#     --batch_size 4 \
#     --zero_shot \
#     --num_demonstrations 10 \
#     --seed 42

# # 43.90827982768193

# # qwen 32b icl seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
#     --tag mmlu_qwen32b_icl_seed42 \
#     --model_name qwen32b \
#     --untrainable_nbit 4 \
#     --batch_size 4 \
#     --num_demonstrations 10 \
#     --seed 42

# # 64.04171141160235





# qwen 32b ctkv15 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_qwen32b_ctkv15_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 15 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42

# qwen 32b ctkv20 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_qwen32b_ctkv20_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 20 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42

# qwen 32b ctkv25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_qwen32b_ctkv25_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 25 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42

# qwen 32b ctkv30 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_qwen32b_ctkv30_lr3e-3_seed42 \
    --model_name qwen32b \
    --untrainable_nbit 4 \
    --batch_size 4 \
    --gs_epochs 30 \
    --gs_batch_size 1 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42






# # test memory
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
#     --tag test \
#     --model_name qwen32b \
#     --untrainable_nbit 4 \
#     --batch_size 4 \
#     --gs_epochs 2 \
#     --gs_batch_size 1 \
#     --num_demonstrations 10 \
#     --seed 42 \
#     --debug
