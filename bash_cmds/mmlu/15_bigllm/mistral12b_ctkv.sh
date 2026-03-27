# python make_sbatch.py --ngpu 1 --time 12 --gb 64 --single --bash_files bash_cmds/mmlu/15_bigllm/seed42/mistral12b.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# # mistral 14b zeroshot seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
#     --tag mmlu_mistral12b_zeroshot_seed42 \
#     --model_name mistral12b \
#     --batch_size 4 \
#     --zero_shot \
#     --num_demonstrations 10 \
#     --seed 42

# # 40.09853907937629

# # mistral 14b icl seed42
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
#     --tag mmlu_mistral12b_icl_seed42 \
#     --model_name mistral12b \
#     --batch_size 4 \
#     --num_demonstrations 10 \
#     --seed 42

# # 53.75969595851222









# mistral 14b ctkv15 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_mistral12b_ctkv15_lr3e-3_seed42 \
    --model_name mistral12b \
    --batch_size 4 \
    --gs_epochs 15 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42

# mistral 14b ctkv20 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_mistral12b_ctkv20_lr3e-3_seed42 \
    --model_name mistral12b \
    --batch_size 4 \
    --gs_epochs 20 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42

# mistral 14b ctkv25 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_mistral12b_ctkv25_lr3e-3_seed42 \
    --model_name mistral12b \
    --batch_size 4 \
    --gs_epochs 25 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42

# mistral 14b ctkv30 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
    --tag mmlu_mistral12b_ctkv30_lr3e-3_seed42 \
    --model_name mistral12b \
    --batch_size 4 \
    --gs_epochs 30 \
    --gs_batch_size 2 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.1 \
    --num_demonstrations 10 \
    --seed 42




# # test memory
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_bigllm/test_time_evaluate.py \
#     --tag test \
#     --model_name mistral12b \
#     --batch_size 4 \
#     --gs_epochs 2 \
#     --gs_batch_size 2 \
#     --num_demonstrations 10 \
#     --seed 42 \
#     --debug
