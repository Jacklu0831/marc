# ARC Prefix Tuning baseline with Qwen3-8B (4-bit quantized) — parts 1, 3, 5
# Gradient checkpointing needed on L40S — uses NonGrowingCache to avoid DynamicCache mutation
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_prefix_qwen3_8b_odd_parts.sh

# arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

#! Submitted batch job 5132531 -> 36_mren -- arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part1
#! Submitted batch job 5132532 -> 219_courant -- arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part3
#! Submitted batch job 5132533 -> 36_mren -- arc_bigllm_prefix_qwen3_8b_e50_lr1e-4_part5
