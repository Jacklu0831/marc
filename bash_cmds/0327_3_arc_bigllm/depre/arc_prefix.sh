# ARC Prefix Tuning baseline with off-the-shelf Qwen2.5-14B-Instruct (4-bit quantized)
# Standard prefix tuning: random KV init (32 tokens), no demo-derived initialization
# Compares to CT-KV (demo-derived KV init) to isolate the benefit of demo initialization
# makesbatch --time 10 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_prefix.sh

# arc_bigllm_prefix_qwen14b_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen14b_part1 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen14b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen14b_part2 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen14b_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen14b_part3 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen14b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen14b_part4 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen14b_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen14b_part5 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32
