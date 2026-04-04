# ARC Prefix Tuning baseline with off-the-shelf Qwen3-14B (non-thinking mode) (4-bit quantized)
# Standard prefix tuning: random KV init (32 tokens), no demo-derived initialization
# Compares to CT-KV (demo-derived KV init) to isolate the benefit of demo initialization
# makesbatch --time 6 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_prefix_qwen3.sh

# arc_bigllm_prefix_qwen3_14b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen3_14b_part2 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

# arc_bigllm_prefix_qwen3_14b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_prefix_qwen3_14b_part4 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_float16 \
    --random_kv token --random_kv_ntokens 32

#! Submitted batch job 5100861 -> 36_cds -- arc_bigllm_prefix_qwen3_14b_part2
#! Submitted batch job 5100862 -> 36_cds -- arc_bigllm_prefix_qwen3_14b_part4
