# ARC ICL baseline with off-the-shelf Qwen3-14B (non-thinking mode, 4-bit quantized)
# Compares to Qwen2.5-14B ICL to see if Qwen3's stronger reasoning helps ARC without fine-tuning
# makesbatch --time 3 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_icl_qwen3.sh

# arc_bigllm_icl_qwen3_14b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_14b_part2 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen3_14b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_14b_part4 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42

#! Submitted batch job 5071379 -> 36_cds -- arc_bigllm_icl_qwen3_14b_part2
#! Submitted batch job 5071381 -> 36_cds -- arc_bigllm_icl_qwen3_14b_part4
