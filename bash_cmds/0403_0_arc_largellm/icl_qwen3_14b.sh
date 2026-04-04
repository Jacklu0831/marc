# ARC ICL baseline with Qwen3-14B (no fine-tuning, 4-bit quantized)
# makesbatch --time 6 --ngpu 1 --gb 128 --no_singularity --bash_file bash_cmds/0403_0_arc_largellm/icl_qwen3_14b.sh

# arc_largellm_icl_qwen3_14b_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_14b_part1 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_14b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_14b_part2 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_14b_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_14b_part3 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_14b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_14b_part4 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_14b_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_14b_part5 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42

#! Submitted batch job 5368409 -> 36_cds -- arc_largellm_icl_qwen3_14b_part1
#! Submitted batch job 5368410 -> 219_courant -- arc_largellm_icl_qwen3_14b_part2
#! Submitted batch job 5368411 -> 36_mren -- arc_largellm_icl_qwen3_14b_part3
#! Submitted batch job 5368412 -> 36_cds -- arc_largellm_icl_qwen3_14b_part4
#! Submitted batch job 5368413 -> 36_cds -- arc_largellm_icl_qwen3_14b_part5
