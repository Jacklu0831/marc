# ARC ICL baseline with Qwen3-8B (all parts)
# makesbatch --time 3 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_icl_qwen3_8b.sh

# arc_bigllm_icl_qwen3_8b_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_8b_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen3_8b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_8b_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen3_8b_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_8b_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen3_8b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_8b_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen3_8b_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen3_8b_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42

#! Submitted batch job 5104617 -> 36_mren -- arc_bigllm_icl_qwen3_8b_part1
#! Submitted batch job 5104618 -> 36_mren -- arc_bigllm_icl_qwen3_8b_part2
#! Submitted batch job 5104619 -> 36_mren -- arc_bigllm_icl_qwen3_8b_part3
#! Submitted batch job 5104620 -> 36_mren -- arc_bigllm_icl_qwen3_8b_part4
#! Submitted batch job 5104621 -> 36_mren -- arc_bigllm_icl_qwen3_8b_part5
