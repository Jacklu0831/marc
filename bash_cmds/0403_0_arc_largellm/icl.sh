# ARC ICL baseline with Qwen3-32B (no fine-tuning, 4-bit quantized)
# makesbatch --time 6 --ngpu 1 --gb 128 --no_singularity --bash_file bash_cmds/0403_0_arc_largellm/icl.sh

# arc_largellm_icl_qwen3_32b_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_32b_part1 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_32b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_32b_part2 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_32b_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_32b_part3 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_32b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_32b_part4 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42

# arc_largellm_icl_qwen3_32b_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_icl_qwen3_32b_part5 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42

#! Submitted batch job 5336884 -> 36_cds -- arc_largellm_icl_qwen3_32b_part1
#! Submitted batch job 5336885 -> 36_mren -- arc_largellm_icl_qwen3_32b_part2
#! Submitted batch job 5336886 -> 36_cds -- arc_largellm_icl_qwen3_32b_part3
#! Submitted batch job 5336887 -> 36_mren -- arc_largellm_icl_qwen3_32b_part4
#! Submitted batch job 5336888 -> 36_cds -- arc_largellm_icl_qwen3_32b_part5
