# ARC ICL baseline with off-the-shelf Qwen2.5-14B-Instruct (4-bit quantized)
# Addresses Reviewer 2: "How does CT-KV perform on ARC using an off-the-shelf model NOT fine-tuned on ARC?"
# makesbatch --time 3 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_icl.sh

# arc_bigllm_icl_qwen14b_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen14b_part1 \
    --model_name qwen14b --untrainable_nbit 4 \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen14b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen14b_part2 \
    --model_name qwen14b --untrainable_nbit 4 \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen14b_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen14b_part3 \
    --model_name qwen14b --untrainable_nbit 4 \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen14b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen14b_part4 \
    --model_name qwen14b --untrainable_nbit 4 \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42

# arc_bigllm_icl_qwen14b_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_icl_qwen14b_part5 \
    --model_name qwen14b --untrainable_nbit 4 \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42

#! Submitted batch job 5043169 -> 36_cds -- arc_bigllm_icl_qwen14b_part1
#! Submitted batch job 5043171 -> 36_mren -- arc_bigllm_icl_qwen14b_part2
#! Submitted batch job 5043173 -> 219_courant -- arc_bigllm_icl_qwen14b_part3
#! Submitted batch job 5043175 -> 36_cds -- arc_bigllm_icl_qwen14b_part4
#! Submitted batch job 5043177 -> 36_cds -- arc_bigllm_icl_qwen14b_part5
