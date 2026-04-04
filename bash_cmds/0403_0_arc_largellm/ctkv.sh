# ARC CT-KV LOO with Qwen3-32B — LR sweep (e50, td=0.1, LOO only)
# makesbatch --time 12 --ngpu 1 --gb 128 --no_singularity --bash_file bash_cmds/0403_0_arc_largellm/ctkv.sh

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part1 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part2 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part3 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part4 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part5 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part1 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part2 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part3 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part4 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part5 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part1 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part2 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part3 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part4 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part5 \
    --model_name qwen3_32b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 5336897 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part1
#! Submitted batch job 5336898 -> 36_mren -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part2
#! Submitted batch job 5336899 -> 219_courant -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part3
#! Submitted batch job 5336900 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part4
#! Submitted batch job 5336901 -> 36_mren -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-4_part5
#! Submitted batch job 5336903 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part1
#! Submitted batch job 5336904 -> 36_mren -- arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part2
#! Submitted batch job 5336905 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part3
#! Submitted batch job 5336906 -> 36_mren -- arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part4
#! Submitted batch job 5336907 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr3e-4_part5
#! Submitted batch job 5336908 -> 36_mren -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part1
#! Submitted batch job 5336909 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part2
#! Submitted batch job 5336910 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part3
#! Submitted batch job 5336911 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part4
#! Submitted batch job 5336912 -> 36_cds -- arc_largellm_ctkv_loo_qwen3_32b_lr1e-3_part5
