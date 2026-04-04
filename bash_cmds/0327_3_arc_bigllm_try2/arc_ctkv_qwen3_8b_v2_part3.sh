# ARC CT-KV with Qwen3-8B — full grid sweep (part3)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_ctkv_qwen3_8b_v2_part3.sh

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part3 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5109801 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part3
#! Submitted batch job 5109802 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part3
#! Submitted batch job 5109803 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part3
#! Submitted batch job 5109804 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part3
#! Submitted batch job 5109805 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part3
#! Submitted batch job 5109806 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part3
#! Submitted batch job 5109807 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part3
#! Submitted batch job 5109808 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part3
#! Submitted batch job 5109809 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part3
#! Submitted batch job 5109810 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part3
#! Submitted batch job 5109811 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part3
#! Submitted batch job 5109812 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part3
#! Submitted batch job 5109813 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part3
#! Submitted batch job 5109814 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part3
#! Submitted batch job 5109815 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part3
#! Submitted batch job 5109816 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part3
#! Submitted batch job 5109817 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part3
#! Submitted batch job 5109818 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part3
#! Submitted batch job 5109819 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part3
#! Submitted batch job 5109820 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part3
#! Submitted batch job 5109821 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part3
#! Submitted batch job 5109822 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part3
#! Submitted batch job 5109823 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part3
#! Submitted batch job 5109824 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part3
