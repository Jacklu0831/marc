# ARC CT-KV with Qwen3-8B — v2 sweep: full grid (part4 only)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_ctkv_qwen3_8b_v2_part4.sh

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5105347 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part4
#! Submitted batch job 5105348 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part4
#! Submitted batch job 5105349 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part4
#! Submitted batch job 5105350 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part4
#! Submitted batch job 5105351 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part4
#! Submitted batch job 5105352 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part4
#! Submitted batch job 5105353 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part4
#! Submitted batch job 5105354 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part4
#! Submitted batch job 5105355 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part4
#! Submitted batch job 5105356 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part4
#! Submitted batch job 5105357 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part4
#! Submitted batch job 5105358 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part4
#! Submitted batch job 5105359 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part4
#! Submitted batch job 5105360 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part4
#! Submitted batch job 5105361 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part4
#! Submitted batch job 5105362 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part4
#! Submitted batch job 5105363 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part4
#! Submitted batch job 5105364 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part4
#! Submitted batch job 5105365 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part4
#! Submitted batch job 5105366 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part4
