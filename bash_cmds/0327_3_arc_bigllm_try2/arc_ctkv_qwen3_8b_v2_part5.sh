# ARC CT-KV with Qwen3-8B — full grid sweep (part5)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_ctkv_qwen3_8b_v2_part5.sh

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part5 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5109833 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part5
#! Submitted batch job 5109834 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part5
#! Submitted batch job 5109835 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part5
#! Submitted batch job 5109836 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part5
#! Submitted batch job 5109837 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part5
#! Submitted batch job 5109838 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part5
#! Submitted batch job 5109839 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part5
#! Submitted batch job 5109840 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part5
#! Submitted batch job 5109841 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part5
#! Submitted batch job 5109842 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part5
#! Submitted batch job 5109843 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part5
#! Submitted batch job 5109844 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part5
#! Submitted batch job 5123643 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part5
#! Submitted batch job 5123644 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part5
#! Submitted batch job 5123645 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part5
#! Submitted batch job 5123646 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part5
#! Submitted batch job 5123647 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part5
#! Submitted batch job 5123648 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part5
#! Submitted batch job 5123650 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part5
#! Submitted batch job 5123651 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part5
#! Submitted batch job 5123652 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part5
#! Submitted batch job 5123653 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part5
#! Submitted batch job 5123654 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part5
#! Submitted batch job 5123655 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part5
