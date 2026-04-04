# ARC CT-KV with Qwen3-8B — full grid sweep (part1)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_ctkv_qwen3_8b_v2_part1.sh

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part1 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5109770 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part1
#! Submitted batch job 5109771 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part1
#! Submitted batch job 5109772 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part1
#! Submitted batch job 5109773 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part1
#! Submitted batch job 5109774 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part1
#! Submitted batch job 5109775 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part1
#! Submitted batch job 5109776 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part1
#! Submitted batch job 5109777 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part1
#! Submitted batch job 5109778 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part1
#! Submitted batch job 5109779 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part1
#! Submitted batch job 5109780 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part1
#! Submitted batch job 5109781 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part1
#! Submitted batch job 5109782 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part1
#! Submitted batch job 5109783 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part1
#! Submitted batch job 5109784 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part1
#! Submitted batch job 5109785 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part1
#! Submitted batch job 5109786 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part1
#! Submitted batch job 5109787 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part1
#! Submitted batch job 5109788 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part1
#! Submitted batch job 5109789 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part1
#! Submitted batch job 5109790 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part1
#! Submitted batch job 5109791 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part1
#! Submitted batch job 5109792 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part1
#! Submitted batch job 5109793 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part1
