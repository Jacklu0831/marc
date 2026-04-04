# ARC CT-KV with Qwen3-8B — v2 sweep: full grid (part2 only)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_ctkv_qwen3_8b_v2_part2.sh

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1


# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0


# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1


# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1


# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0


# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0


# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1


# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1


# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0


# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0


# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1


# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.1


# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0


# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.0


# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1


# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 3e-4 --gs_dropout none --gs_token_dropout 0.1


# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.0


# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.0


# arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1


# arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 100 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5105322 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td01_part2
#! Submitted batch job 5105323 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td0_part2
#! Submitted batch job 5105324 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td01_part2
#! Submitted batch job 5105325 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr3e-4_td01_part2
#! Submitted batch job 5105326 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td0_part2
#! Submitted batch job 5105327 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td0_part2
#! Submitted batch job 5105328 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-3_td01_part2
#! Submitted batch job 5105329 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-3_td01_part2
#! Submitted batch job 5105330 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td0_part2
#! Submitted batch job 5105331 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td0_part2
#! Submitted batch job 5105332 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-4_td01_part2
#! Submitted batch job 5105333 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-4_td01_part2
#! Submitted batch job 5105334 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td0_part2
#! Submitted batch job 5105335 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td0_part2
#! Submitted batch job 5105336 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr3e-4_td01_part2
#! Submitted batch job 5105337 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr3e-4_td01_part2
#! Submitted batch job 5105343 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td0_part2
#! Submitted batch job 5105344 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td0_part2
#! Submitted batch job 5105345 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e100_lr1e-3_td01_part2
#! Submitted batch job 5105346 -> 36_mren -- arc_bigllm_ctkv_noloo_qwen3_8b_e100_lr1e-3_td01_part2
