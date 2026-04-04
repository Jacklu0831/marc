# ARC CT-KV with Qwen3-8B, sweep best configs from local search (parts 2,4)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm_try2/arc_ctkv_qwen3_8b.sh

# Config 1: e50 lr=1e-4 td=0.0 LOO — best from local sweep (+3.6% token, +1 exact)

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.0

# Config 2: e50 lr=1e-4 td=0.1 LOO — standard regularized version

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout train --gs_token_dropout 0.1

# Config 3: e50 lr=3e-4 td=0.0 LOO — higher LR, fastest token acc gains

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.0

# Config 1 no-LOO: e50 lr=1e-4 td=0.0 no LOO — isolate LOO contribution at scale

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part2 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

# arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part4 \
    --model_name qwen3_8b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 50 --gs_lr 1e-4 --gs_dropout none --gs_token_dropout 0.0

#! Submitted batch job 5104669 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part2
#! Submitted batch job 5104670 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td0_part4
#! Submitted batch job 5104671 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part2
#! Submitted batch job 5104672 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr1e-4_td01_part4
#! Submitted batch job 5104673 -> 219_courant -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part2
#! Submitted batch job 5104674 -> 36_mren -- arc_bigllm_ctkv_loo_qwen3_8b_e50_lr3e-4_td0_part4
#! Submitted batch job 5104675 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part2
#! Submitted batch job 5104676 -> 219_courant -- arc_bigllm_ctkv_noloo_qwen3_8b_e50_lr1e-4_td0_part4
