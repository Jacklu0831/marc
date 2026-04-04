# ARC CT-KV with Qwen3-14B, lr=3e-4 sweep (parts 2,4 only)
# makesbatch --time 10 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_ctkv_loo_qwen3_lr3e-4.sh

# arc_bigllm_ctkv_loo_qwen3_14b_lr3e-4_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_14b_lr3e-4_part2 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1 --gs_float16

# arc_bigllm_ctkv_loo_qwen3_14b_lr3e-4_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen3_14b_lr3e-4_part4 \
    --model_name qwen3_14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-4 --gs_dropout train --gs_token_dropout 0.1 --gs_float16

#! Submitted batch job 5078226 -> 36_cds -- arc_bigllm_ctkv_loo_qwen3_14b_lr3e-4_part2
#! Submitted batch job 5078227 -> 36_cds -- arc_bigllm_ctkv_loo_qwen3_14b_lr3e-4_part4
