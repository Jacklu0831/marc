# ARC CT-KV+LOO with off-the-shelf Qwen2.5-14B-Instruct (4-bit quantized)
# Addresses Reviewer 2: "How does CT-KV perform on ARC using an off-the-shelf model NOT fine-tuned on ARC?"
# Uses same hyperparams as ARC CT-KV best config (lr=3e-3, epochs=200, tokdrop=0.1)
# LOO masking enabled (gs_dropout=train) to test whether LOO helps with a stronger base model
# --flash_attn: required for 14B GS — without it, attention activations OOM on H200
# --gs_float16: keep KV in bf16 during GS to save memory
# makesbatch --time 10 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_ctkv_loo.sh

# arc_bigllm_ctkv_loo_qwen14b_part1
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen14b_part1 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1 --gs_float16

# arc_bigllm_ctkv_loo_qwen14b_part2
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen14b_part2 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1 --gs_float16

# arc_bigllm_ctkv_loo_qwen14b_part3
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen14b_part3 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1 --gs_float16

# arc_bigllm_ctkv_loo_qwen14b_part4
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen14b_part4 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1 --gs_float16

# arc_bigllm_ctkv_loo_qwen14b_part5
.venv/bin/accelerate launch --mixed_precision bf16 \
    inference_arc_bigllm/test_time_evaluate.py \
    --tag arc_bigllm_ctkv_loo_qwen14b_part5 \
    --model_name qwen14b --untrainable_nbit 4 --flash_attn --gradient_checkpointing \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos --seed 42 \
    --gs_epochs 200 --gs_lr 3e-3 --gs_dropout train --gs_token_dropout 0.1 --gs_float16
