# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_2_gs_freezeinstruct.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# bbh llama8b gs4 lr1e-3 freezeinstruct
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_lr1e-3_freezeinstruct \
    --model_name llama8b \
    --gs_iters 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_freeze_instruct

# bbh llama8b gs6 lr1e-3 freezeinstruct
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_lr1e-3_freezeinstruct \
    --model_name llama8b \
    --gs_iters 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_freeze_instruct

# bbh llama8b gs8 lr1e-3 freezeinstruct
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_lr1e-3_freezeinstruct \
    --model_name llama8b \
    --gs_iters 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_freeze_instruct

# bbh llama8b gs10 lr1e-3 freezeinstruct
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_freezeinstruct \
    --model_name llama8b \
    --gs_iters 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_freeze_instruct

# bbh llama8b gs12 lr1e-3 freezeinstruct
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_lr1e-3_freezeinstruct \
    --model_name llama8b \
    --gs_iters 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_freeze_instruct


# Submitted batch job 59317920

# -> nothing over 50