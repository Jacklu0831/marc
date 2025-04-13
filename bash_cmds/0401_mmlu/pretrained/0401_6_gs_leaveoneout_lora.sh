# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_6_gs_leaveoneout_lora.sh



# nlp gs5 lr1e-3 leaveoneout lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout lora1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout_lora1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-3 \
    --eval_seeds 100











# nlp gs5 lr1e-3 leaveoneout lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout lora1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout_lora1e-4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-4 \
    --eval_seeds 100








# nlp gs5 lr1e-3 leaveoneout lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs5_lr1e-3_leaveoneout_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# nlp gs25 lr1e-3 leaveoneout lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs25_lr1e-3_leaveoneout_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# nlp gs100 lr1e-3 leaveoneout lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs100_lr1e-3_leaveoneout_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100

# nlp gs250 lr1e-3 leaveoneout lora1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_1_deepthinking_separatekv/test_time_evaluate.py \
    --tag gs250_lr1e-3_leaveoneout_lora1e-5 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_lora \
    --gs_lora_lr 1e-5 \
    --eval_seeds 100



# lora1e-3
# Submitted batch job 59191441
# Submitted batch job 59191442
# Submitted batch job 59191443
# Submitted batch job 59191444

# lora1e-4
# Submitted batch job 59191445
# Submitted batch job 59191446
# Submitted batch job 59191447
# Submitted batch job 59191448

# lora1e-5
# Submitted batch job 59191449
# Submitted batch job 59191450
# Submitted batch job 59191451
# Submitted batch job 59191452