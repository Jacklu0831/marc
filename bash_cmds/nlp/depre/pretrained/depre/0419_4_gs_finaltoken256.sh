# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_4_gs_finaltoken256.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-3 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# nlp gs25 lr1e-3 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# nlp gs100 lr1e-3 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# nlp gs250 lr1e-3 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_final_tokens 256 \
    --eval_seeds 100










# nlp gs5 lr1e-4 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-4_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# nlp gs25 lr1e-4 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-4_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# nlp gs100 lr1e-4 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-4_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# nlp gs250 lr1e-4 finaltoken256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-4_finaltoken256 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_final_tokens 256 \
    --eval_seeds 100

# Submitted batch job 59643133

# lr1e-3
# 0.3741894558637961
# 0.39592772448352326
# 0.397279784942931
# 0.39791190196526655 <-

# lr1e-4
# 0.36126562691963254
# 0.36693059871265776
# 0.37721356685487856
# 0.39727731175521747 <-