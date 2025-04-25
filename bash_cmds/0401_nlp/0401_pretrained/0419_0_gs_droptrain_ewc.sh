# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_0_gs_droptrain_ewc.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e-1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1 \
    --eval_seeds 100





# nlp gs5 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e-2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2 \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3 \
    --eval_seeds 100





# Submitted batch job 59573176
# Submitted batch job 59573178
# trying to beat 0.42813997495334793

# lambda1e0 (too high)
# 0.37533228954544107 <-
# 0.366221744631127
# 0.36232199755453565
# 0.3633051690753347

# lambda1e-1 (too high)
# 0.37862185100348106
# 0.38912873016126537
# 0.3961167396064532 <-
# 0.38762995072097955

# lambda1e-2 (just right)
# 0.40473503880186923
# 0.41722392026211774
# 0.44557514948350246 <-
# 0.43868165641562296

# lambda1e-3
# 0.39954825906174063
# 0.43062596260019587
# 0.44019110190335314 <-
# 0.4395168749117748

# wait, increased to 0.44557514948350246