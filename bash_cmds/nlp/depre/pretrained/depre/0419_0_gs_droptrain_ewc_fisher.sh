# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_0_gs_droptrain_ewc_fisher.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e3_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e3_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e3_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e3_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e2_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e2_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e2_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e2_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --eval_seeds 100












# nlp gs5 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e1_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e0_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher \
    --eval_seeds 100




# Submitted batch job 59650426
# Submitted batch job 59572875
# Submitted batch job 59573175
# trying to beat 0.42813997495334793

# lambda1e7
# 0.38508141570326454
# 0.38749374922955143 <-
# 0.38113339515138994
# 0.3819515111948998

# lambda1e6
# 0.37974366329581344
# 0.39435689663885964
# 0.4136248634390987
# 0.4158084009388634 <-

# lambda1e5
# 0.3914862628594633
# 0.4156599445949746
# 0.43278563763688366 <-
# 0.4141054271829517

# lambda1e4
# time out

# lambda1e3
# 0.3969080292163436
# 0.4253368153650011
# 0.43719228978367525 <-
# 0.4356515626179674

# lambda1e2
# 0.39754950155427
# 0.4294250481452885 <-
# 0.4267851297994569
# 0.4266314312538634

# lambda1e1
# 0.3969621048845574
# 0.4266254906591249
# 0.42634218810483077
# 0.4323655815063771 <-

# lambda1e0
# 0.3961605877196826
# 0.4306966403806777 <-
# 0.42446554897603966
# 0.4275463175584179

# 0.43719228978367525, good but need higher (avg fisher for nlp is 1e-9 1e-8)