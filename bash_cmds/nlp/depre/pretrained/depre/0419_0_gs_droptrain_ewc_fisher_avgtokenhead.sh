# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_0_gs_droptrain_ewc_fisher_avgtokenhead.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr3e-3 droptrain lambda1e7 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e7_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e7 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e7_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e7 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e7_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e7 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e7_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100










# nlp gs5 lr3e-3 droptrain lambda1e6 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e6_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e6 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e6_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e6 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e6_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e6 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e6_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e5 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e5_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e5 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e5_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e5 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e5_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e5 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e5_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100









# nlp gs5 lr3e-3 droptrain lambda1e4 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e4_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e4 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e4_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e4 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e4_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e4 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e4_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100









# nlp gs5 lr3e-3 droptrain lambda1e3 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e3_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e3 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e3_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e3 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e3_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e3 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e3_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e2 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e2_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e2 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e2_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e2 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e2_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e2 fisher avgtokenhead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e2_fisher_avgtokenhead \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 1 3 \
    --eval_seeds 100




# trying to beat 0.42813997495334793

# Submitted batch job 59674799

# lr1e7
# 0.38564524206962003
# 0.3900694583502985
# 0.3836945760321485
# 0.3878521718681111

# lr1e6
# 0.3850146718978892
# 0.39544281173754375
# 0.42074378474924645
# 0.4079333096580079

# lr1e5
# 0.39590597098012653
# 0.4206094991376383
# 0.4359096640603831

# lr1e4
# lr1e3
# lr1e2
