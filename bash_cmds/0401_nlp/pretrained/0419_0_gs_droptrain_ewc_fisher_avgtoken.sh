# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_0_gs_droptrain_ewc_fisher_avgtoken.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr3e-3 droptrain lambda1e7 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e7_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e7 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e7_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e7 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e7_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e7 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e7_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e7 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100










# nlp gs5 lr3e-3 droptrain lambda1e6 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e6_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e6 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e6_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e6 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e6_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e6 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e6_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e6 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e5 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e5_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e5 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e5_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e5 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e5_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e5 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e5_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100









# nlp gs5 lr3e-3 droptrain lambda1e4 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e4_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e4 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e4_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e4 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e4_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e4 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e4_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100









# nlp gs5 lr3e-3 droptrain lambda1e3 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e3_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e3 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e3_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e3 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e3_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e3 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e3_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100











# nlp gs5 lr3e-3 droptrain lambda1e2 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_lambda1e2_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain lambda1e2 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_lambda1e2_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain lambda1e2 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_lambda1e2_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain lambda1e2 fisher avgtoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_lambda1e2_fisher_avgtoken \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher \
    --gs_fisher_avg_dims 3 \
    --eval_seeds 100




# trying to beat 0.42813997495334793

# lr1e7
# 0.3868191008960275
# 0.38503993092859945
# 0.38122562925982817
# 0.3798231657075834

# lr1e6
# 0.3807556584066814
# 0.3938788216449159
# 0.4162461660563947 <-
# 0.40469117465668974

# lr1e5
# 0.3924657306101477
# 0.4159852137006166
# 0.42753405488948976 <-
# 0.410617217287342

# lr1e4
# 0.3962110180712557
# 0.42461910796458696
# 0.4328731096418125 <-
# 0.42554364933209593

# lr1e3
# 0.39710160058351157
# 0.4253796937417139
# 0.43367196478411296
# 0.43674697618856323 <-

# lr1e2
# 0.39626102156381793
# 0.42749045910180633
# 0.42945104086125113 <-
# 0.42734796562548394