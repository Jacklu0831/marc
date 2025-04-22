# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0419_0_gs_droptrain_ewc_fisher_highlr.sh
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

# Submitted batch job 59572875