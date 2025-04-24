# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_0_gs_droptrain_ewc_fisher_highlr.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr1e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs20 lr1e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs30 lr1e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs40 lr1e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs50 lr1e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher











# bbh llama8b gs10 lr1e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs20 lr1e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs30 lr1e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs40 lr1e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs50 lr1e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher













# bbh llama8b gs10 lr1e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs20 lr1e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs30 lr1e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs40 lr1e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs50 lr1e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher











# bbh llama8b gs10 lr1e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs20 lr1e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs30 lr1e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs40 lr1e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs50 lr1e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher




# Submitted batch job 59506563
# Submitted batch job 59642521
# trying to beat 53.22709780710778

# lambda1e3
# 52.50186389684563 <-
# 51.702452167307605
# 51.701942233264944
# 51.93013551936782
# 52.20129954053185

# lambda1e2
# 52.78382005658511 <-
# 51.57238162057038
# 52.0517855604263
# 52.16055097494113
# 52.051062421202005

# 52.78382005658511