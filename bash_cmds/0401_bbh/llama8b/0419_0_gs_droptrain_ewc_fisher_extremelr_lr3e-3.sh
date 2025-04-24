# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_0_gs_droptrain_ewc_fisher_extremelr_lr3e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr3e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs20 lr3e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs30 lr3e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs40 lr3e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs50 lr3e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher










# bbh llama8b gs10 lr3e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs20 lr3e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs30 lr3e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs40 lr3e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs50 lr3e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# lambda1e5
# 46.28825824114956
# 47.806096701078424
# 48.7011210636169
# 49.77083739961772
# 49.308577793954996

# lambda1e4
# 48.97161909333727
# 49.60855089053827
# 50.29328680624793
# 50.385101309827135
# 50.62764308485479

# beat 51.62924586030268