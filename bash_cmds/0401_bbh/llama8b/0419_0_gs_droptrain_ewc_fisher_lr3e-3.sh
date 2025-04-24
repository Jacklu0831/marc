# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_0_gs_droptrain_ewc_fisher_lr3e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs20 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs30 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs40 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher

# bbh llama8b gs50 lr3e-3 droptrain lambda1e3 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e3_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e3 \
    --gs_fisher











# bbh llama8b gs10 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs20 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs30 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs40 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher

# bbh llama8b gs50 lr3e-3 droptrain lambda1e2 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e2_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e2 \
    --gs_fisher













# bbh llama8b gs10 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs20 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs30 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs40 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher

# bbh llama8b gs50 lr3e-3 droptrain lambda1e1 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e1_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e1 \
    --gs_fisher











# bbh llama8b gs10 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs20 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs30 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs40 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# bbh llama8b gs50 lr3e-3 droptrain lambda1e0 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e0_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0 \
    --gs_fisher

# Submitted batch job 59650819
# Submitted batch job 59650820

# beat 51.62924586030268