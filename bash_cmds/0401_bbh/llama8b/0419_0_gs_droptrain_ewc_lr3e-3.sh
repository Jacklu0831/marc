# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_0_gs_droptrain_ewc_lr3e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e0 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0

# bbh llama8b gs20 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e0 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0

# bbh llama8b gs30 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e0 \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0

# bbh llama8b gs40 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e0 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0

# bbh llama8b gs50 lr3e-3 droptrain lambda1e0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e0 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e0











# bbh llama8b gs10 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1

# bbh llama8b gs20 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1

# bbh llama8b gs30 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1

# bbh llama8b gs40 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1

# bbh llama8b gs50 lr3e-3 droptrain lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e-1 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-1







# bbh llama8b gs10 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e-2 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2

# bbh llama8b gs20 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e-2 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2

# bbh llama8b gs30 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e-2 \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2

# bbh llama8b gs40 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e-2 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2

# bbh llama8b gs50 lr3e-3 droptrain lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e-2 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-2











# bbh llama8b gs10 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr3e-3_droptrain_lambda1e-3 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3

# bbh llama8b gs20 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr3e-3_droptrain_lambda1e-3 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3

# bbh llama8b gs30 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr3e-3_droptrain_lambda1e-3 \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3

# bbh llama8b gs40 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr3e-3_droptrain_lambda1e-3 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3

# bbh llama8b gs50 lr3e-3 droptrain lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr3e-3_droptrain_lambda1e-3 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e-3

# beat 51.62924586030268

# lr1e0
# 53.39329454317823
# 53.27285779346263
# 53.02780591206979
# 52.738624954061976
# 51.937489999138386

# lr1e-1
# 53.622534073265214
# 53.9199596976267
# 53.499411377859374
# 53.1702621412658
# 53.10612474745077

# lr1e-2
# 52.84021172813131
# 53.053612531013655
# 53.34783128568439
# 53.12277013755911
# 53.03567911784926

# lr1e-3
# 50.927187573083216
# 51.698486990527535
# 51.16320790714277
# 50.9285393378946
# out of time