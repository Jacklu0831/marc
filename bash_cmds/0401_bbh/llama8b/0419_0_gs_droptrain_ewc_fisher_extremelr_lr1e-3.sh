# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_0_gs_droptrain_ewc_fisher_extremelr.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr1e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs20 lr1e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs30 lr1e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs40 lr1e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher

# bbh llama8b gs50 lr1e-3 droptrain lambda1e5 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_lambda1e5_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e5 \
    --gs_fisher










# bbh llama8b gs10 lr1e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs20 lr1e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs30 lr1e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs40 lr1e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# bbh llama8b gs50 lr1e-3 droptrain lambda1e4 fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_droptrain_lambda1e4_fisher \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train \
    --gs_lambda_param_sqr 1e4 \
    --gs_fisher

# all running

# lambda1e5
# 50.70366282106061
# 51.38128603607168
# 51.660222155403275
# 51.50470546033856
# 52.00742569469721 <-

# lambda1e4
# 51.91025688366997
# 51.745765789052605
# 51.57605006848942
# 51.65246983915978
# 52.10437250857657 <-

# so far 52.10437250857657
# have not converged, need higher lr?