# python make_sbatch.py --ngpu 1 --time 7 --bash_files bash_cmds/0401_arc/smallgridsearch400/lr1e-3_dropnone_tokendrop0.1_lambda0.sh

# arc gs100 lr1e-3 dropnone tokendrop0.1 lambda0 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-3_dropnone_tokendrop0.1_lambda0_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# arc gs150 lr1e-3 dropnone tokendrop0.1 lambda0 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs150_lr1e-3_dropnone_tokendrop0.1_lambda0_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# arc gs200 lr1e-3 dropnone tokendrop0.1 lambda0 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs200_lr1e-3_dropnone_tokendrop0.1_lambda0_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# arc gs250 lr1e-3 dropnone tokendrop0.1 lambda0 400task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-3_dropnone_tokendrop0.1_lambda0_400task \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_lambda_param_sqr 0.0

# Submitted batch job 59850131
# Submitted batch job 59850132
# Submitted batch job 59850133
# Submitted batch job 59850134