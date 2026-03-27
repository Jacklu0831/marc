# python make_sbatch.py --ngpu 1 --time 10 --rtx8000 --bash_files bash_cmds/nlp/numdemon/ctkv_numdemon8.sh

# nlp gs50 lr1e-3 tokendrop0.05 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 8

# nlp gs100 lr1e-3 tokendrop0.05 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 8

# nlp gs150 lr1e-3 tokendrop0.05 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 8

# nlp gs200 lr1e-3 tokendrop0.05 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 8

# Submitted batch job 60534959
# Submitted batch job 60534960
# Submitted batch job 60534961
# Submitted batch job 60534962

# 0.40679349224401806
# 0.4144665248931635
# 0.4122153556526422
# 0.4119471279892104
