# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --bash_files bash_cmds/nlp/numdemon/ctkv_numdemon2.sh

# nlp gs10 lr1e-3 tokendrop0.05 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs10_lr1e-3_tokendrop0.05_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 2

# nlp gs25 lr1e-3 tokendrop0.05 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_tokendrop0.05_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 2

# nlp gs50 lr1e-3 tokendrop0.05 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 2

# nlp gs100 lr1e-3 tokendrop0.05 ndemon2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_ndemon2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 2

# Submitted batch job 60547188
# Submitted batch job 60547189
# Submitted batch job 60547190
# Submitted batch job 60547191

# 0.3492200448896764
# 0.34915523706389423
# cancelled?
# 0.33880317926520476