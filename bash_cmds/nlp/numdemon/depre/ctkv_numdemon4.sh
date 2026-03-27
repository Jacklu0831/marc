# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/nlp/numdemon/ctkv_numdemon4.sh

# nlp gs25 lr1e-3 tokendrop0.05 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_tokendrop0.05_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 4 \
    --eval_seeds 13 21 42 100

# nlp gs50 lr1e-3 tokendrop0.05 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 4 \
    --eval_seeds 13 21 42 100

# nlp gs100 lr1e-3 tokendrop0.05 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 4 \
    --eval_seeds 13 21 42 100

# nlp gs150 lr1e-3 tokendrop0.05 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 4 \
    --eval_seeds 13 21 42 100

# failed
# Submitted batch job 60534953
# Submitted batch job 60534954
# Submitted batch job 60534955
# Submitted batch job 60534956

# new
# Submitted batch job 60547193
# Submitted batch job 60547194
# Submitted batch job 60547195
# Submitted batch job 60547196

# 0.35911957275787093
# 0.3641908038349766
# 0.3668215259158583
# 0.3657902379973486