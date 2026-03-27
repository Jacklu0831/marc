# python make_sbatch.py --ngpu 1 --time 10 --rtx8000 --bash_files bash_cmds/nlp/ablation/noboth.sh

# nlp gs100 lr1e-3 dropnone tokendrop0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_dropnone_tokendrop0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0

# nlp gs150 lr1e-3 dropnone tokendrop0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_dropnone_tokendrop0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0

# nlp gs200 lr1e-3 dropnone tokendrop0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_dropnone_tokendrop0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0

# nlp gs250 lr1e-3 dropnone tokendrop0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_dropnone_tokendrop0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0

# Submitted batch job 60520570
# Submitted batch job 60520571
# Submitted batch job 60520572
# Submitted batch job 60520573