# python make_sbatch.py --ngpu 1 --time 15 --rtx8000 --bash_files bash_cmds/nlp/2_randomhiddenntoken/lr1e-3.sh

# nlp gs200 lr1e-3 randomhidden token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_randomhidden_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32

# nlp gs300 lr1e-3 randomhidden token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gs300_lr1e-3_randomhidden_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 300 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32

# nlp gs400 lr1e-3 randomhidden token ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_hidden/test_time_evaluate.py \
    --tag nlp_gs400_lr1e-3_randomhidden_token_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 400 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32

# Submitted batch job 60096907
# Submitted batch job 60096908
# Submitted batch job 60096909

# 0.4066157349005408
# 0.40778879583858424
# 0.4045095123369372