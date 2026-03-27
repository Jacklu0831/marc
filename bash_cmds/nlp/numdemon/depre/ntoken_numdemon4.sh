# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/nlp/numdemon/ntoken_numdemon4.sh

# nlp gs25 lr1e-3 randomkv token ntoken32 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_randomkv_token_ntoken32_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 4

# nlp gs50 lr1e-3 randomkv token ntoken32 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_randomkv_token_ntoken32_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 4

# nlp gs100 lr1e-3 randomkv token ntoken32 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token_ntoken32_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 4

# nlp gs150 lr1e-3 randomkv token ntoken32 ndemon4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_randomkv_token_ntoken32_ndemon4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 4

# Submitted batch job 60534969
# Submitted batch job 60534970
# Submitted batch job 60534971
# Submitted batch job 60534972

# 0.3561018903258843
# 0.36732170481044696
# 0.378467845182865
# 0.3830193458956788