# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/randommlpinit/prefixuniform_lr3e-3.sh

# nlp gs100 lr3e-3 randomkv uniform ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_randomkv_uniform_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --eval_seeds 100

# nlp gs150 lr3e-3 randomkv uniform ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr3e-3_randomkv_uniform_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --eval_seeds 100

# nlp gs200 lr3e-3 randomkv uniform ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr3e-3_randomkv_uniform_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr3e-3 randomkv uniform ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_randomkv_uniform_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_kv uniform \
    --random_kv_ntokens 32 \
    --eval_seeds 100

# Submitted batch job 60802427
# Submitted batch job 60802428
# Submitted batch job 60802429
# Submitted batch job 60802430