# NOTE: using a100 for high bs
# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_cmds/nlp/numdemon/ntoken_numdemon32.sh

# nlp gs50 lr1e-3 randomkv token ntoken32 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_randomkv_token_ntoken32_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs100 lr1e-3 randomkv token ntoken32 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token_ntoken32_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs150 lr1e-3 randomkv token ntoken32 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_randomkv_token_ntoken32_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs200 lr1e-3 randomkv token ntoken32 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_randomkv_token_ntoken32_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# old
# Submitted batch job 64181711
# Submitted batch job 64181712
# Submitted batch job 64181713
# Submitted batch job 64181714

# Submitted batch job 64205647
# Submitted batch job 64205648
# Submitted batch job 64205649
# Submitted batch job 64205650