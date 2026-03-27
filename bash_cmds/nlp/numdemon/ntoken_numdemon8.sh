# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/numdemon/ntoken_numdemon8.sh

# nlp gs50 lr1e-3 randomkv token ntoken32 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_randomkv_token_ntoken32_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 8 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 8 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs100 lr1e-3 randomkv token ntoken32 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_randomkv_token_ntoken32_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 8 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 8 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs150 lr1e-3 randomkv token ntoken32 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_randomkv_token_ntoken32_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 8 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 8 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs200 lr1e-3 randomkv token ntoken32 ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_randomkv_token_ntoken32_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --num_demonstrations 8 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 8 \
    --task_list data/nlp_high_demo_task_list.txt

# old
# Submitted batch job 64181699
# Submitted batch job 64181700
# Submitted batch job 64181701
# Submitted batch job 64181702

# Submitted batch job 64205634
# Submitted batch job 64205635
# Submitted batch job 64205636
# Submitted batch job 64205637