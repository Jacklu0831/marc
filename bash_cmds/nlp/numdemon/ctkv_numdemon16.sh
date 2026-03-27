# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/numdemon/ctkv_numdemon16.sh

# nlp gs50 lr1e-3 tokendrop0.05 ndemon16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_ndemon16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 16 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs100 lr1e-3 tokendrop0.05 ndemon16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_ndemon16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 16 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs150 lr1e-3 tokendrop0.05 ndemon16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_ndemon16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 16 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs200 lr1e-3 tokendrop0.05 ndemon16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_ndemon16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 16 \
    --task_list data/nlp_high_demo_task_list.txt

# old
# Submitted batch job 64181678
# Submitted batch job 64181679
# Submitted batch job 64181680
# Submitted batch job 64181681

# Submitted batch job 64205607
# Submitted batch job 64205608
# Submitted batch job 64205609
# Submitted batch job 64205610