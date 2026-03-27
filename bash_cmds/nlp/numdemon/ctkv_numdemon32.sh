# NOTE: using a100 for high bs
# python make_sbatch.py --ngpu 1 --time 8 --bash_files bash_cmds/nlp/numdemon/ctkv_numdemon32.sh

# nlp gs50 lr1e-3 tokendrop0.05 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs50_lr1e-3_tokendrop0.05_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs100 lr1e-3 tokendrop0.05 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_tokendrop0.05_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs150 lr1e-3 tokendrop0.05 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp gs200 lr1e-3 tokendrop0.05 ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0.05 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --gs_batch_size 32 \
    --task_list data/nlp_high_demo_task_list.txt

# old
# Submitted batch job 64181695
# Submitted batch job 64181696
# Submitted batch job 64181697
# Submitted batch job 64181698

# Submitted batch job 64205620
# Submitted batch job 64205621
# Submitted batch job 64205622
# Submitted batch job 64205623