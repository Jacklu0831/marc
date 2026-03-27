# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/nlp/numdemon/icl.sh
# MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp icl ndemon8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_ndemon8 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --num_demonstrations 8 \
    --filter_based_on_ndemo 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp icl ndemon16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_ndemon16 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --num_demonstrations 16 \
    --filter_based_on_ndemo 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp icl ndemon24
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_ndemon24 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --num_demonstrations 24 \
    --filter_based_on_ndemo 32 \
    --task_list data/nlp_high_demo_task_list.txt

# nlp icl ndemon32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_icl_ndemon32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --num_demonstrations 32 \
    --filter_based_on_ndemo 32 \
    --task_list data/nlp_high_demo_task_list.txt

# old
# Submitted batch job 64181670
# Submitted batch job 64181671
# Submitted batch job 64181672
# Submitted batch job 64181673

# Submitted batch job 64205625
# Submitted batch job 64205626
# Submitted batch job 64205627
# Submitted batch job 64205628