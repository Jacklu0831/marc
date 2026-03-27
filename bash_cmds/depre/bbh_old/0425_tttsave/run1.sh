# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/0425_tttsave/run1.sh
MASTER_PORT=$(comm -23 <(seq 10000 6125 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter8 save run1 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run1_seed42 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 42

# bbh llama8b ttt iter8 save run1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run1_seed43 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 43

# bbh llama8b ttt iter8 save run1 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run1_seed44 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 44

# bbh llama8b ttt iter8 save run1 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run1_seed45 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 45

# bbh llama8b ttt iter8 save run1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run1_seed46 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 46

# 56.01943947698351
# 54.88261186458261
# 55.10070098346934
# 55.26700761911796
# 56.18257660879794