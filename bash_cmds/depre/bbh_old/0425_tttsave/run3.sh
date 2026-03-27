# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/0425_tttsave/run3.sh
MASTER_PORT=$(comm -23 <(seq 10000 6125 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter8 save run3 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run3_seed42 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 42

# bbh llama8b ttt iter8 save run3 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run3_seed43 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 43

# bbh llama8b ttt iter8 save run3 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run3_seed44 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 44

# bbh llama8b ttt iter8 save run3 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run3_seed45 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 45

# bbh llama8b ttt iter8 save run3 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run3_seed46 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 46

# 55.643613691553206
# 54.990728871586306
# 55.24226702608224
# 55.30699567962779
# 56.1579459153404