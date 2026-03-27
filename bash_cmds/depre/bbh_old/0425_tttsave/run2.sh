# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_bbh/0425_tttsave/run2.sh
MASTER_PORT=$(comm -23 <(seq 10000 6125 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter8 save run2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run2_seed42 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 42

# bbh llama8b ttt iter8 save run2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run2_seed43 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 43

# bbh llama8b ttt iter8 save run2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run2_seed44 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 44

# bbh llama8b ttt iter8 save run2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run2_seed45 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 45

# bbh llama8b ttt iter8 save run2 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8_save_run2_seed46 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 125 \
    --ttt_save \
    --seed 46

# 55.72510862474306
# 55.090278107476514
# 55.10700921925581
# 55.3139523229254
# 55.965406250384646