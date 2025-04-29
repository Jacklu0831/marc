# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_mmlu/0401_tttsave/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mmlu ttt iter20 save seed0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed0 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 100 \
    --ttt_save \
    --seed 0

# mmlu ttt iter20 save seed1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 100 \
    --ttt_save \
    --seed 1

# mmlu ttt iter20 save seed2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed2 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 100 \
    --ttt_save \
    --seed 2

# mmlu ttt iter20 save seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed3 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 100 \
    --ttt_save \
    --seed 3

# mmlu ttt iter20 save seed4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed4 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 100 \
    --ttt_save \
    --seed 4

# 43.35922231474804
# 43.7567565668607
# 42.825031716066675
# 42.65745969731513
# 43.31851720891344