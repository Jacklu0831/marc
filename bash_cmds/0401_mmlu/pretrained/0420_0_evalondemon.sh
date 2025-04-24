# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_mmlu/pretrained/0420_0_evalondemon.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# mmlu evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_evalondemon \
    --eval_on_demonstrations





# mmlu ttt iter5 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter5_evalondemon \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# mmlu ttt iter10 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_evalondemon \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# mmlu ttt iter15 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_evalondemon \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# mmlu ttt iter20 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_evalondemon \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# mmlu ttt iter25 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter25_evalondemon \
    --ttt_iters 25 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# 92.98245614035088

# 95.08771929824562
# 98.94736842105263
# 99.64912280701755
# 99.29824561403508
# 99.64912280701755