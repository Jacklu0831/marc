# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_mmlu/pretrained/0421_2_dt_nopos.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# mmlu dt iter2 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter2_nopos \
    --dt_iters 2 \
    --dt_no_pos

# mmlu dt iter4 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter4_nopos \
    --dt_iters 4 \
    --dt_no_pos

# mmlu dt iter6 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter6_nopos \
    --dt_iters 6 \
    --dt_no_pos

# mmlu dt iter8 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter8_nopos \
    --dt_iters 8 \
    --dt_no_pos

# mmlu dt iter10 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter10_nopos \
    --dt_iters 10 \
    --dt_no_pos

# mmlu dt iter12 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter12_nopos \
    --dt_iters 12 \
    --dt_no_pos

# mmlu dt iter15 nopos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter15_nopos \
    --dt_iters 15 \
    --dt_no_pos

# terrible nah