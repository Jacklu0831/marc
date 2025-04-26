# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_mmlu/pretrained/0421_2_dt.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# mmlu dt iter2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter2 \
    --dt_iters 2

# mmlu dt iter4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter4 \
    --dt_iters 4

# mmlu dt iter6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter6 \
    --dt_iters 6

# mmlu dt iter8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter8 \
    --dt_iters 8

# mmlu dt iter10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter10 \
    --dt_iters 10

# mmlu dt iter12
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter12 \
    --dt_iters 12

# mmlu dt iter15
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_dt_iter15 \
    --dt_iters 15

# 2: 41.4872539378376
# 4: 41.14252599495466
# 6: 41.14502500876742
# 8: 40.41652850100631
# 10: 39.67358911326241
# 12: 39.952832529509145
# 15: 39.61504940848292