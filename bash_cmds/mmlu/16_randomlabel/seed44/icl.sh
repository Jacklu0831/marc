# python make_sbatch.py --ngpu 1 --time 5 --single --bash_files bash_cmds/mmlu/16_randomlabel/seed44/icl.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mmlu normalicl wronglabel0.0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_wronglabel0.0_seed44 \
    --wrong_label 0.0 \
    --seed 44

# mmlu normalicl wronglabel0.25 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_wronglabel0.25_seed44 \
    --wrong_label 0.25 \
    --seed 44

# mmlu normalicl wronglabel0.5 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_wronglabel0.5_seed44 \
    --wrong_label 0.5 \
    --seed 44

# mmlu normalicl wronglabel0.75 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_wronglabel0.75_seed44 \
    --wrong_label 0.75 \
    --seed 44

# mmlu normalicl wronglabel1.0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_wronglabel1.0_seed44 \
    --wrong_label 1.0 \
    --seed 44
