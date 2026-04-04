# ICL baseline across num_demonstrations on MMLU
# makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file bash_cmds/0327_1_adaptive_loo/mmlu_icl_vs_ndemo_s43.sh

# mmlu_icl_s43_nd2
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd2 --seed 43 \
    --num_demonstrations 2

# mmlu_icl_s43_nd3
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd3 --seed 43 \
    --num_demonstrations 3

# mmlu_icl_s43_nd4
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd4 --seed 43 \
    --num_demonstrations 4

# mmlu_icl_s43_nd6
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd6 --seed 43 \
    --num_demonstrations 6

# mmlu_icl_s43_nd8
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd8 --seed 43 \
    --num_demonstrations 8

# mmlu_icl_s43_nd12
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd12 --seed 43 \
    --num_demonstrations 12

# mmlu_icl_s43_nd16
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_icl_s43_nd16 --seed 43 \
    --num_demonstrations 16

#! Submitted batch job 5056399 -> 36_cds -- mmlu_icl_vs_ndemo_s43
