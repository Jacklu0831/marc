# ICL baseline across num_demonstrations on BBH
# makesbatch --time 8 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0327_1_adaptive_loo/bbh_icl_vs_ndemo_s42.sh

# bbh_icl_s42_nd2
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_icl_s42_nd2 --seed 42 \
    --num_demonstrations 2

# bbh_icl_s42_nd3
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_icl_s42_nd3 --seed 42 \
    --num_demonstrations 3

# bbh_icl_s42_nd4
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_icl_s42_nd4 --seed 42 \
    --num_demonstrations 4

# bbh_icl_s42_nd6
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_icl_s42_nd6 --seed 42 \
    --num_demonstrations 6

# bbh_icl_s42_nd8
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_icl_s42_nd8 --seed 42 \
    --num_demonstrations 8

# bbh_icl_s42_nd10
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_icl_s42_nd10 --seed 42 \
    --num_demonstrations 10

#! Submitted batch job 5043766 -> 219_courant -- bbh_icl_vs_ndemo_s42
