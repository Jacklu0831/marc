# Demonstration ordering sensitivity: BBH ICL
# 5 demo orderings (demo_shuffle_seed 0-4), seed=42 (same demo selection)
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_0_demo_ordering/bbh_icl.sh

# bbh_order_icl_dss0
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_icl_dss0 --seed 42 --demo_shuffle_seed 0

# bbh_order_icl_dss1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_icl_dss1 --seed 42 --demo_shuffle_seed 1

# bbh_order_icl_dss2
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_icl_dss2 --seed 42 --demo_shuffle_seed 2

# bbh_order_icl_dss3
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_icl_dss3 --seed 42 --demo_shuffle_seed 3

# bbh_order_icl_dss4
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_icl_dss4 --seed 42 --demo_shuffle_seed 4

#! Submitted batch job 5072887 -> 36_mren -- bbh_icl
