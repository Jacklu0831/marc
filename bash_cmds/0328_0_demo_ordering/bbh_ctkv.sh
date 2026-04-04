# Demonstration ordering sensitivity: BBH CT-KV
# 5 demo orderings (demo_shuffle_seed 0-4), seed=42 (same demo selection)
# CT-KV config from 0327_1_adaptive_loo: lr=1e-3, epochs=20, tokdrop=0.1, batch_size=2
# makesbatch --time 6 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_0_demo_ordering/bbh_ctkv.sh

# bbh_order_ctkv_dss0
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_ctkv_dss0 --seed 42 --demo_shuffle_seed 0 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1

# bbh_order_ctkv_dss1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_ctkv_dss1 --seed 42 --demo_shuffle_seed 1 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1

# bbh_order_ctkv_dss2
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_ctkv_dss2 --seed 42 --demo_shuffle_seed 2 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1

# bbh_order_ctkv_dss3
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_ctkv_dss3 --seed 42 --demo_shuffle_seed 3 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1

# bbh_order_ctkv_dss4
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_order_ctkv_dss4 --seed 42 --demo_shuffle_seed 4 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 5072895 -> 36_mren -- bbh_ctkv
