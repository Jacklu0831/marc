# BBH Prefix Tuning epoch sweep: accuracy vs training epochs on full BBH (all 23 tasks)
# Addresses R4 W2: overfitting risk with large parameterization
# Hyperparams from 0326_2_cross_task (lr=3e-3, batch=2, 32 random tokens, no LOO)
# makesbatch --time 10 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0328_1_epoch_sweep/bbh_prefix.sh

# bbh_epochsweep_prefix_e4
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_epochsweep_prefix_e4 --seed 42 \
    --gs_epochs 4 --gs_lr 3e-3 --gs_batch_size 2 --gs_dropout none \
    --random_kv token --random_kv_ntokens 32

# bbh_epochsweep_prefix_e8
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_epochsweep_prefix_e8 --seed 42 \
    --gs_epochs 8 --gs_lr 3e-3 --gs_batch_size 2 --gs_dropout none \
    --random_kv token --random_kv_ntokens 32

# bbh_epochsweep_prefix_e16
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_epochsweep_prefix_e16 --seed 42 \
    --gs_epochs 16 --gs_lr 3e-3 --gs_batch_size 2 --gs_dropout none \
    --random_kv token --random_kv_ntokens 32

# bbh_epochsweep_prefix_e32
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_epochsweep_prefix_e32 --seed 42 \
    --gs_epochs 32 --gs_lr 3e-3 --gs_batch_size 2 --gs_dropout none \
    --random_kv token --random_kv_ntokens 32

# bbh_epochsweep_prefix_e64
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_epochsweep_prefix_e64 --seed 42 \
    --gs_epochs 64 --gs_lr 3e-3 --gs_batch_size 2 --gs_dropout none \
    --random_kv token --random_kv_ntokens 32

#! Submitted batch job 5073570 -> 36_mren -- bbh_prefix
