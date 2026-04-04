# Demonstration ordering sensitivity: MMLU Prefix Tuning
# 5 demo orderings (demo_shuffle_seed 0-4), seed=42 (same demo selection)
# Same iters as CT-KV (20 epochs), 32 tokens, no LOO
# makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file bash_cmds/0328_0_demo_ordering/mmlu_prefix.sh

# mmlu_order_prefix_dss0
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_order_prefix_dss0 --seed 42 --demo_shuffle_seed 0 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1

# mmlu_order_prefix_dss1
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_order_prefix_dss1 --seed 42 --demo_shuffle_seed 1 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1

# mmlu_order_prefix_dss2
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_order_prefix_dss2 --seed 42 --demo_shuffle_seed 2 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1

# mmlu_order_prefix_dss3
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_order_prefix_dss3 --seed 42 --demo_shuffle_seed 3 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1

# mmlu_order_prefix_dss4
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_order_prefix_dss4 --seed 42 --demo_shuffle_seed 4 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_ntokens 32 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5072931 -> 36_cds -- mmlu_prefix
