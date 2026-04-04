# LOO vs no-LOO across num_demonstrations on MMLU
# Find the crossover point where LOO stops helping
# Ablation config: lr=1e-3, epochs=20, tokdrop=0.1, seed=44
# makesbatch --time 6 --ngpu 1 --gb 64 --single --bash_file bash_cmds/0327_1_adaptive_loo/mmlu_loo_vs_ndemo_s44.sh

# # --- No LOO (gs_dropout=none) ---

# # mmlu_noloo_s44_nd2
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd2 --seed 44 \
#     --num_demonstrations 2 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# # mmlu_noloo_s44_nd3
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd3 --seed 44 \
#     --num_demonstrations 3 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# # mmlu_noloo_s44_nd4
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd4 --seed 44 \
#     --num_demonstrations 4 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# # mmlu_noloo_s44_nd6
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd6 --seed 44 \
#     --num_demonstrations 6 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# # mmlu_noloo_s44_nd8
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd8 --seed 44 \
#     --num_demonstrations 8 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# # mmlu_noloo_s44_nd12
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd12 --seed 44 \
#     --num_demonstrations 12 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# # mmlu_noloo_s44_nd16
# .venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
#     --tag mmlu_noloo_s44_nd16 --seed 44 \
#     --num_demonstrations 16 \
#     --gs_epochs 20 --gs_lr 1e-3 --gs_dropout none --gs_token_dropout 0.1

# --- Full LOO (gs_dropout=train) ---

# mmlu_loo_s44_nd2
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd2 --seed 44 \
    --num_demonstrations 2 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# mmlu_loo_s44_nd3
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd3 --seed 44 \
    --num_demonstrations 3 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# mmlu_loo_s44_nd4
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd4 --seed 44 \
    --num_demonstrations 4 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# mmlu_loo_s44_nd6
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd6 --seed 44 \
    --num_demonstrations 6 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# mmlu_loo_s44_nd8
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd8 --seed 44 \
    --num_demonstrations 8 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# mmlu_loo_s44_nd12
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd12 --seed 44 \
    --num_demonstrations 12 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

# mmlu_loo_s44_nd16
.venv/bin/accelerate launch --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_loo_s44_nd16 --seed 44 \
    --num_demonstrations 16 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 5095275 -> 219_courant -- mmlu_loo_vs_ndemo_s44
