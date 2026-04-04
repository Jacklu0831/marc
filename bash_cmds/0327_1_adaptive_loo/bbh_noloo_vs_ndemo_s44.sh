# No-LOO CT-KV across num_demonstrations on BBH
# Config: lr=1e-3, epochs=20, tokdrop=0.1, batch_size=2, gs_dropout=none
# makesbatch --time 8 --ngpu 1 --gb 64 --l40s --single --bash_file bash_cmds/0327_1_adaptive_loo/bbh_noloo_vs_ndemo_s44.sh

# bbh_noloo_s44_nd2
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_noloo_s44_nd2 --seed 44 \
    --num_demonstrations 2 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout none --gs_token_dropout 0.1

# bbh_noloo_s44_nd3
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_noloo_s44_nd3 --seed 44 \
    --num_demonstrations 3 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout none --gs_token_dropout 0.1

# bbh_noloo_s44_nd4
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_noloo_s44_nd4 --seed 44 \
    --num_demonstrations 4 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout none --gs_token_dropout 0.1

# bbh_noloo_s44_nd6
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_noloo_s44_nd6 --seed 44 \
    --num_demonstrations 6 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout none --gs_token_dropout 0.1

# bbh_noloo_s44_nd8
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_noloo_s44_nd8 --seed 44 \
    --num_demonstrations 8 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout none --gs_token_dropout 0.1

# bbh_noloo_s44_nd10
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_noloo_s44_nd10 --seed 44 \
    --num_demonstrations 10 \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 2 --gs_dropout none --gs_token_dropout 0.1

#! Submitted batch job 5072777 -> 36_mren -- bbh_noloo_vs_ndemo_s44
