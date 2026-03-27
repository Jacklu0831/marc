# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/5_tttnopermute/0.sh

# nlp ttt iter200 seed0 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter200_nopermute_seed0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 200 \
    --ttt_permute_n 1 \
    --seed 0

# nlp ttt iter300 seed0 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter300_nopermute_seed0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 300 \
    --ttt_permute_n 1 \
    --seed 0

# nlp ttt iter400 seed0 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter400_nopermute_seed0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 400 \
    --ttt_permute_n 1 \
    --seed 0

# Submitted batch job 60234540
# Submitted batch job 60234541
# Submitted batch job 60234542