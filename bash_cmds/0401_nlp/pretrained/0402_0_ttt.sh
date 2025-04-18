# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0402_0_ttt.sh

# nlp ttt5 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt5_maxpermute \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ttt25 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt25_maxpermute \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ttt100 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt100_maxpermute \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ttt250 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt250_maxpermute \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ttt500 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt500_maxpermute \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# Submitted batch job 59257389 # 0.405
# Submitted batch job 59257390 # 0.417
# Submitted batch job 59257391 # 0.454
# Submitted batch job 59257392 # 0.461
# Submitted batch job 59257393 # 0.462

# so far 0.462