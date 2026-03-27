# python make_sbatch.py --ngpu 1 --time 3 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0402_0_ttt_lr1e-3.sh
# we hope ttt lr1e-3 fails, that it cannot converge in few iters

# nlp ttt5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt5_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 2000 \
    --ttt_lr 1e-3 \
    --eval_seeds 100

# nlp ttt10 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt10_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 10 \
    --ttt_permute_n 2000 \
    --ttt_lr 1e-3 \
    --eval_seeds 100

# nlp ttt25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt25_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 2000 \
    --ttt_lr 1e-3 \
    --eval_seeds 100

# nlp ttt100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt100_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 2000 \
    --ttt_lr 1e-3 \
    --eval_seeds 100

# nlp ttt250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt250_lr1e-3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 2000 \
    --ttt_lr 1e-3 \
    --eval_seeds 100


# iter5, 10, 25, 100, 250

# Submitted batch job 59298635 # 0.385
# Submitted batch job 59298636 # 0.419
# Submitted batch job 59298637 # 0.429
# Submitted batch job 59298638 # 0.437
# Submitted batch job 59298639 # 0.414