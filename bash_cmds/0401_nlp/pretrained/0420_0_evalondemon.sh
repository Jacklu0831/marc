# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0420_0_evalondemon.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# nlp evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_evalondemon \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --eval_seeds 100 \
    --eval_on_demonstrations





# nlp ttt iter5 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter5_evalondemon \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 5000 \
    --eval_seeds 100 \
    --eval_on_demonstrations

# nlp ttt iter25 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter25_evalondemon \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 5000 \
    --eval_seeds 100 \
    --eval_on_demonstrations

# nlp ttt iter100 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter100_evalondemon \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 5000 \
    --eval_seeds 100 \
    --eval_on_demonstrations

# nlp ttt iter250 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_evalondemon \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 5000 \
    --eval_seeds 100 \
    --eval_on_demonstrations

# Submitted batch job 59510309