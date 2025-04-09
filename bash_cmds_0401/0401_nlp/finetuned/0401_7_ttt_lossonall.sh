# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_7_ttt_lossonall.sh

# nlp ft ttt5 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt5_permuten1000_allloss \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ft ttt25 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt25_permuten1000_allloss \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ft ttt100 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt100_permuten1000_allloss \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ft ttt250 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_allloss \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ft ttt500 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt500_permuten1000_allloss \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# Submitted batch job 59123442
# Submitted batch job 59123443
# Submitted batch job 59123444
# Submitted batch job 59123445
# Submitted batch job 59123446