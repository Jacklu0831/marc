# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0402_0_ttt.sh

# nlp ft ttt5 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt5_maxpermute \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 5 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ft ttt25 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt25_maxpermute \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 25 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ft ttt100 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt100_maxpermute \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 100 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ft ttt250 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_maxpermute \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 250 \
    --ttt_permute_n 2000 \
    --eval_seeds 100

# nlp ft ttt500 maxpermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt500_maxpermute \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --ttt_iters 500 \
    --ttt_permute_n 2000 \
    --eval_seeds 100
