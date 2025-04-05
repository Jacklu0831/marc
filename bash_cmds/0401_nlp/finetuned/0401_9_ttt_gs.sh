# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_9_ttt_gs.sh

# ft nlp ttt250 permuten1000 gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs5_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs25_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs100_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs250_lr1e-3 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100








# ft nlp ttt250 permuten1000 gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs5_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs25_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs100_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# ft nlp ttt250 permuten1000 gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_ttt250_permuten1000_gs250_lr1e-4 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --eval_seeds 100

# Submitted batch job 59034461
# Submitted batch job 59034462
# Submitted batch job 59034463
# Submitted batch job 59034464

# Submitted batch job 59034465
# Submitted batch job 59034466
# Submitted batch job 59034467
# Submitted batch job 59034468