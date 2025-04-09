# python make_sbatch.py --ngpu 1 --time 6 --rtx8000 --bash_files bash_cmds/0401_nlp/pretrained/0401_7_ttt_lossonall.sh

# nlp ttt5 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt5_permuten1000_allloss \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 5 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ttt25 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt25_permuten1000_allloss \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 25 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ttt100 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt100_permuten1000_allloss \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 100 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ttt250 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt250_permuten1000_allloss \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100

# nlp ttt500 permuten1000 allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ttt500_permuten1000_allloss \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 500 \
    --ttt_permute_n 1000 \
    --ttt_loss_type all \
    --eval_seeds 100



# AFTER PRECISION FIX

# Submitted batch job 59139456
# Submitted batch job 59139457
# Submitted batch job 59139458
# Submitted batch job 59139459
# Submitted batch job 59139460