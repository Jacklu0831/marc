# python make_sbatch.py --ngpu 1 --time 20 --rtx8000 --bash_files bash_cmds/0401_nlp/0425_tttsave/0.sh

# nlp ttt iter250 save seed0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_save_seed0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 0

# nlp ttt iter250 save seed1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_save_seed1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 1

# nlp ttt iter250 save seed2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_save_seed2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 2

# nlp ttt iter250 save seed3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_save_seed3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 3

# nlp ttt iter250 save seed4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_iter250_save_seed4 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_iters 250 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 4

# Submitted batch job 59849670
# Submitted batch job 59849671
# Submitted batch job 59849672
# Submitted batch job 59849673
# Submitted batch job 59849674