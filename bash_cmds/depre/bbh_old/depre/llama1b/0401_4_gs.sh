# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_2_gs.sh
# try gs again but with few iters, see if overfitting is truly the problem that explains why permuten is higher

# bbh llama1b gs1 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --gs_iters 1 \
    --gs_lr 1e-4

# bbh llama1b gs2 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --gs_iters 2 \
    --gs_lr 1e-4

# bbh llama1b gs4 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --gs_iters 4 \
    --gs_lr 1e-4

# bbh llama1b gs6 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --gs_iters 6 \
    --gs_lr 1e-4

# bbh llama1b gs8 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --gs_iters 8 \
    --gs_lr 1e-4

# nah, its increasing not only up to 31.39