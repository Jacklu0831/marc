# python make_sbatch.py --ngpu 1 --time 9 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_3_gs_droptrain_tokendropout0.2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-2 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-2_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs25 lr1e-2 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-2_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs100 lr1e-2 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-2_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs250 lr1e-2 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-2_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100










# nlp gs5 lr3e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100









# nlp gs5 lr1e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droptrain tokendropout0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_tokendropout0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.2 \
    --eval_seeds 100

# Submitted batch job 59646965

# lr1e-2
# 0.3897261565156881
# 0.41377924932131754
# 0.4298305831428343 <-
# 0.42470043680465125

# lr3e-3
# 0.3799859515643587
# 0.40563172529564456
# 0.43252941450847565
# 0.4362471331154669 <-

# lr1e-3
# 0.3698838113304158
# 0.3984029047302561
# 0.4309962610820462
# 0.4358783231479722 <-

# so far 0.4362471331154669