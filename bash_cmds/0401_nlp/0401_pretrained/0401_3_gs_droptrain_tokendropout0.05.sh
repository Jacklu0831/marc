# python make_sbatch.py --ngpu 1 --time 9 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_3_gs_droptrain_tokendropout0.05.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-2 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-2_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs25 lr1e-2 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-2_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs100 lr1e-2 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-2_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs250 lr1e-2 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-2_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100










# nlp gs5 lr3e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr3e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs25 lr3e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr3e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs100 lr3e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr3e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs250 lr3e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr3e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100









# nlp gs5 lr1e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droptrain tokendropout0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_tokendropout0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --eval_seeds 100

# Submitted batch job 59643731

# lr1e-2
# 0.4067576747623476
# 0.4234777933036059
# 0.44202929978148897 <-
# 0.4355717073474457

# lr3e-3
# 0.3923759225980958
# 0.4167846395861744
# 0.4397016936314779 <-
# 0.43021512591301087

# lr1e-3
# 0.3783275494175421
# 0.40234566150920253
# 0.4408110929085806 <-
# 0.43464711078365165

# so far 0.44202929978148897