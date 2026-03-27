# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_3_gs_droptrain_ntoken32.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-1 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-1_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-1 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100





# nlp gs5 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-2_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-2_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-2_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-2 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-2_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100










# nlp gs5 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droptrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_ntokens 32 \
    --eval_seeds 100



# Submitted batch job 59571088

# lr1e-1
# 0.3865158977171775
# 0.4040728396360767
# 0.4283765027160612 <-
# 0.4258186768817363

# lr1e-2
# 0.3718639640954817
# 0.3973356525401299
# 0.4106606808129543 <-
# 0.4101624276001236

# lr1e-3
# 0.3518394549447659
# 0.36212507820703066
# 0.3931640524587577
# 0.39718529001400427 <-

# so far 0.4283765027160612