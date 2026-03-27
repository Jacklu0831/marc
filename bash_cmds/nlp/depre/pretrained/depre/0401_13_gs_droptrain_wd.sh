# python make_sbatch.py --ngpu 1 --time 10 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_13_gs_droptrain_wd.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droptrain_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droptrain_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.01 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droptrain wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_wd0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.01 \
    --eval_seeds 100









# nlp gs5 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droptrain_wd0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.05 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droptrain_wd0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.05 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_wd0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.05 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droptrain wd0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_wd0.05 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.05 \
    --eval_seeds 100












# nlp gs5 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droptrain_wd0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.1 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droptrain_wd0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.1 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_wd0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.1 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droptrain wd0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droptrain_wd0.1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_weight_decay 0.1 \
    --eval_seeds 100

# Submitted batch job 59510167

# wd0.01
# 0.3819275676316421
# 0.4103482838025109
# 0.43552358504953825
# 0.4403251063473023 <-

# wd0.05
# 0.37796286449809835
# 0.40966304226737416
# 0.43938054827476186 <-
# 0.43647872159178547

# wd0.1
# 0.3775556827898292
# 0.4067759502407314
# 0.43715616794343104
# 0.43804913808891366 <-

# so far 0.4403251063473023