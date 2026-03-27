# python make_sbatch.py --ngpu 1 --time 8 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_4_gs_droppower_ntoken32.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs5 lr1e-1 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-1_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-1 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-1_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-1 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-1_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-1 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-1_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-1 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100







# nlp gs5 lr1e-2 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-2_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-2 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-2_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-2 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-2_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-2 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-2_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100







# nlp gs5 lr1e-3 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs25 lr1e-3 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs25_lr1e-3_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs100 lr1e-3 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100

# nlp gs250 lr1e-3 droppower ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs250_lr1e-3_droppower_ntoken32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout power \
    --gs_ntokens 32 \
    --eval_seeds 100



# Submitted batch job 59571104

# lr1e-1
# 0.364866815151396
# 0.37010563318123063
# 0.4294395711389191 <-
# 0.42025137012106456

# lr1e-2
# 0.3522461718325785
# 0.37732791601242127
# 0.36280083336786195
# 0.41327068754637253 <-

# lr1e-3
# 0.34678754975824555
# 0.34931998492767924
# 0.36121423774219225 <-
# 0.36503898674090995

# so far 0.4294395711389191