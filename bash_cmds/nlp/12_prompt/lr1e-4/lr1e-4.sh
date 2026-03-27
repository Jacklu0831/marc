# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/12_prompt/lr1e-4.sh

# nlp prompt50 lr1e-4 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt50_lr1e-4_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt100 lr1e-4 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt100_lr1e-4_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt200 lr1e-4 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt200_lr1e-4_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# Submitted batch job 60355309
# Submitted batch job 60355310
# Submitted batch job 60355311

# 0.36312694320034755
# 0.3781395965511447
# 0.38944007904323513
