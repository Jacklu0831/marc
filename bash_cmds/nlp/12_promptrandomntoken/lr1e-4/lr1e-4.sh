# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/12_promptrandomntoken/lr1e-4.sh

# nlp prompt50 lr1e-4 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt50_lr1e-4_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt100 lr1e-4 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt100_lr1e-4_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt200 lr1e-4 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt200_lr1e-4_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# Submitted batch job 60355312
# Submitted batch job 60355313
# Submitted batch job 60355314

# 0.3484452144653923
# 0.34798290359924156
# 0.34941028464674817