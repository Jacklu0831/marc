# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/12_promptrandomntoken/lr1e-3.sh

# nlp prompt50 lr1e-3 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt50_lr1e-3_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt100 lr1e-3 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt100_lr1e-3_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt200 lr1e-3 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt200_lr1e-3_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt250 lr1e-3 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt250_lr1e-3_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt300 lr1e-3 droptrain tokendrop0.1 random evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt300_lr1e-3_droptrain_tokendrop0.1_random_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 300 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# Submitted batch job 60355315
# Submitted batch job 60355316
# Submitted batch job 60355317
# Submitted batch job 60361581
# Submitted batch job 60361582

# 0.35600064837365025
# 0.354011658989917
# 0.39684969750030497
# 0.4035057065425711
# 0.4191079528355776
