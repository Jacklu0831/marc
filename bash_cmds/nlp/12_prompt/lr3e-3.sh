# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/12_prompt/lr3e-3.sh

# nlp prompt50 lr3e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt50_lr3e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt100 lr3e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt100_lr3e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt200 lr3e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt200_lr3e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt250 lr3e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt250_lr3e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt300 lr3e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt300_lr3e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 300 \
    --gs_lr 3e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# Submitted batch job 60393340
# Submitted batch job 60393341
# Submitted batch job 60393342
# Submitted batch job 60393343
# Submitted batch job 60393344

# 0.41788119792182626
# 0.42533528168932394
# 0.4078742480047359
# 0.41949295503199174
# 0.4185205840288054