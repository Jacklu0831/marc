# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/12_prompt/lr1e-3.sh

# nlp prompt50 lr1e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt50_lr1e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt100 lr1e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt100_lr1e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt200 lr1e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt200_lr1e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt250 lr1e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt250_lr1e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# nlp prompt300 lr1e-3 droptrain tokendrop0.1 evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt/test_time_evaluate.py \
    --tag nlp_prompt300_lr1e-3_droptrain_tokendrop0.1_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 300 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.1 \
    --eval_seeds 100

# Submitted batch job 60355306
# Submitted batch job 60355307
# Submitted batch job 60355308
# Submitted batch job 60361572
# Submitted batch job 60361574

# 0.4105313117249326
# 0.418574694947535
# 0.4228652283483122
# 0.4252937658670513
# 0.4323873311254399