# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/randommlpinit/promptuniform_lr1e-3.sh

# nlp prompt100 lr1e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt100_lr1e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt150 lr1e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt150_lr1e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt200 lr1e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt200_lr1e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt250 lr1e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt250_lr1e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# Submitted batch job 60802431
# Submitted batch job 60802432
# Submitted batch job 60802433
# Submitted batch job 60802434