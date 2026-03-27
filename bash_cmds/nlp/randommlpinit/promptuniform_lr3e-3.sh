# python make_sbatch.py --ngpu 1 --time 12 --rtx8000 --bash_files bash_cmds/nlp/randommlpinit/promptuniform_lr3e-3.sh

# nlp prompt100 lr3e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt100_lr3e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt150 lr3e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt150_lr3e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt200 lr3e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt200_lr3e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# nlp prompt250 lr3e-3 droptrain random uniform evalseed100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_prompt_time/test_time_evaluate.py \
    --tag nlp_prompt250_lr3e-3_droptrain_random_uniform_evalseed100 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_batch_size 16 \
    --random_prompt uniform \
    --random_prompt_ntokens 32 \
    --eval_seeds 100

# Submitted batch job 60802435
# Submitted batch job 60802436
# Submitted batch job 60802437
# Submitted batch job 60802438