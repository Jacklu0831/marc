# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_nlp/0401_1_eval_original.sh --rtx8000

# # run locally, just create a model ckpt
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/train.py \
#     --lr_scheduler constant \
#     --tag nlp_pretrained \
#     --eval_pretrained \
#     --num_epochs 0 \
#     --eval_train_test_per_task 1 \
#     --eval_eval_ratio 0.01 \
#     --eval_seeds 100

# nlp gs0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag gs0 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --eval_seeds 100
