# nlp fisher
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp_fisher/test_time_evaluate.py \
    --tag nlp_fisher \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher \
    --eval_ratio 0.0001

# {   'eval/gs_fisher_key': 2.364142347520452e-08,
#     'eval/gs_fisher_val': 6.074722994637063e-08,