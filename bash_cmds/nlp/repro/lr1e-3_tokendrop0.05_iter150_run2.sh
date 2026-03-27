# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/repro/lr1e-3_tokendrop0.05_iter150_run2.sh

# nlp gs150 lr1e-3 tokendrop0.05 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_run2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_float16

# nlp gs150 lr1e-3 tokendrop0.05 run2 float32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_run2_float32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05

# Submitted batch job 60842836
# Submitted batch job 60842837

    # 'eval/score': 0.438423136504806,
    # 'eval/score_list': [   0.4468287318304256,
    #                        0.43980239258703946,
    #                        0.4205783690267192,
    #                        0.4477185426088203,
    #                        0.43718764647102554],

    # 'eval/score': 0.4408397188976688,
    # 'eval/score_list': [   0.4460155513143797,
    #                        0.43470286969595684,
    #                        0.4419959261026961,
    #                        0.44215730236682244,
    #                        0.4393269450084892],