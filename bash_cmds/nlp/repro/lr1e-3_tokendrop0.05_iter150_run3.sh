# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/repro/lr1e-3_tokendrop0.05_iter150_run3.sh

# nlp gs150 lr1e-3 tokendrop0.05 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_run3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_float16

# nlp gs150 lr1e-3 tokendrop0.05 run3 float32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_run3_float32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05

# Submitted batch job 60842838
# Submitted batch job 60842839

    # 'eval/score': 0.437518265103348,
    # 'eval/score_list': [   0.44365770231606777,
    #                        0.4418882394865021,
    #                        0.4182589139174241,
    #                        0.4490043207559535,
    #                        0.4347821490407928],

    # 'eval/score': 0.44149374144165565,
    # 'eval/score_list': [   0.4445059759264233,
    #                        0.43653842751112265,
    #                        0.44282029146472335,
    #                        0.44391731836688697,
    #                        0.43968669393912196],