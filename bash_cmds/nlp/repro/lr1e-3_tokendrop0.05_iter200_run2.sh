# python make_sbatch.py --ngpu 1 --time 15 --rtx8000 --bash_files bash_cmds/nlp/repro/lr1e-3_tokendrop0.05_iter200_run2.sh

# nlp gs200 lr1e-3 tokendrop0.05 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_run2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_float16

# nlp gs200 lr1e-3 tokendrop0.05 run2 float32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_run2_float32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05

# Submitted batch job 60842840
# Submitted batch job 60842841

    # 'eval/score': 0.4419770685961721,
    # 'eval/score_list': [   0.4465194813406365,
    #                        0.4487411121506967,
    #                        0.42854962951347,
    #                        0.447178177495423,
    #                        0.43889694248063466],

    # 'eval/score': 0.4403241186311613,
    # 'eval/score_list': [   0.4422443332796173,
    #                        0.4413055308778072,
    #                        0.4339499236864965,
    #                        0.44558862571453406,
    #                        0.43853217959735147],