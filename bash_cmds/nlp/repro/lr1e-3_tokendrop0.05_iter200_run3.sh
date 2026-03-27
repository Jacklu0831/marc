# python make_sbatch.py --ngpu 1 --time 15 --rtx8000 --bash_files bash_cmds/nlp/repro/lr1e-3_tokendrop0.05_iter200_run3.sh

# nlp gs200 lr1e-3 tokendrop0.05 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_run3 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_float16

# nlp gs200 lr1e-3 tokendrop0.05 run3 float32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs200_lr1e-3_tokendrop0.05_run3_float32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05

# Submitted batch job 60842845
# Submitted batch job 60842846

    # 'eval/score': 0.44262792203746715,
    # 'eval/score_list': [   0.4482771610778203,
    #                        0.4469099658638128,
    #                        0.42960539720551505,
    #                        0.44751164046012964,
    #                        0.4408354455800579],

    # 'eval/score': 0.4403241186311613,
    # 'eval/score_list': [   0.4422443332796173,
    #                        0.4413055308778072,
    #                        0.4339499236864965,
    #                        0.44558862571453406,
    #                        0.43853217959735147],