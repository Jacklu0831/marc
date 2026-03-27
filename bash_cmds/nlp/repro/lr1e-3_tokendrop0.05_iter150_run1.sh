# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --bash_files bash_cmds/nlp/repro/lr1e-3_tokendrop0.05_iter150_run1.sh

# nlp gs150 lr1e-3 tokendrop0.05 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_run1 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05 \
    --gs_float16

# nlp gs150 lr1e-3 tokendrop0.05 run1 float32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs150_lr1e-3_tokendrop0.05_run1_float32 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_token_dropout 0.05

# Submitted batch job 60840314
# Submitted batch job 60840315

# 'eval/score': 0.43780929940324287,
# 'eval/score_list': [   0.44431290049257427,
#                        0.4405994264588031,
#                        0.42202161449206055,
#                        0.44694741196132787,
#                        0.43516514361144815],

# 'eval/score': 0.44103147944413224,
# 'eval/score_list': [   0.4467622403240646,
#                        0.43663164412688976,
#                        0.43875968928002507,
#                        0.44132214347090115,
#                        0.4416816800187808],