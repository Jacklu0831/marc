# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_nokeysearchallseed/lr1e-3_droptrain_nokey_tokendrop0.01.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp gs20 lr1e-3 droptrain nokey tokendrop0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs20_lr1e-3_droptrain_nokey_tokendrop0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_no_key \
    --gs_token_dropout 0.01

# nlp gs40 lr1e-3 droptrain nokey tokendrop0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs40_lr1e-3_droptrain_nokey_tokendrop0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_no_key \
    --gs_token_dropout 0.01

# nlp gs60 lr1e-3 droptrain nokey tokendrop0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs60_lr1e-3_droptrain_nokey_tokendrop0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 60 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_no_key \
    --gs_token_dropout 0.01

# nlp gs80 lr1e-3 droptrain nokey tokendrop0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs80_lr1e-3_droptrain_nokey_tokendrop0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 80 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_no_key \
    --gs_token_dropout 0.01

# nlp gs100 lr1e-3 droptrain nokey tokendrop0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs100_lr1e-3_droptrain_nokey_tokendrop0.01 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train \
    --gs_no_key \
    --gs_token_dropout 0.01
