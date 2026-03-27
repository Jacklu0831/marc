# python make_sbatch.py --ngpu 1 --time 24 --rtx8000 --single --bash_files bash_cmds/0401_nlp/0425_tttkvsearch/lr1e-4_droptrain_tokendrop0.2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# nlp ttt gs5 lr1e-4 droptrain tokendrop0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_gs5_lr1e-4_droptrain_tokendrop0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_weight_dir eval_nlp_ttt_iter250_save_seed2_nlp_pretrained \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.2

# nlp ttt gs10 lr1e-4 droptrain tokendrop0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_gs10_lr1e-4_droptrain_tokendrop0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_weight_dir eval_nlp_ttt_iter250_save_seed2_nlp_pretrained \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.2

# nlp ttt gs15 lr1e-4 droptrain tokendrop0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_gs15_lr1e-4_droptrain_tokendrop0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_weight_dir eval_nlp_ttt_iter250_save_seed2_nlp_pretrained \
    --gs_epochs 15 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.2

# nlp ttt gs20 lr1e-4 droptrain tokendrop0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_gs20_lr1e-4_droptrain_tokendrop0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_weight_dir eval_nlp_ttt_iter250_save_seed2_nlp_pretrained \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.2

# nlp ttt gs25 lr1e-4 droptrain tokendrop0.2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_ttt_gs25_lr1e-4_droptrain_tokendrop0.2 \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --ttt_weight_dir eval_nlp_ttt_iter250_save_seed2_nlp_pretrained \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout train \
    --gs_token_dropout 0.2
