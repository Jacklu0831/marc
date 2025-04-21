# python make_sbatch.py --ngpu 1 --time 2 --rtx8000 --single --bash_files bash_cmds/0401_nlp/pretrained/0401_10_gs_numpermute_concat.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# nlp numpermute2 concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_numpermute2_concat \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --num_permute 2 \
    --permute_batch_size 2 \
    --eval_seeds 100 \
    --batch_size 8

# nlp numpermute4 concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag nlp_gs5_lr1e-3_numpermute4_concat \
    --weight_dir nlp_pretrained \
    --weight_epoch 0 \
    --num_permute 4 \
    --permute_batch_size 4 \
    --eval_seeds 100 \
    --batch_size 4

# Submitted batch job 59510166