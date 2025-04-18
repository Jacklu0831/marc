# python make_sbatch.py --ngpu 1 --time 2 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_10_gs_numpermute_concat.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)



# bbh llama8b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b \
    --model_name llama8b

# bbh llama8b numpermute2 concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat

# bbh llama8b numpermute4 concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute4_concat \
    --model_name llama8b \
    --num_permute 4 \
    --permute_batch_size 4 \
    --permute_concat

# bbh llama8b numpermute4 concat batchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute4_concat_batchsize2 \
    --model_name llama8b \
    --num_permute 4 \
    --permute_batch_size 4 \
    --permute_concat \
    --batch_size 2


# runs baseline on top
# check that batchsize4 and batchsize2 is the same