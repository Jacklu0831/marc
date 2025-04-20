# python make_sbatch.py --ngpu 1 --time 9 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_12_gs_numpermute_permuteback_strip.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b gs10 lr1e-2 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-2_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs20 lr1e-2 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-2_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs30 lr1e-2 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-2_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs40 lr1e-2 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-2_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs50 lr1e-2 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-2_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position













# bbh llama8b gs10 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs20 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs20_lr1e-3_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs30 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs30_lr1e-3_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs40 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs40_lr1e-3_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# bbh llama8b gs50 lr1e-3 numpermute128 permuteback strip
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs50_lr1e-3_numpermute128_permuteback_strip \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --num_permute 128 \
    --permute_batch_size 2 \
    --permute_back \
    --permute_back_strip_position

# Submitted batch job 59411894 # OOM

# Submitted batch job 59464110