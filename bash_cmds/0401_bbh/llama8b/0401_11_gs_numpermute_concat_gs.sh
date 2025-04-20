# python make_sbatch.py --ngpu 1 --time 16 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_11_gs_numpermute_concat_gs.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




# bbh llama8b numpermute2 concat gs10 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs10_lr1e-3 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs20 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs20_lr1e-3 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs30 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs30_lr1e-3 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 30 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs40 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs40_lr1e-3 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs50 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs50_lr1e-3 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2











# bbh llama8b numpermute2 concat gs10 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs10_lr1e-4 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs20 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs20_lr1e-4 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs30 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs30_lr1e-4 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 30 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs40 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs40_lr1e-4 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b numpermute2 concat gs50 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_numpermute2_concat_gs50_lr1e-4 \
    --model_name llama8b \
    --num_permute 2 \
    --permute_batch_size 2 \
    --permute_concat \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# Submitted batch job 59408958
# Submitted batch job 59408957

# lr1e-3
# 47.43275244812302
# 47.35510180217725
# 47.000451467467094
# 47.84974441754103 <-
# 46.65870334323308

# lr1e-4
# 48.62958303220849
# 48.550125637197745
# 48.59041042656862
# 49.13969511219428 <-
# 48.04916599408124

# so far 49.13969511219428
# <5hrs for each lr