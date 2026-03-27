# python make_sbatch.py --ngpu 1 --time 8 --single --bash_files bash_cmds/0401_bbh/llama8b_hidden/0401_3_gs.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)













# bbh llama8b gshidden10 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-2 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_batch_size 2

# bbh llama8b gshidden20 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-2 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_batch_size 2

# bbh llama8b gshidden30 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-2 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2

# bbh llama8b gshidden40 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-2 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-2 \
    --gs_batch_size 2

# bbh llama8b gshidden50 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-2 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2








# bbh llama8b gshidden10 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gshidden20 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gshidden30 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gshidden40 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2

# bbh llama8b gshidden50 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-3 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2












# bbh llama8b gshidden10 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gshidden20 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gshidden30 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gshidden40 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_batch_size 2

# bbh llama8b gshidden50 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-4 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_batch_size 2










# bbh llama8b gshidden10 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-5 \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-5 \
    --gs_batch_size 2

# bbh llama8b gshidden20 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-5 \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-5 \
    --gs_batch_size 2

# bbh llama8b gshidden30 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-5 \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-5 \
    --gs_batch_size 2

# bbh llama8b gshidden40 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-5 \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-5 \
    --gs_batch_size 2

# bbh llama8b gshidden50 lr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-5 \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-5 \
    --gs_batch_size 2
