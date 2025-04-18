# python make_sbatch.py --ngpu 1 --time 6 --single --bash_files bash_cmds/0401_bbh/llama8b/0401_7_gs_curriculum_lr1e-3.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)





# bbh llama8b gscurriculum1 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum1_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 1 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum2 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum2_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 2 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum3 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum3_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 3 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum4 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum4_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 4 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum5_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 5 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum6 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum6_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 6 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum7 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum7_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 7 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum8 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum8_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 8 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2
