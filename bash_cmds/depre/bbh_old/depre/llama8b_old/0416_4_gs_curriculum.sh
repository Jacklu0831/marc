# python make_sbatch.py --ngpu 1 --time 6 --single --bash_files bash_cmds/0401_bbh/llama8b/0416_4_gs_curriculum.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# before running:
# check if OOM with the full batch
# check if pastkeyvalues grow as expected
# check speed, is 6 hour enough


# bbh llama8b gscurriculum epoch1 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum_epoch1_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 1 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2

# bbh llama8b gscurriculum epoch2 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gscurriculum_epoch2_lr1e-3 \
    --model_name llama8b \
    --gs_curriculum_epochs 2 \
    --gs_curriculum_lr 1e-3 \
    --gs_curriculum_batch_size 2