# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_bbh/llama8b/0422_1_ttt_gs.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter8 \
    --model_name llama8b \
    --ttt_iters 8 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000



accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_debug/test_time_evaluate.py \
    --tag test \
    --model_name llama1b \
    --ttt_iters 2 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --gs_epochs 5