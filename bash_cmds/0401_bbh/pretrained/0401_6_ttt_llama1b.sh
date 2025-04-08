# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_bbh/pretrained/0401_6_ttt_llama1b.sh

# bbh llama1b ttt allloss
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_ttt_allloss \
    --model_name llama1b \
    --ttt_iters 40 \
    --ttt_loss_type all
