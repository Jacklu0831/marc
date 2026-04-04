# Many-shot ICL on BBH (75-shot, 8 shortest tasks), seed 42
# makesbatch --time 2 --ngpu 1 --gb 64 --bash_file bash_cmds/0326_1_bbh_manyshot/bbh_icl_75shot.sh

# bbh_manyshot75_icl_seed42
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot75_icl_seed42 \
    --seed 42 \
    --num_demonstrations 75 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate

#! Submitted batch job 5056327 -> 36_mren -- bbh_manyshot75_icl_seed42
