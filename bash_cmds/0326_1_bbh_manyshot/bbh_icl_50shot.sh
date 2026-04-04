# Many-shot ICL on BBH (50-shot, 8 shortest tasks), seeds 42-44
# makesbatch --time 2 --ngpu 1 --gb 64 --bash_file bash_cmds/0326_1_bbh_manyshot/bbh_icl_50shot.sh

# bbh_manyshot_icl_seed42_zhenbang
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot_icl_seed42_zhenbang \
    --seed 42 \
    --num_demonstrations 50 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate

#! Submitted batch job 5030848 -> 36_cds -- bbh_manyshot_icl_seed42_zhenbang