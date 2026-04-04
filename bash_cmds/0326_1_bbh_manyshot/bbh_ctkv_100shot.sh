# Many-shot CT-KV sweep on BBH (100-shot, 8 shortest tasks), seed 42
# Sweep: epochs {10, 20, 40} x token_dropout {0.1, 0.2, 0.3}
# makesbatch --time 4 --ngpu 1 --gb 128 --bash_file bash_cmds/0326_1_bbh_manyshot/bbh_ctkv_100shot.sh

# bbh_manyshot100_ctkv_e10_td0.1_zhenbang
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e10_td0.1_zhenbang \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot100_ctkv_e20_td0.1_zhenbang
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e20_td0.1_zhenbang \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot100_ctkv_e40_td0.1_zhenbang
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e40_td0.1_zhenbang \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 40 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 5020976 -> 36_cds -- bbh_manyshot100_ctkv_e10_td0.1_zhenbang
#! Submitted batch job 5020979 -> 36_mren -- bbh_manyshot100_ctkv_e20_td0.1_zhenbang
#! Submitted batch job 5020982 -> 36_cds -- bbh_manyshot100_ctkv_e40_td0.1_zhenbang