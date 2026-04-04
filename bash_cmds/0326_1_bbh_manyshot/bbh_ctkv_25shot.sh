# Many-shot CT-KV epoch sweep on BBH (25-shot, 8 shortest tasks), seed 42
# td=0.1 fixed, epochs {1, 3, 5, 10, 20, 40, 60}
# makesbatch --time 2 --ngpu 1 --gb 64 --bash_file bash_cmds/0326_1_bbh_manyshot/bbh_ctkv_25shot.sh

# bbh_manyshot25_ctkv_e1_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e1_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 1 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot25_ctkv_e3_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e3_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 3 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot25_ctkv_e5_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e5_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 5 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot25_ctkv_e10_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e10_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 10 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot25_ctkv_e20_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e20_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 20 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot25_ctkv_e40_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e40_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 40 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot25_ctkv_e60_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_ctkv_e60_td0.1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 60 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 5042017 -> 36_cds -- bbh_manyshot25_ctkv_e1_td0.1
#! Submitted batch job 5042018 -> 36_mren -- bbh_manyshot25_ctkv_e3_td0.1
#! Submitted batch job 5042019 -> 36_cds -- bbh_manyshot25_ctkv_e5_td0.1
#! Submitted batch job 5042020 -> 36_mren -- bbh_manyshot25_ctkv_e10_td0.1
#! Submitted batch job 5042021 -> 36_cds -- bbh_manyshot25_ctkv_e20_td0.1
#! Submitted batch job 5042022 -> 36_cds -- bbh_manyshot25_ctkv_e40_td0.1
#! Submitted batch job 5042023 -> 36_cds -- bbh_manyshot25_ctkv_e60_td0.1
