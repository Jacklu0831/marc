# Many-shot prefix tuning epoch sweep on BBH (25-shot, 8 shortest tasks), seed 42
# random_kv=token, 32 tokens, gs_dropout=none, lr=3e-3
# makesbatch --time 2 --ngpu 1 --gb 64 --bash_file bash_cmds/0326_1_bbh_manyshot/bbh_prefix_25shot.sh

# bbh_manyshot25_prefix_e1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e1 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 1 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

# bbh_manyshot25_prefix_e3
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e3 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 3 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

# bbh_manyshot25_prefix_e5
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e5 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 5 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

# bbh_manyshot25_prefix_e10
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e10 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 10 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

# bbh_manyshot25_prefix_e20
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e20 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 20 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

# bbh_manyshot25_prefix_e40
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e40 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 40 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

# bbh_manyshot25_prefix_e60
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot25_prefix_e60 \
    --seed 42 \
    --num_demonstrations 25 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 60 --gs_lr 3e-3 \
    --gs_dropout none --random_kv token --random_kv_ntokens 32

#! Submitted batch job 5071506 -> 36_mren -- bbh_manyshot25_prefix_e1
#! Submitted batch job 5071507 -> 36_cds -- bbh_manyshot25_prefix_e3
#! Submitted batch job 5071508 -> 36_mren -- bbh_manyshot25_prefix_e5
#! Submitted batch job 5071509 -> 36_cds -- bbh_manyshot25_prefix_e10
#! Submitted batch job 5071510 -> 36_mren -- bbh_manyshot25_prefix_e20
#! Submitted batch job 5071522 -> 36_cds -- bbh_manyshot25_prefix_e40
#! Submitted batch job 5071523 -> 36_mren -- bbh_manyshot25_prefix_e60
