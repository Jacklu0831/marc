# Many-shot CT-KV epoch sweep (td=0.1 fixed, epochs {1,3,5,60}), seed 42
# Complements the main sweep which covered {10,20,40}
# makesbatch --time 4 --ngpu 1 --gb 128 --bash_file bash_cmds/0326_1_bbh_manyshot/bbh_ctkv_epoch_sweep.sh

# bbh_manyshot_ctkv_e1_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot_ctkv_e1_td0.1 \
    --seed 42 \
    --num_demonstrations 50 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 1 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot_ctkv_e3_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot_ctkv_e3_td0.1 \
    --seed 42 \
    --num_demonstrations 50 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 3 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot_ctkv_e5_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot_ctkv_e5_td0.1 \
    --seed 42 \
    --num_demonstrations 50 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 5 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot_ctkv_e60_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot_ctkv_e60_td0.1 \
    --seed 42 \
    --num_demonstrations 50 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 60 --gs_lr 1e-3 --gs_batch_size 16 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot100_ctkv_e1_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e1_td0.1 \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 1 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot100_ctkv_e3_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e3_td0.1 \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 3 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot100_ctkv_e5_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e5_td0.1 \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 5 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

# bbh_manyshot100_ctkv_e60_td0.1
.venv/bin/accelerate launch --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_manyshot100_ctkv_e60_td0.1 \
    --seed 42 \
    --num_demonstrations 100 --max_seq_len 8192 \
    --select_tasks boolean_expressions sports_understanding multistep_arithmetic_two hyperbaton object_counting dyck_languages web_of_lies navigate \
    --gs_epochs 60 --gs_lr 1e-3 --gs_batch_size 8 \
    --gs_dropout train --gs_token_dropout 0.1

#! Submitted batch job 5041916 -> 36_cds -- bbh_manyshot_ctkv_e1_td0.1
#! Submitted batch job 5041917 -> 36_mren -- bbh_manyshot_ctkv_e3_td0.1
#! Submitted batch job 5041918 -> 36_cds -- bbh_manyshot_ctkv_e5_td0.1
#! Submitted batch job 5041919 -> 36_mren -- bbh_manyshot_ctkv_e60_td0.1
#! Submitted batch job 5041920 -> 36_cds -- bbh_manyshot100_ctkv_e1_td0.1
#! Submitted batch job 5041921 -> 36_mren -- bbh_manyshot100_ctkv_e3_td0.1
#! Submitted batch job 5041922 -> 36_cds -- bbh_manyshot100_ctkv_e5_td0.1
#! Submitted batch job 5041923 -> 36_cds -- bbh_manyshot100_ctkv_e60_td0.1
