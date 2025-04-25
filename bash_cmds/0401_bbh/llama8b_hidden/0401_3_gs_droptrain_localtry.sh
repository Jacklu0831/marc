# bbh llama8b gshidden10 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-2_droptrain_local \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden20 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-2_droptrain_local \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden30 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-2_droptrain_local \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden40 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-2_droptrain_local \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden50 lr1e-2 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-2_droptrain_local \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_dropout train








# bbh llama8b gshidden10 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-3_droptrain_local \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden20 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-3_droptrain_local \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden30 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-3_droptrain_local \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden40 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-3_droptrain_local \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden50 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-3_droptrain_local \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_dropout train










# bbh llama8b gshidden10 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-4_droptrain_local \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden20 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-4_droptrain_local \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden30 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-4_droptrain_local \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden40 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-4_droptrain_local \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden50 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-4_droptrain_local \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_dropout train










# bbh llama8b gshidden10 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden10_lr1e-5_droptrain_local \
    --model_name llama8b \
    --gs_epochs 10 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden20 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden20_lr1e-5_droptrain_local \
    --model_name llama8b \
    --gs_epochs 20 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden30 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden30_lr1e-5_droptrain_local \
    --model_name llama8b \
    --gs_epochs 25 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden40 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden40_lr1e-5_droptrain_local \
    --model_name llama8b \
    --gs_epochs 40 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_dropout train

# bbh llama8b gshidden50 lr1e-5 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_llama8b_gshidden50_lr1e-5_droptrain_local \
    --model_name llama8b \
    --gs_epochs 50 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_dropout train


# lr1e-3
# 9.742623979912116
# 10.412892715152603
# 10.77997489014438 <-
# 10.08553044569994
# 10.734724837832184

# lr1e-3
# 42.13523011213274
# 39.76402142778015 <-
# 37.100596359070934
# 34.982860061789935
# 33.538808178638696

# lr1e-4
# 51.97582077400953
# 51.3871964353852
# 52.44900879372466
# 52.08203871630259
# 52.68225965841452 <-

# lr1e-5
# 49.81050015737619
# 50.25172278578725
# 50.492022609420424
# 50.63803518896572
# 50.9183230731087 <-

# so far 52.68225965841452