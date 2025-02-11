# TODO: change weight epoch to best from the base run
# time: <4 hrs on single a100
# disk memory: 293M for lora * 5 epoch * 80 task = 118GB

# base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --weight_dir 0209_noprogram_base \
    --weight_epoch 14 \
    --tag base \
    --num_epochs 5 \
    --save_epochs 1

# TOREMOVE, just testing
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --weight_dir test_noprogram \
    --weight_epoch 1 \
    --tag test \
    --num_epochs 5 \
    --save_epochs 1 \
    --aug_type extra
