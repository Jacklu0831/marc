# test train.py has same loss as before with padleft==padright, and same gen with padleft (original script)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --lr_scheduler constant \
    --no_bos \
    --tag test \
    --pad_side left \
    --lr_other 0.0 \
    --lr_embedding 0.0 \
    --samples_per_epoch 8

# test train.py has same loss as before with padleft==padright, and same gen with padleft (original script)
accelerate launch --main_process_port $MASTER_PORT inference_arc/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --lr_scheduler constant \
    --no_bos \
    --tag test \
    --pad_side left

# now that it works, save a quick train.py ckpt for eval
accelerate launch --main_process_port $MASTER_PORT inference_arc/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --lr_scheduler constant \
    --no_bos \
    --tag test \
    --eval_pretrained \
    --debug_no_resume \
    --num_epochs 0
# {"0c9aba6e-0": [["93\n000\n000\n000\n000\n000\n000\n000\n000\n000", "64\n0080\n0008\n0880\n0000\n0800\n8800"]], "3b4c2228-0": [["3\n100\n010\n00", "33\n100\n010\n000"]], "3b4c2228-1": [["3\n100\n010\n001\ninput8\ninput8", "33\n100\n010\n001"]], "662c240a-0": [["3\noutput3output3\noutput3output3\noutput3output3\noutput3", "33\n544\n454\n454"]], "662c240a-1": [["3\noutput3output3\noutput3output3\noutput3output3\noutput3", "33\n111\n454\n454"]], "995c5fa3-0": [["3\n", "33\n444\n333\n888"]], "995c5fa3-1": [["3\n", "33\n444\n333\n777"]]}

# use test_time_evaluate to make sure same output
accelerate launch --main_process_port $MASTER_PORT inference_arc/test_time_evaluate.py \
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --tag test \
    --weight_dir test \
    --weight_epoch 0 \
    --pad_side left # generation can only use left

# {"0c9aba6e-0": [["93\n000\n000\n000\n000\n000\n000\n000\n000\n000", "64\n0080\n0008\n0880\n0000\n0800\n8800"]], "3b4c2228-0": [["3\n100\n010\n00", "33\n100\n010\n000"]], "3b4c2228-1": [["3\n100\n010\n001\ninput8\ninput8", "33\n100\n010\n001"]], "662c240a-0": [["3\noutput3output3\noutput3output3\noutput3output3\noutput3", "33\n544\n454\n454"]], "662c240a-1": [["3\noutput3output3\noutput3output3\noutput3output3\noutput3", "33\n111\n454\n454"]], "995c5fa3-0": [["3\n", "33\n444\n333\n888"]], "995c5fa3-1": [["3\n", "33\n444\n333\n777"]]}