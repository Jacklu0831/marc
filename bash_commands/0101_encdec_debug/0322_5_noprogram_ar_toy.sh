# toy experiment
# - remove tokenizer, no need at all, no bos related stuff
# - remove color permute, pair permute, any augmentation etc, no more dataset ratio stuff
# - remove lora, gradient checkpointing
# - remove (un)trainable nbit, use float32
# - configurable model size
# - can still have debug pad, where pad is a fixed 0.1234 vector
# - keep notf32
# - keep stuff as much as possible, like program loss and stuff, no dry run

# makesbatch should only use 2 workers
# for runs: search over magnitudes of numtrainnet -1, then from 1000 to 100000
#           default lr 2e-4, try double and quadruple with numtrainnet -1


# check if cache is consistency across multigpu on rtx8000
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --tag test \
    --num_train_net 1000 \
    --samples_per_epoch 32 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --debug_random_pad \
    --num_workers 0 \
    --min_eval_num_pair 3 \
    --pad_side left



# tested ntokens0 == ntokens-1 without attention args
# tested padleft == padright for ntokens-1
# tested padleft == padright for ntokens2 attention_cutoff attend_prev_programs
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --tag test \
    --samples_per_epoch 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --min_eval_num_pair 3 \
    --ntokens 2 \
    --attention_cutoff \
    --attend_prev_programs



# tested gs padleft == padright (with random pad len)
# tested gslr0 equivalent to nogs
# works, tho it has a hard time optimizing some particular tasks, whatever
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy/train.py \
    --tag test \
    --samples_per_epoch 32 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --min_eval_num_pair 3 \
    --num_eval_net 5 \
    --eval_gs_lr 1.0 \
    --eval_gs_batch_size 100000 \
    --eval_gs_optimizer sgd \
    --eval_gs_max_grad_norm 1e8 \
    --eval_gs_iters 50 \
    --eval_gs_take_best \
    --debug_pad_len 40 \
    --pad_side right







# check if cache is consistency across multigpu on rtx8000
# tested padleft + debugrandompad == padright + debugpadlen5
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_toy/train.py \
    --tag test \
    --num_train_net 1000 \
    --samples_per_epoch 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --min_train_num_pair 10 \
    --max_train_num_pair 10 \
    --min_eval_num_pair 3 \
    --max_eval_num_pair 10 \
    --ntokens 0 \
    --lr_other 0.0 \
    --num_workers 0 \
    --pad_side left \
    --debug_random_pad

# tested arlongcache token0 equals baseline
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_toy_debug/train.py \
    --tag test \
    --num_train_net 1000 \
    --samples_per_epoch 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --min_train_num_pair 10 \
    --max_train_num_pair 10 \
    --min_eval_num_pair 3 \
    --max_eval_num_pair 10 \
    --lr_other 0.0 \
    --num_workers 0 \
    --pad_side right \
    --debug_pad_len 40
