# python make_sbatch.py --ngpu 1 --time 4 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_15_gs_numpermute.sh

# for the special 1e-4 long runs with iters 300 350 400
# python make_sbatch.py --ngpu 1 --time 7 --rtx8000 --bash_files bash_cmds/0401_nlp/finetuned/0401_15_gs_numpermute.sh


# # nlp ft gs5 lr1e-2 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs5_lr1e-2_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 5 \
#     --gs_lr 1e-2 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs25 lr1e-2 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs25_lr1e-2_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 25 \
#     --gs_lr 1e-2 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs100 lr1e-2 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs100_lr1e-2_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 100 \
#     --gs_lr 1e-2 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs250 lr1e-2 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs250_lr1e-2_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 250 \
#     --gs_lr 1e-2 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100





# # nlp ft gs5 lr1e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs5_lr1e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 5 \
#     --gs_lr 1e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs25 lr1e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs25_lr1e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 25 \
#     --gs_lr 1e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs100 lr1e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs100_lr1e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 100 \
#     --gs_lr 1e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs250 lr1e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs250_lr1e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 250 \
#     --gs_lr 1e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100










# # nlp ft gs5 lr1e-4 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs5_lr1e-4_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 5 \
#     --gs_lr 1e-4 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs25 lr1e-4 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs25_lr1e-4_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 25 \
#     --gs_lr 1e-4 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs100 lr1e-4 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs100_lr1e-4_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 100 \
#     --gs_lr 1e-4 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs250 lr1e-4 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs250_lr1e-4_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 250 \
#     --gs_lr 1e-4 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# nlp ft gs300 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs300_lr1e-4_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 300 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs350 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs350_lr1e-4_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 350 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100

# nlp ft gs400 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
    --tag ft_gs400_lr1e-4_permuten1024 \
    --weight_dir 0401_nlp_gpt2_notruncate \
    --weight_epoch 5 \
    --gs_iters 400 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --eval_seeds 100








# # nlp ft gs5 lr3e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs5_lr3e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 5 \
#     --gs_lr 3e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs25 lr3e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs25_lr3e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 25 \
#     --gs_lr 3e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs100 lr3e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs100_lr3e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 100 \
#     --gs_lr 3e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100

# # nlp ft gs250 lr3e-3 permuten1024
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_nlp/test_time_evaluate.py \
#     --tag ft_gs250_lr3e-3_permuten1024 \
#     --weight_dir 0401_nlp_gpt2_notruncate \
#     --weight_epoch 5 \
#     --gs_iters 250 \
#     --gs_lr 3e-3 \
#     --gs_num_permute 1024 \
#     --eval_seeds 100



# lr1e-2
# Submitted batch job 59084506 # 0.383
# Submitted batch job 59084507 # 0.411
# Submitted batch job 59084508 # 0.431
# Submitted batch job 59084509 # 0.419

# lr1e-3
# Submitted batch job 59084510 # 0.378
# Submitted batch job 59084511 # 0.392
# Submitted batch job 59084512 # 0.436
# Submitted batch job 59084513 # 0.440

# lr1e-4
# Submitted batch job 59084514 # 0.371
# Submitted batch job 59084515 # 0.370
# Submitted batch job 59084516 # 0.379
# Submitted batch job 59084517 # 0.403
# Submitted batch job 59128727
# Submitted batch job 59128728
# Submitted batch job 59128729

# lr3e-3
# Submitted batch job 59128665
# Submitted batch job 59128666
# Submitted batch job 59128667
# Submitted batch job 59128668


# so far 0.440
# pretrained model benefits from numpermute, but not finetuned? lets diagnose this
# because permute averages over so many KV, its memory is obscured
# both lr1e-2 and lr1e-3 were too high for nopermute, suffering from memorization, permute is better!
# but nopermute performs 0.003 better in lr1e-4 because its initial KV is well tuned for ICL, here lr1e-4 simply haven't converged

# overall numpermute has a bit of potential for higher if trained for the proper amount,
# and it just misses the "mark" with lr1e-3,
# lets try to justify using it for all NLP tasks (ARC kinda has no hope)




# AFTER PRECISION FIX

# lr1e-2
# Submitted batch job 59139552 # 0.385
# Submitted batch job 59139553 # 0.422
# Submitted batch job 59139554 # 0.428
# Submitted batch job 59139555 # 0.421

# lr1e-3
# Submitted batch job 59139556 # 0.377
# Submitted batch job 59139557 # 0.387
# Submitted batch job 59139558 # 0.431
# Submitted batch job 59139559 # 0.441

# lr1e-4
# Submitted batch job 59139560 # 0.373
# Submitted batch job 59139561 # 0.371
# Submitted batch job 59139562 # 0.379
# Submitted batch job 59139563 # 0.399
# Submitted batch job 59139569 # 0.405
# Submitted batch job 59139570 # 0.413
# Submitted batch job 59139571 # 0.423

# lr1e-5
# Submitted batch job 59139564 # 0.388
# Submitted batch job 59139565 # 0.410
# Submitted batch job 59139566 # 0.435
# Submitted batch job 59139567 # 0.434

# so far 0.441