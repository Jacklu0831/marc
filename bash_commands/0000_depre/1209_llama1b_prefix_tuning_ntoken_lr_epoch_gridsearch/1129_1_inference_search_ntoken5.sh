# python make_sbatch.py --time 48 --bash_files bash_commands/1209_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1129_1_inference_search_ntoken5.sh

# lr1e-2 ntoken5 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken5_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken5 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken5_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken5 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken5_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken5 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken5_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken5 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken5_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn













# lr3e-3 ntoken5 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken5_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken5 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken5_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken5 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken5_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken5 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken5_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken5 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken5_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn











# lr1e-3 ntoken5 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken5_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken5 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken5_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken5 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken5_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken5 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken5_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken5 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken5_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn











# lr3e-4 ntoken5 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken5_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken5 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken5_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken5 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken5_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken5 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken5_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken5 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken5_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# best is 2
# Submitted batch job 54625357
# Submitted batch job 54625358
# Submitted batch job 54625359
# Submitted batch job 54625360
# Submitted batch job 54625361

# Submitted batch job 54625362
# Submitted batch job 54625363
# Submitted batch job 54625364
# Submitted batch job 54625365
# Submitted batch job 54625366

# Submitted batch job 54625367
# Submitted batch job 54625368
# Submitted batch job 54625369
# Submitted batch job 54625370
# Submitted batch job 54625371

# Submitted batch job 54625382
# Submitted batch job 54625383
# Submitted batch job 54625384
# Submitted batch job 54625385
# Submitted batch job 54625386