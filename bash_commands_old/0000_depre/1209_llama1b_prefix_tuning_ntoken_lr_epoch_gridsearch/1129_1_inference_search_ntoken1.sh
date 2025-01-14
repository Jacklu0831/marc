# python make_sbatch.py --time 48 --bash_files bash_commands/1209_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1129_1_inference_search_ntoken1.sh

# lr1e-2 ntoken1 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken1 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken1_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken1 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken1_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken1 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken1_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken1 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken1_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn













# lr3e-3 ntoken1 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken1 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken1_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken1 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken1_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken1 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken1_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken1 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken1_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn











# lr1e-3 ntoken1 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken1 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken1_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken1 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken1_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken1 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken1_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken1 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken1_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn











# lr3e-4 ntoken1 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken1 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken1_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken1 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken1_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken1 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken1_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken1 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken1_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn


# best gets 4
# Submitted batch job 54625291
# Submitted batch job 54625292
# Submitted batch job 54625293
# Submitted batch job 54625294
# Submitted batch job 54625295

# Submitted batch job 54625296
# Submitted batch job 54625297
# Submitted batch job 54625298
# Submitted batch job 54625299
# Submitted batch job 54625300

# Submitted batch job 54625301
# Submitted batch job 54625302
# Submitted batch job 54625303
# Submitted batch job 54625304
# Submitted batch job 54625305

# Submitted batch job 54625316
# Submitted batch job 54625317
# Submitted batch job 54625318
# Submitted batch job 54625319
# Submitted batch job 54625320