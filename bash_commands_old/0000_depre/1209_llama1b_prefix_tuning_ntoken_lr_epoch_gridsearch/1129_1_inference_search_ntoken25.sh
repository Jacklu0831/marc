# python make_sbatch.py --time 48 --bash_files bash_commands/1209_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1129_1_inference_search_ntoken25.sh

# lr1e-2 ntoken25 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken25_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken25 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken25_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken25 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken25_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken25 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken25_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-2 ntoken25 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-2_ntoken25_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn













# lr3e-3 ntoken25 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken25_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken25 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken25_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken25 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken25_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken25 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken25_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-3 ntoken25 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-3_ntoken25_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn











# lr1e-3 ntoken25 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken25_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken25 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken25_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken25 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken25_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken25 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken25_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr1e-3 ntoken25 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr1e-3_ntoken25_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr1e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn











# lr3e-4 ntoken25 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken25_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken25 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken25_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken25 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken25_epoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken25 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken25_epoch8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# lr3e-4 ntoken25 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1129_1_inference_search_lr3e-4_ntoken25_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1129_0_ttt_search_lr3e-4_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn
