# python make_sbatch.py --time 48 --bash_files bash_commands/1201_llama8b_finetuned_prefix_tuning_lr5e-3_float16/1201_2_llama8b_inference_prefix_tuning_ntoken5_lr5e-3.sh

# ntoken5 epoch1 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch1 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch2 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch2 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch3 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch3 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch4 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch4 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format
