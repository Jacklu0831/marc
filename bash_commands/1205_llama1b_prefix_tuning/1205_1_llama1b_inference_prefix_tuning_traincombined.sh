# python make_sbatch.py --time 48 --bash_files bash_commands/1205_llama1b_prefix_tuning/1205_1_llama1b_inference_prefix_tuning_traincombined.sh
# this is just a sanity check that training data is properly trained WITH LEAKAGE

# inference prefix tuning with llama1b lr1e-2 traincombined
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_traincombined_lr1e-2_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr1e-2 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --solution_file kaggle_dataset/arc-agi_training_solutions.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr5e-3 traincombined
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_traincombined_lr5e-3_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr5e-3 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --solution_file kaggle_dataset/arc-agi_training_solutions.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr1e-3 traincombined
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_traincombined_lr1e-3_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr1e-3 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --solution_file kaggle_dataset/arc-agi_training_solutions.json \
    --include_n=1 \
    --new_format
