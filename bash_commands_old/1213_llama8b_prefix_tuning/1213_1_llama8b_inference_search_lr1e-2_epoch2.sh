# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama8b_prefix_tuning/1213_1_llama8b_inference_search_lr1e-2_epoch2.sh

# inference search lr1e-2 epoch2 ntoken1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_llama8b_inference_search_lr1e-2_epoch2_ntoken1 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1213_0_llama8b_ttt_search_lr1e-2_epoch2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference search lr1e-2 epoch2 ntoken5
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_llama8b_inference_search_lr1e-2_epoch2_ntoken5 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1213_0_llama8b_ttt_search_lr1e-2_epoch2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference search lr1e-2 epoch2 ntoken10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_llama8b_inference_search_lr1e-2_epoch2_ntoken10 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1213_0_llama8b_ttt_search_lr1e-2_epoch2_ntoken10 \
    --num_virtual_tokens 10 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference search lr1e-2 epoch2 ntoken50
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_llama8b_inference_search_lr1e-2_epoch2_ntoken50 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1213_0_llama8b_ttt_search_lr1e-2_epoch2_ntoken50 \
    --num_virtual_tokens 50 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# Submitted batch job 54878526
# Submitted batch job 54878527
# Submitted batch job 54878528
# Submitted batch job 54878529