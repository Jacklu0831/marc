from transformers.models import sam
from test_time_train import main as ttt_main
from predict import main as predict_main

ttt_main(
    data_file="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
    base_checkpoint_dir="checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/",
    experiment_folder="experiments/ttt/debug/",
    lora_config="configs/ttt/8.1B_lora_single_device.yaml",
    batch_size=2,
    epochs=2,
    learning_rate=1e-4,
    lora_rank=128,
    lora_alpha=16.0,
    lora_to_output=True,
    barc_format=True,
    new_format=False,
    # ids=["0a1d4ef5"],
)

# predict_main(
#     data_file="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
#     solution_file="/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
#     pretrained_checkpoint="checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/",
#     experiment_folder="experiments/tti/debug/",
#     lora_checkpoints_folder="experiments/ttt/debug/",
#     temperature=0.0,
#     n_sample=1,
#     max_lora_rank=128,
#     include_n=[1],
#     barc_format=True,
#     new_format=False,
#     ids=["0a1d4ef5"],
# )