import os
from test_time_train import main as ttt_main

assert os.path.exists("data.json")

ttt_main(
    data_file="data.json",
    base_checkpoint_dir="/checkpoints/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/",
    experiment_folder="experiments/ttt/modal/",
    lora_config="configs/ttt/8.1B_lora_single_device.yaml",
    batch_size=2,
    epochs=2,
    learning_rate=1e-4,
    lora_rank=128,
    lora_alpha=16.0,
    lora_to_output=True,
    barc_format=True,
    new_format=False,
)