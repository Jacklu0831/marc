import os
from predict import main as predict_main

assert os.path.exists("data.json")
assert os.path.exists("experiments/ttt/modal/")

predict_main(
    data_file="data.json",
    solution_file=None,
    pretrained_checkpoint="/checkpoints/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/",
    experiment_folder="experiments/tti/modal/",
    lora_checkpoints_folder="experiments/ttt/modal/",
    temperature=0.0,
    n_sample=1,
    max_lora_rank=128,
    include_n=[1],
    barc_format=True,
    new_format=False,
)