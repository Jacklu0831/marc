# This code is editable! Click the Run button below to execute.

import modal
from test_time_train import main as ttt_main

app = modal.App("marc-torchtune-custom-container")

# cuda_version = "12.6.0"  # should be no greater than host CUDA version
# flavor = "devel"  #  includes full CUDA toolkit
# operating_sys = "ubuntu22.04"
# tag = f"{cuda_version}-{flavor}-{operating_sys}"
# image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")

image= modal.Image.debian_slim(python_version="3.10")
image = image.apt_install("git")
image = image.pip_install("numpy", "scipy", "matplotlib",  "tqdm", "torchtune@git+https://github.com/ekinakyurek/torchtune.git@ekin/llama32")  # add our neural network libraries
image = image.pip_install("torch", "torchao", extra_options="--pre --upgrade --index-url https://download.pytorch.org/whl/nightly/cu121") # unsafe version


@app.function(gpu="A10G", image=image)
def check_cuda():
    import torchtune
    import torch  # installed as dependency of transformers

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    ttt_main(
        data_file="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json", # needs to be read from volume
        base_checkpoint_dir="checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/", # needs to be read from volume
        experiment_folder="experiments/ttt/debug/", # needs to write to volume, should be readable by other containers
        lora_config="configs/ttt/8.1B_lora_single_device.yaml", # not sure if this will be copied
        batch_size=2,
        epochs=2,
        learning_rate=1e-4,
        lora_rank=128,
        lora_alpha=16.0,
        lora_to_output=True,
        barc_format=True,
        new_format=False,
        ids=["0a1d4ef5"],
    )



@app.local_entrypoint()
def main():
    check_cuda.remote()