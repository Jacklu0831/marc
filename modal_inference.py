# This code is editable! Click the Run button below to execute.

import modal
from predict import main as predict_main

app = modal.App("marc-vllm-custom-container")

# cuda_version = "12.6.0"  # should be no greater than host CUDA version
# flavor = "devel"  #  includes full CUDA toolkit
# operating_sys = "ubuntu22.04"
# tag = f"{cuda_version}-{flavor}-{operating_sys}"
# image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")

# debian slim
image= modal.Image.debian_slim(python_version="3.10")

image = image.apt_install("git")
#image = image.pip_install("numpy", "scipy", "matplotlib",  "tqdm", "vllm@git+https://github.com/ekinakyurek/vllm.git@ekin/torchtunecompat")  # add our neural network libraries
image = image.pip_install("numpy", "scipy", "matplotlib",  "tqdm", "vllm==0.5.5")  # add our neural network libraries

@app.function(gpu="A10G", image=image)
def check_cuda():
    import vllm
    print("VLLM version:", vllm.__version__)
    predict_main(
        data_file="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
        solution_file="/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json",
        pretrained_checkpoint="checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/",
        experiment_folder="experiments/tti/debug/",
        lora_checkpoints_folder="experiments/ttt/debug/",
        temperature=0.0,
        n_sample=1,
        max_lora_rank=128,
        include_n=[1],
        barc_format=True,
        new_format=False,
        ids=["0a1d4ef5"],
    )




@app.local_entrypoint()
def main():
    check_cuda.remote()