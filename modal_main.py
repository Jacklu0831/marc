# This code is editable! Click the Run button below to execute.
from typing import Dict, List
import modal
import os
import subprocess
import json
# from test_time_train import main as ttt_main

app = modal.App("marc-torchtune-custom-container")

# cuda_version = "12.6.0"  # should be no greater than host CUDA version
# flavor = "devel"  #  includes full CUDA toolkit
# operating_sys = "ubuntu22.04"
# tag = f"{cuda_version}-{flavor}-{operating_sys}"
# image = modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
image = modal.Image.from_registry("hanguo97/marc:0.2")
# image = image.apt_install("git")
# image = image.pip_install("numpy", "scipy", "matplotlib",  "tqdm", "torchtune@git+https://github.com/ekinakyurek/torchtune.git@ekin/llama32")  # add our neural network libraries
# image = image.pip_install("torch", "torchao", extra_options="--pre --upgrade --index-url https://download.pytorch.org/whl/nightly/cu121") # unsafe version


# image.shell("conda create -n torchtune python=3.10")
# image.shell("conda activate torchtune")
# image.shell("pip install torchao")
# image = modal.Image.debian_slim(python_version="3.10")

# image = image.run_commands("conda create -n vllm python=3.10;conda activate vllm;pip install numpy scipy matplotlib tqdm vllm==0.5.5;conda activate base")



@app.function(gpu="A10G", image=image)
def pipeline(data: Dict):
    # Use subprocess to execute the script
    with open("/workspace/main/data.json", "w") as f:
        json.dump(data, f)

    # with open("/workspace/main/ids.txt", "w") as f:
    #     f.write("\n".join(ids))

    cmd = """cd /workspace/main/ && ls &&. /opt/venv-ttt/bin/activate && python modal_ttt.py"""
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    cmd = """cd /workspace/main/ && ls && . /opt/venv-inference/bin/activate && python modal_predict.py"""
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    with open("/workspace/main/experiments/tti/modal/submission.json", "r") as f:
        result = f.read()

    return result




@app.local_entrypoint()
def main():
    with open("/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json", "r") as f:
        data = json.load(f)

    ids = list(data.keys())

    MAX_MACHINES = 20

    # Split the data into chunks
    chunk_size = len(data) // MAX_MACHINES + (len(data) % MAX_MACHINES > 0)
    data_chunks = [dict(list(data.items())[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]

    list_of_submissions = pipeline.starmap([(chunk,) for chunk in data_chunks])

    # merge the submissions
    submission = {}
    for sub in list_of_submissions:
        submission.update(sub)

    with open(f"marc_submission.json", "w") as f:
        f.write(json.dumps(submission))





