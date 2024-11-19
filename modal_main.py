from typing import Dict, List
import modal
import os
import subprocess
import json

app = modal.App("marc-torchtune-custom-container")
image = modal.Image.from_registry("hanguo97/marc:0.5")
volume = modal.Volume.from_name("checkpoints")

# TODO: make H100/A100 for real submission
@app.function(gpu="H100", image=image, volumes={"/checkpoints": volume})
def pipeline(data: Dict):
    # Use subprocess to execute the script
    with open("/workspace/main/data.json", "w") as f:
        json.dump(data, f)

    cmd = """cd /workspace/main/ && ls && . /opt/venv-inference/bin/activate && python modal_ttt.py"""
    result = subprocess.run(cmd, shell=True,  capture_output=True, text=True)

    cmd = """cd /workspace/main/ && ls && . /opt/venv-inference/bin/activate && python modal_predict.py"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    with open("/workspace/main/experiments/tti/modal/submission.json", "r") as f:
        result = f.read()

    return result


@app.local_entrypoint()
def main():
    MAX_MACHINES = 20

    with open("/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json", "r") as f:
        data = json.load(f)

    # Split the data into chunks
    chunk_size = len(data) // MAX_MACHINES + (len(data) % MAX_MACHINES > 0)
    data_chunks = [dict(list(data.items())[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]

    pipeline.remote(data_chunks[0])
    # list_of_submissions = pipeline.starmap([(chunk,) for chunk in data_chunks[:2]], return_exceptions=True)

    # # merge the submissions
    # submission = {}
    # for sub in list_of_submissions:
    #     submission.update(sub)

    # with open(f"marc_submission.json", "w") as f:
    #     f.write(json.dumps(submission))





