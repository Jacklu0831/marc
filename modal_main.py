from typing import Dict, List
import modal
import os
import subprocess
import json

VOLUME_NAME = "checkpoints"
# modal volume create {VOLUME_NAME}
# modal volume put {VOLUME_NAME} Llama-3.1-ARC-Potpourri-Transduction-8B checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B
VOLUME_DIR = "/checkpoints"


app = modal.App("marc-torchtune-custom-container")
image = modal.Image.from_registry("hanguo97/marc:0.8")
image = image.run_commands(["cd /workspace/main/ && git pull origin modal"])
image = image.run_commands([
    f"[ ! -e /workspace/main/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B ] && ln -s {VOLUME_DIR}/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B /workspace/main/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B"
])
volume = modal.Volume.from_name(VOLUME_NAME)


# TODO: make sure the timout is enough
@app.function(gpu="H100", image=image, volumes={VOLUME_DIR: volume}, timeout=1200)
def pipeline(data: Dict):
    # Use subprocess to execute the script
    with open("/workspace/main/data.json", "w") as f:
        json.dump(data, f)

    cmd = """cd /workspace/main/ && ls && . /opt/venv-ttt/bin/activate && python modal_ttt.py"""
    result = subprocess.run(cmd, shell=True,  capture_output=True, text=True)
    print(result)

    cmd = """cd /workspace/main/ && ls && . /opt/venv-inference/bin/activate && python modal_predict.py"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result)

    with open("/workspace/main/experiments/tti/modal/submission.json", "r") as f:
        result = f.read()

    print(result)

    return result


@app.local_entrypoint()
def main():
    # TODO: Adjust this for optimal strategy
    MAX_MACHINES = 400

    with open("/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json", "r") as f:
        data = json.load(f)

    # Split the data into chunks
    chunk_size = len(data) // MAX_MACHINES + (len(data) % MAX_MACHINES > 0)
    data_chunks = [dict(list(data.items())[i:i + chunk_size]) for i in range(0, len(data), chunk_size)]
    print(f"Split data into {len(data_chunks)} chunks")

    # Test for the first 5 chunks
    list_of_submissions = pipeline.starmap([(chunk,) for chunk in data_chunks[:2]], return_exceptions=True)

    merged_submission = {}
    for submission in list_of_submissions:
        # NOTE: seems like the remote output is serialized as string, so we need to deserialize it
        if isinstance(submission, str):
            try:
                submission = json.loads(submission)
                for key, value in submission.items():
                    merged_submission[key] = value
            except Exception as err:
                print("error in submission: ", submission, " error: ", err)
                continue


        for key, value in submission.items():
            merged_submission[key] = value

    with open(f"marc_submission.json", "w") as f:
        f.write(json.dumps(merged_submission))





