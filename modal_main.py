from asyncio import sleep
from typing import Dict, List
import modal
import os
import subprocess
import json
import time


VOLUME_NAME = "marc_checkpoints"
VOLUME_DIR = "/checkpoints"
BATCH_SIZE = 1
MAX_CONCURRENCY = 10
CHALLENGE_FILE = "/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json"
MAX_NUM_BATCHES = 2


app = modal.App("marc-torchtune-custom-container")

image = modal.Image.from_registry("hanguo97/marc:0.8")\
    .run_commands(["cd /workspace/main/ && git pull origin modal"])\
    .run_commands([
    f"[ ! -e /workspace/main/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B ] && mkdir -p /workspace/main/checkpoints/pretrained/ && ln -s {VOLUME_DIR}/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B /workspace/main/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B"
])\
    .pip_install("huggingface_hub[cli]")
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(image=image, volumes={VOLUME_DIR: volume}, timeout=240)
def download():
    # check if checkpoint path exists
    if not os.path.exists(f"{VOLUME_DIR}/checkpoints/pretrained/"):
        # download the checkpoint with huggingface
        cmd = f"mkdir -p {VOLUME_DIR}/checkpoints/pretrained/ && huggingface-cli download ekinakyurek/Llama-3.1-ARC-Potpourri-Transduction-8B  --include '*'  --local-dir {VOLUME_DIR}/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result)
    else:
        # maybe a parallel process is downloading the checkpoint
        TIMEOUT=180
        FILES = ["model-0000{0}-of-00004.safetensors".format(i+1) for i in range(4)]
        start_time = time.time()
        # list dir
        print("BEFORE: ", os.listdir(f"{VOLUME_DIR}/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B"))
        while True:
            if all(os.path.exists(f"{VOLUME_DIR}/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B/{file}") for file in FILES):
                print("All checkpoint files are available.")
                break
            elif time.time() - start_time > TIMEOUT:
                raise TimeoutError("Timeout reached while waiting for checkpoint files.")
            else:
                print("Waiting for checkpoint files to be available...")
                time.sleep(5)

        print("AFTER: ", os.listdir(f"{VOLUME_DIR}/checkpoints/pretrained/Llama-3.1-ARC-Potpourri-Transduction-8B"))
        time.sleep(5)


# TODO: make sure the timout is enough
@app.function(gpu="H100", image=image, volumes={VOLUME_DIR: volume}, timeout=1200, concurrency_limit=MAX_CONCURRENCY)
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
    # call the download function
    download.remote()

    with open(CHALLENGE_FILE, "r") as f:
        data = json.load(f)

    # Split the data into chunks
    data_chunks = [dict(list(data.items())[i:i + BATCH_SIZE]) for i in range(0, len(data), BATCH_SIZE)]
    print(f"Split data into {len(data_chunks)} chunks")
    # print length of each chunk
    for i, chunk in enumerate(data_chunks):
        print(f"Chunk {i} has {len(chunk)} items")

    # Test for the first 5 chunks
    max_num_batches = MAX_NUM_BATCHES if MAX_NUM_BATCHES else len(data_chunks)
    list_of_submissions = pipeline.starmap([(chunk,) for chunk in data_chunks[:max_num_batches]], return_exceptions=True)

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





