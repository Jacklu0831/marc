# accelerate launch --main_process_port $MASTER_PORT test_gpu_async_error.py
# NCCL_TIMEOUT=1 accelerate launch --main_process_port $MASTER_PORT test_gpu_async_error.py

import time
from accelerate import Accelerator, PartialState
from accelerate.utils import ProjectConfiguration, gather_object

project_config = ProjectConfiguration(project_dir="./temp")
accelerator = Accelerator(project_config=project_config)

all_data = list(range(143))

distributed_state = PartialState()
results = []

with accelerator.split_between_processes(all_data) as data:
    print(accelerator.process_index, 'has data:', data)
    for d in data:
        results.append(d ** 2)
    time.sleep(accelerator.process_index * 1000) # 2nd process wait 20 seconds

distributed_state.wait_for_everyone()
results = gather_object(results)
if accelerator.is_main_process:
    assert results == [i ** 2 for i in range(143)]