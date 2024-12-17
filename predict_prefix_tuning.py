import argparse
import glob
import json
import os
os.system('nvidia-smi')
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.prompt_adapter.request import PromptAdapterRequest
from peft import get_peft_model, PrefixTuningConfig
import arclib.messagers
from arclib.arc import (
    make_submission,
    read_tasks_from_single_file,
    to_list,
    to_tuple,
)
import arclib.augmenters  # noqa: F401 to prevent removal by black
from arclib.eval import evaluate
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.representers import (
    DiffExampleRepresenter,
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
)
from arclib.voting import vote
from inference.engine import get_sampling_params
from inference.preprocess import get_preprocessed_tasks
from accelerate.utils import set_seed


parser = argparse.ArgumentParser(description="Process some integers.")
# data
parser.add_argument("--data_file", type=str, default="kaggle_dataset/arc-agi_evaluation_challenges_selected.json", help="Data file path to evaluate")
parser.add_argument("--solution_file", type=str, default="kaggle_dataset/arc-agi_evaluation_solutions_selected.json", help="Solution file path to evaluate")
parser.add_argument("--num_tasks", type=int, default=10000, help="Number of examples to process for limited evaluation.")
parser.add_argument("--new_format", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--barc_format", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--add_diff_format", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--include_n", type=int, nargs="+", default=[1], help="Which leave-n tasks to include, it is generally 1 for test time trained model, 0 for base model")
parser.add_argument("--permute_n", type=int, default=2, help="Number of permutations to generate for each leave-n task")
# model
parser.add_argument("--pretrained_checkpoint", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct", help="path to the pretrained checkpoint")
parser.add_argument("--pt_checkpoints_folder", type=str, default=None, help="Prefix tuning checkpoints folder, if none then base model is used")
parser.add_argument("--pt_epoch", type=int, default=0, help="Prompt tuning checkpoints folder, if none then base model is used")
parser.add_argument("--train_mode", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--flash_attn", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--float16", action="store_true", help="Whether to use the new format or not")
# prefix tuning
parser.add_argument("--num_virtual_tokens", type=int, default=4, help="number of virtual tokens")
parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
# sampling
parser.add_argument("--n_sample", type=int, default=1, help="Number of samples to generate per input")
# misc
parser.add_argument("--experiment_folder", type=str, default="experiments/tti/new/", help="submission folder")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--limit_tokens", action="store_true", help="Whether to use the new format or not")
args = parser.parse_args()

# print args
print("Arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

os.makedirs(args.experiment_folder, exist_ok=True)
set_seed(args.seed)


tasks = read_tasks_from_single_file(args.data_file, solution_file=args.solution_file, test=True)
tasks = tasks[: args.num_tasks]

formatters = []
if args.new_format:
    messager = GPTTextMessageRepresenterV2(
        task_representer=TextTaskRepresenter(
            example_representer=TextExampleRepresenter(
                io_sep=" -> ",
                input_header="",
                output_header="",
                output_footer="#",
                grid_representer=PythonListGridRepresenter(),
            )
        )
    )
    formatters.append(messager)
elif args.barc_format:
    messages = arclib.messagers.GPTTextMessageRepresenterForBarc(
        task_representer=arclib.representers.TextTaskRepresenter(
            example_representer=arclib.representers.TextExampleRepresenter(
            grid_representer=arclib.representers.WordGridRepresenter(),
            input_header="Input:\n",
            output_header="\nOutput:\n",
            io_sep="\n"
        )))
    formatters.append(messages)
else:
    messager = arclib.messagers.GPTTextMessageRepresenterV2()
    formatters.append(messager)

if args.add_diff_format:
    diff_formatter = TextTaskRepresenter(
        example_representer=DiffExampleRepresenter(
            use_output=False,
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )
    input_diff_formatter = GPTTextMessageRepresenterV2(task_representer=diff_formatter)

    formatters.append(input_diff_formatter)

if args.flash_attn:
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_checkpoint, cache_dir=f'{args.pretrained_checkpoint}_cache', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
else:
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_checkpoint, cache_dir=f'{args.pretrained_checkpoint}_cache', torch_dtype=torch.bfloat16)

# get pt paths and filter tasks if necessary
id_to_pt_path = {}
if args.pt_checkpoints_folder is not None:
    for pt_path in glob.glob(f"{args.pt_checkpoints_folder}/*/checkpoint-epoch*.pt"):
        epoch = int(pt_path[pt_path.rfind('checkpoint-epoch'):][16:-3])
        if epoch == args.pt_epoch:
            pt_id = pt_path.split('/')[-2]
            id_to_pt_path[pt_id] = pt_path

task_name_to_processed_data = get_preprocessed_tasks(
    tasks,
    tokenizer,
    formatters,
    max_tokens=args.max_tokens,
    id_to_pt_path=id_to_pt_path,
    include_n=args.include_n,
    permute_n=args.permute_n,
)
valid_tasks = [info for key, info in task_name_to_processed_data.items() if info["valid"]]
invalid_tasks = [info for key, info in task_name_to_processed_data.items() if not info["valid"]]

# prediction has an extra part
for i in range(len(valid_tasks)):
    for query in valid_tasks[i]['queries']:
        split_text = query['text'].split('\n')
        split_text.pop(1)
        split_text.pop(1)
        split_text.pop(1)
        query['text'] = '\n'.join(split_text)

print("Len of valid tasks:", len(valid_tasks))
print("Len of invalid tasks:", len(invalid_tasks))
# for each valid task print the length of queries
for info in valid_tasks:
    print(f"{info['task'].name}: Number of Queries: {len(info['queries'])}")

example_task = valid_tasks[0]
example_task_id = example_task["task"].name.split("-")[0]

print()
print("Example Task Information:")
print(f"Task Name: {example_task['task'].name}")
print(f"Number of Queries: {len(example_task['queries'])}")
print("Example Query")
print(example_task["queries"][0]["text"])

# pt config
pt_path_idxs = list(id_to_pt_path.keys())

# abstract away
inputs_to_the_engine = []
inputs_to_remember = {}
for i, info in enumerate(valid_tasks):
    name = info["task"].name
    idx, no = name.split("-")
    # prompt tuning or no
    if args.pt_checkpoints_folder is not None:
        pt_path = id_to_pt_path[idx]
        # pt_path = os.path.dirname(pt_path)
        pt_index = pt_path_idxs.index(idx)
        pt_request = PromptAdapterRequest(
            prompt_adapter_name=idx + no,
            prompt_adapter_id=pt_index,
            prompt_adapter_local_path=pt_path,
            prompt_adapter_num_virtual_tokens=args.num_virtual_tokens
        )
    else:
        pt_request = None
    test_inputs = info["queries"]
    for j, test_input in enumerate(test_inputs):
        input_token_length = len(tokenizer.encode(test_input["text"])) - 1
        sampling_param = get_sampling_params(
            tokenizer,
            input_token_length,
            args.max_tokens,
            temperature=args.temperature,
            n=args.n_sample,
        )
        new_idx = name + "-" + str(j)
        inputs_to_the_engine.append(
            (test_input["text"], sampling_param, pt_request, new_idx)
        )
        assert new_idx not in inputs_to_remember
        inputs_to_remember[new_idx] = test_input


# base model
tokenizer.pad_token = tokenizer.eos_token
if args.flash_attn:
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_checkpoint, cache_dir=f'{args.pretrained_checkpoint}_cache', device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
else:
    model = AutoModelForCausalLM.from_pretrained(args.pretrained_checkpoint, cache_dir=f'{args.pretrained_checkpoint}_cache', device_map="auto", torch_dtype=torch.bfloat16)

# prefix tuning
if args.pt_checkpoints_folder is not None:
    prefix_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=args.num_virtual_tokens,
        encoder_hidden_size=model.config.hidden_size,
        inference_mode=True,
    )
    model = get_peft_model(model, prefix_config)
if args.float16:
    model = model.to(torch.bfloat16)
model.print_trainable_parameters()

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

max_length, avg_length, max_tokens, avg_tokens = 0, 0, 0, 0
for text, sampling_params, _, _ in inputs_to_the_engine:
    find_start = text.find("<|begin_of_text|>") + len("<|begin_of_text|>")
    text = text[find_start:]
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    avg_length += inputs['input_ids'].shape[1]
    max_length = max(max_length, inputs['input_ids'].shape[1])
    max_tokens = max(max_tokens, sampling_params.max_tokens)
    avg_tokens += sampling_params.max_tokens
print(f'max input length {max_length} avg input length {avg_length / len(inputs_to_the_engine)}')
print(f'max gen tokens {max_tokens} avg gen tokens {avg_tokens / len(inputs_to_the_engine)}')

# inference
print(f"Number of input queries to the engine: {len(inputs_to_the_engine)}")
outputs_by_key = {}
if not args.train_mode:
    model.eval()
for text, sampling_params, pt_request, new_idx in tqdm(inputs_to_the_engine):
    # load embeds
    if args.pt_checkpoints_folder is not None:
        loaded_embeds = torch.load(pt_request.prompt_adapter_local_path, weights_only=True).to(model.device)
        assert model.prompt_encoder.default.embedding.weight.data.shape == loaded_embeds.shape
        model.prompt_encoder.default.embedding.weight.data = loaded_embeds
    # text process?
    find_start = text.find("<|begin_of_text|>") + len("<|begin_of_text|>")
    text = text[find_start:]
    # inference
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    max_new_tokens = sampling_params.max_tokens
    if args.limit_tokens:
        max_new_tokens = min(max_new_tokens, inputs['input_ids'].shape[1])
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_new_tokens,
        num_return_sequences=sampling_params.n,
        do_sample=False,
        eos_token_id=terminators,
        # repetition_penalty=sampling_params.repetition_penalty,
        # pad_token_id=tokenizer.eos_token_id,
    )
    input_len = inputs["input_ids"].shape[-1] # BS1
    assert all(torch.equal(output[:input_len], inputs['input_ids'][0]) for output in outputs)
    decoded_outputs = [tokenizer.decode(output[input_len:], skip_special_tokens=True) for output in outputs]
    print(decoded_outputs[0])
    print()
    outputs_by_key[new_idx] = decoded_outputs


for key in list(outputs_by_key.keys()):
    inverter = inputs_to_remember[key]["inverter"]
    if inverter is not None:
        inverter_fn = eval("arclib.augmenters." + inverter)
    else:
        inverter_fn = np.array

    outputs = outputs_by_key[key]
    outputs_by_key[key] = []
    current_formatter_repr = inputs_to_remember[key]["formatter"]
    input = inputs_to_remember[key]["input"]["content"]
    current_formatter = eval(current_formatter_repr)

    for output in outputs:
        output = output.replace("#", "")
        output = output.replace("  ", " ")
        if "```" in output:
            # get things between ``` and ```
            output = output.split("```")[1]
            output = output.strip()
            input = input.split("Here is the input grid for the test example:\nInput:\n")[-1]
            input = input.split("\n\n\nDirectly provide")[0]
            input = input.strip()

        try:
            decoded = current_formatter.task_representer.example_representer.decode(
                (input, output)
            )
        except Exception as e:
            print(f"Cannot Decode: {e}")
            print(f"Input: {input}")
            print(f"Output: {output}")
            continue

        try:
            pred = to_tuple(inverter_fn(decoded.output))
        except Exception as e:
            print(f"Error: {e}")
            continue

        if decoded is not None:
            outputs_by_key[key].append(
                {
                    "output": to_tuple(inverter_fn(decoded.output)),
                    "inverter": inverter,
                    "formatter": current_formatter_repr,
                }
            )

outputs_by_key = {key: outputs for key, outputs in outputs_by_key.items() if len(outputs) > 0}

# save
all_predictions_file = os.path.join(args.experiment_folder, "all_predictions.json")

with open(all_predictions_file, "w") as f:
    json.dump(outputs_by_key, f)

outputs = {}
for task in tasks:
    name = task.name

    to_vote = [out for key, out in outputs_by_key.items() if name in key]
    to_vote = [out for sublist in to_vote for out in sublist]

    if len(to_vote) == 0:
        outputs[name] = [[[0]], [[0]]]
        continue
    else:
        attempt_1, attempt_2 = vote(to_vote)
        outputs[name] = [to_list(attempt_1), to_list(attempt_2)]

predictions = [outputs[task.name] for task in tasks]

submission_file = os.path.join(args.experiment_folder, "submission.json")
make_submission(tasks, predictions, submission_file, number_of_attempts=2)

print(f"Submission file is saved to {submission_file}")

# evaluate
if args.solution_file is not None:
    evaluate(args.data_file, args.solution_file, submission_file)
