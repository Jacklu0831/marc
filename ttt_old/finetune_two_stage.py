import numpy as np
from tqdm import tqdm
from collections import Counter
import glob
import argparse
import copy
import functools
import json
import os
os.system('nvidia-smi')
from multiprocessing import Pool
import gc

import torch
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.datasets import arc_dataset

import arclib.messagers
from arclib.arc import (
    make_submission,
    read_tasks_from_single_file,
    to_list,
    to_tuple,
)
from arclib.eval import evaluate
from arclib.messagers import GPTTextMessageRepresenterV2
from arclib.representers import (
    PythonListGridRepresenter,
    TextExampleRepresenter,
    TextTaskRepresenter,
)
from utils.preprocess import get_augmenters, process_task

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from vllm.prompt_adapter.request import PromptAdapterRequest
from peft import PrefixTuningConfig, LoraConfig, get_peft_model

from accelerate.utils import set_seed
import wandb

from arclib.voting import vote
from inference.engine import get_sampling_params
from inference.preprocess import get_formatted_tasks
from accelerate.utils import set_seed


parser = argparse.ArgumentParser(description="Process some integers.")
# model
parser.add_argument("--base_checkpoint_dir", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--tokenizer_path", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model")
parser.add_argument("--flash_attn", action="store_true", help="Whether to use the new format or not")
# data
parser.add_argument("--data_file", type=str, default="kaggle_dataset/arc-agi_training_challenges.json")
parser.add_argument("--solution_file", type=str, default="kaggle_dataset/arc-agi_training_solutions.json")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
parser.add_argument("--new_format", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--barc_format", action="store_true", help="Whether to use the barc format or not")
parser.add_argument("--no_transform", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--unmask_outputs", type=bool, default=True, help="Unmask outputs setting")
parser.add_argument("--train_on_input", type=bool, default=False, help="Train on input setting")
parser.add_argument("--permute_n", type=int, default=1, help="Permute n")
parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
parser.add_argument("--num_tasks", type=int, default=10000, help="Number of tasks to process for limited evaluation.")
parser.add_argument("--num_max_per_task", type=int, default=250, help="Number of tasks to process for limited evaluation.")
parser.add_argument("--extra_leave_n", type=int, default=0, help="Train on input setting")
parser.add_argument("--cache_dataset", action='store_true', help="whether to log wandb")
# train args
parser.add_argument("--outer_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--prefix_steps", type=int, default=-1, help="Number of epochs")
parser.add_argument("--net_steps", type=int, default=-1, help="Number of epochs")
parser.add_argument("--prefix_epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--net_epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--logging_steps", type=int, default=10, help="Number of epochs")
parser.add_argument("--prefix_lr", type=float, default=5e-3, help="Learning rate")
parser.add_argument("--net_lr", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
# prefix tuning
parser.add_argument("--num_virtual_tokens", type=int, default=25, help="number of virtual tokens")
parser.add_argument("--reuse_prefix", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--pt_checkpoints_folder", type=str, default=None, help="Prefix tuning checkpoints folder, if none then base model is used")
parser.add_argument("--pt_epoch", type=int, default=0, help="Prompt tuning checkpoints folder, if none then base model is used")
# lora
parser.add_argument("--lora_rank", type=int, default=128)
parser.add_argument("--lora_alpha", type=float, default=16.0)
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument('--lora_target_modules', type=str, nargs="+", default=['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'])
parser.add_argument("--use_lora", action="store_true", help="Whether to use the new format or not")
# misc
parser.add_argument("--experiment_folder", type=str, default="experiments/ttt/new/", help="submission folder")
parser.add_argument("--save_every", type=int, default=1, help="Random seed")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--float16", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--wandb", action='store_true', help="whether to log wandb")
parser.add_argument("--tracker_project_name", type=str, default="arc", help="The `project_name` argument passed to tor.init_trackers")
# eval
parser.add_argument("--eval_every", type=int, default=5, help="Random seed")


args = parser.parse_args()
os.makedirs(args.experiment_folder, exist_ok=True)
set_seed(args.seed)

print("#### BEGIN ALL ARGUMENTS ####")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("#### END ALL ARGUMENTS ####\n")


if args.wandb:
    wandb.init(project=args.tracker_project_name,
               name=args.experiment_folder.split()[-1],
               config=dict(vars(args)))
    wandb.define_metric('Steps')
    wandb.define_metric("*", step_metric="Steps")


##### BEGIN DATA

arc_test_tasks = read_tasks_from_single_file(args.data_file, test=True)
arc_test_tasks = [task for task in arc_test_tasks if "-0" in task.name]
arc_test_tasks = arc_test_tasks[: args.num_tasks]
arc_test_ids = [task.name.replace("-0", "") for task in arc_test_tasks]
print("Number of train tasks: ", len(arc_test_tasks))

if args.new_format:
    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )
    formatter = GPTTextMessageRepresenterV2(task_representer=standard_formatter)
elif args.barc_format:
    formatter = arclib.messagers.GPTTextMessageRepresenterForBarc(
        prompt = (
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 26 Jul 2024\n\n"
            "You are a world-class puzzle solver with exceptional pattern recognition skills. "
            "Your task is to analyze puzzles, spot patterns, and provide direct solutions."
        ),
        task_representer=arclib.representers.TextTaskRepresenter(
            example_representer=arclib.representers.TextExampleRepresenter(
            grid_representer=arclib.representers.WordGridRepresenter(),
            input_header="Input:\n",
            output_header="\nOutput:\n",
            io_sep="\n"
        )))
else:
    formatter = arclib.messagers.GPTTextMessageRepresenterV2()

tokenizer = llama3_tokenizer(args.tokenizer_path)
augmenters_to_apply = []
if not args.no_transform:
    augmenters_to_apply = get_augmenters(include_basic=True, include_size=True, include_chain=True, include_repeat=True)
processor = functools.partial(
    process_task,
    augmenters=augmenters_to_apply,
    formatter=formatter,
    tokenizer=tokenizer,
    permute_n=args.permute_n,
    Nmax=args.num_max_per_task,
    seed=args.seed,
    num_virtual_tokens=args.num_virtual_tokens,
    extra_leave_n=args.extra_leave_n,
)
with Pool(args.num_workers) as p:
    # this does use the tokenizer, but only for figuring out lengths
    data = p.map(processor, arc_test_tasks)
# data = [processor(task) for task in arc_test_tasks]
assert len(data) == len(arc_test_tasks)

empty_tasks = set()
for task, task_train_data in zip(arc_test_tasks, data):
    task_id = task.name.replace("-0", "")
    os.makedirs(f"{args.experiment_folder}/{task_id}", exist_ok=True)
    # save data for torchtune
    with open(f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "w") as f:
        if len(task_train_data) == 0:
            print(f'{task_id} has no data')
            empty_tasks.add(task_id)
        for td in task_train_data:
            print(json.dumps(td), file=f)
    # placeholder test file for torchtune
    with open(f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "r") as src, open(f"{args.experiment_folder}/{task_id}/td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "w") as dst:
        first_line = src.readline()
        dst.write(first_line)

# data tokenizer and collator
def padded_collate_sft(batch, padding_idx=0, ignore_idx=-100):
    input_ids = pad_sequence(
        [torch.tensor(x["input_ids"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )
    return {"input_ids": input_ids.long(), "labels": labels.long()}

collate_fn = functools.partial(padded_collate_sft, padding_idx=tokenizer.pad_id, ignore_idx=-100)

##### END DATA





##### START MODEL

if args.flash_attn:
    model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint_dir, cache_dir=f'{args.base_checkpoint_dir}_cache', device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
else:
    model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint_dir, cache_dir=f'{args.base_checkpoint_dir}_cache', device_map="auto", torch_dtype=torch.bfloat16)

# add lora
if args.use_lora:
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,  # Rank of the LoRA layer
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
# add prefix tuning
prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=args.num_virtual_tokens,
    encoder_hidden_size=model.config.hidden_size
)
# model.enable_input_require_grads()
model = get_peft_model(model, prefix_config)
# cast float16
if args.float16:
    model = model.to(torch.bfloat16)

def set_requires_grad(params, requires_grad):
    for p in params:
        p.requires_grad = requires_grad

def count_params(params):
    return sum(p.numel() for p in params)

set_requires_grad(model.parameters(), False)
net_params = [p for n, p in model.named_parameters() if 'lora' in n] if args.use_lora else [p for n, p in model.named_parameters() if 'prompt_encoder' not in n]
prompt_encoder_params = [p for n, p in model.named_parameters() if 'prompt_encoder' in n]
print(f'model has {count_params(model.parameters())} params in total')
print(f'found {count_params(net_params)} net params')
print(f'found {count_params(prompt_encoder_params)} prompt encoder params')
# found 1,236,224,000 net params
# found 76,546,048 net params
# found 409600 prompt encoder params

# get pt paths and filter tasks if necessary
id_to_pt_path = {}
if args.pt_checkpoints_folder is not None:
    # load pt paths
    for pt_path in glob.glob(f"{args.pt_checkpoints_folder}/*/checkpoint-epoch*.pt"):
        epoch = int(pt_path[pt_path.rfind('checkpoint-epoch'):][16:-3])
        if epoch == args.pt_epoch:
            pt_id = pt_path.split('/')[-2]
            id_to_pt_path[pt_id] = pt_path
    print(f'loaded {len(id_to_pt_path)} pt paths')

    # filter tasks
    for t in arc_test_tasks:
        task_id = t.name.replace("-0", "")
        if task_id not in id_to_pt_path:
            assert task_id in empty_tasks

##### END MODEL




##### BEGIN TRAIN

saved_model_forward = model.forward
init_prefix = copy.deepcopy(model.prompt_encoder.default.embedding.weight.data)
task_id_to_last_prefix = {}

task_to_step = Counter()
task_id_to_dataset = {}

for outer_epoch in range(1, args.outer_epochs + 1):
    print(f'\n################# BEGINNING EPOCH {outer_epoch} #################')
    num_train_tasks, average_train1_loss, average_train2_loss = 0, 0.0, 0.0

    for task in arc_test_tasks:
        task_id = task.name.replace("-0", "")
        if task_id in empty_tasks:
            continue

        output_dir = f"{args.experiment_folder}/{task_id}"
        print(f"\nTraining task {task_id}")

        # get dataset
        if task_id not in task_id_to_dataset:
            task_id_to_dataset[task_id] = arc_dataset(
                tokenizer=tokenizer,
                source=output_dir,
                train_on_input=args.train_on_input,
                unmask_outputs=args.unmask_outputs,
                change_dataset_key=True,
            )
        train_dataset = task_id_to_dataset[task_id]

        ##### BEGIN STAGE ONE
        print("begin stage one...")

        if outer_epoch == 1 or not args.reuse_prefix:
            if args.pt_checkpoints_folder is not None:
                loaded_prefix = torch.load(id_to_pt_path[task_id], weights_only=True).to(model.device)
                assert model.prompt_encoder.default.embedding.weight.data.shape == loaded_prefix.shape
                model.prompt_encoder.default.embedding.weight.data = loaded_prefix
                print('loaded prefix at', id_to_pt_path[task_id])
            else:
                model.prompt_encoder.default.embedding.weight.data = init_prefix
                print('setting prefix to the random initialization')
        else:
            last_prefix = torch.load(task_id_to_last_prefix[task_id], weights_only=True).to(model.device)
            assert model.prompt_encoder.default.embedding.weight.data.shape == last_prefix.shape
            model.prompt_encoder.default.embedding.weight.data = last_prefix
            print('loaded prefix at', task_id_to_last_prefix[task_id])

        set_requires_grad(net_params, False)
        set_requires_grad(prompt_encoder_params, True)
        for p in model.parameters(): p.grad = None

        # training
        training_args1 = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            save_strategy="no",
            learning_rate=args.prefix_lr,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.prefix_epochs,
            max_steps=args.prefix_steps,
            weight_decay=args.weight_decay,
            logging_dir=output_dir,
            logging_steps=max(min(args.logging_steps, args.prefix_steps), 1),
            report_to="none",
            lr_scheduler_type='constant',
            dataloader_num_workers=args.num_workers,
            bf16=args.flash_attn,
        )
        trainer1 = Trainer(
            model=model,
            args=training_args1,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=collate_fn,
        )
        train_out1 = trainer1.train()

        if args.wandb:
            for history in trainer1.state.log_history[:-1]:
                wandb.log({f"train/loss_{task_id}": history['loss'], 'Steps': task_to_step[task_id]})
                task_to_step[task_id] += args.logging_steps

        del trainer1
        gc.collect()
        model.forward = saved_model_forward

        output_path = os.path.join(output_dir, f'prefix-outer-epoch{outer_epoch}.pt')
        torch.save(model.prompt_encoder.default.embedding.weight.data, output_path)
        if task_id in task_id_to_last_prefix:
            os.remove(task_id_to_last_prefix[task_id])
            print('removed prefix at', task_id_to_last_prefix[task_id])
        task_id_to_last_prefix[task_id] = output_path
        print('saved prefix to', output_path)
        ##### END STAGE ONE

        ##### BEGIN STAGE TWO
        print("begin stage two...")
        set_requires_grad(net_params, True)
        set_requires_grad(prompt_encoder_params, False)
        for p in model.parameters(): p.grad = None
        # training
        training_args2 = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",
            save_strategy="no",
            learning_rate=args.net_lr,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.net_epochs,
            max_steps=args.net_steps,
            weight_decay=args.weight_decay,
            logging_dir=output_dir,
            logging_steps=max(min(args.logging_steps, args.net_steps), 1),
            report_to="none",
            lr_scheduler_type='constant',
            dataloader_num_workers=args.num_workers,
            bf16=args.flash_attn,
        )
        trainer2 = Trainer(
            model=model,
            args=training_args2,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=collate_fn,
        )
        train_out2 = trainer2.train()

        if args.wandb:
            for history in trainer2.state.log_history[:-1]:
                wandb.log({f"train/loss_{task_id}": history['loss'], 'Steps': task_to_step[task_id]})
                task_to_step[task_id] += args.logging_steps

        del trainer2
        gc.collect()
        model.forward = saved_model_forward

        # accumulate train loss
        num_train_tasks += 1
        average_train1_loss += train_out1.training_loss
        average_train2_loss += train_out2.training_loss
        ##### END STAGE TWO

        if not args.cache_dataset:
            del task_id_to_dataset[task_id]

    # log avg losses
    average_train1_loss /= num_train_tasks
    average_train2_loss /= num_train_tasks
    print('average train1 loss across tasks', average_train1_loss)
    print('average train2 loss across tasks', average_train2_loss)
    if args.wandb:
        wandb.log({"train/avg_loss1": average_train1_loss, 'Steps': outer_epoch})
        wandb.log({"train/avg_loss2": average_train2_loss, 'Steps': outer_epoch})

    # save lora
    if outer_epoch % args.save_every == 0:
        output_path = os.path.join(args.experiment_folder, f'checkpoint-outer-epoch{outer_epoch}.pt')
        if args.use_lora:
            lora_state_dict = {n: p for n, p in model.state_dict().items() if "lora" in n}
            torch.save(lora_state_dict, output_path)
            print('lora saved to', output_path)
        else:
            unet_state_dict = {n: p for n, p in model.state_dict().items() if "prompt_encoder" not in n}
            torch.save(unet_state_dict, output_path)
            print('unet saved to', output_path)

    if outer_epoch % args.eval_every == 0:
        tasks = read_tasks_from_single_file(args.data_file, solution_file=args.solution_file, test=True)

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

        if args.flash_attn:
            hf_tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint_dir, cache_dir=f'{args.base_checkpoint_dir}_cache', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint_dir, cache_dir=f'{args.base_checkpoint_dir}_cache', torch_dtype=torch.bfloat16)

        id_to_pt_path = task_id_to_last_prefix
        task_name_to_processed_data = get_formatted_tasks(
            tasks,
            hf_tokenizer,
            formatters,
            max_tokens=args.max_tokens,
            id_to_pt_path=id_to_pt_path,
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
                input_token_length = len(hf_tokenizer.encode(test_input["text"])) - 1
                sampling_param = get_sampling_params(
                    hf_tokenizer,
                    input_token_length,
                    args.max_tokens,
                    temperature=0.0,
                    n=1,
                )
                new_idx = name + "-" + str(j)
                inputs_to_the_engine.append(
                    (test_input["text"], sampling_param, pt_request, new_idx)
                )
                assert new_idx not in inputs_to_remember
                inputs_to_remember[new_idx] = test_input

        terminators = [
            hf_tokenizer.eos_token_id,
            hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # inference
        print(f"Number of input queries to the engine: {len(inputs_to_the_engine)}")
        outputs_by_key = {}
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
            inputs = hf_tokenizer(text, return_tensors="pt").to("cuda")
            max_new_tokens = sampling_params.max_tokens
            max_new_tokens = min(max_new_tokens, inputs['input_ids'].shape[1])
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                num_return_sequences=sampling_params.n,
                do_sample=False,
                eos_token_id=terminators,
            )
            input_len = inputs["input_ids"].shape[-1] # BS1
            assert all(torch.equal(output[:input_len], inputs['input_ids'][0]) for output in outputs)
            decoded_outputs = [hf_tokenizer.decode(output[input_len:], skip_special_tokens=True) for output in outputs]
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
                    continue

                try:
                    pred = to_tuple(inverter_fn(decoded.output))
                except Exception as e:
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
        evaluate(args.data_file, args.solution_file, submission_file)



##### END TRAIN