import argparse
import copy
import functools
import json
import os
from multiprocessing import Pool
import random
import numpy as np

import torch
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.datasets import arc_dataset

import arclib.messagers
from arclib.arc import read_tasks_from_single_file
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
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import PrefixTuningConfig, LoraConfig, get_peft_model



parser = argparse.ArgumentParser(description="Process some integers.")
# model
parser.add_argument("--base_checkpoint_dir", type=str, default="checkpoints/pretrained/multi_format_model/")
parser.add_argument("--tokenizer_path", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model")
parser.add_argument("--flash_attn", action="store_true", help="Whether to use the new format or not")
# data
parser.add_argument("--data_file", type=str, default="kaggle_dataset/arc-agi_training_combined.json")
parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
parser.add_argument("--new_format", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--barc_format", action="store_true", help="Whether to use the barc format or not")
parser.add_argument("--no_transform", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--unmask_outputs", type=bool, default=True, help="Unmask outputs setting")
parser.add_argument("--train_on_input", type=bool, default=False, help="Train on input setting")
parser.add_argument("--permute_n", type=int, default=1, help="Permute n")
parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
parser.add_argument("--num_tasks", type=int, default=10000, help="Number of tasks to process for limited evaluation.")
parser.add_argument("--num_max_per_task", type=int, default=250, help="Number of tasks to process for limited evaluation.")
# train args
parser.add_argument("--outer_epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--inner_epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
# prefix tuning
parser.add_argument("--num_virtual_tokens", type=int, default=25, help="number of virtual tokens")
# lora
parser.add_argument("--lora_rank", type=int, default=128)
parser.add_argument("--lora_alpha", type=float, default=16.0)
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument('--lora_target_modules', type=str, nargs="+", default=['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'])
parser.add_argument("--use_lora", action="store_true", help="Whether to use the new format or not")
# misc
parser.add_argument("--experiment_folder", type=str, default="experiments/ttt/new/", help="submission folder")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--float16", action="store_true", help="Whether to use the new format or not")


args = parser.parse_args()
os.makedirs(args.experiment_folder, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("#### BEGIN ALL ARGUMENTS ####")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("#### END ALL ARGUMENTS ####\n")






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
    include_leave_3=True,
)
with Pool(args.num_workers) as p:
    # this does use the tokenizer, but only for figuring out lengths
    data = p.map(processor, arc_test_tasks)
# data = [processor(task) for task in arc_test_tasks]
assert len(data) == len(arc_test_tasks)

for task, task_train_data in zip(arc_test_tasks, data):
    task_id = task.name.replace("-0", "")
    os.makedirs(f"{args.experiment_folder}/{task_id}", exist_ok=True)
    # save data for torchtune
    with open(f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "w") as f:
        if len(task_train_data) == 0:
            print(f'{task_id} has no data')
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
    model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint_dir, cache_dir=f'{args.base_checkpoint_dir}_cache', device_map="auto", attn_implementation="flash_attention_2")
else:
    model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint_dir, cache_dir=f'{args.base_checkpoint_dir}_cache', device_map="auto", torch_dtype=torch.bfloat16)

# add prefix tuning
prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=args.num_virtual_tokens,
    encoder_hidden_size=model.config.hidden_size
)
model = get_peft_model(model, prefix_config)
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
    # prefix tuning weight still requires grad
    model.prompt_encoder.default.embedding.weight.requires_grad = True
else:
    for p in model.parameters():
        p.requires_grad = True
# cast float16
if args.float16:
    model = model.to(torch.bfloat16)
model.print_trainable_parameters()

##### END MODEL







##### BEGIN TRAIN

saved_prefix = copy.deepcopy(model.prompt_encoder['default'].embedding.weight.data)

for outer_epoch in range(args.outer_epochs):
    print(f'BEGINNING EPOCH {outer_epoch}')

    for task in arc_test_tasks:
        task_id = task.name.replace("-0", "")
        output_dir = f"{args.experiment_folder}/{task_id}"
        print(f"Trying task {task_id}")

        try:
            # reset prompt embeddings, not loras
            # TODO: option to reuse prompt embeddings
            model.prompt_encoder['default'].embedding.weight.data = saved_prefix
            model.prompt_encoder['default'].embedding.weight.grad = None

            # get dataset
            ds = arc_dataset(
                tokenizer=tokenizer,
                source=output_dir,
                train_on_input=args.train_on_input,
                unmask_outputs=args.unmask_outputs,
                cache_dir='.cache/'
            )
            all_input_ids, all_label = [], []
            for i in range(len(ds)):
                data = ds[i]
                all_input_ids.append(data['tokens'])
                all_label.append(data['labels'])
            train_dataset = Dataset.from_dict({'input_ids': all_input_ids, 'labels': all_label})

            # training
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="no",
                save_strategy="no",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.inner_epochs,
                weight_decay=args.weight_decay,
                logging_dir=output_dir,
                logging_steps=10,
                report_to="none",
                lr_scheduler_type='constant',
                dataloader_num_workers=args.num_workers,
                bf16=args.flash_attn,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=collate_fn,
            )
            trainer.train()

        except Exception as e:
            print(e)
            print("Error training for ", task_id)
            continue

    output_path = os.path.join(args.experiment_folder, f'checkpoint-outer-epoch{outer_epoch}.pt')
    if args.use_lora:
        lora_state_dict = {n: p for n, p in model.state_dict().items() if "lora" in n}
        torch.save(lora_state_dict, output_path)
        print('lora saved to', output_path)
    else:
        unet_state_dict = {n: p for n, p in model.state_dict().items() if "prompt_encoder" not in n}
        torch.save(unet_state_dict, output_path)
        print('unet saved to', output_path)

##### END TRAIN