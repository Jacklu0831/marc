import glob
import argparse
import copy
import functools
import json
import os
from multiprocessing import Pool
import random
import numpy as np
import gc

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
parser.add_argument("--base_checkpoint_dir", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--tokenizer_path", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model")
parser.add_argument("--flash_attn", action="store_true", help="Whether to use the new format or not")
# data
parser.add_argument("--data_file", type=str, default="kaggle_dataset/arc-agi_training_combined.json")
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

for outer_epoch in range(1, args.outer_epochs + 1):
    print(f'\nBEGINNING EPOCH {outer_epoch}')

    for task in arc_test_tasks:
        task_id = task.name.replace("-0", "")
        if task_id in empty_tasks:
            continue

        output_dir = f"{args.experiment_folder}/{task_id}"
        print(f"Training task {task_id}")

        # get dataset
        ds = arc_dataset(
            tokenizer=tokenizer,
            source=output_dir,
            train_on_input=args.train_on_input,
            unmask_outputs=args.unmask_outputs,
        )
        all_input_ids, all_label = [], []
        for i in range(len(ds)):
            data = ds[i]
            all_input_ids.append(data['tokens'])
            all_label.append(data['labels'])
        train_dataset = Dataset.from_dict({'input_ids': all_input_ids, 'labels': all_label})

        ##### BEGIN STAGE ONE
        print("\nbegin stage one...")

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
            logging_steps=args.logging_steps,
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
        trainer1.train()

        del trainer1
        gc.collect()
        model.forward = saved_model_forward

        if args.reuse_prefix:
            output_path = os.path.join(output_dir, f'prefix-outer-epoch{outer_epoch}.pt')
            torch.save(model.prompt_encoder.default.embedding.weight.data, output_path)
            if task_id in task_id_to_last_prefix:
                os.remove(task_id_to_last_prefix[task_id])
                print('removed prefix at', task_id_to_last_prefix[task_id])
            task_id_to_last_prefix[task_id] = output_path
            print('saved prefix to', output_path)
        ##### END STAGE ONE

        ##### BEGIN STAGE TWO
        print("\nbegin stage two...")
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
            logging_steps=args.logging_steps,
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
        trainer2.train()

        del trainer2
        gc.collect()
        model.forward = saved_model_forward
        ##### END STAGE TWO

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

##### END TRAIN