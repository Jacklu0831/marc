import argparse
import copy
import functools
import json
import os
from multiprocessing import Pool
import random
import numpy as np
from typing import Dict

import torch
from torchtune.config._parse import TuneRecipeArgumentParser
from torchtune.config._utils import _merge_yaml_and_cli_args
from torchtune.models.llama3 import llama3_tokenizer
from torchtune import config

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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import PrefixTuningConfig, get_peft_model



parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--data_file",
    type=str,
    default="/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json",
    help="Data file path to evaluate",
)
parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks to process for limited evaluation.")
parser.add_argument("--num_max_per_task", type=int, default=250, help="Number of tasks to process for limited evaluation.")
parser.add_argument("--quantization", type=str, default=None, help="Quantization type bitsandbytes or none")
parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
parser.add_argument("--cpus", type=int, default=16, help="Number of cpus")
parser.add_argument(
    "--pt_config",
    type=str,
    default="configs/ttt/1B_prefix_tuning_single_device.yaml",
    help="LoRA config file",
)
parser.add_argument(
    "--experiment_folder", type=str, default="experiments/ttt/new/", help="submission folder"
)
parser.add_argument(
    "--formatter",
    type=str,
    default="arclib.messagers.GPTTextMessageRepresenterV2",
    help="formatter for the task, better to be same with the one used for training",
)
parser.add_argument("--unmask_outputs", type=bool, default=True, help="Unmask outputs setting")
parser.add_argument("--train_on_input", type=bool, default=False, help="Train on input setting")
parser.add_argument("--permute_n", type=int, default=1, help="Permute n")
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--compile", type=bool, default=True, help="Compile setting")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument(
    "--base_checkpoint_dir",
    type=str,
    default="checkpoints/pretrained/multi_format_model/",
    help="Checkpoint directory",
)
parser.add_argument("--new_format", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--barc_format", action="store_true", help="Whether to use the barc format or not")
parser.add_argument("--no_transform", action="store_true", help="Whether to use the new format or not")
# prompt tuning
parser.add_argument("--num_virtual_tokens", type=int, default=4, help="number of virtual tokens")


args = parser.parse_args()
os.makedirs(args.experiment_folder, exist_ok=True)

arc_test_tasks = read_tasks_from_single_file(args.data_file, test=True)
arc_test_tasks = [task for task in arc_test_tasks if "-0" in task.name]
if args.num_tasks is not None:
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

# Load config
conf = _merge_yaml_and_cli_args(
    *TuneRecipeArgumentParser(
        description="PromptTuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    ).parse_known_args(["--config={}".format(args.pt_config)])
)

# Update conf with argparse settings
conf.dataset.unmask_outputs = args.unmask_outputs # true
conf.dataset.train_on_input = args.train_on_input # false
conf.epochs = args.epochs
conf.batch_size = args.batch_size
conf.gradient_accumulation_steps = args.gradient_accumulation_steps
conf.optimizer.lr = args.learning_rate
conf.compile = False  # we will do it ourselves
conf.seed = args.seed
conf.output_dir = f"{args.experiment_folder}"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# print conf
from pprint import pprint
pprint(dict(conf))

# data
tokenizer = llama3_tokenizer(conf.tokenizer.path)
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
)
with Pool(args.cpus) as p:
    # this does use the tokenizer, but only for figuring out lengths
    data = p.map(processor, arc_test_tasks)
# data = [processor(task) for task in arc_test_tasks]
assert len(data) == len(arc_test_tasks)
del tokenizer

for task, task_train_data in zip(arc_test_tasks, data):
    task_id = task.name.replace("-0", "")
    os.makedirs(f"{args.experiment_folder}/{task_id}", exist_ok=True)
    # save data for torchtune
    with open(f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "w") as f:
        for td in task_train_data:
            print(json.dumps(td), file=f)
    # placeholder test file for torchtune
    with open(f"{args.experiment_folder}/{task_id}/td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "r") as src, open(f"{args.experiment_folder}/{task_id}/td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl", "w") as dst:
        first_line = src.readline()
        dst.write(first_line)







# base model
tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint_dir, torch_dtype=torch.bfloat16, cache_dir=f'{args.base_checkpoint_dir}_cache')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base_checkpoint_dir, torch_dtype=torch.bfloat16, cache_dir=f'{args.base_checkpoint_dir}_cache', device_map="auto")

# prefix tuning
prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=args.num_virtual_tokens,
    encoder_hidden_size=model.config.hidden_size
)
model = get_peft_model(model, prefix_config)
model.print_trainable_parameters()

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




def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
    if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
        logs: Dict[str, float] = {}

        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss

        logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        if grad_norm is not None:
            logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        logs["learning_rate"] = self._get_learning_rate()

        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.store_flos()

        self.log(logs)

    metrics = None
    if self.control.should_evaluate:
        metrics = self._evaluate(trial, ignore_keys_for_eval)

    if self.control.should_save:
        # begin custom code
        output_path = os.path.join(self.args.output_dir, f'checkpoint-epoch{epoch}.pt')
        torch.save(self.model.prompt_encoder['default'].embedding.weight.data, output_path)
        print('checkpoint saved to', output_path)
        # end custom code
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)

Trainer._maybe_log_save_evaluate = _maybe_log_save_evaluate


# train
torchtune_tokenizer = config.instantiate(conf.tokenizer)
collate_fn = functools.partial(padded_collate_sft, padding_idx=torchtune_tokenizer.pad_id, ignore_idx=-100)

saved_prefix = copy.deepcopy(model.prompt_encoder['default'].embedding.weight.data)
for task in arc_test_tasks:
    task_id = task.name.replace("-0", "")
    print(f"Trying task {task_id}")

    try:
        # reset model
        model.prompt_encoder['default'].embedding.weight.data = saved_prefix
        model.prompt_encoder['default'].embedding.weight.grad = None

        # get dataset
        conf.dataset.source = f"{args.experiment_folder}/{task_id}"
        ds = config.instantiate(conf.dataset, torchtune_tokenizer)
        all_input_ids, all_label = [], []
        for i in range(len(ds)):
            data = ds[i]
            all_input_ids.append(data['tokens'])
            all_label.append(data['labels'])
        train_dataset = Dataset.from_dict({'input_ids': all_input_ids, 'labels': all_label})

        # training
        training_args = TrainingArguments(
            output_dir=f"{args.experiment_folder}/{task_id}",
            evaluation_strategy="no",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            logging_dir=f"{args.experiment_folder}/{task_id}",
            logging_steps=1,
            report_to="none",
            lr_scheduler_type='constant',
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=torchtune_tokenizer,
            data_collator=collate_fn,
        )
        trainer.train()

    except Exception as e:
        print(e)
        print("Error training for ", task_id)
        continue
