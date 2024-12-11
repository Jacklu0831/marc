import argparse
import copy
import functools
import json
import os
from multiprocessing import Pool
from typing import Dict
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
from accelerate.utils import set_seed
import wandb


parser = argparse.ArgumentParser(description="Process some integers.")
# model
parser.add_argument("--base_checkpoint_dir", type=str, default="checkpoints/pretrained/multi_format_model/")
parser.add_argument("--tokenizer_path", type=str, default="downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model")
parser.add_argument("--flash_attn", action="store_true", help="Whether to use the new format or not")
# data
parser.add_argument("--data_file", type=str, default="kaggle_dataset/arc-agi_evaluation_challenges.json")
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
parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--grad_accum", type=int, default=1, help="Batch size")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--float16", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--logging_steps", type=int, default=10, help="Batch size")
parser.add_argument("--lr_scheduler_type", type=str, default='constant', help="Batch size")
parser.add_argument("--warmup_steps", type=int, default=0, help="Batch size")
# prefix tuning
parser.add_argument("--num_virtual_tokens", type=int, default=4, help="number of virtual tokens")
# lora
parser.add_argument("--lora_rank", type=int, default=128)
parser.add_argument("--lora_alpha", type=float, default=16.0)
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument('--lora_target_modules', type=str, nargs="+", default=['q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'])
parser.add_argument("--lora_ckpt", type=str, default=None, help="submission folder")
# misc
parser.add_argument("--experiment_folder", type=str, default="experiments/ttt/new/", help="submission folder")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--reuse_prefix", action="store_true", help="Whether to use the new format or not")
parser.add_argument("--wandb", action='store_true', help="whether to log wandb")
parser.add_argument("--tracker_project_name", type=str, default="arc", help="The `project_name` argument passed to tor.init_trackers")


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

if args.lora_ckpt != None:
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

model.print_trainable_parameters()

##### END MODEL






##### BEGIN CUSTOM TRAINER SAVE CKPT

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

    if self.control.should_save:
        # begin custom code
        output_path = os.path.join(self.args.output_dir, f'checkpoint-epoch{epoch+1}.pt')
        torch.save(self.model.prompt_encoder.default.embedding.weight.data, output_path)
        print('checkpoint saved to', output_path)
        # end custom code
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)

Trainer._maybe_log_save_evaluate = _maybe_log_save_evaluate

##### END CUSTOM TRAINER SAVE CKPT





##### BEGIN TRAIN

saved_model_forward = model.forward
saved_prefix = copy.deepcopy(model.prompt_encoder.default.embedding.weight.data)
num_train_tasks, average_train_loss = 0, 0.0

for task in arc_test_tasks:
    task_id = task.name.replace("-0", "")
    if task_id in empty_tasks:
        continue

    output_dir = f"{args.experiment_folder}/{task_id}"
    print(f"Training task {task_id}")

    # reset model
    if args.reuse_prefix:
        model.prompt_encoder.default.embedding.weight.data = saved_prefix
    else:
        model.prompt_encoder.default.embedding.weight.data.copy_(saved_prefix)
    model.prompt_encoder.default.embedding.weight.grad = None

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

    # training
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=output_dir,
        logging_steps=args.logging_steps,
        report_to="none",
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=args.num_workers,
        bf16=args.flash_attn,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
    )
    train_out = trainer.train()

    if args.wandb:
        for step, history in enumerate(trainer.state.log_history[:-1]):
            step *= args.logging_steps
            wandb.log({f"train/loss_{task_id}": history['loss'], 'Steps': step})

    del trainer
    gc.collect()
    model.forward = saved_model_forward

    # accumulate train loss
    num_train_tasks += 1
    average_train_loss += train_out.training_loss

average_train_loss /= num_train_tasks
print('average train loss across tasks', average_train_loss)

if args.wandb:
    wandb.log({"train/avg_loss": average_train_loss, 'Steps': 0})

##### END TRAIN