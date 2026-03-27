# ARC with Off-the-Shelf Large LLM (2026-03-27)

## Reviewer Question

R2 Weakness: "Base Model Selection for ARC: For the ARC benchmark, the authors used a Llama3.2-1B model that was explicitly fine-tuned on the ARC training split. Although this choice is justified by the base model's inability to solve ARC tasks zero-shot, the pre-tuning blurs the line between inference-time optimization and task-specific fine-tuning."

R2 Question: "How does CT-KV perform on ARC (or similar symbolic reasoning tasks) when applied to a larger, off-the-shelf open-source model that possesses stronger native spatial reasoning capabilities but has not undergone any task-specific fine-tuning?"

## Experiment Design

**Model**: Qwen2.5-14B-Instruct (4-bit NF4 quantized, ~10GB VRAM) — off-the-shelf, no ARC fine-tuning.

**Why Qwen2.5-14B-Instruct**: Strong reasoning capabilities, used in BBH/MMLU bigllm experiments. 14B at 4-bit fits comfortably on 1 GPU.

**Benchmark**: ARC evaluation split (400 tasks, 5 parts of 80).

**Key difference from original ARC setup**:
- Original: Llama-3.2-1B fine-tuned on ARC training set, custom 45-token vocabulary, resized embeddings
- New: Qwen2.5-14B-Instruct off-the-shelf, model's native BPE tokenizer, full vocabulary, text-based grid encoding

**Grid encoding format**:
```
input
3 4
1234
5678
9012
output
2 3
123
456
```
Grids encoded as plain text — dimensions on one line, then rows of concatenated digits. Each demo pair follows this format sequentially.

## Hyperparameters

Using the same ARC CT-KV best config from the paper:
- `gs_lr`: 3e-3
- `gs_epochs`: 200
- `gs_token_dropout`: 0.1
- `gs_dropout`: none (LOO masking disabled — ARC has <4 demos on average, LOO hurts in this regime per paper Section 5.7)

## Code Changes

Created `inference_arc_bigllm/` directory with:
- `arc_utils.py` — utility functions extracted from `inference_arc/train.py`
- `data_utils.py` — text-based grid encoding using model's native tokenizer (replaces ARCTokenizer)
- `test_time_evaluate.py` — multi-model support, optional LoRA loading, no embedding resizing
- `custom_llama.py` — copied from `inference_arc/` (only used for Llama models)
- `arclib/` — copied from `inference_arc/` (ARC task/augmenter classes)

**Backwards compatible**: `inference_arc/` is untouched. The `--weight_dir`/`--weight_epoch` args are optional — omit them for off-the-shelf models.

## bash_cmds

```
bash_cmds/0327_3_arc_bigllm/
  arc_icl.sh    # ICL baseline: 5 parts x 1 seed
  arc_ctkv.sh   # CT-KV: 5 parts x 1 seed (lr=3e-3, 200 epochs, tokdrop=0.1)
```

**Submission**:
```bash
makesbatch --time 3 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_icl.sh
makesbatch --time 6 --ngpu 1 --gb 64 --no_singularity --bash_file bash_cmds/0327_3_arc_bigllm/arc_ctkv.sh
```

## Expected Results

Even with an off-the-shelf 14B model (no ARC fine-tuning):
1. ICL accuracy should be low but non-trivial (the model sees demo pairs as in-context examples)
2. CT-KV should improve over ICL, demonstrating that KV optimization helps independently of fine-tuning
3. This directly refutes the reviewer's concern that CT-KV's gains on ARC are confounded with task-specific fine-tuning

## Rebuttal Framing

"We conduct additional experiments on ARC using Qwen2.5-14B-Instruct, a larger off-the-shelf model with no task-specific fine-tuning. Table X shows that CT-KV improves over ICL by Y% on this fully off-the-shelf model, demonstrating that the benefits of KV cache optimization are not contingent on task-specific pre-tuning. The improvement persists because CT-KV optimizes the context representation via gradient descent on demonstration pairs, a mechanism orthogonal to how the base model was trained."
