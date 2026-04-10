"""
TTFT Benchmark: ICL vs CT-KV (pre-loaded KV cache from disk).

Measures time-to-first-token on real BBH and MMLU data with Llama-3.2-1B-Instruct.
Same data setup as the evaluation scripts (same seed, same demos, same tokenization).

Usage:
    .venv/bin/python random_utils/ttft_benchmark.py --benchmark bbh --seed 42
    .venv/bin/python random_utils/ttft_benchmark.py --benchmark mmlu --seed 42
"""
import argparse
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def create_dummy_kv_cache(
    num_layers: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> DynamicCache:
    """Create a DynamicCache with dummy values matching model architecture.

    Args:
        num_layers: number of transformer layers
        num_kv_heads: number of key-value heads (GQA)
        seq_len: sequence length (number of demo tokens)
        head_dim: dimension per attention head
        dtype: tensor dtype

    Returns:
        DynamicCache with random values on CPU
    """
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        key = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype)
        value = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype)
        cache.update(key, value, layer_idx)
    return cache


def time_icl_ttft(
    model: torch.nn.Module,
    full_input_ids: torch.Tensor,
    n_warmup: int,
    n_measure: int,
) -> List[float]:
    """Measure ICL TTFT: forward pass over full [demo + query] input.

    Args:
        model: the language model
        full_input_ids: (1, D+Q) token IDs for demos + query
        n_warmup: number of warmup iterations
        n_measure: number of timed iterations

    Returns:
        list of TTFT measurements in seconds
    """
    attention_mask = torch.ones_like(full_input_ids)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids=full_input_ids, attention_mask=attention_mask, use_cache=False)

    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(input_ids=full_input_ids, attention_mask=attention_mask, use_cache=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def time_ctkv_load(
    kv_path: str,
    n_warmup: int,
    n_measure: int,
) -> List[float]:
    """Measure KV cache load time from disk to GPU.

    Args:
        kv_path: path to saved KV cache file
        n_warmup: number of warmup iterations
        n_measure: number of timed iterations

    Returns:
        list of load time measurements in seconds
    """
    for _ in range(n_warmup):
        _ = torch.load(kv_path, map_location='cuda', weights_only=True)

    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.load(kv_path, map_location='cuda', weights_only=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def time_ctkv_load_cpu(
    kv_cpu: tuple,
    n_warmup: int,
    n_measure: int,
) -> List[float]:
    """Measure KV cache transfer time from CPU to GPU.

    Args:
        kv_cpu: KV cache as tuple of (key, value) pairs on CPU
        n_warmup: number of warmup iterations
        n_measure: number of timed iterations

    Returns:
        list of transfer time measurements in seconds
    """
    for _ in range(n_warmup):
        _ = tuple((k.cuda(), v.cuda()) for k, v in kv_cpu)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = tuple((k.cuda(), v.cuda()) for k, v in kv_cpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def time_ctkv_forward(
    model: torch.nn.Module,
    query_input_ids: torch.Tensor,
    kv_path: str,
    demo_seq_len: int,
    n_warmup: int,
    n_measure: int,
) -> List[float]:
    """Measure CT-KV forward TTFT: forward pass over query with pre-loaded KV.

    Args:
        model: the language model
        query_input_ids: (1, Q) token IDs for query only
        kv_path: path to saved KV cache for loading
        demo_seq_len: length of demo tokens in the KV cache
        n_warmup: number of warmup iterations
        n_measure: number of timed iterations

    Returns:
        list of forward time measurements in seconds
    """
    query_len = query_input_ids.shape[1]
    attention_mask = torch.ones(1, demo_seq_len + query_len, device='cuda', dtype=torch.long)
    position_ids = torch.arange(demo_seq_len, demo_seq_len + query_len, device='cuda', dtype=torch.long).unsqueeze(0)

    for _ in range(n_warmup):
        loaded_kv = torch.load(kv_path, map_location='cuda', weights_only=True)
        kv_cache = DynamicCache.from_legacy_cache(loaded_kv)
        with torch.no_grad():
            model(
                input_ids=query_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=False,
            )

    times = []
    for _ in range(n_measure):
        loaded_kv = torch.load(kv_path, map_location='cuda', weights_only=True)
        kv_cache = DynamicCache.from_legacy_cache(loaded_kv)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(
                input_ids=query_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=False,
            )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times


def load_bbh_dataset(seed: int, tokenizer: AutoTokenizer) -> 'EvalDataset':
    """Load BBH dataset with same args as evaluation scripts."""
    # inference_bbh/data_utils.py does `from tasks import TASKS` — needs inference_bbh on path
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'inference_bbh'))
    from inference_bbh.data_utils import EvalDataset
    return EvalDataset(
        data_dir='data/BIG-Bench-Hard',
        seed=seed,
        tokenizer=tokenizer,
        debug_random_pad=False,
        debug_pad_len=-1,
        debug_max_len=False,
        pad_side='left',
        max_seq_len=2048,
        num_demonstrations=10,
        eval_ratio=1.0,
        eval_on_demonstrations=False,
    )


def load_mmlu_dataset(seed: int, tokenizer: AutoTokenizer) -> 'EvalDataset':
    """Load MMLU dataset with same args as evaluation scripts."""
    from inference_mmlu.data_utils import EvalDataset
    return EvalDataset(
        seed=seed,
        tokenizer=tokenizer,
        debug_random_pad=False,
        debug_pad_len=-1,
        debug_max_len=False,
        pad_side='left',
        max_seq_len=4096,
        eval_test_per_task=10000000,
        eval_ratio=0.25,
        delimiter='\n',
        num_demonstrations=16,
        filter_based_on_ndemo=16,
        wrong_label=0.0,
        eval_on_demonstrations=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='TTFT benchmark: ICL vs CT-KV')
    parser.add_argument('--benchmark', type=str, choices=['bbh', 'mmlu'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_warmup', type=int, default=3)
    parser.add_argument('--n_measure', type=int, default=10)
    parser.add_argument('--examples_per_task', type=int, default=10,
                        help='Number of random examples to measure per task')
    parser.add_argument('--load_from', type=str, choices=['disk', 'cpu'], default='disk',
                        help='Where to load KV cache from: disk (torch.load) or cpu (CPU→GPU transfer)')
    args = parser.parse_args()

    import pprint
    pprint.pprint(vars(args))

    # Initialize accelerate state (needed by data_utils logging)
    from accelerate import PartialState
    PartialState()

    # Load model and tokenizer (same as evaluation scripts)
    model_path = 'meta-llama/Llama-3.2-1B-Instruct'
    print(f'\nLoading model: {model_path}')
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='./encoder_decoder_cache')
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path, cache_dir='./encoder_decoder_cache')
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        cache_dir='./encoder_decoder_cache',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad = False

    # KV cache shape info
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    print(f'KV cache shape per layer: (1, {num_kv_heads}, seq_len, {head_dim})')
    print(f'Number of layers: {num_layers}')

    # Load dataset
    print(f'\nLoading {args.benchmark.upper()} dataset with seed={args.seed}...')
    if args.benchmark == 'bbh':
        dataset = load_bbh_dataset(seed=args.seed, tokenizer=tokenizer)
    else:
        dataset = load_mmlu_dataset(seed=args.seed, tokenizer=tokenizer)
    print(f'Dataset size: {len(dataset)} examples')

    # Group examples by task, randomly sample N per task
    import random
    task_all_indices: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(dataset)):
        task_all_indices[dataset[i]['task']].append(i)
    rng = random.Random(args.seed)
    task_examples: Dict[str, List[int]] = {}
    for task, indices in sorted(task_all_indices.items()):
        if len(indices) <= args.examples_per_task:
            task_examples[task] = indices
        else:
            task_examples[task] = sorted(rng.sample(indices, args.examples_per_task))
    n_tasks = len(task_examples)
    n_examples = sum(len(v) for v in task_examples.values())
    print(f'Measuring {n_examples} examples across {n_tasks} tasks '
          f'({args.examples_per_task} per task)')

    # Run benchmark
    all_icl_times: List[float] = []
    all_load_times: List[float] = []
    all_forward_times: List[float] = []
    all_demo_lens: List[int] = []
    all_query_lens: List[int] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for task_idx, (task, indices) in enumerate(sorted(task_examples.items())):
            for ex_idx in indices:
                example = dataset[ex_idx]
                demon_ids = example['demon_input_ids']
                gen_ids = example['gen_input_ids']
                demo_len = len(demon_ids)
                query_len = len(gen_ids)

                # ICL: full input
                full_ids = torch.cat([demon_ids, gen_ids]).unsqueeze(0).cuda()

                # CT-KV: create dummy KV cache
                dummy_kv = create_dummy_kv_cache(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    seq_len=demo_len,
                    head_dim=head_dim,
                    dtype=torch.bfloat16,
                )
                legacy_cache = tuple(
                    (dummy_kv.key_cache[i], dummy_kv.value_cache[i])
                    for i in range(num_layers)
                )

                if args.load_from == 'disk':
                    kv_path = os.path.join(tmpdir, f'kv_{task_idx}_{ex_idx}.pt')
                    torch.save(legacy_cache, kv_path)
                del dummy_kv

                # Query input for CT-KV
                query_ids = gen_ids.unsqueeze(0).cuda()

                # Measure
                icl_times = time_icl_ttft(
                    model=model, full_input_ids=full_ids,
                    n_warmup=args.n_warmup, n_measure=args.n_measure,
                )
                if args.load_from == 'disk':
                    load_times = time_ctkv_load(
                        kv_path=kv_path,
                        n_warmup=args.n_warmup, n_measure=args.n_measure,
                    )
                    forward_times = time_ctkv_forward(
                        model=model, query_input_ids=query_ids, kv_path=kv_path,
                        demo_seq_len=demo_len,
                        n_warmup=args.n_warmup, n_measure=args.n_measure,
                    )
                else:
                    load_times = time_ctkv_load_cpu(
                        kv_cpu=legacy_cache,
                        n_warmup=args.n_warmup, n_measure=args.n_measure,
                    )
                    # For forward timing with CPU source, load once to GPU then time forward
                    kv_path = os.path.join(tmpdir, f'kv_{task_idx}_{ex_idx}.pt')
                    torch.save(legacy_cache, kv_path)
                    forward_times = time_ctkv_forward(
                        model=model, query_input_ids=query_ids, kv_path=kv_path,
                        demo_seq_len=demo_len,
                        n_warmup=args.n_warmup, n_measure=args.n_measure,
                    )
                del legacy_cache

                all_icl_times.extend(icl_times)
                all_load_times.extend(load_times)
                all_forward_times.extend(forward_times)
                all_demo_lens.append(demo_len)
                all_query_lens.append(query_len)

                icl_ms = sum(icl_times) / len(icl_times) * 1000
                load_ms = sum(load_times) / len(load_times) * 1000
                fwd_ms = sum(forward_times) / len(forward_times) * 1000
                print(f'  [{task_idx+1}/{n_tasks}] {task} '
                      f'(demo={demo_len}, query={query_len}) | '
                      f'ICL={icl_ms:.1f}ms, load={load_ms:.1f}ms, fwd={fwd_ms:.1f}ms')

    # Summary
    def stats(times: List[float]) -> Tuple[float, float]:
        """Return (mean_ms, std_ms)."""
        t = torch.tensor(times) * 1000  # to ms
        return t.mean().item(), t.std().item()

    icl_mean, icl_std = stats(all_icl_times)
    load_mean, load_std = stats(all_load_times)
    fwd_mean, fwd_std = stats(all_forward_times)
    total_mean = load_mean + fwd_mean
    avg_demo_len = sum(all_demo_lens) / len(all_demo_lens)
    avg_query_len = sum(all_query_lens) / len(all_query_lens)

    print(f'\n{"=" * 60}')
    print(f'TTFT Benchmark Results: {args.benchmark.upper()}')
    print(f'{"=" * 60}')
    print(f'Model: Llama-3.2-1B-Instruct (bf16)')
    print(f'Tasks: {n_tasks}, Examples: {n_examples}')
    print(f'KV cache load from: {args.load_from}')
    print(f'Measurements per example: {args.n_measure}')
    print(f'Avg demo tokens: {avg_demo_len:.0f}, Avg query tokens: {avg_query_len:.0f}')
    print(f'{"─" * 60}')
    print(f'ICL TTFT:          {icl_mean:>8.2f} ± {icl_std:.2f} ms')
    print(f'CT-KV disk load:   {load_mean:>8.2f} ± {load_std:.2f} ms')
    print(f'CT-KV forward:     {fwd_mean:>8.2f} ± {fwd_std:.2f} ms')
    print(f'CT-KV total:       {total_mean:>8.2f} ms')
    print(f'{"─" * 60}')
    print(f'Speedup (ICL / CT-KV total): {icl_mean / total_mean:.2f}x')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
