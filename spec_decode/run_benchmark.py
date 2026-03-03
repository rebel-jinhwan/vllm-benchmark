#!/usr/bin/env python3
"""Run speculative decoding benchmark for one dataset + config combination.

Data: Hugging Face vrvrv/vllm-benchmark-datasets (dataclaw, spider, humaneval, novita).
Requires the vLLM server to be running (start with launch_server.sh).
See README.md for full usage.

Usage:
    python3 run_benchmark.py --dataset dataclaw --config baseline
    python3 run_benchmark.py --dataset novita --config ngram_k5 --port 8000

Config: label for the run (must match CONFIG used in launch_server.sh). Presets in configs/*.yaml.

Results saved to: spec_decode/results/{dataset}_{config}.jsonl
"""

import json
import argparse
import sys
import time
from pathlib import Path

# Ensure unbuffered output for pipe compatibility (tee, etc.)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
HF_DATASET = "vrvrv/vllm-benchmark-datasets"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_MAX_TOKENS = 256

MODEL = "Qwen/Qwen3-30B-A3B"
SEED = 42


def get_config_choices():
    """Return list of config names: baseline + configs/*.json stems (for --config label)."""
    choices = ["baseline"]
    if CONFIGS_DIR.is_dir():
        for f in sorted(CONFIGS_DIR.glob("*.json")):
            choices.append(f.stem)
    return choices


def _sharegpt_to_messages(conversations):
    """Convert ShareGPT conversations [{"from": "human"|"gpt", "value": "..."}] to OpenAI messages."""
    role_map = {"human": "user", "gpt": "assistant"}
    return [
        {"role": role_map.get(t["from"], "user"), "content": t.get("value", "") or ""}
        for t in conversations
    ]


def load_dataset_prompts(dataset_name):
    """Load prompts from vrvrv/vllm-benchmark-datasets. Returns list of {id, messages, max_tokens}."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install Hugging Face datasets: pip install datasets"
        ) from e

    ds = load_dataset(HF_DATASET, dataset_name, split="train", trust_remote_code=True)
    prompts = []
    for row in ds:
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        messages = _sharegpt_to_messages(conversations)
        # osl is fixed 256 in this dataset
        max_tokens = int(row.get("osl", DEFAULT_MAX_TOKENS))
        prompts.append({
            "id": row["id"],
            "messages": messages,
            "max_tokens": max_tokens,
        })
    return prompts


def check_server(port=8000, timeout=5):
    """Check if server is reachable. Raises RuntimeError if not."""
    import urllib.request
    try:
        req = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=timeout)
        if req.status == 200:
            return
    except Exception as e:
        raise RuntimeError(
            f"Server not reachable at http://localhost:{port}. "
            "Start it with: ./launch_server.sh (see README.md)"
        ) from e


def send_request(messages, max_tokens, port=8000, timeout=600):
    """Send a non-streaming request and measure TPOT.

    TPOT = elapsed / completion_tokens (ms/token).
    Note: includes prefill time in numerator (~3% overhead for 256-token outputs).
    Streaming is broken for Gemma-3-27B on vLLM 0.15.x (empty content chunks).
    """
    import urllib.request

    url = f"http://localhost:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": SEED,
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.perf_counter() - t_start}

    elapsed = time.perf_counter() - t_start
    choice = data["choices"][0]
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    output_text = choice["message"]["content"]

    # TPOT = total_elapsed / completion_tokens (includes prefill)
    tpot_mean = (
        (elapsed / completion_tokens * 1000) if completion_tokens > 0 else float("inf")
    )

    return {
        "output_text": output_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_s": elapsed,
        "tpot_mean_ms": tpot_mean,
        "n_chunks": 0,
    }


def run_benchmark(dataset_file, config_name, port=8000, max_prompts=None):
    """Run full benchmark for one dataset + config. Server must already be running (launch_server.sh)."""
    print(f"[Data] Loading {dataset_file} from {HF_DATASET}...")
    try:
        prompts = load_dataset_prompts(dataset_file)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)

    # Optionally limit number of prompts
    if max_prompts and len(prompts) > max_prompts:
        print(f"[Limit] Capping from {len(prompts)} to {max_prompts} prompts")
        prompts = prompts[:max_prompts]

    print(f"\n{'=' * 60}")
    print(f"Benchmark: {dataset_file} / {config_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"{'=' * 60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{dataset_file}_{config_name}.jsonl"

    # Check for existing partial results
    completed_ids = set()
    if result_path.exists():
        with open(result_path) as f:
            for line in f:
                r = json.loads(line)
                completed_ids.add(r["id"])
        print(f"[Resume] Found {len(completed_ids)} existing results")

    remaining = [p for p in prompts if p["id"] not in completed_ids]
    if not remaining:
        print("[Done] All prompts already completed")
        return

    check_server(port=port)

    for i, prompt in enumerate(remaining):
        result = send_request(
            messages=prompt["messages"],
            max_tokens=prompt.get("max_tokens", 256),
            port=port,
        )
        result["id"] = prompt["id"]
        result["dataset"] = dataset_file
        result["config"] = config_name

        with open(result_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        status = "OK" if "error" not in result else f"ERR: {result['error'][:50]}"
        tokens = result.get("completion_tokens", 0)
        tpot = result.get("tpot_mean_ms", 0)
        print(
            f"  [{i + 1}/{len(remaining)}] {prompt['id'][:40]:40s} | {tokens:4d} tok | {tpot:6.1f} ms/tok | {status}"
        )

    # Summary
    results = []
    with open(result_path) as f:
        results = [json.loads(line) for line in f]

    ok_results = [r for r in results if "error" not in r]
    if ok_results:
        tpots = [
            r["tpot_mean_ms"]
            for r in ok_results
            if r.get("tpot_mean_ms", float("inf")) < float("inf")
        ]
        avg_tpot = sum(tpots) / len(tpots) if tpots else 0
        p50_tpot = sorted(tpots)[len(tpots) // 2] if tpots else 0
        total_tokens = sum(r["completion_tokens"] for r in ok_results)
        print(f"\n[Summary] {dataset_file}/{config_name}")
        print(f"  Completed: {len(ok_results)}/{len(results)}")
        print(f"  Mean TPOT: {avg_tpot:.1f} ms (elapsed/tokens, includes prefill)")
        print(f"  P50 TPOT: {p50_tpot:.1f} ms")
        print(f"  Total Output Tokens: {total_tokens}")


def main():
    parser = argparse.ArgumentParser(
        description="Run speculative decoding benchmark (server must be running: launch_server.sh)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dataclaw", "spider", "humaneval", "novita"],
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=get_config_choices(),
        help="Run label (match CONFIG used in launch_server.sh). Presets: configs/*.json",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port (must match launch_server.sh)")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts")
    args = parser.parse_args()

    run_benchmark(
        dataset_file=f"{args.dataset}-*.parquet",
        config_name=args.config,
        port=args.port,
        max_prompts=args.max_prompts,
    )


if __name__ == "__main__":
    main()
