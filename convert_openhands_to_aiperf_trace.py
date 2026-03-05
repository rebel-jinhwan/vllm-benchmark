#!/usr/bin/env python3
"""
Convert nebius/SWE-rebench-openhands-trajectories → AIPerf mooncake_trace JSONL.

핵심: 하나의 trajectory를 턴별로 쪼개서, 각 LLM call을
      "누적 messages + 해당 턴의 실제 output token 수"로 변환합니다.

OpenHands trajectory 구조:
  system → assistant(tool_calls) → tool → assistant(tool_calls) → tool → ...

각 assistant 턴이 하나의 LLM call:
  - messages: 해당 assistant 응답 직전까지의 모든 메시지 (누적)
  - output_length: 해당 assistant 응답의 실제 토큰 수 추정치
  - tools: OpenAI function-calling 도구 정의 (bash, str_replace_editor)
  - session_id: 동일 trajectory는 같은 session

AIPerf main branch 기준 (messages + tools 모드)

Usage:
  pip install git+https://github.com/ai-dynamo/aiperf.git datasets
  python convert_openhands_to_aiperf_trace.py [--max-rows N] [--resolved-only]
"""

import argparse
import json
from collections import Counter

from datasets import load_dataset


DATASET_ID = "nebius/SWE-rebench-openhands-trajectories"

# Per the dataset card
ROLE_FIELDS = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}


def estimate_tokens(text: str) -> int:
    """Rough token estimation: chars / 4"""
    return max(1, len(text) // 4)


def clean_message(msg: dict) -> dict:
    """Clean a message: keep only valid fields, deserialize tool_calls arguments."""
    role = msg["role"]
    cleaned = {k: msg[k] for k in ROLE_FIELDS[role] if k in msg}

    # Deserialize tool_calls[].function.arguments from JSON string
    if role == "assistant" and cleaned.get("tool_calls"):
        for i, tc in enumerate(cleaned["tool_calls"]):
            func = tc.get("function", {})
            if isinstance(func.get("arguments"), str):
                try:
                    cleaned["tool_calls"][i]["function"]["arguments"] = json.loads(
                        func["arguments"]
                    )
                except json.JSONDecodeError:
                    pass

    # Remove None tool_calls (assistant msgs without tool use)
    if role == "assistant" and cleaned.get("tool_calls") is None:
        del cleaned["tool_calls"]

    return cleaned


def split_trajectory_into_calls(trajectory: list[dict]) -> list[dict]:
    """
    Trajectory를 개별 LLM call로 분할.

    Returns list of:
      {
        "context": [msg, msg, ...],   # assistant 응답 직전까지의 누적 messages
        "response_tokens": int,       # 해당 assistant 응답의 추정 토큰 수
      }
    """
    calls = []
    context_so_far = []

    for msg in trajectory:
        cleaned = clean_message(msg)

        if msg["role"] == "assistant":
            # context가 비어있으면 skip
            if not context_so_far:
                context_so_far.append(cleaned)
                continue

            # assistant의 응답 크기 추정 (content + tool_calls)
            response_content = msg.get("content") or ""
            tool_calls_str = json.dumps(msg.get("tool_calls") or [], ensure_ascii=False)
            total_response = response_content + tool_calls_str
            response_tokens = estimate_tokens(total_response)

            calls.append(
                {
                    "context": list(context_so_far),
                    "response_tokens": response_tokens,
                }
            )

            # assistant 응답도 이후 턴의 context에 포함
            context_so_far.append(cleaned)
        else:
            context_so_far.append(cleaned)

    return calls


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenHands trajectories to AIPerf trace (per-turn)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of trajectories (default: all 67K)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL filename",
    )
    args = parser.parse_args()
    output_file = args.output or "openhands_trace.jsonl"

    # ── Load ──
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  Loaded {len(ds)} trajectories")


    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))
        print(f"  Limited to {len(ds)} trajectories")

    # ── Convert ──
    total_entries = 0
    total_trajectories = 0
    skipped = 0
    input_token_counts = []
    output_token_counts = []
    calls_per_traj = []

    with open(output_file, "w") as f:
        for row in ds:
            trajectory = row["trajectory"]
            if not trajectory:
                skipped += 1
                continue

            calls = split_trajectory_into_calls(trajectory)
            if not calls:
                skipped += 1
                continue

            session_id = row.get("trajectory_id") or f"traj-{total_trajectories}"
            tools = row.get("tools")
            calls_per_traj.append(len(calls))

            for call in calls:
                entry = {}

                entry["messages"] = call["context"]
                entry["output_length"] = max(1, call["response_tokens"])
                entry["session_id"] = session_id

                if tools:
                    entry["tools"] = tools

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_entries += 1

                input_tokens = estimate_tokens(
                    json.dumps(call["context"], ensure_ascii=False)
                )
                input_token_counts.append(input_tokens)
                output_token_counts.append(call["response_tokens"])

            total_trajectories += 1

    # ── Stats ──
    print(f"\n{'=' * 60}")
    print(f"Output:       {output_file}")
    print(f"Trajectories: {total_trajectories} processed, {skipped} skipped")
    print(f"Trace entries: {total_entries} (individual LLM calls)")

    print(f"\nCalls per trajectory:")
    print(f"  min:    {min(calls_per_traj):>8}")
    print(f"  median: {sorted(calls_per_traj)[len(calls_per_traj) // 2]:>8}")
    print(f"  mean:   {sum(calls_per_traj) // len(calls_per_traj):>8}")
    print(f"  max:    {max(calls_per_traj):>8}")

    print(f"\nInput tokens per call (estimated):")
    print(f"  min:    {min(input_token_counts):>8,}")
    print(f"  median: {sorted(input_token_counts)[len(input_token_counts) // 2]:>8,}")
    print(f"  mean:   {sum(input_token_counts) // len(input_token_counts):>8,}")
    print(f"  max:    {max(input_token_counts):>8,}")

    print(f"\nOutput tokens per call (estimated):")
    print(f"  min:    {min(output_token_counts):>8,}")
    print(f"  median: {sorted(output_token_counts)[len(output_token_counts) // 2]:>8,}")
    print(f"  mean:   {sum(output_token_counts) // len(output_token_counts):>8,}")
    print(f"  max:    {max(output_token_counts):>8,}")

    # Input token distribution
    buckets = Counter()
    for t in input_token_counts:
        if t < 2000:
            buckets["<2K"] += 1
        elif t < 8000:
            buckets["2K-8K"] += 1
        elif t < 32000:
            buckets["8K-32K"] += 1
        elif t < 65000:
            buckets["32K-64K"] += 1
        else:
            buckets["64K+"] += 1

    print(f"\nInput token distribution:")
    for bucket in ["<2K", "2K-8K", "8K-32K", "32K-64K", "64K+"]:
        count = buckets.get(bucket, 0)
        pct = count / total_entries * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>8}: {count:>6} ({pct:5.1f}%) {bar}")

    # Output token distribution
    buckets_out = Counter()
    for t in output_token_counts:
        if t < 100:
            buckets_out["<100"] += 1
        elif t < 500:
            buckets_out["100-500"] += 1
        elif t < 1000:
            buckets_out["500-1K"] += 1
        elif t < 4000:
            buckets_out["1K-4K"] += 1
        else:
            buckets_out["4K+"] += 1

    print(f"\nOutput token distribution:")
    for bucket in ["<100", "100-500", "500-1K", "1K-4K", "4K+"]:
        count = buckets_out.get(bucket, 0)
        pct = count / total_entries * 100
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>8}: {count:>6} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
