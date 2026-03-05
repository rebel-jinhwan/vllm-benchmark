#!/usr/bin/env python3
"""
Convert SWE-bench/SWE-smith-trajectories → AIPerf mooncake_trace JSONL.

핵심: 하나의 trajectory를 턴별로 쪼개서, 각 LLM call을
      "누적 messages + 해당 턴의 실제 output token 수"로 변환합니다.

Trajectory 구조:
  system → user → assistant → user/tool → assistant → ...
                  ^^^^^^^^^^              ^^^^^^^^^^
                  LLM call 1              LLM call 2

각 LLM call에 대해:
  - messages: 해당 assistant 응답 직전까지의 모든 메시지 (누적)
  - output_length: 해당 assistant 응답의 실제 토큰 수 추정치
  - session_id: 동일 trajectory는 같은 session
  - timestamp/delay: 첫 턴은 timestamp, 이후 턴은 delay

AIPerf main branch 기준 (messages 모드)

Usage:
  pip install git+https://github.com/ai-dynamo/aiperf.git datasets
  python convert_swe_smith_to_aiperf_trace.py [--split tool|xml|ticks] [--max-rows N]
"""

import json
import random
from collections import Counter
import tiktoken
from datasets import load_dataset

encoding = tiktoken.get_encoding("gpt-4o")

DATASET_ID = "SWE-bench/SWE-smith-trajectories"
MAX_ROWS = 20
MEAN_INTER_ARRIVAL_MS = 5000  # session 간 평균 간격
MEAN_TURN_DELAY_MS = 2000  # 턴 간 평균 delay (에이전트 thinking time)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: chars / 4"""
    return len(encoding.encode(text))


def split_trajectory_into_calls(messages: list[dict]) -> list[dict]:
    """
    Trajectory를 개별 LLM call로 분할.

    Returns list of:
      {
        "context": [msg, msg, ...],   # assistant 응답 직전까지의 누적 messages
        "response": "...",            # assistant의 실제 응답 content
        "output_tokens": int,         # 응답의 추정 토큰 수
      }
    """
    calls = []
    context_so_far = []

    for msg in messages:
        if msg["role"] == "assistant":
            response_content = msg.get("content") or ""

            # context가 비어있으면 skip (첫 메시지가 assistant인 경우)
            if not context_so_far:
                context_so_far.append(msg)
                continue

            calls.append(
                {
                    "context": list(context_so_far),
                    "response": response_content,
                    "output_tokens": estimate_tokens(response_content),
                }
            )

            # assistant 응답도 이후 턴의 context에 포함
            context_so_far.append(msg)
        else:
            context_so_far.append(msg)

    return calls


random.seed(42)
output_file = f"swe_smith_trace_{MAX_ROWS}.jsonl"

# ── Load ──
print(f"Loading {DATASET_ID} ...")
ds = load_dataset(DATASET_ID, split='tool')
print(f"  Loaded {len(ds)} trajectories")

ds = ds.select(range(min(MAX_ROWS, len(ds))))
print(f"  Limited to {len(ds)} trajectories")

# ── Convert ──
current_ts = 0
total_entries = 0
total_trajectories = 0
skipped = 0
input_token_counts = []
output_token_counts = []
calls_per_traj = []

with open(output_file, "w") as f:
    for row in ds:
        # Parse messages
        raw = row["messages"]
        if isinstance(raw, str):
            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                skipped += 1
                continue
        else:
            messages = raw

        if not messages:
            skipped += 1
            continue

        # Split into individual LLM calls
        calls = split_trajectory_into_calls(messages)
        if not calls:
            skipped += 1
            continue

        session_id = (
            row.get("traj_id")
            or row.get("instance_id")
            or f"traj-{total_trajectories}"
        )
        calls_per_traj.append(len(calls))

        for i, call in enumerate(calls):
            if i == 0:
                # 첫 턴: absolute timestamp
                entry = {"timestamp": current_ts}
            else:
                # 이후 턴: delay (이전 턴의 응답 후 대기 시간)
                delay = int(random.expovariate(1.0 / MEAN_TURN_DELAY_MS))
                entry = {"delay": delay}

            entry["messages"] = call["context"]
            entry["output_length"] = max(1, call["output_tokens"])
            entry["session_id"] = session_id

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total_entries += 1

            input_tokens = estimate_tokens(
                json.dumps(call["context"], ensure_ascii=False)
            )
            input_token_counts.append(input_tokens)
            output_token_counts.append(call["output_tokens"])

        total_trajectories += 1
        # 다음 session까지의 간격
        current_ts += int(random.expovariate(1.0 / MEAN_INTER_ARRIVAL_MS))

# ── Stats ──
print(f"\n{'=' * 60}")
print(f"Output:       {output_file}")
print(f"Trajectories: {total_trajectories} processed, {skipped} skipped")
print(f"Trace entries: {total_entries} (individual LLM calls)")
print(
    f"Total trace duration: {current_ts / 1000:.0f}s ({current_ts / 60000:.1f}min)"
)

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
