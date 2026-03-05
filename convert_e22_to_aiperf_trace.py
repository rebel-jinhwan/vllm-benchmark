#!/usr/bin/env python3
"""
Convert novita/agentic_code_dataset_22 (e22_sessions_openai.json) → AIPerf mooncake_trace JSONL.

데이터 구조:
  {
    "sessions": [
      {
        "turns": [
          {
            "system": "...",              # optional system prompt
            "messages": [msg, msg, ...],  # 해당 턴의 전체 messages (누적 context 포함)
            "output_seq_len": N,          # 또는 "max_tokens"
          },
          ...
        ]
      },
      ...
    ]
  }

각 turn이 이미 하나의 LLM call이므로, split_trajectory_into_calls 없이
  - messages: system + turn.messages
  - output_length: output_seq_len 또는 max_tokens
  - session_id: 세션 인덱스
  - timestamp/delay: 첫 턴은 timestamp, 이후 턴은 delay

AIPerf main branch 기준 (messages 모드)

Usage:
  pip install huggingface_hub
  python convert_e22_to_aiperf_trace.py [--skip-first-turns 0] [--max-turns 40]
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATASET_REPO = "novita/agentic_code_dataset_22"
DATA_FILE = "e22_sessions_openai.json"
LOCAL_DIR = Path("data")

MEAN_INTER_ARRIVAL_MS = 5000  # session 간 평균 간격
MEAN_TURN_DELAY_MS = 2000  # 턴 간 평균 delay


def estimate_tokens(text: str) -> int:
    """Rough token estimation: chars / 4"""
    return max(1, len(text) // 4)


def ensure_data_file(local_dir: Path) -> Path:
    """데이터 파일이 없으면 HuggingFace에서 다운로드."""
    local_path = local_dir / DATA_FILE
    if local_path.exists():
        return local_path

    print(f"Downloading {DATA_FILE} from {DATASET_REPO} ...")
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATA_FILE,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    print(f"  Saved to {local_path}")
    return local_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert e22_sessions_openai.json to AIPerf trace"
    )
    parser.add_argument(
        "--skip-first-turns",
        type=int,
        default=0,
        help="세션당 앞 N턴 건너뛰기 (default: 0)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="건너뛴 뒤 세션당 최대 턴 수 (default: all)",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=None,
        help="사용할 세션 수 (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL filename (default: auto-generated)",
    )
    args = parser.parse_args()

    # ── Load ──
    data_path = ensure_data_file(LOCAL_DIR)
    print(f"Loading {data_path} ...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = data.get("sessions", [])
    print(f"  세션 수: {len(sessions)}")
    print(f"  총 턴 수: {sum(len(s.get('turns', [])) for s in sessions)}")

    if args.num_sessions is not None:
        sessions = sessions[: args.num_sessions]
        print(f"  제한: {len(sessions)} sessions")

    # ── Output filename ──
    if args.output:
        output_file = args.output
    else:
        parts = ["e22_trace"]
        if args.num_sessions is not None:
            parts.append(f"s{args.num_sessions}")
        if args.skip_first_turns > 0:
            parts.append(f"skip{args.skip_first_turns}")
        if args.max_turns is not None:
            parts.append(f"max{args.max_turns}")
        output_file = f"data/{'_'.join(parts)}.jsonl"

    # ── Convert ──
    random.seed(42)
    current_ts = 0
    total_entries = 0
    total_sessions = 0
    skipped_sessions = 0
    skipped_turns = 0
    input_token_counts = []
    output_token_counts = []
    turns_per_session = []

    with open(output_file, "w", encoding="utf-8") as f:
        for session_idx, session in enumerate(sessions):
            turns = session.get("turns", [])

            if args.skip_first_turns > 0:
                turns = turns[args.skip_first_turns :]
            if args.max_turns is not None:
                turns = turns[: args.max_turns]

            if not turns:
                skipped_sessions += 1
                continue

            session_id = f"e22-session-{session_idx}"
            session_turn_count = 0

            for turn_idx, turn in enumerate(turns):
                messages = []

                # system prompt
                if turn.get("system"):
                    messages.append({"role": "system", "content": turn["system"]})

                # turn messages
                turn_msgs = turn.get("messages", [])
                if not turn_msgs:
                    skipped_turns += 1
                    continue

                messages.extend(turn_msgs)

                # output token count
                output_length = (
                    turn.get("output_seq_len") or turn.get("max_tokens") or 256
                )

                # AIPerf trace entry
                if turn_idx == 0:
                    entry = {"timestamp": current_ts}
                else:
                    delay = int(random.expovariate(1.0 / MEAN_TURN_DELAY_MS))
                    entry = {"delay": delay}

                entry["messages"] = messages
                entry["output_length"] = max(1, output_length)
                entry["session_id"] = session_id

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_entries += 1
                session_turn_count += 1

                # Stats collection
                input_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
                input_token_counts.append(input_tokens)
                output_token_counts.append(output_length)

            if session_turn_count > 0:
                turns_per_session.append(session_turn_count)
                total_sessions += 1
                # 다음 session까지의 간격
                current_ts += int(random.expovariate(1.0 / MEAN_INTER_ARRIVAL_MS))
            else:
                skipped_sessions += 1

    # ── Stats ──
    print(f"\n{'=' * 60}")
    print(f"Output:       {output_file}")
    print(f"Sessions:     {total_sessions} processed, {skipped_sessions} skipped")
    print(f"Turns skipped: {skipped_turns}")
    print(f"Trace entries: {total_entries} (individual LLM calls)")
    print(
        f"Total trace duration: {current_ts / 1000:.0f}s ({current_ts / 60000:.1f}min)"
    )

    if not turns_per_session:
        print("\nNo data to report.")
        return

    print(f"\nTurns per session:")
    print(f"  min:    {min(turns_per_session):>8}")
    print(f"  median: {sorted(turns_per_session)[len(turns_per_session) // 2]:>8}")
    print(f"  mean:   {sum(turns_per_session) // len(turns_per_session):>8}")
    print(f"  max:    {max(turns_per_session):>8}")

    print(f"\nInput tokens per call (estimated):")
    print(f"  min:    {min(input_token_counts):>8,}")
    print(f"  median: {sorted(input_token_counts)[len(input_token_counts) // 2]:>8,}")
    print(f"  mean:   {sum(input_token_counts) // len(input_token_counts):>8,}")
    print(f"  max:    {max(input_token_counts):>8,}")

    print(f"\nOutput tokens per call (output_length in trace):")
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
