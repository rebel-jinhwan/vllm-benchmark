# vllm-benchmark

## 데이터 준비

이 레포에서는 현재 [novita/agentic_code_dataset_22](https://huggingface.co/datasets/novita/agentic_code_dataset_22) 하나의 데이터셋을 지원합니다.
GuideLLM을 통한 벤치마킹을 위해 **`data/export_guidellm_jsonl.ipynb`** 노트북으로 후가공해 벤치마크용 JSONL을 만듭니다.

1. [novita/agentic_code_dataset_22](https://huggingface.co/datasets/novita/agentic_code_dataset_22)에서 JSON 파일 다운로드 → `data/` 디렉터리 등 로컬에 저장 (노트북에서 사용하는 원본 파일명 예: `e22_sessions_openai.json`)
2. **`data/export_guidellm_jsonl.ipynb`** 실행 → 턴 단위 행(`messages`, `output_seq_len`)이 포함된 `data/e22_sessions_guidellm.jsonl` 생성
3. 아래 GuideLLM 벤치마크에서 `--data`로 이 JSONL을 지정

## GuideLLM으로 벤치마크 (멀티턴 데이터)

`data/e22_sessions_guidellm.jsonl` 같은 멀티턴 메시지 형식으로 벤치마크하려면 **GuideLLM을 PR [#618](https://github.com/vllm-project/guidellm/pull/618)(multi-turn 지원) 기준으로 editable install** 해야 합니다.

```bash
# PR #618 브랜치 클론 후 editable install
git clone https://github.com/rebel-jinhwan/guidellm.git
cd guidellm
git fetch origin pull/618/head:pr-618
git checkout pr-618
# 또는 upstream fork 브랜치: git checkout feat/support-multi-turn
pip install -e .
# 또는 uv: uv pip install -e .
```

이후 `spec_decode/README.md`에 있는 대로 `guidellm benchmark --data ~/vllm-benchmark/data/e22_sessions_guidellm.jsonl ...` 로 실행하면 됩니다.
