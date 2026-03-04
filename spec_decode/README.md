# Speculative Decoding Benchmark

vLLM 서버를 띄운 뒤, GuideLLM을 이용해 메트릭을 측정합니다.

## 사전 준비

- **데이터**: [novita/agentic_code_dataset_22](https://huggingface.co/datasets/novita/agentic_code_dataset_22)에서 JSON 파일을 로컬에 저장한 뒤, `data/export_guidellm_jsonl.ipynb` 노트북으로 후가공해 `e22_sessions_guidellm.jsonl`을 생성해 사용합니다. (vrvrv/vllm-benchmark-datasets는 사용하지 않음)
- **의존성**: GuideLLM 벤치마크 시 [PR #618](https://github.com/vllm-project/guidellm/pull/618) 기준 editable install 필요 (루트 `README.md` 참고)
- vLLM 설치 및 사용할 모델 준비

## 사용 순서

### 1. 서버 기동 (포그라운드)

```bash
cd spec_decode
./launch_server.sh
```

**다른 설정으로 서버 띄우기 (configs/ 프리셋)**

```bash
# configs/ngram_k5.json 사용
CONFIG=ngram_k5 ./launch_server.sh

# configs/suffix_decoding.json, configs/ngram_k3.json 등
CONFIG=suffix_decoding ./launch_server.sh

# 다른 포트 / GPU
PORT=8001 CUDA_VISIBLE_DEVICES=0,1 ./launch_server.sh
```

**환경 변수 (기본값)**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CONFIG` | baseline | Speculative decoding 프리셋. `configs/<CONFIG>.json` 사용 (baseline이면 미사용) |
| `PORT` | 8000 | API 서버 포트 |
| `CUDA_VISIBLE_DEVICES` | 0,1,2,3 | 사용 GPU |
| `MODEL` | Qwen/Qwen3-30B-A3B | 모델명 |
| `TP_SIZE` | 1 | Tensor parallel size |
| `MAX_MODEL_LEN` | 32768 | 최대 시퀀스 길이 |
| `GPU_MEM_UTIL` | 0.80 | GPU 메모리 사용률 |
| `ENABLE_EP` | 1 | Expert parallel (MoE) |

### 2. 벤치마크 실행 (GuideLLM 권장)

서버가 준비된 **다른 터미널**에서 **GuideLLM**으로 벤치마크를 실행할 수 있습니다. 멀티턴 대화 데이터(`e22_sessions_guidellm.jsonl`)를 그대로 사용합니다.

```bash
guidellm benchmark \
  --target "http://localhost:8000" \
  --profile constant \
  --rate 4,5,6,7,8,9,10 \
  --max-seconds 30 \
  --data ~/vllm-benchmark/data/e22_sessions_guidellm.jsonl \
  --warmup 0.1 \
  --cooldown 0.1 \
  --output-path result.json
```

- `--profile constant` — 일정 부하 프로필
- `--rate 4,8,16` — 초당 요청 수를 4, 8, 16으로 스윕
- `--data` — GuideLLM 형식 JSONL (각 행에 `messages` 배열 + `output_seq_len` 등)
- `--warmup` / `--cooldown` — 구간 전후 안정화 시간(초)

**서버 프리셋 (configs/*.json)**

- `baseline` — speculative decoding 미사용
- `suffix_decoding` — configs/suffix_decoding.json
- `ngram_k5` / `ngram_k3` — n-gram PLD

## 데이터셋

벤치마크에는 [novita/agentic_code_dataset_22](https://huggingface.co/datasets/novita/agentic_code_dataset_22)(22개 Claude Code 세션, OpenAI 형식)를 사용합니다. 이 데이터셋의 JSON 파일을 로컬에 저장한 뒤 **`data/export_guidellm_jsonl.ipynb`** 노트북으로 후가공해 GuideLLM용 JSONL(`e22_sessions_guidellm.jsonl`)을 만듭니다.

1. Hugging Face에서 `novita/agentic_code_dataset_22` JSON 파일 다운로드 → `data/` 등 로컬에 저장
2. `data/export_guidellm_jsonl.ipynb` 실행 → 턴 단위로 분리된 행(`messages`, `output_seq_len`)이 포함된 `data/e22_sessions_guidellm.jsonl` 생성
3. 아래처럼 `--data`에 해당 JSONL 경로를 지정해 벤치마크 실행
