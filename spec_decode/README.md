# Speculative Decoding Benchmark

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
  --rate 1,2,3,4,5,6,7,8 \
  --max-seconds 60 \
  --data ~/vllm-benchmark/data/e22_sessions_guidellm.jsonl \
  --warmup 0.1 \
  --cooldown 0.1 \
  --output-path result.json
```

- `--profile constant` — 일정 부하 프로필
- `--rate 4,8,16` — 초당 요청 수를 4, 8, 16으로 스윕
- `--data` — GuideLLM 형식 JSONL (각 행에 `messages` 배열 + `output_seq_len` 등)
- `--warmup` / `--cooldown` — 구간 전후 안정화 시간(초)

### 3. 설정 스윕 자동 실행 (sweep.sh)

여러 설정(baseline, suffix_decoding, ngram_k4)에 대해 **서버 기동 → 벤치마크 → 서버 종료**를 자동으로 반복하고, 결과를 `results/<timestamp>/` 아래에 저장합니다. 데이터 파일 경로는 인자로 넘깁니다.

```bash
cd spec_decode

# 데이터 JSONL 경로를 첫 번째 인자로 지정 (필수)
./sweep.sh ../data/e22_sessions_guidellm.jsonl

# rate당 최대 시간만 줄이고 싶을 때 (기본 60초)
MAX_SECONDS=30 ./sweep.sh ../data/e22_sessions_guidellm.jsonl
```

- **인자**: `./sweep.sh <data.jsonl>` — GuideLLM 벤치마크용 JSONL 경로 (필수)
- **환경 변수**: `MAX_SECONDS` — rate당 최대 실행 시간(초), 기본값 60
- **결과**: `vllm-benchmark/results/YYYYMMDD-HHMMSS/result-baseline.json`, `result-suffix_decoding.json`, `result-ngram_k4.json`

**서버 프리셋 (configs/*.json)**

- `baseline` — speculative decoding 미사용
- `suffix_decoding` — configs/suffix_decoding.json
- `ngram_k5` / `ngram_k4` — n-gram PLD

## 데이터셋

벤치마크에는 [novita/agentic_code_dataset_22](https://huggingface.co/datasets/novita/agentic_code_dataset_22)(22개 Claude Code 세션, OpenAI 형식)를 사용합니다. 이 데이터셋의 JSON 파일을 로컬에 저장한 뒤 **`data/export_guidellm_jsonl.ipynb`** 노트북으로 후가공해 GuideLLM용 JSONL(`e22_sessions_guidellm.jsonl`)을 만듭니다.

### 데이터셋 저장 (bash 명령어)

**1. Hugging Face에서 원본 데이터셋 다운로드**

```bash
cd data
huggingface-cli download novita/agentic_code_dataset_22 --repo-type dataset --local-dir .
```

다운로드된 파일 중 OpenAI 형식 세션 JSON이 있다면 `e22_sessions_openai.json`으로 두거나, 노트북의 `INPUT_JSON`을 해당 경로에 맞춥니다.

**2. GuideLLM용 JSONL 생성**

Jupyter에서 `data/export_guidellm_jsonl.ipynb`를 열어 실행합니다. 노트북에 설정된 `OUTPUT_JSONL` 경로(예: `data/e22_sessions_guidellm.jsonl`)에 JSONL이 생성됩니다.

**3. 벤치마크에서 사용**

- 수동 실행: `guidellm benchmark --data data/e22_sessions_guidellm.jsonl ...`
- 스윕: `./sweep.sh data/e22_sessions_guidellm.jsonl`
