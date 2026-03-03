# Speculative Decoding Benchmark

vLLM 서버를 포그라운드로 띄운 뒤, 다른 터미널에서 벤치마크 스크립트로 TPOT(Time Per Output Token) 등을 측정합니다.

## 사전 준비

- **데이터**: [vrvrv/vllm-benchmark-datasets](https://huggingface.co/datasets/vrvrv/vllm-benchmark-datasets) 사용 (dataclaw, spider, humaneval, novita). 별도 다운로드 없이 스크립트 실행 시 자동 로드
- **의존성**: `pip install datasets` (벤치마크)
- vLLM 설치 및 사용할 모델 준비

## 사용 순서

### 1. 서버 기동 (포그라운드)

```bash
cd spec_decode
./launch_server.sh
```

- 서버는 **포그라운드**로 실행됩니다. 로그는 이 터미널에 출력되며, **Ctrl+C**로 종료할 수 있습니다.
- 기동이 끝날 때까지 "Application startup complete" 메시지를 확인한 뒤, **다른 터미널**에서 벤치마크를 실행하세요.

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
| `TP_SIZE` | 2 | Tensor parallel size |
| `DP_SIZE` | 2 | Data parallel size |
| `MAX_MODEL_LEN` | 32768 | 최대 시퀀스 길이 |
| `GPU_MEM_UTIL` | 0.80 | GPU 메모리 사용률 |
| `ENABLE_EP` | 1 | Expert parallel (MoE) |

### 2. 서버 종료

- **포그라운드**로 띄운 경우: 서버가 보이는 터미널에서 **Ctrl+C**
- 포트를 쓰는 프로세스만 정리하려면: `./launch_server.sh stop` (다른 포트면 `PORT=8001 ./launch_server.sh stop`)

### 3. 벤치마크 실행

서버가 준비된 **다른 터미널**에서:

```bash
cd spec_decode
python3 run_benchmark.py --dataset dataclaw --config baseline
python3 run_benchmark.py --dataset novita --config ngram_k5 --port 8000
```

- `--config`는 결과 파일 이름용 라벨이며, 서버를 띄울 때 쓴 `CONFIG`와 맞추면 좋습니다 (baseline, suffix_decoding, ngram_k5, ngram_k3 등).
- `--port`는 `launch_server.sh`에서 쓴 `PORT`와 동일해야 합니다.

**프리셋 (configs/*.json)**

- `baseline` — CONFIG 없음. speculative decoding 미사용
- `suffix_decoding` — suffix decoding (configs/suffix_decoding.json)
- `ngram_k5` — n-gram PLD (configs/ngram_k5.json)
- `ngram_k3` — n-gram PLD (configs/ngram_k3.json)

**기타 옵션**

- `--max-prompts N` — 처리할 프롬프트 수 제한

결과는 `spec_decode/results/{dataset}_{config}.jsonl`에 추가됩니다. 이미 완료된 id는 건너뛰고 이어서 실행됩니다.

## 데이터셋

벤치마크는 [vrvrv/vllm-benchmark-datasets](https://huggingface.co/datasets/vrvrv/vllm-benchmark-datasets)를 사용합니다.

| Split       | 설명 |
|------------|------|
| **dataclaw**  | Multi-turn chat (Claude Code 세션) |
| **spider**    | Text-to-SQL |
| **humaneval** | Python 함수 완성 (코드 생성) |
| **novita**    | Agentic coding (22 Claude Code 세션) |

## 요약

1. 한 터미널에서 `./launch_server.sh` 로 서버 기동 (필요 시 `CONFIG=ngram_k5` 등 설정)
2. "Application startup complete" 확인 후, 다른 터미널에서 `python3 run_benchmark.py --dataset ... --config ...` 실행
3. 서버 종료는 서버 터미널에서 **Ctrl+C**
