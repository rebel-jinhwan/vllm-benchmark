# vllm-benchmark


## GuideLLM 설치

GuideLLM으로 멀티턴 데이터를 벤치마크하려면 [PR #618](https://github.com/vllm-project/guidellm/pull/618) 기준 설치가 필요합니다.

```bash
git clone https://github.com/rebel-jinhwan/guidellm.git
cd guidellm
git fetch origin pull/618/head:pr-618
git checkout pr-618
pip install -e .
```

## Benchmarking

vLLM speculative decoding 벤치마크. 서버 기동, 벤치마크 실행, 데이터셋 준비 등은 **[spec_decode/README.md](spec_decode/README.md)** 를 참고하세요.
