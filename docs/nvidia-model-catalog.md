# NVIDIA NIM 수업용 모델 카탈로그

확인일: 2026-07-29

아래 모델은 NVIDIA 공식 카탈로그에서 Free Endpoint가 활성화되어 있고, 수업 계정의
`/v1/models` 목록에도 있는 모델이다. `공식 언어 목록 미기재`는 모델 문서에서 지원
언어를 별도로 지정하지 않았다는 뜻이다.

## 멀티모달 모델

| 모델 | 개발사 | 규모 | Context | 입력 | 출력 | 공식 언어 정보 | 주요 활용 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | NVIDIA | 33B/A3.1B | 262K | text·image·video·audio | text | 영어만 | PDF Q&A·OCR·표·차트·영상·음성 |
| [`nvidia/nemotron-nano-12b-v2-vl`](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl) | NVIDIA | 13B | 131K | text·image·video | text | 영어만 | 짧은 다중 이미지 Q&A·문서 이해 |
| [`meta/llama-3.2-11b-vision-instruct`](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct) | Meta | 11B | 131K | text·image | text | text-only 8개 언어, image+text 영어만 | 이미지 VQA·DocVQA |
| [`google/diffusiongemma-26b-a4b-it`](https://build.nvidia.com/google/diffusiongemma-26b-a4b-it) | Google | 25.2B/A3.8B | 262K | text·image·video | text | 35개 이상 언어 | 빠른 생성·PDF parsing·OCR·structured JSON |
| [`google/gemma-4-31b-it`](https://build.nvidia.com/google/gemma-4-31b-it) | Google | 33B | 262K | text·image·video | text | 35개 이상 언어 | 문서·이미지·영상 이해·coding·agent |
| [`minimaxai/minimax-m3`](https://build.nvidia.com/minimaxai/minimax-m3) | MiniMax | 427B | 1M | text·image·video | text | 공식 언어 목록 미기재 | 장문 멀티모달 추론·coding·tool calling |
| [`stepfun-ai/step-3.7-flash`](https://build.nvidia.com/stepfun-ai/step-3.7-flash) | StepFun | 약 200B/A11B | 262K | text·image | text | 공식 언어 목록 미기재 | 차트·GUI·coding·agent |
| [`meta/llama-3.2-90b-vision-instruct`](https://build.nvidia.com/meta/llama-3.2-90b-vision-instruct) | Meta | 89B | 131K | text·image | text | text-only 8개 언어, image+text 영어만 | 대형 VQA·DocVQA 비교 |
| [`moonshotai/kimi-k2.6`](https://build.nvidia.com/moonshotai/kimi-k2.6) | Moonshot AI | 1T | 262K | text·image·video | text | 공식 언어 목록 미기재 | 장기 agent·coding·이미지·영상 이해 |

Llama 3.2 Vision의 text-only 공식 언어는 영어, 독일어, 프랑스어, 이탈리아어,
포르투갈어, 힌디어, 스페인어, 태국어다. image+text의 공식 지원 언어는 영어다.
Gemma 4 계열은 35개 이상 언어를 바로 지원하며 140개 이상 언어로 사전 학습됐다.

## Text 모델

| 모델 | 개발사 | 규모 | Context | 입력 | 출력 | 공식 언어 정보 | 주요 활용 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`openai/gpt-oss-20b`](https://build.nvidia.com/openai/gpt-oss-20b) | OpenAI | 21B/A3.6B | 131K | text | text | 공식 언어 목록 미기재 | 추론·수학·structured output·tool calling |
| [`deepseek-ai/deepseek-v4-flash`](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash) | DeepSeek AI | 284B/A13B | 1M | text | text | 공식 언어 목록 미기재 | 빠른 coding·reasoning·agent |
| [`deepseek-ai/deepseek-v4-pro`](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro) | DeepSeek AI | 1.6T/A49B | 1M | text | text | 공식 언어 목록 미기재 | 고난도 coding·reasoning·agent |
