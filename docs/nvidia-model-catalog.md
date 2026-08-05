# NVIDIA NIM 수업용 모델 카탈로그

확인일: 2026-08-04

아래 모델은 NVIDIA 공식 카탈로그에서 Free Endpoint가 활성화되어 있고, 수업 계정의
`/v1/models` 목록에도 있는 모델이다. `공식 언어 목록 미기재`는 모델 문서에서 지원
언어를 별도로 지정하지 않았다는 뜻이다.

## 멀티모달 모델

| 모델 | 개발사 | 규모 | 문맥 길이(Context) | 입력 | 출력 | 공식 언어 정보 | 주요 활용 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | NVIDIA | 33B/A3.1B | 262K | 텍스트·이미지·영상·음성 | 텍스트 | 영어만 | PDF Q&A·OCR·표·차트·영상·음성 |
| [`nvidia/nemotron-nano-12b-v2-vl`](https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl) | NVIDIA | 13B | 131K | 텍스트·이미지·영상 | 텍스트 | 영어만 | 짧은 다중 이미지 Q&A·문서 이해 |
| [`meta/llama-3.2-11b-vision-instruct`](https://build.nvidia.com/meta/llama-3.2-11b-vision-instruct) | Meta | 11B | 131K | 텍스트·이미지 | 텍스트 | 텍스트만 8개 언어, 이미지+텍스트 영어만 | 이미지 VQA·DocVQA |
| [`google/diffusiongemma-26b-a4b-it`](https://build.nvidia.com/google/diffusiongemma-26b-a4b-it) | Google | 25.2B/A3.8B | 262K | 텍스트·이미지·영상 | 텍스트 | 다국어 | 빠른 생성·PDF 분석·OCR·구조화 JSON |
| [`google/gemma-4-31b-it`](https://build.nvidia.com/google/gemma-4-31b-it) | Google | 33B | 262K | 텍스트·이미지·영상 | 텍스트 | 즉시 지원 35개 이상, 사전학습 140개 이상 | 문서·이미지·영상 이해·코딩·에이전트 |
| [`minimaxai/minimax-m3`](https://build.nvidia.com/minimaxai/minimax-m3) | MiniMax | 427B | 1M | 텍스트·이미지·영상 | 텍스트 | 공식 언어 목록 미기재 | 장문 멀티모달 추론·코딩·도구 호출 |
| [`stepfun-ai/step-3.7-flash`](https://build.nvidia.com/stepfun-ai/step-3.7-flash) | StepFun | 약 200B/A11B | 262K | 텍스트·이미지 | 텍스트 | 공식 언어 목록 미기재 | 차트·GUI·코딩·에이전트 |
| [`meta/llama-3.2-90b-vision-instruct`](https://build.nvidia.com/meta/llama-3.2-90b-vision-instruct) | Meta | 89B | 131K | 텍스트·이미지 | 텍스트 | 텍스트만 8개 언어, 이미지+텍스트 영어만 | 대형 VQA·DocVQA 비교 |
| [`moonshotai/kimi-k2.6`](https://build.nvidia.com/moonshotai/kimi-k2.6) | Moonshot AI | 1T | 262K | 텍스트·이미지·영상 | 텍스트 | 공식 언어 목록 미기재 | 장기 에이전트·코딩·이미지·영상 이해 |

Llama 3.2 Vision의 텍스트 전용 공식 언어는 영어, 독일어, 프랑스어, 이탈리아어,
포르투갈어, 힌디어, 스페인어, 태국어다. 이미지+텍스트의 공식 지원 언어는 영어다.
한국어 데이터 실습의 수업 기준 모델은 Gemma 4 31B IT다. Gemma 4 공식 카드는 35개가
넘는 언어의 즉시 지원과 140개가 넘는 언어의 사전학습을 구분해 명시한다.

### Kimi K2.6 Week 2 설정

`moonshotai/kimi-k2.6`은 NVIDIA 무료 접속 주소(Free Endpoint)와 2026-08-04 수업 계정의
`/v1/models`에서 확인했다. JPEG·PNG 다중 이미지와 텍스트 출력을 지원하지만 공식
한국어 지원 목록은 없다. 따라서 한국어 적합성은 모델 소개만으로 주장하지 않고 같은
AIHub 40건에서 직접 평가한다.

수업 설정은 `configs/nvidia-nim-kimi-k2.6.yaml`이다. 짧은 JSON만 필요한 과제라 Kimi의
즉시 응답 모드(Instant mode) 권장값인 `temperature: 0.6`, `top_p: 0.95`,
`thinking_mode: disabled`를 사용하고 NVIDIA 예제와 같이 난수 기준값(`seed: 0`)을 고정한다.
Kimi 전용 지시문(prompt)은 Gemma의 `/no_think` 지시만 제거하고 나머지 답변·근거·JSON
규칙을 유지한다. 이 호출 경로(route)는 지시문과 생성 설정(sampling)도 달라지므로 세 모델
표는 모델 하나만 바꾼 인과 비교가 아니라 실무 설정별 진단 결과로 해석한다.

같은 날 이 수업 계정의 실제 추론 소규모 사전 확인(probe)은 NVIDIA 내부 처리 시스템의
모델 함수 404로 끝났다. 키를 교체한 뒤 한 번 더 확인해도 같은 오류였다. 즉
`/v1/models`에 표시되는 것과 이 계정에서 호출 가능한 것은 같지 않았다. 성공 응답·실제
처리 모델(actual model)·사용량(usage)이 없으므로 40건 전체 실행은 하지 않았으며,
NVIDIA 계정의 Kimi 접속 주소 접근을 복구한 뒤 소규모 사전 확인부터 다시 해야 한다.

### DiffusionGemma 26B A4B IT 대체 호출 경로

Kimi 대신 `google/diffusiongemma-26b-a4b-it`을 세 번째 호출 경로로 선택했다. 공식 모델
카드가 다국어 추론과 문서·PDF 분석, 다국어 OCR, 차트 이해, 구조화 JSON을
명시해 이 실습 조건에 직접 맞는다. 한국어를 개별 지원 언어로 보장하는
자료는 아니므로 한국어 적합성은 실제 데이터 결과로만 판단한다.

수업 설정은 `configs/nvidia-nim-diffusiongemma.yaml`이다. NVIDIA hosted 예제의
`temperature: 1.0`, `top_p: 0.95`를 사용한다. 이번 과제는 추론 과정을 보여 주는 것이
아니라 JSON 하나만 필요하므로 `chat_template_kwargs.enable_thinking: false`를 LiteLLM `extra_body`로
전달한다. Kimi용으로 만들었던 지시문에는 모델 전용 토큰이 없어
`prompts/pdf-question-answer-json-only.md`로 이름을 바꾸고 두 호출 경로에서 함께 사용한다.

첫 소규모 사전 확인에서는 9페이지 문서가 모델의 지시문당 최대 이미지 8장 제한을 넘어 API가
응답 전에 거절했다. 페이지를 버리지 않고 8·9페이지를 위아래로 합친 한 이미지로 보내며,
원래 페이지 번호와 합성 이미지 바이트 수·SHA-256을 결과 목록 파일에 남긴다. 이 입력
묶음 처리(packing)도 호출 경로 차이이므로 모델 단독 효과로 해석하지 않는다.

묶음 처리를 적용한 소규모 사전 확인은 JSON·정답·근거와 실제 처리 모델을 모두 통과했다.
전체 40건에서는 37건이 응답했고 17건이 전체 성공(`task_success`)을 통과했으며, 마지막
보도자료 3건은 429로 품질 분모에서 제외했다. 합쳐진 8·9페이지를 정답 근거로 쓰는 3건은
모두 실패해 묶음 처리 영향의 진단 신호가 남았지만, 별도 요인 제거 비교(ablation) 없이
묶음 처리만을 원인으로 단정하지 않는다.

## 텍스트 모델

| 모델 | 개발사 | 규모 | 문맥 길이(Context) | 입력 | 출력 | 공식 언어 정보 | 주요 활용 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| [`openai/gpt-oss-20b`](https://build.nvidia.com/openai/gpt-oss-20b) | OpenAI | 21B/A3.6B | 131K | 텍스트 | 텍스트 | 공식 언어 목록 미기재 | 추론·수학·구조화 출력·도구 호출 |
| [`deepseek-ai/deepseek-v4-flash`](https://build.nvidia.com/deepseek-ai/deepseek-v4-flash) | DeepSeek AI | 284B/A13B | 1M | 텍스트 | 텍스트 | 공식 언어 목록 미기재 | 빠른 코딩·추론·에이전트 |
| [`deepseek-ai/deepseek-v4-pro`](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro) | DeepSeek AI | 1.6T/A49B | 1M | 텍스트 | 텍스트 | 공식 언어 목록 미기재 | 고난도 코딩·추론·에이전트 |
