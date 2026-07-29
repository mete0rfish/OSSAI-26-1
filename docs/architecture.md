# 코드 구조 안내

이 프로젝트는 수업 순서와 폴더 순서를 맞췄다.

| 순서 | 실행 파일 | 주요 코드 | 결과 |
| ---: | --- | --- | --- |
| 1 | `prepare_documents.py` | `preprocessing/pdf.py` | PNG, API용 JPEG, page text |
| 2 | `prepare_cases.py` | `data/dataset.py` | 질문 40건 JSONL |
| 3 | `inspect_inputs.py` | schema와 manifest | EDA JSON |
| 4 | `preflight_nvidia.py` | NVIDIA `/v1/models` | 모델 사용 가능 여부 |
| 5 | `run_nvidia_nim.py` | `providers/litellm_provider.py` | 실제 응답과 즉시 평가 |
| 6 | `freeze_recorded_responses.py` | 실제 observation | 회귀 fixture |
| 7 | `run_workflow.py` | `providers/recorded.py` | API 없는 replay |
| 8 | `evaluate_failures.py` | `evaluation/scoring.py` | 실패 주입 결과 |

## Python package

```text
src/verifiable_ai_workflow/
├── config/          YAML과 .env를 읽는다.
├── schemas/         문서, 질문, 응답과 평가 결과 계약
├── preprocessing/   PDF image와 text를 API 호출 전에 준비
├── data/            사람이 편집하는 YAML을 JSONL로 변환
├── providers/       실제 LiteLLM 또는 recorded response
├── workflow/        PDF 페이지, 질문과 provider 연결
└── evaluation/      Pydantic, 정량 metric과 DeepEval
```

## 실제 호출 한 건의 흐름

```text
EvaluationCase
→ PreparedDocument manifest
→ page JPEG를 base64 image input으로 변환
→ system prompt + 질문 + 전체 페이지
→ LiteLLM NVIDIA NIM
→ raw response 즉시 저장
→ JSON 정리와 Pydantic
→ 정답 exact·ANLS·token F1·숫자·페이지·quote 점수
→ DeepEval TestRun
```

먼저 `scripts/run_nvidia_nim.py`를 읽고, 다음 순서로 들어가면 된다.

1. `workflow/runner.py`
2. `workflow/inputs.py`
3. `providers/litellm_provider.py`
4. `schemas/models.py`
5. `evaluation/scoring.py`
6. `evaluation/deepeval_runner.py`
