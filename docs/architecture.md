# 코드 구조 안내

이 프로젝트는 수업 순서와 폴더 순서를 맞췄다.

| 순서 | 실행 파일 | 주요 코드 | 결과 |
| ---: | --- | --- | --- |
| 1 | `prepare_documents.py` | `preprocessing/pdf.py` | PNG, API용 JPEG, 라벨 점검용 페이지 텍스트 |
| 2 | `prepare_cases.py` | `data/dataset.py` | 질문 40건 JSONL |
| 3 | `inspect_inputs.py` | 출력 형식(schema)과 목록 파일(manifest) | 탐색 결과(EDA) JSON |
| 4 | `preflight_nvidia.py` | NVIDIA `/v1/models` | 모델 사용 가능 여부 |
| 5 | `run_nvidia_nim.py` | `providers/litellm_provider.py` | 실제 응답과 즉시 평가 |
| 6 | `freeze_recorded_responses.py` | 실제 관찰 결과(observation) | 회귀검사용 고정 응답(fixture) |
| 7 | `run_workflow.py` | `providers/recorded.py` | API 없는 저장 응답 재생(replay) |
| 8 | `evaluate_failures.py` | `evaluation/scoring.py` | 실패 주입 결과 |
| 9 | `evaluate_recorded_provider_routes.py` | `provider_evaluation.py` | 저장 응답 비교와 장애 상황(fault) 6건 |
| 10 | `compare_live_provider_routes.py` | `live_provider_comparison.py` | 두 provider 실제 비교 |

## Python 패키지

```text
src/verifiable_ai_workflow/
├── config/          YAML과 .env를 읽는다.
├── schemas/         문서, 질문, 응답과 평가 결과 형식
├── preprocessing/   PDF 페이지 이미지와 라벨 점검용 텍스트를 분리해 준비
├── data/            사람이 편집하는 YAML을 JSONL로 변환
├── providers/       실제 LiteLLM 또는 저장 응답
├── workflow/        PDF 페이지, 질문과 provider 연결
└── evaluation/      Pydantic, 정량 지표(metric)와 DeepEval
```

## 실제 호출 한 건의 흐름

```text
평가 사례(EvaluationCase)
→ 준비 문서 목록(PreparedDocument manifest)
→ 페이지 JPEG를 base64 이미지 입력으로 변환
→ 시스템 지시문(system prompt) + 질문 + 전체 페이지
→ LiteLLM NVIDIA NIM
→ 원본 응답(raw response) 즉시 저장
→ JSON 정리와 Pydantic
→ 정답 완전 일치(exact)·ANLS·토큰 F1·숫자·페이지·인용문 자체 일관성 점수
→ DeepEval 평가 실행(TestRun)
```

모델에는 페이지 JPEG와 질문만 보낸다. PDF 추출 문장은 모델 입력과 고정 규칙 채점에 사용하지
않는다. Week 2 실제 비교는 `live_execution.py`의 요청·토큰·비용·시간 상한과
`model_identity.py`의 실제 처리 모델(actual model) 확인을 추가한다.

먼저 `scripts/run_nvidia_nim.py`를 읽고, 다음 순서로 들어가면 된다.

1. `workflow/runner.py`
2. `workflow/inputs.py`
3. `providers/litellm_provider.py`
4. `schemas/models.py`
5. `evaluation/scoring.py`
6. `evaluation/deepeval_runner.py`
