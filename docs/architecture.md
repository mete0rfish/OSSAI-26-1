# 코드 구조

학습자는 `scripts/`의 명령을 수업 순서대로 실행한다. 각 명령은
`src/verifiable_ai_workflow/`의 공통 구현을 호출한다. 별도 평가 엔진은 없다.

## Week 1–2 실행 순서

| 순서 | 학습자 실행 파일 | 하는 일 | 주요 결과 |
| ---: | --- | --- | --- |
| 1 | `prepare_documents.py` | PDF를 페이지 이미지로 준비 | 문서별 `manifest.json`, JPEG |
| 2 | `prepare_cases.py` | 사람이 편집하는 YAML을 실행용 JSONL로 변환 | `cases.jsonl` |
| 3 | `inspect_inputs.py` | 문서·질문·이미지 제한 검사 | `eda.json` |
| 4 | `inspect_deterministic_scoring_case.py` | 저장 응답 한 건의 채점 과정 확인 | 터미널 JSON |
| 5 | `preflight_nvidia.py` | 현재 설정 모델이 NVIDIA 목록에 있는지 확인 | 사용 가능 여부 |
| 6 | `run_nvidia_nim.py` | NIM 실제 호출과 즉시 채점 | 원응답·점수·요약 |
| 7 | `run_workflow.py` | 저장 응답 재실행(replay) | 시험 전용 원응답 |
| 8 | `evaluate_workflow.py` | 저장 응답 채점과 DeepEval 저장 | 사례별 점수·요약 |
| 9 | `evaluate_failures.py` | 의도적으로 깨진 네 응답 채점 | 실패 주입 결과 |
| 10 | `compare_gemma_prompts.py` | 같은 Gemma의 지시문 A/B 비교 | 사례별 변화 |
| 11 | `compare_live_provider_routes.py` | Gemma와 Gemini 실제 비교 | 호출 경로별 결과·비교 |
| 12 | `evaluate_recorded_provider_routes.py` | 저장된 API 장애 상황 확인 | 장애 상황 결과 |

`inspect_*.py`는 학습자 화면에 필요한 질문·답·점수만 보여 준다. 실제 실행의 식별값,
비용과 오류 기록은 `run_*.py`와 결과 파일에 남기되 첫 개념 설명에 섞지 않는다.

## 내부 코드

```text
src/verifiable_ai_workflow/
├── config/          YAML 설정과 .env 읽기
├── schemas/         질문·모델 답·평가 결과의 출력 형식(schema)
├── preprocessing/   PDF 페이지 이미지 준비
├── data/            수업 데이터 읽기와 JSONL 생성
├── providers/       LiteLLM 실제 호출과 저장 응답 제공자(provider)
├── workflow/        질문·페이지 이미지·모델 호출 연결
└── evaluation/      고정 규칙 점수 계산과 DeepEval 저장
```

실제 모델에 전달되는 값은 질문, 지시문(prompt)과 페이지 JPEG다. PDF 추출 문장은 원본·라벨
점검에만 쓰며 모델 입력과 채점에 넣지 않는다.

## 한 사례의 실제 흐름

```text
평가 사례
→ 문서의 페이지 JPEG 읽기
→ 지시문 + 질문 + 이미지 구성
→ LiteLLM을 통해 작업 모델 호출
→ 원응답 저장
→ Pydantic 출력 형식 검사
→ 고정 규칙 채점
→ DeepEval 결과 저장
```

채점기의 필수 지표와 진단 지표는
[수업 도구·채점기·용어](terms-tools-and-scoring.md#고정-규칙-채점기와-평가지표)에 있다.

## Week 6까지 유지하는 공통 기능

다음 코드는 Week 1의 핵심 개념은 아니지만 실제 호출 결과를 Week 6까지 비교하려면 필요해
유지한다.

| 파일 | 기능 명세 | 사용하는 시점 |
| --- | --- | --- |
| `live_execution.py` | 요청·토큰·비용·시간 상한을 호출 전에 검사하고, 중단 시 누적값을 보존한다. 같은 실행을 동시에 쓰지 않게 잠그고 JSON을 원자적으로 저장한다. | Week 1–2 실제 호출, Week 6 재현성 확인 |
| `model_identity.py` | 요청 모델과 API가 반환한 실제 처리 모델(actual model)이 같은지 확인한다. | Week 1–2 품질 판정, Week 6 자동 검사 |
| `comparison.py` | 두 실행의 데이터·지시문·출력 형식·채점 조건이 같은지 확인하고 사례별 변화를 계산한다. | Week 2 비교, Week 4·6 회귀 비교 |
| `prompt_comparison.py` | 같은 모델에서 지시문만 다른 두 전체 실행을 비교한다. | Week 2 |
| `live_provider_comparison.py` | 서로 다른 두 API 제공자를 같은 입력과 상한으로 실행한다. | Week 2, Week 6 정기 비교 |

이 기능들은 학습자가 직접 다시 구현하지 않는다. 해당 주차에서는 결과 파일에서 기능이
지켜졌는지만 확인한다.
