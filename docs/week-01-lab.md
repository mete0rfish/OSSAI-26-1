# Week 1 실행형 실습

## 목표

AIHub PDF와 준비된 전체 질문을 NVIDIA NIM에 실제로 전달하고, 모델이 반환한 답과 근거를
Pydantic과 DeepEval로 평가한다. 저장 응답은 실제 실행 이후에 회귀 fixture로 만든다.
이 저장소의 reference 데이터는 현재 40건이지만, 실습 완료 여부는 고정 개수가 아니라
EDA의 `case_count`와 실행 결과의 `target_count`를 기준으로 판단한다.

## 환경과 API key

`uv`가 없다면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)로
먼저 설치한다.

```bash
uv python install 3.12
uv sync --locked --dev
cp .env.example .env
```

`.env`의 `NVIDIA_NIM_API_KEY`에 발급받은 key를 넣는다. key를 notebook, Python,
commit 또는 화면 캡처에 넣지 않는다.

```bash
uv run python scripts/check_environment.py
uv run python scripts/preflight_nvidia.py
```

완료 기준:

- Python 3.12와 필수 package 확인
- `configured model: google/gemma-4-31b-it`
- `available now: True`

## PDF 전처리

AIHub sample을 `local-data/aihub/source/`에 넣고 실행한다.

```bash
uv run python scripts/prepare_documents.py
uv run python scripts/prepare_cases.py
```

전처리 결과:

- 보고서 9페이지와 보도자료 3페이지 전체 PNG
- API용 175KB 이하 JPEG 12개
- 페이지별 PDF 추출 텍스트
- 문서별 `manifest.json`
- 준비된 전체 질문 `local-data/aihub/cases.jsonl`

## EDA와 계약 확인

```bash
uv run python scripts/inspect_inputs.py
```

`reports/week-01/eda.json`에서 다음을 확인한다. 아래 개수는 현재 reference 데이터의
예시이며, 다른 데이터로 실습할 때는 실제 보고서 값을 사용한다.

- 문서 2개, 질문 40건
- development 32건, validation 8건
- 정답형 36건, 답변 보류형 4건
- 기대 근거 페이지가 PDF 범위 안에 있음
- 모델 입력 이미지가 모두 175KB 이하
- 추출 텍스트가 빈 페이지 없음
- prompt와 `StructuredAnswer` field가 일치함
- 정답 30건의 text layer와 라벨 페이지 자동 일치
- text layer로 확인할 수 없는 표·복합 날짜 6건은 이미지 수동 검토
- 답변 보류 4건은 문서 전체 수동 검토
- `anomalies=[]`

다음 파일을 함께 읽는다.

- `data/cases/week-01-aihub.yaml`
- `prompts/pdf-question-answer.md`
- `src/verifiable_ai_workflow/schemas/models.py`

준비된 모든 기대 답과 근거 페이지는 실제 호출 전에 두 사람이 검토한다.

## 대표 1건을 입력부터 평가까지 읽기

```bash
uv run --locked python scripts/show_week01_case.py
```

출력 한 건을 `input` → `model_output.raw_response` → `parsed_answer` → `expected` →
`evaluation_design` → `evaluation_result` 순서로 읽는다. 이 명령은 저장된 실제 응답을
다시 채점하므로 `test_only`이며, 현재 모델의 실제 품질 증거는 아니다. 전체 40건과
DeepEval 평가는 아래 `evaluate_workflow.py`에서 실행한다. `input`에는 실제로 준비한
page JPEG의 경로·byte 크기와 기대 page의 짧은 추출 문장만 나오며, 이미지 base64와
문서 전체 문장은 출력하지 않는다.

## 실제 첫 1건

```bash
uv run python scripts/run_nvidia_nim.py --live --limit 1
```

첫 raw response를 열어 다음을 찾는다.

- `raw_output`
- `model_error`
- `model_call.actual_model`
- `model_call.latency_ms`
- input/output token
- retry 횟수

`results.jsonl`에서 Pydantic 검증, 정답, 근거 페이지와 근거 문장 점수를 확인한다.
모델의 형식 또는 답이 틀린 것은 실습 실패가 아니라 관찰할 품질 결과다. Provider 오류만
`inconclusive`로 분리한다.

## 실제 나머지 질문

```bash
uv run python scripts/run_nvidia_nim.py --live --resume
```

처음부터 다시 실행하지 않는다. 완료된 `sample_id`는 건너뛰고 나머지만 순차 실행한다.
각 network attempt는 최소 3초 간격을 두며 429에는 5초, 10초, 20초 backoff를 적용한다.

완료 기준:

- `observations.jsonl`의 행 수가 `summary.json`의 `target_count`와 같음
- 모든 sample에 raw response 또는 명확한 provider 오류
- actual model, latency, token과 retry 기록
- `passed / failed / inconclusive` 구분

## DeepEval 분석

```bash
uv run deepeval inspect reports/week-01-nvidia/deepeval
```

| 지표 | 의미 |
| --- | --- |
| `json_object_only` | Markdown fence 없이 JSON만 반환했는가 |
| `schema_validity` | 정리된 응답이 Pydantic을 통과하는가 |
| `answer_exact` | 정답과 완전히 같은가 |
| `answer_similarity` | 정규화 후 문자 유사도 |
| `answer_anls` | DocVQA 방식의 편집거리 유사도 |
| `answer_token_f1` | 정답 token precision과 recall의 조화평균 |
| `numeric_match` | 핵심 숫자 집합이 일치하는가 |
| `abstention_correct` | 답변/보류 선택이 맞는가 |
| `evidence_page_precision` | 제시 페이지 중 맞는 페이지 비율 |
| `evidence_page_recall` | 가능한 페이지 중 인용한 비율 |
| `evidence_page_f1` | precision과 recall의 조화평균 |
| `evidence_coverage` | 가능한 근거 페이지를 하나 이상 인용했는가 |
| `quote_answer_support` | 인용문에 답의 핵심 문자열·숫자가 있는가 |
| `quote_verifiability` | PDF 추출 텍스트로 검증 가능한 인용 비율 |
| `quote_grounding` | 검증 가능한 인용문이 실제 페이지와 일치하는 정도 |
| `task_success` | 필수 계약·정답·근거 기준 통과 |

`answer_exact`, ANLS, token F1은 함께 보고 표현 차이와 실제 오답을 구분한다. 표·차트
숫자가 PDF text layer에 없으면 `quote_verifiability=0`으로 남기고 그 이유만으로 모델을
실패시키지 않는다. 추출 가능한 인용만 `task_success`의 근거 문장 조건에 사용한다.

## 실제 응답 회귀 fixture

```bash
uv run python scripts/freeze_recorded_responses.py
uv run python scripts/run_workflow.py
uv run python scripts/evaluate_workflow.py
```

실제 NIM raw response 전체를 고정해 API 없이 동일 parser와 scorer를 다시 실행한다.
`live_quality`와 `test_only` 결과를 혼동하지 않는다.

## 실패 주입

```bash
uv run python scripts/evaluate_failures.py
uv run pytest
uv run ruff check .
```

깨진 JSON, confidence 범위 위반, 오답과 잘못된 페이지가 각각 어떤 metric을 실패시키는지
확인한다.

## 실습에서 생성되는 결과

- `reports/week-01/eda.json`
- `reports/week-01-nvidia/observations.jsonl`
- `reports/week-01-nvidia/results.jsonl`
- `reports/week-01-nvidia/summary.json`
- DeepEval TestRun
- 실제 NIM 응답 기반 recorded fixture
- 실패 사례 결과

위 목록은 실습을 완료하면 만들어지는 전체 결과다.

## Reference 실행 결과

2026-07-29 실제 실행에서는 40건 모두 응답을 받았고 429, 재시도와 provider 오류는
없었다. 필수 기준은 16건 통과, 24건 실패였다. 평균 `answer_correct`는 0.4000,
`evidence_page_f1`은 0.6667, `quote_verifiability`는 0.7250이었다. 이는 정답 40건을
그대로 복사한 fixture가 아니라 실제 NIM 원응답을 채점한 결과다.

## Week 1 실습해보기

### 실습 목표

수업에서 사용한 기준 모델과 설정을 바꾸지 않고, 자기 환경에서 PDF 전처리부터 실제 응답
평가까지 한 번 완료한다. 점수를 높이는 것이 아니라 각 단계의 입력과 출력이 어디에
저장되는지 확인하고, 통과와 실패 사례를 구분하는 것이 목표다.

### 1. 데이터와 전처리 결과 확인

AIHub 샘플을 `local-data/aihub/source/`에 넣고 다음 명령을 실행한다.

```bash
uv run python scripts/prepare_documents.py
uv run python scripts/prepare_cases.py
uv run python scripts/inspect_inputs.py
```

다음 세 가지를 확인한다.

- `local-data/aihub/prepared/` 아래에 문서별 `manifest.json`이 있다.
- `reports/week-01/eda.json`의 `case_count`가 준비한 질문 수와 같다.
- `reports/week-01/eda.json`의 `anomalies`가 빈 목록이다.

`anomalies`가 비어 있지 않으면 임의로 데이터를 고치지 말고 해당 항목과 오류 메시지를
확인한다.

### 2. 기준 모델 전체 실행

먼저 1건을 실행해 API key와 모델 응답을 확인한다.

```bash
uv run python scripts/run_nvidia_nim.py --live --limit 1
```

문제가 없으면 나머지 질문을 이어서 실행한다.

```bash
uv run python scripts/run_nvidia_nim.py --live --resume
```

`reports/week-01-nvidia/summary.json`에서 다음 값을 확인한다.

- `record_count`
- `target_count`
- `status_counts`의 `passed`, `failed`, `inconclusive`
- `requested_model`
- `evidence_kind`
- `judge_status`

모든 질문이 `passed`일 필요는 없다. `record_count`와 `target_count`가 같고,
`passed`, `failed`, `inconclusive`의 합계가 `target_count`와 같으면 실행은 완료된
것이다. API 오류가 남았다면 오류가 발생한 `sample_id`와 메시지를 확인한다.
`status_counts`에 표시되지 않은 상태는 0건이다.

### 3. 결과 2건 확인

`reports/week-01-nvidia/results.jsonl`에서 `passed` 1건과 `failed` 또는 `inconclusive`
1건을 선택한다. 실패나 미확정 결과가 하나도 없다면 `passed` 2건을 선택한다.

질문과 기대 답은 `local-data/aihub/cases.jsonl`에서 확인하고, 실제 답과 `reasons`는
`reports/week-01-nvidia/results.jsonl`에서 확인한다.

각 사례의 `sample_id`, `status`, 기대 답, 실제 답과 `task_success`를 비교한다. 통과
사례는 답과 근거 페이지가 왜 맞는지 확인한다. 실패 사례는 `reasons`에서 점수가 0인
항목 하나를 찾아 무엇이 달랐는지 확인한다.

### 4. 저장 응답과 실패 사례 실행

다음 명령으로 API를 다시 호출하지 않는 저장 응답 평가와 실패 사례를 실행한다.

```bash
uv run python scripts/run_workflow.py
uv run python scripts/evaluate_workflow.py
uv run python scripts/evaluate_failures.py
```

다음을 확인한다.

- `reports/week-01/summary.json`의 `evidence_kind`는 `test_only`다.
- `reports/week-01/summary.json`의 `judge_status`는 `not_requested`다.
- `reports/week-01-failures/results.json`의 네 사례는 모두 `task_success=0.0`이다.

### 결과 확인 파일

- `reports/week-01/eda.json`
- `reports/week-01-nvidia/summary.json`
- `reports/week-01-failures/results.json`

### 완료 기준

- 데이터와 전체 질문을 오류 없이 준비했거나 발견한 데이터 오류를 확인했다.
- `record_count`와 `target_count`가 같고 모든 질문에 실행 상태가 있다.
- 결과 2건의 기대 답, 실제 답과 점수를 비교했다.
- 실제 응답의 `live_quality`와 저장 응답의 `test_only`를 구분했다.
- 의도적으로 만든 네 실패 사례가 모두 실패하는 것을 확인했다.
