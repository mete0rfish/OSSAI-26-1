# Week 1 실행형 실습

## 목표

PDF를 페이지 이미지로 바꾸고 NVIDIA NIM의 Nemotron VLM에 질문과 이미지를 보내고
구조화된 답을 고정 규칙 채점기(deterministic scorer)로 평가한다.

Week 1에서는 Gemma를 사용하지 않는다. Week 2부터 모델을 Gemma로 바꾸고 지시문(prompt)을
비교한다.

## 1. 환경 준비

저장소 루트에서 실행한다.

```bash
uv python install 3.12
uv sync --locked --dev
cp .env.example .env
uv run --locked python scripts/check_environment.py
```

`.env`의 `NVIDIA_NIM_API_KEY`에 발급받은 키를 입력한다. API 키는 Python 코드, Notebook,
커밋이나 화면 캡처에 넣지 않는다.

`uv sync --locked --dev`의 뜻과 설치되는 라이브러리는
[수업 도구·채점기·용어](terms-tools-and-scoring.md#라이브러리와-수업-도구)에 정리되어 있다.

## 2. PDF와 질문 준비

AIHub 샘플을 `local-data/aihub/source/`에 둔 뒤 실행한다.

```bash
uv run --locked python scripts/prepare_documents.py
uv run --locked python scripts/prepare_cases.py
uv run --locked python scripts/inspect_inputs.py
```

여기까지의 준비·저장 응답 실습은 `configs/week-01.yaml`의 `recorded` provider를 사용한다.
실제 Nemotron 호출부터는 같은 질문·이미지·지시문·채점기를 유지한 채
`configs/nvidia-nim.yaml`의 NVIDIA provider와 별도 결과 폴더를 사용한다.

다음 세 파일을 연다.

1. `local-data/aihub/prepared/MI2_240819_TY1_0012/manifest.json`
2. `local-data/aihub/cases.jsonl`
3. `reports/week-01/eda.json`

그다음 실제 모델 입력이 어떻게 만들어지는지 차례로 확인한다.

1. `local-data/aihub/prepared/MI2_240819_TY1_0012/model-pages/page-0001.jpg`를
   열어 모델에 보낼 JPEG를 본다.
2. `src/verifiable_ai_workflow/workflow/inputs.py`의 `build_page_input()`을 열어 JPEG를
   `image_url` 입력으로 만드는 부분을 찾는다.
3. `src/verifiable_ai_workflow/workflow/runner.py`의 `build_messages()`와 `run_cases()`를
   열어 질문과 페이지 이미지가 메시지에 들어간 뒤 API 제공자에게 어떻게 전달되는지 찾는다.

아래 내용을 확인한다.

- 문서 2개가 페이지 JPEG로 준비됐다.
- 질문은 40건이며 답을 찾아야 하는 36건과 답변을 보류해야 하는 4건이 있다.
- `eda.json`의 `anomalies`가 빈 목록이다.
- 기대 근거 페이지가 실제 페이지 범위 안에 있다.
- 모델 입력 이미지는 설정한 크기 제한을 넘지 않는다.

모델 입력은 질문과 페이지 JPEG뿐이다. 전처리 폴더의 텍스트는 원본과 라벨을 사람이
점검하기 위한 보조 자료이며 모델 입력과 채점에 사용하지 않는다.

## 3. 대표 사례 한 건으로 채점 이해

```bash
uv run --locked python scripts/inspect_deterministic_scoring_case.py
```

출력을 다음 순서로 읽는다.

1. `input.prepared_input_status`: 로컬 페이지 이미지 준비 여부
2. `input.question`: 모델에 묻는 질문
3. `model_output.raw_response`: 저장된 모델 원응답
4. `model_output.parsed_answer`: 출력 형식 검사 뒤의 답
5. `expected`: 기대 답·답변 보류 여부·근거 페이지
6. `scoring.required_scores`: 통과 판정 지표
7. `scoring.diagnostic_scores`: 실패 원인 보조값
8. `scoring.failed_requirements`: 실패한 필수 지표와 이유

이 명령은 API를 호출하지 않고 저장 응답(fixture)을 다시 채점한다. 따라서 현재 모델의
품질 측정이 아니라 시험 전용 증거(`test_only`)다. PDF를 아직 준비하지 않았어도 저장 응답
채점은 실행되며 이때 `input.prepared_input_status`는 `not_prepared`다.

### 최종 통과 조건

한 사례의 전체 성공(`task_success`)은 다음 네 항목이 모두 1일 때만 1이다.

```text
출력 형식 유효성(schema_validity)
AND 답변 보류 정확성(abstention_correct)
AND 정답 허용 기준(answer_correct)
AND 근거 페이지 포함(evidence_coverage)
```

문자 유사도와 숫자 일치는 진단값이며 실패 원인을 찾는 데 쓴다. 최종 통과를 직접 결정하지 않는다.
각 값의 계산 목적은
[고정 규칙 채점기와 평가지표](terms-tools-and-scoring.md#고정-규칙-채점기와-평가지표)를
참고한다.

`answer_correct`는 기준 답 또는 데이터에 명시한 `accepted_answers`와 정규화 완전일치할
때만 1이다. 문자 유사도와 숫자 일치는 오류 분석용이며 통과 조건이 아니다. 예를 들어
`71.6%`를 `71.6명`으로 답하면 숫자가 같아도 실패다.

`quote_answer_support`는 모델이 쓴 `answer`와 `evidence[].quote`가 서로 맞는지만 확인한다.
인용문을 페이지 JPEG와 다시 대조하지 않으므로 실제 이미지에 있는 근거라는 보장은 아니다.

## 4. 모델 목록 사전 점검

다음 명령은 모델 추론 없이 NVIDIA의 현재 모델 목록만 조회한다.

```bash
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim.yaml
```

다음 두 줄을 확인한다.

```text
configured model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
available now: True
```

`False`면 실제 API를 호출하지 않는다. 그다음 `configs/nvidia-nim.yaml`의
`pricing_source_url`을 열어 `billing_basis`와 입력·출력 단가가 같은지 확인한다.
모델 목록과 가격을 오늘 확인했을 때만 두 날짜를 기록한다.

```bash
CATALOG_DATE=$(date +%F)
PRICING_DATE=$(date +%F)
```

## 5. 실제 API로 한 사례 실행

한 사례 probe는 탐색 실행이다. 여기서는 API 연결과 결과 형식을 확인한다. 작업 파일을 수정한
상태에서도 실행할 수 있지만 `summary.json`의 상태는 항상 `inconclusive`이며 전체 품질
근거로 사용하지 않는다. 오류와 실제 모델 불일치 없이 응답을 저장했다면
`observed_status=complete`이고 명령은 정상 종료 코드 0을 반환한다.

```bash
uv run --locked python scripts/run_nvidia_nim.py \
  --config configs/nvidia-nim.yaml \
  --live \
  --sample-id aihub-report-r01 \
  --max-requests 1 \
  --max-input-tokens 20000 \
  --max-output-tokens 500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 120 \
  --max-retries 0 \
  --catalog-verified-on "$CATALOG_DATE" \
  --pricing-verified-on "$PRICING_DATE"
```

명령 마지막의 `run directory`가 이번 결과 폴더다. 폴더 이름을 따로 만들거나 입력할 필요가
없다. 그 폴더에서 다음 파일을 순서대로 연다.

1. `observations.jsonl`: `raw_output`, `model_error`, `model_call.actual_model`
2. `results.jsonl`: `status`, `scores`, `reasons`
3. `summary.json`: 실행 건수와 상태 개수

모델 답이 틀리거나 JSON 형식이 깨진 것은 관찰할 품질 결과다. API 오류나 요청 모델과 실제
처리 모델의 불일치는 품질을 판정할 수 없어 판단 보류(`inconclusive`)로 기록한다.

## 6. 준비된 전체 40건 결과 분석

수업에서는 40건을 다시 호출하지 않는다. 튜터가 같은 모델과 지시문으로 원본을 미리 만들어
두며 수업에서는 그 원본을 연다.

```bash
FULL_RUN_DIR="local-data/week-01-full-runs/nemotron"
test -f "$FULL_RUN_DIR/summary.json"
test -f "$FULL_RUN_DIR/observations.jsonl"
test -f "$FULL_RUN_DIR/results.jsonl"
```

`summary.json`에서 확인한다.

- `record_count`와 `target_count`가 모두 40인가
- 통과(`passed`)·실패(`failed`)·판단 보류(`inconclusive`) 합이 40인가
- `observed_status=complete`로 40건 실행 기록이 완성됐는가
- `status`가 전부 성공이면 `pass`, 실패가 하나라도 있으면 `fail`인가
- `provider_error_count`와 `model_drift_count`는 몇 건인가
- `quality_eligible_count`는 품질 평균에 사용한 응답 수와 같은가. API 오류는 뺐는가
- `score_averages.task_success`는 그 응답만의 성공률로 얼마인가

그다음 `results.jsonl`에서 통과 1건과 실패 1건을 골라 같은 `sample_id`의
`observations.jsonl` 원응답을 확인한다. 평균보다 사례를 먼저 읽고 실패한 필수 지표와
`reasons`를 한 문장으로 설명한다.

현재 준비 결과는 14건 통과, 26건 실패다. `status=fail`은 40건 호출이 중단됐다는 뜻이
아니라 고정 규칙을 통과하지 못한 실제 답이 있다는 뜻이다. 새 전체 실행의 명령과
승인 절차는 튜터 runbook에 둔다.

DeepEval 결과도 함께 저장되지만 이 주차에서는 trace 화면을 열지 않는다. 학습자는 위
`summary.json`, `results.jsonl`, `observations.jsonl` 세 파일에서 점수와 이유를 직접 연결한다.

## 7. API 없이 회귀평가와 실패 주입

저장소의 고정 응답으로 같은 파서와 채점기를 다시 실행한다.

```bash
uv run --locked python scripts/evaluate_workflow.py
uv run --locked python scripts/evaluate_failures.py
```

확인할 결과:

- `reports/week-01/summary.json`의 `evidence_kind`는 `test_only`다.
- 같은 파일의 `judge_status`는 `not_requested`다.
- `reports/week-01-failures/results.json`의 깨진 JSON, 잘못된 신뢰도 범위, 오답,
  잘못된 페이지는 모두 `task_success=0.0`이다.

마지막으로 API를 호출하지 않는 코드 검사를 실행한다.

```bash
uv run --locked pytest
uv run --locked ruff check .
```

## 완료 기준

- PDF 문장이 아닌 페이지 JPEG와 질문이 모델 입력임을 설명할 수 있다.
- Nemotron 한 사례를 실행하고 준비된 40건 결과 폴더를 분석한다.
- 기대 답, 모델 원응답, 필수 지표와 실패 이유를 한 사례에서 연결해 설명한다.
- 실제 품질 증거(`live_quality`)와 저장 응답의 시험 전용 증거(`test_only`)를 구분한다.
- 모델 기반 채점기 없이 고정 규칙 채점기로 무엇을 판정했는지 설명한다.
