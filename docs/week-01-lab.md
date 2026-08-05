# Week 1 실행형 실습

## 목표

AIHub PDF와 준비된 전체 질문을 NVIDIA NIM에 실제로 전달하고, 모델이 반환한 답과 근거를
Pydantic과 DeepEval로 평가한다. 저장 응답은 실제 실행 이후에 회귀검사용 고정
응답(fixture)으로 만든다. 이 저장소의 기준 데이터는 현재 40건이지만, 실습 완료 여부는
고정 개수가 아니라 데이터 사전 점검(EDA)의 `case_count`와 실행 결과의 `target_count`를
기준으로 판단한다.

## 환경과 API 키

`uv`가 없다면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)로
먼저 설치한다.

```bash
uv python install 3.12
uv sync --locked --dev
cp .env.example .env
```

`.env`의 `NVIDIA_NIM_API_KEY`에 발급받은 키를 넣는다. 키를 노트북, Python 코드,
커밋(commit) 또는 화면 캡처에 넣지 않는다.

```bash
uv run python scripts/check_environment.py
uv run python scripts/preflight_nvidia.py
```

완료 기준:

- Python 3.12와 필수 패키지 확인
- `configured model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- `available now: True`

## PDF 전처리

AIHub 샘플을 `local-data/aihub/source/`에 넣고 실행한다.

```bash
uv run python scripts/prepare_documents.py
uv run python scripts/prepare_cases.py
```

전처리 결과:

- 보고서 9페이지와 보도자료 3페이지 전체 PNG
- API용 175KB 이하 JPEG 12개
- 라벨 작성·사람 점검용 PDF 추출 텍스트
- 문서별 목록 파일(`manifest.json`)
- 준비된 전체 질문 `local-data/aihub/cases.jsonl`

## 데이터 사전 점검(EDA)과 입출력 기준 확인

```bash
uv run python scripts/inspect_inputs.py
```

`reports/week-01/eda.json`에서 다음을 확인한다. 아래 개수는 현재 기준 데이터의
예시이며, 다른 데이터로 실습할 때는 실제 보고서 값을 사용한다.

- 문서 2개, 질문 40건
- 개발용(development) 32건, 검증용(validation) 8건
- 정답형 36건, 답변 보류형 4건
- 기대 근거 페이지가 PDF 범위 안에 있음
- 모델 입력 이미지가 모두 175KB 이하
- 추출 텍스트가 빈 페이지 없음
- 지시문(prompt)과 `StructuredAnswer` 필드(field)가 일치함
- 정답 30건의 PDF 텍스트 계층(text layer)과 라벨 페이지 자동 일치
- 텍스트 계층으로 확인할 수 없는 표·복합 날짜 6건은 이미지 수동 검토
- 답변 보류 4건은 문서 전체 수동 검토
- `anomalies=[]`

다음 파일을 함께 읽는다.

- `data/cases/week-01-aihub.yaml`
- `prompts/pdf-question-answer.md`
- `src/verifiable_ai_workflow/schemas/models.py`

준비된 모든 기대 답과 근거 페이지는 실제 호출 전에 두 사람이 검토한다.
추출 텍스트는 이 작성 단계에서만 보조 자료로 쓰며 모델 입력과 고정 규칙
채점기(deterministic scorer)에는 넣지 않는다. 실제 VLM 입력은 질문과 페이지 JPEG뿐이다.

## 대표 1건을 입력부터 평가까지 읽기

```bash
uv run --locked python scripts/inspect_deterministic_scoring_case.py
```

출력 한 건을 `input` → `model_output.raw_response` → `parsed_answer` → `expected` →
`evaluation_design` → `evaluation_result` 순서로 읽는다. 이 명령은 저장된 실제 응답을
다시 채점하므로 시험 전용 증거(`test_only`)이며, 현재 모델의 실제 품질 증거는 아니다.
전체 40건과 DeepEval 평가는 아래 `evaluate_workflow.py`에서 실행한다. `input`에는 실제
VLM 실행에서 사용할 페이지 JPEG의 경로·바이트 크기·SHA-256만 나오며 PDF 추출 문장은
나오지 않는다.
저장 응답 명령 자체는 이미지를 API에 다시 보내지 않는다.

## 실제 첫 1건

```bash
uv run python scripts/run_nvidia_nim.py \
  --live \
  --sample-id aihub-report-r01 \
  --max-requests 1 \
  --max-input-tokens 20000 \
  --max-output-tokens 500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 120 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

`--catalog-verified-on`은 `preflight_nvidia.py`가 성공한 실제 날짜로 바꾼다. 결과는
`reports/week-01-nvidia/runs/<run-id>/`에 생성된다.

첫 원본 응답(raw response)을 열어 다음을 찾는다.

- `raw_output`
- `model_error`
- `model_call.actual_model`
- `model_call.latency_ms`
- 입력/출력 토큰(token)
- 재시도(retry) 횟수

`results.jsonl`에서 Pydantic 검증, 정답, 근거 페이지와 근거 문장 점수를 확인한다.
모델의 형식 또는 답이 틀린 것은 실습 실패가 아니라 관찰할 품질 결과다. Provider 오류만
판단 보류(`inconclusive`)로 분리한다.

## 실제 전체 질문

```bash
uv run python scripts/run_nvidia_nim.py \
  --live \
  --max-requests 40 \
  --max-input-tokens 800000 \
  --max-output-tokens 20000 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

소규모 사전 확인(probe)과 전체 실행은 서로 다른 실행(run)이다. 전체 실행이 중단되면
출력에 표시된 `run_id`, 최초 실행과 같은 상한(cap), `--resume`을 사용한다. 저장된
`sample_id`는 다시 호출하지 않는다. 각 네트워크 시도(network attempt)는 프로그램 재개
전후를 포함해 최소 3초 간격을 둔다. 실행 전에
Git에 기록된 40건과 생성된 `cases.jsonl`의 모든 필드가 같은지 확인하고,
봉인 평가 데이터(`sealed_test`)가 있으면 네트워크 호출 전에 중단한다.

완료 기준:

- `observations.jsonl`의 행 수가 `summary.json`의 `target_count`와 같음
- 모든 샘플에 원본 응답 또는 명확한 provider 오류
- 실제 처리 모델(actual model), 응답 시간(latency), 토큰과 재시도 기록
- 통과(`passed`) / 실패(`failed`) / 판단 보류(`inconclusive`) 구분

## 채점기 두 종류: 고정 규칙과 모델 기반

답을 만드는 과정과 답을 채점하는 과정을 먼저 분리한다.

```text
질문·페이지 JPEG → 작업 모델(task model) → 저장된 원본 응답(raw response)
저장된 원본 응답 + 기대 답 → 채점기(scorer) → 점수와 이유
```

같은 질문을 작업 모델에 다시 보내 답이 달라지는 것은 **답 생성의 변동성**이다. 저장된
같은 원본 응답을 다시 평가할 때 결과가 달라지는지는 **채점기의 결정성**에 관한
문제다.

| 구분 | 고정 규칙 채점기(deterministic scorer) | 결과가 달라질 수 있는 모델 기반 채점기(model-based scorer, stochastic scorer) |
| --- | --- | --- |
| 계산 주체 | Python 규칙과 수식 | 별도 LLM 평가 모델(Judge) |
| 같은 저장 답 재평가 | 입력·코드·채점 기준(profile)이 같으면 같은 결과 | 점수나 이유가 달라질 수 있음 |
| API·비용 | 채점 API가 필요 없음 | 보통 평가 모델 API·토큰·비용이 필요 |
| 알맞은 항목 | JSON 구조, 숫자, 답변 보류, 근거 페이지 | 설명의 완전성, 근거 충실성, 유용성 |
| 주의점 | 잘못 만든 규칙도 일관되게 같은 오판을 냄 | 사람 평가와 맞추기 전에는 진단용 |

`temperature=0`은 작업 모델이 답을 생성할 때 변화를 줄이는 설정이다. 이 값을 0으로
고정해도 모델 출력의 완전한 재현을 보장하지 않으며, 채점기가 같은 결과를 내는지 여부도
정하지 않는다.

이번 AIHub VQA에는 짧은 기대 답, 답변 보류 여부와 기대 근거 페이지가 있다. 따라서
`aihub-vqa-deterministic-v2` 고정 규칙 채점 기준(profile)을 사용한다. 현재 DeepEval은
아래에서 계산한 점수와 이유를 평가 실행 기록(TestRun)으로 저장할 뿐 LLM 평가 모델을
호출하지 않는다.

다음 명령을 두 번 실행해 같은 `evaluation_result.scores`가 나오는지 확인한다. API는
호출하지 않는다.

```bash
uv run --locked python scripts/inspect_deterministic_scoring_case.py
uv run --locked python scripts/inspect_deterministic_scoring_case.py
```

결과가 반복된다는 사실은 규칙이 올바르다는 뜻이 아니다. 규칙이 업무 의미를 제대로
반영하는지는 실패 사례를 사람이 확인해야 한다.

## DeepEval 분석

```bash
uv run deepeval inspect reports/week-01-nvidia/runs/<run-id>/deepeval
```

### 수식에서 쓰는 기호

- `I(조건)`: 조건이 참이면 1, 아니면 0
- `A`, `E`: 각각 실제 답과 기대 답을 NFKC·소문자화하고 공백·문장부호를 제거한 문자열
- `T_A`, `T_E`: 실제 답과 기대 답에서 뽑은 숫자·영문·한글 토큰(token)의 중복을 보존한 목록
- `P_A`, `P_E`: 실제 근거 페이지 집합과 기대 근거 페이지 집합
- `|S|`: 문자열 길이 또는 집합·목록의 원소 수

`answer_similarity`는 `A`와 `E`가 모두 비었을 때 1이다. ANLS는 둘 중 하나라도 비어
있으면 두 문자열이 같은 경우에만 1이다. 토큰 F1도 두 토큰 목록이 모두 비었을 때 1,
한쪽만 비었을 때 0으로 계산한다.

### 출력 형식과 답 점수

| 지표 | 역할 | 현재 계산식·규칙 |
| --- | --- | --- |
| 순수 JSON(`json_object_only`) | 진단 | 원본 응답이 객체이거나, 문자열의 처음과 끝이 `{`, `}`이면 1 |
| 출력 형식 통과(`schema_validity`) | 전체 성공에 직접 사용 | JSON으로 읽은 뒤 `StructuredAnswer`의 필드·자료형(type)·범위를 통과하면 1 |
| 완전 일치(`answer_exact`) | 진단, 답변 보류 정답 계산에 사용 | `I(출력 형식 통과 AND 답변 보류 선택 일치 AND A = E)` |
| 문자열 유사도(`answer_similarity`) | 일반 정답 계산에 사용 | Python `SequenceMatcher`의 `2M / (|A| + |E|)`. `M`은 순서를 보존한 일치 문자 블록 길이의 합 |
| OCR 유사도(`answer_anls`) | OCR 표현 차이 진단 | `s = 1 - Levenshtein거리(A,E) / max(|A|,|E|)`. `s ≥ 0.5`면 `s`, 아니면 0 |
| 토큰 F1(`answer_token_f1`) | 여러 단어 답 진단 | 중복을 고려한 공통 토큰 수를 `O`라 할 때 `precision=O/|T_A|`, `recall=O/|T_E|`, `F1=2PR/(P+R)` |
| 숫자 일치(`numeric_match`) | 숫자 정답 계산에 사용 | 기대 답에 숫자가 없으면 1. 숫자가 있으면 실제·기대 숫자 목록의 값과 순서가 모두 같을 때 1 |
| 답변 보류 판정(`abstention_correct`) | 전체 성공에 직접 사용 | `I(실제 abstained = 기대 abstained)` |
| 정답 판정(`answer_correct`) | 전체 성공에 직접 사용 | 아래 질문 유형별 허용 기준을 통과하면 1 |

정답 판정은 하나의 유사도 점수만 보지 않고 다음 순서로 계산한다.

```text
기대 동작이 답변 보류        → answer_exact
숫자가 있고 숫자를 뺀 E 길이 ≤ 3
                              → 출력 형식 통과 AND numeric_match
숫자가 있고 숫자를 뺀 E 길이 > 3
                              → 출력 형식 통과 AND numeric_match AND answer_similarity ≥ 0.65
숫자가 없는 기대 답         → 출력 형식 통과 AND answer_similarity ≥ 0.75
```

기대 답에 숫자가 없을 때 `numeric_match=1`인 것은 “확인할 숫자가 없음”이라는 뜻이다.
정답이라는 뜻이 아니므로 `answer_correct`와 함께 본다.

### 근거와 전체 성공 점수

먼저 `H = |P_A ∩ P_E|`를 실제 답이 맞힌 근거 페이지 수라고 둔다.

| 지표 | 역할 | 현재 계산식·규칙 |
| --- | --- | --- |
| 근거 페이지 정밀도(`evidence_page_precision`) | 불필요한 페이지 진단 | `H / |P_A|`. 실제 페이지가 없으면 기대 페이지도 없을 때만 1 |
| 근거 페이지 재현율(`evidence_page_recall`) | 놓친 기대 페이지 진단 | `H / |P_E|`. 기대 페이지가 없으면 실제 페이지도 없을 때만 1 |
| 근거 페이지 F1(`evidence_page_f1`) | 정밀도·재현율 종합 진단 | `2PR / (P+R)`. 두 값의 합이 0이면 0 |
| 근거 포함(`evidence_coverage`) | 전체 성공에 직접 사용 | 기대 페이지가 있으면 교집합이 하나 이상일 때 1. 기대 페이지가 없으면 실제 페이지도 없어야 1 |
| 답과 인용문의 자체 일관성(`quote_answer_support`) | 출력 내부 일관성 진단 | 일반 답변은 각 인용문이 답의 모든 숫자 또는 정규화한 답 문자열을 포함하는지 0/1로 검사한 뒤 평균. 답변 보류는 1, 구조를 읽지 못하면 0 |
| 전체 성공(`task_success`) | 최종 문제별 판정 | `schema_validity AND abstention_correct AND answer_correct AND evidence_coverage` |

`answer_exact`, ANLS, 토큰 F1은 함께 보고 표현 차이와 실제 오답을 구분한다. 표·차트
OCR은 VLM이 페이지 JPEG에서 직접 수행한다. 답과 인용문의 자체 일관성 점수는 출력 내부의
진단일 뿐 이미지에 그 문구가 있다는 사실을 증명하거나 전체 성공을 결정하지 않는다.
순수 JSON, ANLS, 토큰 F1과 근거 페이지 F1도 원인을 찾는 진단 점수이며 단독
통과 기준이 아니다.

모델이 문서 전체 페이지 수보다 큰 근거 페이지를 내면 `evidence_coverage=0`과
전체 성공을 0(`task_success=0`)으로 고친다. API 제공사 오류가 나면 상태를 판단 보류
(`inconclusive`)로 남기고 품질 분모에서 제외한다. 이때 기록된 0점은 모델 오답으로
집계하지 않는다.

## 실제 응답 회귀검사용 고정 응답(fixture)

```bash
uv run python scripts/freeze_recorded_responses.py --run-id <완료된-run-id>
uv run python scripts/run_workflow.py
uv run python scripts/evaluate_workflow.py
```

실제 NIM 원본 응답 전체를 고정해 API 없이 같은 구문 분석기(parser)와 채점기를 다시
실행한다. 실제 호출 품질(`live_quality`)과 시험 전용(`test_only`) 결과를 혼동하지 않는다.

## 실패 주입

```bash
uv run python scripts/evaluate_failures.py
uv run pytest
uv run ruff check .
```

깨진 JSON, 신뢰도(confidence) 범위 위반, 오답과 잘못된 페이지가 각각 어떤 지표(metric)를
실패시키는지 확인한다.

## 실습에서 생성되는 결과

- `reports/week-01/eda.json`
- `reports/week-01-nvidia/runs/<run-id>/observations.jsonl`
- `reports/week-01-nvidia/runs/<run-id>/records.jsonl`
- `reports/week-01-nvidia/runs/<run-id>/results.jsonl`
- `reports/week-01-nvidia/runs/<run-id>/summary.json`
- DeepEval 평가 실행(TestRun)
- 실제 NIM 응답 기반 저장 고정 응답(recorded fixture)
- 실패 사례 결과

위 목록은 실습을 완료하면 만들어지는 전체 결과다.

## Week 1의 범위

Week 1의 작업 모델은 NVIDIA Nemotron 3 Nano Omni다. 이 주차에서는 PDF를 페이지 이미지로
준비하고, Nemotron 응답을 구조화한 뒤 고정 규칙 채점기로 평가하는 데까지만 진행한다.
Gemma 모델 변경과 지시문 비교는 [Week 2 실습 안내](week-02-lab.md)에서 처음 시작한다.

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
uv run python scripts/run_nvidia_nim.py \
  --live \
  --sample-id aihub-report-r01 \
  --max-requests 1 \
  --max-input-tokens 20000 \
  --max-output-tokens 500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 120 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

문제가 없으면 별도 실행(run)으로 전체 질문을 실행한다.

```bash
uv run python scripts/run_nvidia_nim.py \
  --live \
  --max-requests 40 \
  --max-input-tokens 800000 \
  --max-output-tokens 20000 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

출력에 표시된 `reports/week-01-nvidia/runs/<run-id>/summary.json`에서 다음 값을
확인한다.

- `record_count`
- `target_count`
- `status_counts`의 통과(`passed`), 실패(`failed`), 판단 보류(`inconclusive`)
- `requested_model`
- `evidence_kind`
- `judge_status`

모든 질문이 통과할 필요는 없다. `record_count`와 `target_count`가 같고 통과, 실패,
판단 보류의 합계가 `target_count`와 같으면 실행은 완료된 것이다. API 오류가 남았다면
오류가 발생한 `sample_id`와 메시지를 확인한다.
`status_counts`에 표시되지 않은 상태는 0건이다.

### 3. 결과 2건 확인

해당 실행의 `results.jsonl`에서 통과 1건과 실패 또는 판단 보류 1건을 선택한다. 실패나
판단 보류 결과가 하나도 없다면 통과 2건을 선택한다.

질문과 기대 답은 `local-data/aihub/cases.jsonl`에서 확인하고, 실제 답과 `reasons`는
해당 run의 `results.jsonl`에서 확인한다.

각 사례의 `sample_id`, 상태(`status`), 기대 답, 실제 답과 전체 성공(`task_success`)을
비교한다. 통과
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

- `reports/week-01/summary.json`의 증거 종류(`evidence_kind`)는 시험 전용(`test_only`)이다.
- `reports/week-01/summary.json`의 평가 모델 상태(`judge_status`)는 요청하지 않음
  (`not_requested`)이다.
- `reports/week-01-failures/results.json`의 네 사례는 모두 `task_success=0.0`이다.

### 결과 확인 파일

- `reports/week-01/eda.json`
- `reports/week-01-nvidia/runs/<run-id>/summary.json`
- `reports/week-01-failures/results.json`

### 완료 기준

- 데이터와 전체 질문을 오류 없이 준비했거나 발견한 데이터 오류를 확인했다.
- `record_count`와 `target_count`가 같고 모든 질문에 실행 상태가 있다.
- 결과 2건의 기대 답, 실제 답과 점수를 비교했다.
- 실제 응답의 실제 호출 품질(`live_quality`)과 저장 응답의 시험 전용(`test_only`)을 구분했다.
- 의도적으로 만든 네 실패 사례가 모두 실패하는 것을 확인했다.
