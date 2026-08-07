# Week 2 실행형 실습

## 목표

Week 1의 Nemotron을 Gemma 4로 처음 바꾸고, 실패 사례를 근거로 지시문(prompt)을
개선한다. 이어서 같은 입력과 채점기로 NVIDIA NIM Gemma와 Google AI Studio Gemini를
비교한다.

실습에서 한 번에 바꾸는 것은 하나다.

```text
Nemotron → Gemma: 모델 변경 실험
Gemma 기준 지시문 → 개선 지시문: 지시문 변경 실험
NIM Gemma → Gemini: API 제공자와 모델 호출 경로 변경 실험
```

## 1. Week 1 결과에서 출발하기

Week 1 결과 폴더의 `summary.json`을 열어 다음을 확인한다.

- `requested_model`이 `nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`이다.
- `record_count`와 `target_count`가 같다.
- `provider_error_count`와 `model_drift_count`가 기록되어 있다.
- 고정 규칙 채점 기준은 `aihub-vqa-deterministic-v2`다.

Week 1에서는 Gemma를 실행하지 않았다. 따라서 저장된 Week 1 응답을 Gemma 결과라고 부르지
않는다.

대표 사례를 다시 보면 Week 2에서 무엇을 개선할지 쉽게 연결할 수 있다.

```bash
uv run --locked python scripts/inspect_deterministic_scoring_case.py
```

## 2. Nemotron에서 Gemma로 바뀐 값 확인

```bash
diff -u configs/nvidia-nim.yaml configs/nvidia-nim-gemma4-baseline.yaml || true
```

| 항목 | Week 1 | Week 2 기준 |
| --- | --- | --- |
| 요청 모델 | `nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `nvidia_nim/google/gemma-4-31b-it` |
| 예상 실제 모델 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `google/gemma-4-31b-it` |
| 결과 폴더 | `reports/week-01-nvidia` | `reports/week-02-gemma-baseline` |

질문 40건, 페이지 JPEG, 기준 지시문, 출력 형식(schema)과 채점기는 그대로다. 첫 비교에서
모델과 결과 폴더만 바뀌었음을 확인한다.

## 3. Gemma 모델 목록 사전 점검

다음 명령은 모델 추론을 하지 않고 NVIDIA의 현재 모델 목록만 조회한다.

```bash
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim-gemma4-baseline.yaml
```

```text
configured model: google/gemma-4-31b-it
available now: True
```

`False`면 실제 호출을 진행하지 않는다. 성공한 날짜를 수업 당일 날짜로 적는다.

```bash
CATALOG_DATE=2026-08-06
```

한 사례 probe는 prompt를 수정한 상태에서도 실행할 수 있다. 이 결과는 연결과 출력 변화를
보는 탐색 실행이며 전체 품질 근거가 아니다. 전체 40건 실행만 변경 사항이 없는 commit에서
허용한다.

## 4. Gemma 기준 지시문으로 같은 사례 5회 실행

소규모 사전 실행(probe)은 흔히 dry run이라고 부르지만 여기서는 실제 API를 호출한다.
같은 `aihub-report-r01`을 독립 실행 5회 보내 출력 형식과 답의 변동을 관찰한다.

```bash
for trial in 01 02 03 04 05; do
  uv run --locked python scripts/run_nvidia_nim.py \
    --config configs/nvidia-nim-gemma4-baseline.yaml \
    --live \
    --sample-id aihub-report-r01 \
    --trial-id "week02-gemma-baseline-probe-${trial}" \
    --max-requests 1 \
    --max-input-tokens 20000 \
    --max-output-tokens 500 \
    --max-cost-usd 0.01 \
    --max-wall-seconds 120 \
    --max-retries 0 \
    --catalog-verified-on "$CATALOG_DATE"
done
```

`<YYYY-MM-DD>` 같은 꺾쇠 자리표시자를 입력하지 않는다. zsh에서 `<`는 파일 입력 기호로
해석된다. 위처럼 `CATALOG_DATE` 변수에 실제 날짜를 넣으면 그대로 실행할 수 있다.

각 실행 마지막에 자동 생성된 `run directory`가 출력된다. 학습자가 실행 식별자(`run_id`)를
직접 만들 필요는 없다. 다섯 폴더에서 `observations.jsonl`과 `results.jsonl`의 첫 행을 보고
아래 표를 채운다.

| 반복 | 실제 답 | JSON 단독 반환 | 출력 형식 유효 | 숫자 일치 | 전체 성공 | 응답 시간(ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 | | | | | | |
| 02 | | | | | | |
| 03 | | | | | | |
| 04 | | | | | | |
| 05 | | | | | | |

확인할 필드:

- `observations.jsonl`: `raw_output`, `model_error`, `model_call.actual_model`,
  `model_call.latency_ms`
- `results.jsonl`: `json_object_only`, `schema_validity`, `numeric_match`,
  `answer_correct`, `task_success`

한 사례는 전체 품질을 대표하지 않으므로 `summary.json`의 실행 상태는
`inconclusive`다. API 호출 자체가 정상인지 여부는 `observed_status=complete`,
`provider_error_count=0`, `model_drift_count=0`으로 확인한다.

## 5. 준비된 Gemma 기준 지시문 전체 결과 분석

수업 전에 승인된 환경에서 만든 40건 결과를 연다. 직접 전체 실행을 승인받은 경우에만
`git status --short` 출력이 없는지 확인하고 다음 명령을 실행한다.

```bash
uv run --locked python scripts/run_nvidia_nim.py \
  --config configs/nvidia-nim-gemma4-baseline.yaml \
  --live \
  --max-requests 40 \
  --max-input-tokens 800000 \
  --max-output-tokens 20000 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --max-retries 0 \
  --catalog-verified-on "$CATALOG_DATE"
```

출력된 기준 전체 결과 폴더를 메모한다. `summary.json`에서 전체 수, API 제공자 오류,
출력 형식 유효성 평균, 정답 허용 기준 평균과 전체 성공률을 확인한다. 그다음
`results.jsonl`에서 필수 지표가 0인 사례를 고르고 같은 `sample_id`의 원응답을
`observations.jsonl`에서 읽는다.

## 6. 실패를 보고 지시문 직접 수정하기

기준 지시문을 Git에서 제외되는 학습자 파일로 복사한다.

```bash
cp prompts/pdf-question-answer.md local-data/week-02-prompt.md
```

`local-data/week-02-prompt.md`를 열고 5회 probe에서 관찰한 실패 하나만 고친다. 채점 기준이나
질문은 바꾸지 않는다.

개선 후보는 관찰한 실패를 다음 규칙으로 다룬다.

| 관찰한 실패 | 지시문 변경 |
| --- | --- |
| 답에 질문의 연도·설명이 다시 들어감 | 필요한 값·단위·기관명만 `answer`에 쓴다. |
| 표·차트의 다른 값을 선택함 | 대상·행·열·단위를 다시 확인한다. |
| JSON 앞뒤에 설명이나 코드 블록이 붙음 | JSON 객체 하나만 반환한다. |
| 답이 없는데 표의 다른 값을 추측함 | 문서에서 확인할 수 없으면 정해진 형식으로 보류한다. |

채점기가 긴 답에서 기대 숫자만 골라내도록 느슨하게 바꾸지 않는다. 모델이 업무에 필요한
짧은 답을 내도록 지시문을 바꾸는 실험이다.

수정한 지시문으로 같은 사례를 한 번 실행한다. `--prompt`에는 `local-data` 아래 파일만
사용할 수 있으며 한 사례 probe에서만 허용된다.

```bash
uv run --locked python scripts/run_nvidia_nim.py \
  --config configs/nvidia-nim-gemma4-baseline.yaml \
  --prompt local-data/week-02-prompt.md \
  --live \
  --sample-id aihub-report-r01 \
  --trial-id week02-my-prompt-probe \
  --max-requests 1 \
  --max-input-tokens 20000 \
  --max-output-tokens 500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 120 \
  --max-retries 0 \
  --catalog-verified-on "$CATALOG_DATE"
```

원응답과 필수 점수를 기준 probe와 비교하고, 고친 규칙이 실제 출력에 어떤 영향을 줬는지
한 문장으로 기록한다.

준비된 개선안과도 비교한다.

```bash
diff -u local-data/week-02-prompt.md prompts/pdf-question-answer-gemma4.md || true
uv run --locked python scripts/inspect_prompt_comparison_case.py
```

저장 응답 비교는 현재 모델 품질 주장이 아니라 코드 학습용이다.

## 7. 개선 지시문 확인과 준비된 전체 결과 비교

먼저 같은 사례 한 건으로 출력이 의도대로 바뀌었는지 확인한다.

```bash
uv run --locked python scripts/run_nvidia_nim.py \
  --config configs/nvidia-nim-gemma4.yaml \
  --live \
  --sample-id aihub-report-r01 \
  --trial-id week02-gemma-improved-probe \
  --max-requests 1 \
  --max-input-tokens 20000 \
  --max-output-tokens 500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 120 \
  --max-retries 0 \
  --catalog-verified-on "$CATALOG_DATE"
```

한 건이 성공해도 지시문을 채택하지 않는다. 전체 40건은 수업 전에 승인된 환경에서 한 번
실행한 기준·개선 결과를 사용한다. 직접 전체 실행을 승인받은 경우에만 아래 명령을 사용한다.

```bash
uv run --locked python scripts/run_nvidia_nim.py \
  --config configs/nvidia-nim-gemma4.yaml \
  --live \
  --max-requests 40 \
  --max-input-tokens 800000 \
  --max-output-tokens 20000 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --max-retries 0 \
  --catalog-verified-on "$CATALOG_DATE"
```

터미널에 출력된 기준·개선 전체 결과 폴더를 변수에 그대로 붙여 넣는다.

```bash
BASELINE_RUN_DIR="reports/week-02-gemma-baseline/runs/기준-전체-폴더명"
CANDIDATE_RUN_DIR="reports/week-02-gemma-improved/runs/개선-전체-폴더명"

uv run --locked python scripts/compare_gemma_prompts.py \
  --baseline-run "$BASELINE_RUN_DIR" \
  --candidate-run "$CANDIDATE_RUN_DIR" \
  --rescore-current \
  --output reports/week-02/gemma-prompt-comparison.json
```

`gemma-prompt-comparison.json`에서 다음을 읽는다.

- `metric_deltas.task_success`: 전체 성공률 차이
- `new_success_ids`: 기준 실패에서 개선 성공으로 바뀐 사례
- `new_failure_ids`: 새로 실패한 사례
- `not_comparable_ids`: API 오류 등으로 비교할 수 없는 사례
- `automated_status`: 자동 통과·실패·판단 보류

평균이 올라도 새 실패나 비교 불가 사례가 있으면 자동 채택하지 않는다.

## 8. 두 API 제공자가 공유할 지시문 확인

Gemma 전용 지시를 다른 모델에 그대로 보내면 모델과 지시문이 동시에 달라진다. 두 API
제공자 비교에는 모델 이름에 의존하지 않는 공통 지시문을 사용한다.

```bash
diff -u prompts/pdf-question-answer-gemma4.md prompts/pdf-question-answer-json-only.md || true
```

`configs/week-02-live.yaml`을 열어 다음을 확인한다.

- 두 호출 경로가 같은 질문 40건, 페이지 JPEG와 공통 지시문을 쓴다.
- 출력 형식과 고정 규칙 채점기가 같다.
- 기준은 NVIDIA NIM Gemma, 후보는 Google AI Studio Gemini다.
- 두 호출 경로는 서로 다른 API 키와 접속 주소(endpoint)를 쓴다.

## 9. Gemma와 Gemini 세 사례 비교

`.env`에 두 API 키가 있어야 한다. 결과 폴더 이름에 쓸 짧은 실행 표식은 학습자가 정한다.
이 값은 모델 실행 식별자가 아니라 로컬 폴더가 겹치지 않게 하는 이름이다.

```bash
RUN_TAG=class-01
```

같은 표식을 이미 사용했다면 `class-02`처럼 바꾼다.

```bash
uv run --locked python scripts/compare_live_provider_routes.py \
  --live --probe-sample-id aihub-report-r01 \
  --max-requests 2 --max-input-tokens 40000 --max-output-tokens 1000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 240 \
  --catalog-verified-on "$CATALOG_DATE" \
  --output "reports/week-02-live/probe-r01-${RUN_TAG}"

uv run --locked python scripts/compare_live_provider_routes.py \
  --live --probe-sample-id aihub-report-r03 \
  --max-requests 2 --max-input-tokens 40000 --max-output-tokens 1000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 240 \
  --catalog-verified-on "$CATALOG_DATE" \
  --output "reports/week-02-live/probe-r03-${RUN_TAG}"

uv run --locked python scripts/compare_live_provider_routes.py \
  --live --probe-sample-id aihub-report-r31 \
  --max-requests 2 --max-input-tokens 40000 --max-output-tokens 1000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 240 \
  --catalog-verified-on "$CATALOG_DATE" \
  --output "reports/week-02-live/probe-r31-${RUN_TAG}"
```

각 폴더의 `summary.json`에서 두 API 제공자가 각각 한 응답을 남겼고, API 오류·출력 형식
오류·실제 처리 모델 불일치가 모두 0인지 확인한다.

## 10. 준비된 Gemma와 Gemini 전체 결과 비교

40건씩 총 80회 호출은 수업 전에 승인된 환경에서 한 번 수행한다. 학습자는 결과를 분석한다.
직접 전체 실행을 승인받은 경우에만 다음 명령을 사용한다.

```bash
uv run --locked python scripts/compare_live_provider_routes.py \
  --live \
  --max-requests 80 --max-input-tokens 1600000 --max-output-tokens 40000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 3600 \
  --catalog-verified-on "$CATALOG_DATE" \
  --output "reports/week-02-live/full-${RUN_TAG}"
```

결과를 다음 순서로 읽는다.

1. `summary.json`: 두 호출 경로의 응답·오류·전체 상태
2. `comparison.json`: 성공률 차이와 사례별 변화
3. `baseline-results.jsonl`, `candidate-results.jsonl`: 점수와 실패 이유
4. `baseline-observations.jsonl`, `candidate-observations.jsonl`: 실제 원응답
5. `baseline-provenance.json`, `candidate-provenance.json`: 토큰·비용·실제 처리 모델

`comparison.json`의 `new_success`, `new_failure`, `unchanged`, `not_comparable` 건수를 적고,
공통 실패 한 건과 두 모델의 답이 달라진 한 건을 직접 읽는다. 상대 비교의 자동 상태가
`pass`여도 절대 실패가 남아 있으면 출시 승인이 아니다.

## 11. API 오류를 오답과 분리하기

실제 API를 다시 호출하지 않고 저장된 장애 상황을 실행한다.

```bash
uv run --locked python scripts/evaluate_recorded_provider_routes.py
```

`reports/week-02/faults.json`에서 인증 실패, 요청 제한, 시간 초과와 대체 경로 성공이 모델
오답으로 집계되지 않는지 확인한다. 저장 상황을 썼으므로 이 결과는 시험 전용
증거(`test_only`)다.

## 완료 기준

- Week 1의 Nemotron에서 Week 2의 Gemma로 바뀐 설정 세 가지를 설명할 수 있다.
- 같은 Gemma 사례를 기준 지시문으로 5회 실행하고 출력 변동을 표로 정리했다.
- 실패 사례를 보고 학습자 지시문을 직접 수정하고 같은 사례를 다시 실행했다.
- 준비된 기준·개선 지시문 전체 결과에서 새 성공, 새 실패와 비교 불가 사례를 찾았다.
- 공통 지시문으로 Gemma와 Gemini를 같은 조건에서 비교했다.
- API 오류와 출력 형식 오류, 정답 실패를 서로 다른 상태로 설명할 수 있다.
- AIHub 실습에서는 모델 기반 채점기 없이 고정 규칙 채점기를 쓰는 이유를 설명할 수 있다.
