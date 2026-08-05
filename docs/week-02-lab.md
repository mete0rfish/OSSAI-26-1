# Week 2 실행형 실습

## 목표

Week 1에서 사용한 Nemotron을 Gemma 4로 처음 변경하고, 소규모 사전 실행에서 발견한 실패를
근거로 입력 지시문(prompt)을 개선한다. 같은 Gemma에서 기준·개선 지시문을 전체 데이터로
비교한 뒤, 확정한 공통 지시문으로 NVIDIA NIM Gemma와 Google AI Studio Gemini를 비교한다.

실습을 마치면 다음을 할 수 있다.

- Week 1의 Nemotron 설정과 Week 2의 Gemma 설정에서 실제로 바뀐 값을 설명한다.
- 모델 목록 확인과 실제 API를 호출하는 소규모 사전 실행(probe)을 구분한다.
- Gemma 기준 지시문을 같은 사례로 5회 실행하고 출력 변동을 확인한다.
- 실행 폴더에서 원본 응답, 실제 처리 모델, 점수와 오류를 찾는다.
- 실패 근거로 입력 지시문을 별도 후보로 개선한다.
- 같은 모델에서 지시문만 바꾼 결과와 두 provider 결과를 구분해 비교한다.

## 시작 조건

Week 1에서 다음까지 완료했다고 가정한다.

- 작업 모델: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- 실제 실행 설정: `configs/nvidia-nim.yaml`
- 실행 결과: `reports/week-01-nvidia/runs/<WEEK1_RUN_ID>/`
- 모델 입력: 질문과 페이지 JPEG
- 평가: `aihub-vqa-deterministic-v2` 고정 규칙 채점기

Week 1에서는 Gemma를 실행하거나 Gemma 지시문을 비교하지 않았다. Gemma 도입은 이 문서의
첫 실험부터 시작한다.

## 1. 환경과 Week 1 결과 확인

저장소 루트에서 실행한다.

```bash
uv sync --locked --dev
```

Week 1에서 만든 `reports/week-01-nvidia/runs/<WEEK1_RUN_ID>/summary.json`을 열고 다음을
확인한다.

- `requested_model`이 Nemotron인가
- `actual_models`에 실제 처리 모델이 기록됐는가
- `record_count`와 `target_count`가 같은가
- `provider_error_count`와 `model_drift_count`는 몇 건인가
- 통과·실패·판단 보류 합이 전체 건수와 같은가

Week 1 결과는 Gemma 기준 결과로 재해석하지 않는다.

## 2. Nemotron에서 Gemma로 변경한 설정 확인

```bash
diff -u configs/nvidia-nim.yaml configs/nvidia-nim-gemma4-baseline.yaml || true
```

반드시 확인할 변경은 다음 세 가지다.

| 항목 | Week 1 | Week 2 첫 기준 |
| --- | --- | --- |
| 요청 모델 | `nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `nvidia_nim/google/gemma-4-31b-it` |
| 예상 실제 모델 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `google/gemma-4-31b-it` |
| 결과 위치 | `reports/week-01-nvidia` | `reports/week-02-gemma-baseline` |

첫 실험에서는 입력 지시문, 질문 40건, 페이지 JPEG, 출력 형식과 채점기를 바꾸지 않는다.
모델과 결과 경로만 Gemma용으로 바꾼다.

## 3. 모델 추론 없는 사전 점검

다음 명령은 NVIDIA 모델 목록만 조회한다. 질문과 페이지 이미지를 모델에 보내지 않는다.

```bash
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim-gemma4-baseline.yaml
```

터미널에서 다음을 확인한다.

```text
configured model: google/gemma-4-31b-it
available now: True
```

`available now: False`이면 API 실행으로 넘어가지 않는다. 모델 목록과 가격을 확인한 실제
날짜를 이후 명령의 `<YYYY-MM-DD>`에 사용한다.

## 4. Gemma 기준 지시문으로 5회 사전 실행

이 단계는 수업의 dry run 역할을 하지만 실제 NVIDIA NIM API를 호출한다. 같은
`aihub-report-r01`을 재시도 없이 독립 실행 5회 호출해, 한 번의 우연한 출력만 보고
지시문을 바꾸지 않도록 한다.

먼저 외부 전송과 다음 상한을 확인한다.

- 요청: 5회
- 재시도: 0회
- 비용 상한 합계: USD 0.05
- 실행별 시간 상한: 120초
- 입력: 질문과 페이지 JPEG
- 보내지 않는 값: PDF 추출 문장, 기대 답, 채점 결과, API 키

`<YYYY-MM-DD>`를 실제 확인일로 바꾸고 실행한다.

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
    --catalog-verified-on <YYYY-MM-DD>
done
```

각 명령 마지막에 출력된 `run directory` 다섯 개를 기록한다.

```text
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_PROBE_RUN_ID_01>/
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_PROBE_RUN_ID_02>/
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_PROBE_RUN_ID_03>/
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_PROBE_RUN_ID_04>/
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_PROBE_RUN_ID_05>/
```

각 폴더에서 다음 순서로 파일을 연다.

1. `summary.json`
2. `observations.jsonl`
3. `results.jsonl`
4. `run-manifest.json`

`summary.json`에서 확인할 값:

- `probe_only: true`
- `status: inconclusive`
- `observed_status: complete`
- `requested_model: nvidia_nim/google/gemma-4-31b-it`
- `actual_models`
- `provider_error_count`
- `model_drift_count`
- `budget.request_count`

한 건 사전 실행은 품질 완료 근거가 아니므로 정답을 맞혀도 `status`는 `inconclusive`다.
이 때문에 명령이 종료 코드 2를 반환할 수 있다. API 실행 성공 여부는 종료 코드만 보지
말고 `observed_status`, `provider_error_count`와 `model_drift_count`로 확인한다.

다섯 실행의 `observations.jsonl`과 `results.jsonl`을 보고 표를 채운다.

| trial | 실제 답 | JSON만 출력 | 구조 통과 | 숫자 일치 | 전체 성공 | 응답 시간 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 | | | | | | |
| 02 | | | | | | |
| 03 | | | | | | |
| 04 | | | | | | |
| 05 | | | | | | |

확인할 필드는 다음과 같다.

- `observations.jsonl`: `raw_output`, `model_call.actual_model`,
  `model_call.latency_ms`, 입력·출력 토큰, `model_error`
- `results.jsonl`: `json_object_only`, `schema_validity`, `numeric_match`,
  `answer_correct`, `evidence_page_f1`, `task_success`

근거 페이지가 맞아도 답에 질문의 연도처럼 불필요한 숫자가 포함되면 숫자 일치와 전체
성공은 실패할 수 있다. Markdown 코드 블록을 벗겨 JSON을 읽을 수 있어도
`json_object_only=0`은 보존한다.

## 5. Gemma 기준 지시문 전체 40건 실행

5회 모두 provider 오류와 실제 처리 모델 불일치가 없을 때 별도 전체 실행을 시작한다.

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
  --catalog-verified-on <YYYY-MM-DD>
```

출력된 전체 실행 ID를 `<GEMMA_BASELINE_FULL_RUN_ID>`로 기록하고 다음 파일을 연다.

```text
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_FULL_RUN_ID>/summary.json
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_FULL_RUN_ID>/observations.jsonl
reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_FULL_RUN_ID>/results.jsonl
```

`summary.json`의 실제 값을 적는다.

| 확인 항목 | Gemma 기준 지시문 |
| --- | ---: |
| 전체 대상 `target_count` | |
| 저장 결과 `record_count` | |
| `provider_error_count` | |
| `score_averages.schema_validity` | |
| `score_averages.answer_correct` | |
| `score_averages.task_success` | |

실패 사례는 `results.jsonl`의 `scores`와 `reasons`로 고른 뒤 같은 `sample_id`의
`raw_output`을 `observations.jsonl`에서 확인한다. 평균만 보고 지시문을 수정하지 않는다.

## 6. 실패 근거로 Gemma 지시문 변경 확인

기준 파일은 비교를 위해 그대로 둔다.

- 기준: `prompts/pdf-question-answer.md`
- 개선 후보: `prompts/pdf-question-answer-gemma4.md`

```bash
diff -u prompts/pdf-question-answer.md prompts/pdf-question-answer-gemma4.md || true
```

개선 후보에서 확인할 변경은 다음과 같다.

1. 숫자·날짜·기관명 질문은 필요한 값과 단위만 `answer`에 쓴다.
2. 질문에 있는 연도와 분기를 답에 반복하지 않는다.
3. 표·차트는 대상, 행, 열과 단위를 다시 확인한다.
4. 근거에는 답을 포함하는 연속 원문 한 구절을 넣는다.
5. 문서에서 찾을 수 없으면 정해진 답변 보류 형식을 사용한다.
6. 첫 글자 `{`, 마지막 글자 `}`인 JSON 객체 하나만 반환한다.
7. Markdown 코드 블록으로 감싼 JSON 예시를 사용하지 않는다.

기준 지시문에는 코드 블록을 쓰지 말라는 규칙과 코드 블록 예시가 함께 있다. 개선 후보는
예시까지 한 줄 JSON 객체로 바꾼다. 긴 답에서 기대 숫자만 골라내거나 마지막 JSON만
추출하도록 채점기를 느슨하게 바꾸지 않는다.

## 7. 개선 지시문으로 1건 확인 실행

개선 후보는 먼저 같은 사례 한 건으로 출력 구조가 의도대로 바뀌었는지 확인한다. 지시문
효과의 최종 판단은 다음 절의 전체 40건 A/B에서 하므로 이 단계는 요청 1회만 사용한다.

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
  --catalog-verified-on <YYYY-MM-DD>
```

결과 위치는 다음과 같다.

```text
reports/week-02-gemma-improved/runs/<GEMMA_IMPROVED_PROBE_RUN_ID>/
```

기준 5회에서 반복해서 나타난 문제와 개선 확인 실행의 다음 항목을 비교한다.

| 비교 항목 | 기준 5회 관찰 | 개선 확인 실행 |
| --- | --- | --- |
| 실제 답 | | |
| 순수 JSON | | |
| 구조 통과 | | |
| 숫자 일치 | | |
| 전체 성공 | | |

개선 확인 한 건이 성공해도 지시문이 채택된 것은 아니다.

## 8. 개선 지시문 전체 실행과 A/B 비교

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
  --catalog-verified-on <YYYY-MM-DD>
```

출력된 실행 ID를 `<GEMMA_IMPROVED_FULL_RUN_ID>`로 기록한다. 두 전체 실행을 현재
고정 규칙 채점기로 비교한다.

```bash
uv run --locked python scripts/compare_gemma_prompts.py \
  --baseline-run reports/week-02-gemma-baseline/runs/<GEMMA_BASELINE_FULL_RUN_ID> \
  --candidate-run reports/week-02-gemma-improved/runs/<GEMMA_IMPROVED_FULL_RUN_ID> \
  --rescore-current \
  --output reports/week-02/gemma-prompt-comparison.json
```

`reports/week-02/gemma-prompt-comparison.json`에서 확인할 값:

- `quality_eligible_counts`
- `provider_error_counts`
- `metric_deltas.task_success`
- `metric_deltas.schema_validity`
- `new_success_ids`
- `new_failure_ids`
- `not_comparable_ids`
- `automated_status`와 `invalid_reasons`

새 실패나 비교 불가가 있으면 평균이 올라도 자동 채택하지 않는다. 이 결과는 같은 Gemma에서
지시문만 바꾼 비교다.

## 9. 두 provider가 공유할 지시문 확정

Gemma 전용 후보의 `/no_think`를 Gemini에도 보내면 provider와 지시문이 동시에 달라진다.
두 provider 비교에는 `prompts/pdf-question-answer-json-only.md`를 사용한다.

```bash
diff -u prompts/pdf-question-answer-gemma4.md prompts/pdf-question-answer-json-only.md || true
```

`configs/week-02-live.yaml`에서 두 호출 경로가 같은 질문, 페이지 JPEG, 공통 지시문,
출력 형식과 채점기를 사용하는지 확인한다. 달라지는 것은 provider와 모델 호출 경로다.

## 10. Gemma와 Gemini 3건 사전 비교

`.env`에 승인된 `NVIDIA_NIM_API_KEY`와 `GEMINI_API_KEY`가 있어야 한다.

```bash
uv run --locked python scripts/compare_live_provider_routes.py \
  --live --probe-sample-id aihub-report-r01 \
  --max-requests 2 --max-input-tokens 40000 --max-output-tokens 1000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 240 \
  --catalog-verified-on <YYYY-MM-DD> \
  --output reports/week-02-live/probe-r01-<RUN_ID>

uv run --locked python scripts/compare_live_provider_routes.py \
  --live --probe-sample-id aihub-report-r03 \
  --max-requests 2 --max-input-tokens 40000 --max-output-tokens 1000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 240 \
  --catalog-verified-on <YYYY-MM-DD> \
  --output reports/week-02-live/probe-r03-<RUN_ID>

uv run --locked python scripts/compare_live_provider_routes.py \
  --live --probe-sample-id aihub-report-r31 \
  --max-requests 2 --max-input-tokens 40000 --max-output-tokens 1000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 240 \
  --catalog-verified-on <YYYY-MM-DD> \
  --output reports/week-02-live/probe-r31-<RUN_ID>
```

각 출력 폴더의 `summary.json`에서 다음을 확인한다.

- baseline과 candidate 관찰 결과가 각각 1건
- provider 오류 0건
- invalid output 0건
- 실제 처리 모델 불일치 0건

하나라도 통과하지 않으면 전체 비교를 시작하지 않는다.

## 11. Gemma와 Gemini 전체 비교

세 사전 실행이 모두 정상일 때만 실행한다.

```bash
uv run --locked python scripts/compare_live_provider_routes.py \
  --live \
  --max-requests 80 --max-input-tokens 1600000 --max-output-tokens 40000 \
  --max-retries 0 --max-cost-usd 0.01 --max-wall-seconds 3600 \
  --catalog-verified-on <YYYY-MM-DD> \
  --output reports/week-02-live/full-<RUN_ID>
```

최종 결과는 다음 순서로 확인한다.

1. `reports/week-02-live/full-<RUN_ID>/summary.json`
2. `comparison.json`
3. `baseline-provenance.json`과 `candidate-provenance.json`
4. `baseline-results.jsonl`과 `candidate-results.jsonl`
5. `baseline-observations.jsonl`과 `candidate-observations.jsonl`
6. `run-manifest.json`

`summary.json`의 실제 값으로 표를 채운다.

| 확인 항목 | NIM Gemma | Gemini |
| --- | ---: | ---: |
| 응답 수: `summary.*_observation_count` | | |
| 품질 판정 가능 수: `comparison.*.quality_eligible_count` | | |
| provider 오류: `summary.*_provider_errors` | | |
| invalid output: `summary.*_invalid_outputs` | | |
| 전체 성공률: `comparison.*.task_success_rate` | | |
| 평균 응답 시간: `comparison.*.average_latency_ms` | | |
| 입력/출력 토큰: `*-provenance.json`의 `budget.observed_*_tokens` | | |

| 문제별 변화 | 건수 |
| --- | ---: |
| `new_success` | |
| `new_failure` | |
| `unchanged` | |
| `not_comparable` | |

문제별 건수는 `comparison.json`의 `classification_counts`, 개별 사례는 `case_diffs`에서
확인한다. 재시도·비용·모델 불일치는 두 `*-provenance.json`에서 확인한다. 실패 사례의
점수와 이유는 `*-results.jsonl`, 실제 원문은 같은 `sample_id`의 `*-observations.jsonl`에서
확인한다.

상대 비교의 `automated_status=pass`는 출시 승인이 아니다. 공통 실패, candidate 절대 실패와
고위험 거짓 통과를 확인한 뒤 사람이 `SHIP`, `HOLD`, `ROLLBACK` 중 하나를 결정한다.

## 12. API 오류 처리 연습

실제 API를 다시 호출하지 않고 저장된 장애 시나리오를 실행한다.

```bash
uv run --locked python scripts/evaluate_recorded_provider_routes.py
```

다음 파일을 확인한다.

- `reports/week-02/faults.json`
- `reports/week-02/comparison.json`
- `reports/week-02/summary.json`

이 결과는 시험 전용(`test_only`)이다. 인증 실패, timeout과 fallback 성공을 모델 오답으로
계산하지 않는지 확인하는 코드 연습이다.

## 채점기 구분

이번 AIHub 데이터는 짧은 기대 답, 답변 보류 여부와 근거 페이지가 있으므로 고정 규칙
채점기를 사용한다.

| 구분 | 고정 규칙 채점기 | 모델 기반 채점기 |
| --- | --- | --- |
| 같은 저장 응답 재평가 | 같은 규칙이면 같은 결과 | 결과가 달라질 수 있음 |
| 알맞은 항목 | JSON 구조, 숫자, 정답, 답변 보류, 근거 페이지 | 설명의 충실성·유용성처럼 정답 하나로 표현하기 어려운 항목 |
| 이번 실습 | 사용 | 사용하지 않음 |

API를 다시 호출해 작업 모델의 답이 달라지는 현상과 저장된 같은 답을 채점하는 규칙의
결정성을 혼동하지 않는다. 이번 실행의 `judge_status`는 `not_requested`다.

## 완료 기준

- Week 1의 작업 모델이 Nemotron이었음을 결과 파일에서 확인했다.
- Week 2에서 Gemma 설정으로 모델 목록 확인을 완료했다.
- Gemma 기준 지시문은 같은 사례로 5회 실행해 모두 기록하고, 개선 지시문은 1건 확인했다.
- Gemma 기준 전체 실행 ID와 개선 전체 실행 ID를 기록했다.
- 실패 사례와 지시문 변경 규칙을 연결해 설명했다.
- `gemma-prompt-comparison.json`에서 새 성공·새 실패·비교 불가를 확인했다.
- 공통 지시문으로 Gemma와 Gemini의 사전 실행과 전체 비교 결과를 확인했다.
- 저장 장애 시나리오와 실제 API 품질 결과를 구분했다.
