# Week 2 실습 — 점수 차이를 설명 가능한 비교로 바꾸기

## 이번 주에 배우는 것

Week 1에서는 AI 답을 고정 규칙으로 채점했다. Week 2에서는 점수가 달라졌을 때
**무엇을 바꿨고, 그 결과로 어디까지 말할 수 있는지** 배운다.

수업이 끝나면 다음 세 문장을 설명할 수 있어야 한다.

1. 한 번에 여러 조건을 바꾸면 점수 차이의 원인을 하나로 말할 수 없다.
2. 한 사례의 성공은 가설이고, 같은 전체 사례의 새 성공과 새 실패가 검증이다.
3. API 오류는 모델의 오답이 아니며 품질 결론을 보류해야 한다.

명령을 외우는 것이 목표가 아니다. 모든 활동은
`관찰 → 원인 가설 → 바꾼 것 → 결과 → 말할 수 없는 것`을 연결하는 데 쓴다.

## 1. 비교가 필요한 이유

### 무엇을 비교하는가

바꾸기 전 조건과 결과를 **기준선(baseline)**, 비교하려고 바꾼 조건과 결과를
**후보(candidate)**라고 부른다. 같은 문제끼리 짝을 맞추고 확인하려는 조건 외에는
가능한 한 그대로 둔다.

| 비교 | 바뀌는 것 | 이 수업에서 답할 질문 |
| --- | --- | --- |
| Gemma 기준·개선 지시문 | 지시문 묶음 | 같은 모델과 같은 40건에서 새 성공과 새 실패가 무엇인가? |
| NIM Gemma·AI Studio Gemini | 모델·API 제공자·접속 경로·출력 방식 | 두 호출 경로 묶음의 결과가 어디서 다른가? |

Week 1 Nemotron과 Week 2 Gemma의 점수도 볼 수 있지만, 두 저장 실행은 실행 코드 버전도
같지 않다. 따라서 이것은 “통제하지 않은 조건이 섞이면 원인을 좁힐 수 없다”는 반례로만
사용한다. Gemma 지시문 비교가 이번 주의 주된 통제 실험이다.

### 왜 평균만 보면 안 되는가

전체 통과 수가 늘어도 기존에 맞던 중요한 사례가 틀릴 수 있다. 같은 `sample_id`를
짝지으면 다음 네 경우가 생긴다.

| 기준 | 후보 | 뜻 |
| --- | --- | --- |
| 실패 | 성공 | 새 성공 |
| 성공 | 실패 | 새 실패, 즉 회귀 |
| 성공 | 성공 | 성공 유지 |
| 실패 | 실패 | 남은 실패 |

한 사례는 실패 원인을 이해하고 규칙을 만드는 데 좋다. 전체 40건은 그 규칙 묶음이 다른
문제를 망가뜨렸는지 확인하는 데 필요하다. 두 역할을 섞지 않는다.

### 왜 API 오류를 따로 보는가

| 관찰 | 해석 |
| --- | --- |
| 정상 응답이 왔지만 채점 규칙을 통과하지 못함 | 모델 답의 품질 실패 |
| 인증 실패·시간 초과로 답이 없음 | 품질을 판단할 수 없는 실행 오류 |
| 원 경로 실패 뒤 대체 경로가 응답함 | 서비스 연결은 확인했지만 원 모델 품질은 판단 불가 |

응답이 없는데 오답으로 세면 모델 품질과 서비스 상태가 섞인다. 반대로 API가 정상이라고
답이 좋은 것도 아니다.

## 2. 실습 준비

실행 프로젝트 저장소 최상위에서 환경을 준비한다.

```bash
uv sync --locked --dev

ls -d local-data/week-02-full-runs/{gemma-baseline,gemma-improved,provider-comparison}
```

세 폴더는 튜터가 수업 전에 실제 API로 만든 **이론 설명용 저장 결과**다. 하나라도 없으면
남은 파일로 수치를 추정하지 말고 튜터에게 요청한다. 이 중 고정된 개선 수치 예시는 뒤에서
가설과 회귀를 설명하는 데 쓴다. AIHub 이용 조건과 실제 응답 때문에 이 폴더들은 Git commit에
포함되지 않으며, 튜터가 허용된 과정 자료 묶음으로 따로 전달한다.

이 실습은 AIHub 샘플의 경제전망 보고서와 보도자료를 사용한다. 질문 40건 가운데 36건은
문서에서 답을 찾고, 4건은 근거가 없을 때 답변을 보류하는 문제다. 원본 이용 조건, 두 문서와
질문의 구성, PDF를 API용 JPEG로 준비하는 방법은 [AIHub 데이터 안내](aihub-data.md)에 있다.

튜터 자료 묶음에 준비본이 없다면 AIHub 안내대로 원본을 넣고 다음 두 명령을 한 번 실행한다.
이 단계는 API를 호출하지 않는다. 이미 준비본을 받았다면 다시 만들지 않고 `ls`만 확인한다.

```bash
uv run --locked python scripts/prepare_documents.py
uv run --locked python scripts/prepare_cases.py
ls local-data/aihub/cases.jsonl local-data/aihub/prepared/*/manifest.json
```

각 학습자는 자기 지시문(prompt)으로 40건 전체를 한 번 실행한다. 여러 사람이 같은 장비를
쓸 때 파일이 섞이지 않도록 본인의 영문·숫자 과정 별칭을 정한다. 별칭에는 영문, 숫자,
`.`, `_`, `-`만 쓴다.

```bash
STUDENT_ALIAS=student01
STUDENT_DIR="local-data/week-02-students/$STUDENT_ALIAS"
STUDENT_PROMPT="$STUDENT_DIR/prompt.md"
mkdir -p "$STUDENT_DIR"

test -e "$STUDENT_PROMPT" || cp prompts/pdf-question-answer.md "$STUDENT_PROMPT"
```

`student01`은 반드시 본인에게 배정된 고유 별칭으로 바꾼다. 이후 실습도 같은
터미널에서 이어 실행해 이 변수들을 유지한다.

학습 기록이 없다면 한 번만 복사한다.

```bash
test -e local-data/learning-progress.md ||
  cp templates/week-02-learning-note.md local-data/learning-progress.md
```

## 3. 사례 한 건에서 가설 만들기

먼저 개선 결과를 열지 않고 Gemma 기준 지시문의 `aihub-report-r01`만 읽는다.

```bash
rg '"sample_id":"aihub-report-r01"' \
  local-data/week-02-full-runs/gemma-baseline/results.jsonl |
  rg -o '"answer":"[^"]*"|"answer_correct":[0-9.]+|"task_success":[0-9.]+'
```

이 사례의 기준 답은 `71.6%`다. 모델은 숫자를 포함했지만 설명 문장까지
`answer`에 넣었다. 구조와 근거는 맞았지만 업무 계약보다 답의 범위가 넓어
`answer_correct=0`이었다.

결과를 보고 다음을 기록한다.

1. **관찰:** 어느 출력이 어떤 규칙에서 실패했는가?
2. **원인 가설:** 모델이 왜 그 형태로 답했을 가능성이 있는가?
3. **변경 제안:** 질문·출력 형식·채점기는 그대로 두고 지시문 규칙 하나를 어떻게 바꿀까?
4. **예측:** 같은 사례가 어떻게 달라지고, 다른 사례에는 어떤 새 실패가 생길 수 있을까?

이 사례에서는 다음 가설을 사용한다.

> 숫자를 묻는 질문에는 값과 단위만 answer에 쓴다.

미리 준비한 두 응답을 열어 이 가설이 한 사례에서 어떤 차이를 만드는지 확인한다.

```bash
uv run --locked python scripts/inspect_prompt_comparison_case.py
```

이 명령은 API를 호출하지 않으며 Git에 포함된 고정 응답(`test_only`)을 사용한다.
후보의 `71.6%`와 통과 결과는 **가설을 설명하는 시험용 예시**다. 학습자가 수정할
파일의 실행 결과가 아니다.

위에서 만든 개인 파일을 열고 기준 지시문에 이 규칙을 추가한다.

```bash
diff -u prompts/pdf-question-answer.md "$STUDENT_PROMPT" || true
```

`diff`는 무엇을 바꿨는지만 증명한다. 고정 예시를 개인 수정의 성공 증거로 쓰지
않는다. 개인 수정의 실제 효과는 5절의 본인 40건 실행과 저장 baseline 비교로 확인한다.

## 4. 저장 예시에서 새 성공과 새 실패 이해하기

튜터가 준비한 개선 지시문에는 방금 작성한 규칙 외에도 날짜·기관·목록 규칙이 들어 있다.
따라서 아래 결과는 **준비된 지시문 묶음 전체**의 효과이며 한 줄 규칙의 단독 효과가 아니다.

```bash
STORED_EXAMPLE_COMPARISON="$STUDENT_DIR/stored-example-comparison.json"

uv run --locked python scripts/compare_gemma_prompts.py \
  --baseline-run local-data/week-02-full-runs/gemma-baseline \
  --candidate-run local-data/week-02-full-runs/gemma-improved \
  --rescore-current \
  --output "$STORED_EXAMPLE_COMPARISON" >/dev/null

rg -n -A 5 '"classification_counts"' "$STORED_EXAMPLE_COMPARISON"
rg -n '"(invalid_reasons|automated_status)"' "$STORED_EXAMPLE_COMPARISON"

rg -n '"(abstention_correct|evidence_coverage|task_success)"' local-data/week-02-full-runs/gemma-{baseline,improved}/summary.json

for SAMPLE_ID in aihub-report-r29 aihub-report-r01 aihub-report-r07; do
  rg -n -A 6 "\"sample_id\": \"$SAMPLE_ID\"" "$STORED_EXAMPLE_COMPARISON"
done
```

세 사례는 서로 다른 역할을 한다.

| 사례 | 관찰 목적 |
| --- | --- |
| `r29` | 두 지시문에서 성공이 유지됐는가? |
| `r01` | 기준 실패가 후보 성공으로 바뀌었는가? |
| `r07` | 개선 뒤에도 왜 실패가 남았는가? |

두 폴더는 튜터가 실제 API로 만든 품질 증거(`live_quality`)다. 지시문 변경의 원리를
설명하기 위해 고정한 저장 예시의 결과는 다음과 같다. 개인 실행과 비교할 같은-release
baseline은 5절에서 별도로 지정한다.

- 기준 3/40 → 개선 25/40
- 새 성공 22건, 새 실패 0건, 남은 실패 15건
- `task_success`는 올랐지만 `abstention_correct`와
  `evidence_coverage`는 낮아짐

따라서 “이번 40건에서는 정한 비교 조건을 통과했다”고 말할 수 있다. 다음 주제까지
일반화해서는 안 된다.

- 이 지시문이 모든 문서와 질문에서 더 좋다는 주장
- 한 줄 규칙만으로 22건이 좋아졌다는 주장
- 한 번의 확률적 실행이 반복되어도 같다는 주장

## 5. 내 지시문으로 같은 40건 전체 실행하기

### 5-1. 승인과 같은 release baseline 확인

이 절은 선택 활동이 아니다. 튜터가 외부 이미지 전송, 계정 할당량과 반 전체 합산 상한을
승인한 뒤, 각 학습자가 실행한다. 튜터의 화면 시연은 `r01` 한 건뿐이지만, 학습자의
품질 비교는 본인 지시문으로 40건을 모두 호출해야 한다.

튜터가 **현재 수업 release와 같은 clean commit에서 수업 전에 만든 baseline 폴더**의 정확한
경로를 알려 준다. 다음 예시의 값은 튜터가 알려 준 경로로 바꾼다.

```bash
WEEK2_BASELINE_RUN=local-data/week-02-full-runs/gemma-release-baseline-RELEASE_SHA
```

뒤의 실행 명령은 이 폴더가 40건 완결본인지, clean Git에서 만들어졌는지, 현재 Git SHA와
같은지 요청 전에 검사한다. 현재 저장소에 tracked 변경이 있어도 요청 전에 차단한다. 개인
지시문은 `local-data` 아래라 Git 상태를 더럽히지 않는다.

`.env`가 없다면 `.env.example`을 복사하고 본인의 `NVIDIA_NIM_API_KEY`를 입력한다. key는 제출하거나
화면에 출력하지 않는다. 모델 목록과 [NVIDIA NIM 이용 안내](https://docs.api.nvidia.com/nim/docs/product)의
가격·이용 조건을 실제로 확인한 오늘 날짜만 기록한다.

```bash
test -e .env || cp .env.example .env
```

```bash
unset CATALOG_DATE PRICING_DATE
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim-gemma4-baseline.yaml &&
  CATALOG_DATE=$(date +%F) &&
  PRICING_DATE="$CATALOG_DATE"
```

사전 점검이 `available now: True`가 아니거나 확인 날짜를 사실대로 기록할 수 없으면
날짜 변수는 비어 있고 뒤의 실행기도 호출 전에 멈춘다. 사전 점검의 모델 목록 metadata 조회
1회는 아래 40회의 모델 추론 요청과 별도다.

### 5-2. 40건 실행과 자동 폴더 찾기

아래 명령은 재시도 없이 최대 40회 요청한다. 전체 상한은 입력 800,000 token, 출력 20,000
token, 비용 안전장치 0.01달러, 7,200초다. `$0.01`은 예상 청구액이 아니라 설정 오류나
예상 밖 사용량을 상한 전에 막는 안전장치다. 실제 이용 조건과 가격은 실행 당일 확인한다.
완료 실행의 실제 시도 수도 40회다.

| 옵션 | 이 실습에서 하는 일 |
| --- | --- |
| `--config` | 모델·데이터·출력 형식을 기준 실행과 같게 둔다. |
| `--prompt` | 비교에서 바꾸는 유일한 조건인 개인 지시문을 지정한다. |
| `--baseline-run` | 같은 release의 40건 기준 실행이 맞는지 요청 전에 확인한다. |
| `--live`, `--trial-id` | 실제 호출임을 명시하고 실행을 고유하게 식별한다. |
| `--max-*`, 확인 날짜 | 요청·token·비용·시간을 승인 범위 안에 묶는다. |

```bash
TRIAL_ID="week02-${STUDENT_ALIAS}-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$STUDENT_DIR/${TRIAL_ID}.log"

uv run --locked python scripts/run_nvidia_nim.py \
  --config configs/nvidia-nim-gemma4-baseline.yaml --prompt "$STUDENT_PROMPT" \
  --baseline-run "$WEEK2_BASELINE_RUN" --live --trial-id "$TRIAL_ID" \
  --max-requests 40 --max-retries 0 \
  --max-input-tokens 800000 --max-output-tokens 20000 \
  --max-cost-usd 0.01 --max-wall-seconds 7200 \
  --catalog-verified-on "$CATALOG_DATE" \
  --pricing-verified-on "$PRICING_DATE" 2>&1 | tee "$RUN_LOG"
```

터미널의 `status`는 답의 품질, `observed status`는 40건 수집 완료 여부다.
`status=fail`이어도 `observed status=complete`이면 오답을 포함한 40건이
정상적으로 저장된 것이다. API 오류나 중단이 생겨도 이미 만든 폴더는 지우지 않는다.

`tee`로 보존한 터미널 출력에서 runner가 표시한 실제 폴더를 그대로 가져온다.

```bash
LEARNER_RUN_DIR=$(sed -n 's/^run directory: //p' "$RUN_LOG" | tail -n 1)
printf 'learner run directory: %s\n' "$LEARNER_RUN_DIR"
```

실행기는 폴더를 만든 직후 이 경로를 출력한다. 값이 비어 있으면 호출 전에 차단된 것이므로
경로를 추측하지 말고 로그를 튜터에게 보여 준다.

### 5-3. 원본과 완전성 확인

이번 실행의 네 핵심 파일은 다음 위치에 있다.

| 파일 | 위치 | 확인할 것 |
| --- | --- | --- |
| 실제 지시문 | `$LEARNER_RUN_DIR/prompt.md` | 내 수정본과 정확히 같은가? |
| 원응답 | `$LEARNER_RUN_DIR/observations.jsonl` | 40개 응답·오류 기록이 있는가? |
| 채점 결과 | `$LEARNER_RUN_DIR/results.jsonl` | 같은 40개 사례의 점수가 있는가? |
| 전체 요약 | `$LEARNER_RUN_DIR/summary.json` | `observed_status`가 `complete`인가? |

```bash
rg -n '"(status|observed_status|record_count|target_count|provider_error_count|model_drift_count)"' \
  "$LEARNER_RUN_DIR/summary.json"

rg -n '"(request_count|attempt_count|actual_input_tokens|actual_output_tokens|actual_cost_usd|wall_seconds)"' \
  "$LEARNER_RUN_DIR/summary.json"
```

| 확인 항목 | 완료 기준 | 뜻 |
| --- | --- | --- |
| 수집 상태 | `observed_status=complete` | API 오류 없이 40건을 모두 모았다. |
| 사례 수 | `record_count=target_count=40` | 일부 사례만 본 결과가 아니다. |
| 호출 수 | `request_count=attempt_count=40` | 재시도 없이 사례마다 한 번 호출했다. |
| 모델 상태 | `provider_error_count=model_drift_count=0` | 접속 실패나 다른 모델 응답을 품질 점수에 섞지 않았다. |
| 답의 품질 | `status=pass` 또는 `fail` | 수집 완료와 정답 품질은 서로 다른 판정이다. |

`observed_status=complete`가 실행 완결 판정이다. 이 값이 `complete`이면
`status=fail`이어도 40건 원본은 완결된 품질 증거다. API 오류나 실제
처리 모델 불일치가 있으면 `observed_status`는 `complete`가 아니므로 품질
비교를 진행하지 않는다. 다음 절의 비교기가 네 파일, 저장한 지시문, 40건, 요청·attempt와
token·비용·시간 상한을 다시 검사하므로 문서에서 긴 검증 코드를 반복하지 않는다.

### 5-4. 저장 baseline과 내 후보 비교

비교 결과도 학생별 고유 경로에 쓴다. 비교기는 같은 model, 40개 `sample_id`, 요청
상한, 입력 manifest, lockfile, workflow, dataset, schema와 scorer hash를 확인한다.
`--rescore-current`는 저장 점수를 그대로 믿지 않고 두 실행의 원응답을 현재 저장소의
같은 고정 규칙 채점기로 다시 계산한다.

```bash
COMPARISON_OUTPUT="reports/week-02/students/$STUDENT_ALIAS/${TRIAL_ID}-baseline-vs-candidate.json"

uv run --locked python scripts/compare_gemma_prompts.py \
  --baseline-run "$WEEK2_BASELINE_RUN" \
  --candidate-run "$LEARNER_RUN_DIR" \
  --rescore-current \
  --output "$COMPARISON_OUTPUT" >/dev/null

rg -n -A 5 '"classification_counts"' "$COMPARISON_OUTPUT"
rg -n '"(invalid_reasons|automated_status)"' "$COMPARISON_OUTPUT"
printf 'comparison file: %s\n' "$COMPARISON_OUTPUT"
```

`invalid_reasons`가 빈 목록일 때만 지시문만 바꾼 비교로 해석한다.
`automated_status=fail`은 새 실패가 있거나 성공률이 오르지 않았다는 유효한 결과일 수
있다. 반면 `invalid_reasons`가 하나라도 있으면 비교 상태는
`inconclusive`다. 특히 저장 baseline이 다른 commit이나 runner에서 만들어졌다면
개인 지시문의 효과라고 주장하지 말고 같은 release baseline을 튜터에게 요청한다.

2026-08-17 실제 리허설에서는 baseline 2/40에서 개인 후보 23/40으로 바뀌었고,
새 성공 21건·새 실패 0건이었다. `invalid_reasons=[]`, `automated_status=pass`였지만, 개인이
작성한 지시문에 따라 수치는 달라질 수 있다.

## 6. 호출 경로 묶음 비교하기

이번에는 한 요소만 바꾸지 않는다.

| 경로 | 함께 달라지는 것 |
| --- | --- |
| NVIDIA NIM Gemma | Gemma 모델, NIM 접속 경로와 출력 설정 |
| Google AI Studio Gemini | Gemini 모델, AI Studio 접속 경로와 출력 설정 |

저장된 비교 결과와 새 실패 두 건을 연다.

```bash
rg -n '"(record_count|provider_error_count|task_success_rate|invalid_comparison_reasons|automated_status)"' \
  local-data/week-02-full-runs/provider-comparison/comparison.json
rg -n -A 5 '"classification_counts"' \
  local-data/week-02-full-runs/provider-comparison/comparison.json

rg -n -A 8 '"sample_id": "aihub-press-p0(3|5)"' \
  local-data/week-02-full-runs/provider-comparison/comparison.json
```

현재 정본에서 NIM Gemma가 24/40, Gemini가 31/40을 통과했다. 새 성공은 9건,
새 실패는 `p03`, `p05` 두 건이다. 두 경로 모두 API 오류와 실제 모델
불일치는 0건이었다. 4절의 개선 Gemma 25/40과 이 절의 NIM Gemma 24/40은 서로 다른 시점과
목적의 저장 실행이므로 같은 결과가 아니다.

이 결과만으로 “Gemini 모델이 17.5%p 더 우수하다”고 말할 수 없다. 모델뿐 아니라 제공자,
접속 경로와 출력 방식이 함께 달랐기 때문이다. 말할 수 있는 것은 “이 40건과 이 실행
조건에서 두 호출 경로 묶음의 결과가 이렇게 달랐다”까지다.

`p03`과 `p05`는 의미상 타당해 보여도 허용 문자열과 달라 실패했다.
짧은 답의 허용 표현은 사람이 보완할 수 있지만, 긴 설명의 모든 표현을 문자열로 등록하기는
어렵다. Week 3에서는 이 한계를 사람의 상대 비교와 LLM Judge로 다룬다.

## 7. 강의자 시연 — 반 전체 대표 live 한 건

승인 조건이 모두 맞을 때만 튜터가 별도로 `r01` 한 건을 실행한다. 전체 명령과 안전
확인은 강의자용 Week 2 runbook에만 둔다. 이 시연은 원응답과 저장 형식을 함께 보는 용도이며,
각 학습자의 40건 실행을 대신하지 않는다.

실행했다면 다음 세 가지만 확인한다.

1. 실제 사용한 지시문과 원응답이 저장됐는가?
2. API 오류인지 품질 실패인지 구분할 수 있는가?
3. 한 건의 좋은 답을 전체 품질 개선으로 일반화하지 않았는가?

시연이 승인되지 않으면 강의자 요청 0건이 정상이다. 학습자 40건 실행 승인은 별도로 확인한다.
학습자 실행 승인이 없거나 provider 장애로 완결되지 않으면 저장 예시 분석은 계속하되, 개인
비교 완료로 표시하지 않고 승인된 보충 실행 시간을 정한다.

## 8. 제출 — 세 줄로 주장 범위 정리하기

`local-data/learning-progress.md`의 Week 2 표를 다음 형식으로 채운다.

| 비교 | 바뀐 것 | 관찰 | 말할 수 없는 것 |
| --- | --- | --- | --- |
| 저장 baseline·내 prompt 40건 |  |  |  |
| NIM Gemma·AI Studio Gemini |  |  |  |
| 품질 실패·API 오류 |  |  |  |

## 완료 기준

- 한 사례의 개선과 40건의 회귀 비교가 다른 역할임을 설명했다.
- 학생별 지시문과 비교 결과를 학생별 고유 경로에 저장했다.
- 내 40건 `summary.json`의 `observed_status=complete`와 40개
  `observations`·`results`를 확인했다.
- 같은 release baseline 비교의 `invalid_reasons`가 비어 있는지 확인하고 내
  새 성공·새 실패·남은 실패를 기록했다.
- 호출 경로 묶음의 차이를 모델 하나의 효과라고 주장하지 않았다.
- API 오류를 모델 오답으로 세지 않았고, 말할 수 없는 범위를 표에 적었다.
