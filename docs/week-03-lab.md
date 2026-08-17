# Week 3 실습 — 실제 답 두 개를 LLM Judge로 비교하기

## 이번 주에 배우는 것

설명형 답은 뜻이 같아도 표현이 다를 수 있다. 수치 일부만 맞거나 중요한 비교를 빠뜨린 답도
단순 문자열 비교만으로는 다루기 어렵다. Week 3에서는 같은 차트와 질문에 두 지시문을 적용해
NVIDIA NIM의 Gemma로 실제 답을 만들고, 사람과 Gemini Judge가 같은 두 답을 비교한다.

OpenCQA 사람이 작성한 `abstractive_answer`는 **기대 답(reference)**이다. 후보 A나 B로 쓰지
않는다. 후보는 각 학습자가 같은 Gemma 모델에 baseline과 improved prompt를 각각 적용해 만든다.
모델과 입력을 고정하고 prompt만 바꾸므로, 두 답의 차이를 prompt 변화와 연결해 볼 수 있다.

수업이 끝나면 다음 네 문장을 설명할 수 있어야 한다.

1. 고정 규칙 채점과 LLM Judge가 각각 맡아야 할 문제가 다르다.
2. 작업 모델의 답과 사람이 쓴 기대 답은 역할이 다르다.
3. Judge 결과를 보기 전에 사람 판단을 고정해야 결과에 맞춰 답을 바꾸는 일을 막을 수 있다.
4. 후보 순서 교환과 같은 입력 반복은 서로 다른 종류의 흔들림을 찾는다.

## 1. 무엇을 만들고, 왜 두 모델을 나누는가

### 한 학습자의 전체 흐름

```text
OpenCQA 차트·질문 30개
→ NIM Gemma + baseline prompt로 답 30개
→ 같은 NIM Gemma + improved prompt로 답 30개
→ 답 두 개를 출처가 보이지 않는 A/B 30쌍으로 묶음
→ 배정된 한 쌍을 사람이 먼저 판단하고 파일 SHA-256을 잠금
→ Gemini 3.5 Flash Lite가 30쌍을 A/B·B/A 순서로 두 번씩 판단
→ 사람 판단·순서 충돌·반복 충돌·API 오류를 비교
```

Gemma는 업무 답을 만드는 **작업 모델**이고, Gemini는 두 업무 답을 비교하는 **Judge 모델**이다.
같은 모델이 자기 답을 평가하지 않도록 역할과 API 제공자를 분리한다. Gemini 3.5 Flash Lite는
현재 잠긴 LiteLLM adapter로 요청 모델·실제 처리 모델과 구조화 출력을 확인할 수 있는 Judge
모델이다.

### 고정 규칙과 Judge의 역할

| 확인할 것 | 맡길 방법 | 이유 |
| --- | --- | --- |
| JSON 형식·ID·숫자 범위·실행 누락·hash | Python 고정 규칙 | 답이 정해져 있고 같은 입력에 같은 판정이 필요함 |
| 설명의 사실성·질문 적합성·핵심 비교 누락 | 사람과 LLM Judge | 문맥을 고려해 여러 표현의 의미를 비교해야 함 |

Judge는 모든 채점을 대신하는 만능 도구가 아니다. 코드로 확실히 검사할 수 있는 항목까지
Judge에 맡기면 재현 가능한 오류도 확률적으로 판정하게 된다.

고정 Judge 기준은 네 단계다. 차트·질문을 먼저 보고 기대 답은 보조 기준으로만 쓰며, 수치와
비교 대상의 정확성·중요한 누락·차트 밖 주장을 확인한다. 사실성과 완결성이 같으면 `tie`,
순서만 바뀌어 선택이 달라지면 `review`로 보낸다. 실제 문구는
[`week-03-judge-rubric.yaml`](../configs/week-03-judge-rubric.yaml)에 고정돼 있다.

### 왜 순서를 바꾸고 반복하는가

| 검사 | 무엇을 바꾸는가 | 찾으려는 문제 |
| --- | --- | --- |
| A/B → B/A | 같은 두 답의 화면 위치 | 앞이나 뒤에 놓인 답을 선호하는 위치 민감도 |
| trial 1 → trial 2 | 아무 조건도 바꾸지 않음 | 같은 입력에도 결과가 달라지는 출력 변동 |

한 pair에는 `trial 1 A/B`, `trial 1 B/A`, `trial 2 A/B`, `trial 2 B/A`의 네 판단이 생긴다.
한 trial에서 순서만 바꿨는데 실제 후보 선택이 달라지면 **순서 충돌**, 같은 순서의 두 trial이
다르면 **반복 충돌**이다. 충돌이 하나라도 있으면 다수결로 결론을 만들지 않고 `review`로 보낸다.
일관된 결과도 정답이라는 뜻은 아니다. 같은 방식으로 일관되게 틀릴 수 있다.

## 2. 실습 준비와 승인 확인

받은 저장소의 최상위에서 실행한다. 상위 과정 폴더는 필요하지 않다.

```bash
uv sync --locked --dev
```

### OpenCQA 원본 받기

[OpenCQA](https://github.com/vis-nlp/OpenCQA)는 차트 이미지와 설명형 질문·사람 작성 답을 제공하는
공개 데이터다. 이 수업은 GPL-3.0 원본의 고정 revision에서 30개를 골라 development 18개,
validation 6개, test 6개로 나눈다. 사람 작성 `abstractive_answer`는 Judge가 참고할 기대 답이며
Gemma 후보로 쓰지 않는다. 자세한 필드와 라이선스는 [OpenCQA 데이터 준비](open-cqa-data.md)에서
확인한다.

처음 한 번만 공식 저장소를 프로젝트 옆에 받고, 수업에서 검증한 revision으로 이동한다.
전체 clone은 현재 약 2GB이므로 수업 전에 네트워크와 빈 디스크 공간을 확인한다.

```bash
git clone https://github.com/vis-nlp/OpenCQA.git ../OpenCQA
git -C ../OpenCQA checkout 28db0fd26a12fd376f6c30b7feb8a4db32313424
uv run --locked python scripts/prepare_opencqa.py --source-root ../OpenCQA
```

준비 명령은 선택한 차트 30개를 JPEG로 바꾸고, 질문·기대 답·출처와 이미지 SHA-256을
`local-data/opencqa/`에 기록한다. 이미 clone한 경우에는 `git clone`을 다시 하지 않고
`checkout`과 준비 명령만 실행한다. 다음 입력과 수업 파일이 모두 보여야 다음 단계로 간다.

```bash
ls -d local-data/opencqa/images
ls local-data/opencqa/week-03-cases.jsonl \
  prompts/open-cqa-answer-{baseline,improved}.md \
  configs/google-gemini-3.5-flash-lite-judge.yaml \
  templates/judge-human-label-template.yaml
```

파일이 없으면 수치를 추정하거나 다른 자료로 바꾸지 말고 튜터에게 요청한다. 과정용 별칭과
튜터가 배정한 1부터 30 사이 번호를 정한다.

```bash
STUDENT_ALIAS="course-alias"
PAIR_NUMBER=1

WEEK3_STUDENT_DIR="local-data/week-03-student-judges/$STUDENT_ALIAS"
HUMAN_LABEL="$WEEK3_STUDENT_DIR/human-label.yaml"
INTERPRETATION="$WEEK3_STUDENT_DIR/interpretation.md"
WEEK3_RUN_DIR="reports/week-03/student-full/$STUDENT_ALIAS-$(date +%Y%m%d-%H%M%S)"
CANDIDATE_RUN_DIR="$WEEK3_RUN_DIR/candidates"
JUDGE_RUN_DIR="$WEEK3_RUN_DIR/judge"
mkdir -p "$WEEK3_STUDENT_DIR"
```

`course-alias`는 본인 별칭으로 바꾸고 공백이나 `/`는 쓰지 않는다. `PAIR_NUMBER`에는 튜터가
배정한 1~30 번호를 쓴다. 이후 명령은 이 변수가 남아 있는 같은 터미널에서 실행한다.

`.env`가 없다면 `.env.example`을 복사해 본인의 `NVIDIA_NIM_API_KEY`와 `GEMINI_API_KEY`를
입력한다. key는 제출하거나 화면에 출력하지 않는다.

```bash
test -e .env || cp .env.example .env
```

두 실제 실행은 별도 승인을 받는다.

| 단계 | 외부로 보내는 자료 | 개인 상한 |
| --- | --- | --- |
| NIM Gemma 후보 생성 | OpenCQA JPEG·질문·각 prompt | 요청·attempt 60/60, 입력 1,200,000 token, 출력 30,000 token, 비용 안전장치 $0.01, 7,200초, 재시도 0 |
| Gemini 3.5 Flash Lite Judge | OpenCQA JPEG·질문·기대 답·익명 Gemma 후보 A/B·고정 rubric | 실제 요청 120~240회, 요청·attempt 최대 240/240, 입력 1,200,000 token, 출력 120,000 token, 비용 안전장치 $0.01, 10,800초, 요청당 재시도 1회 |

여기서 요청은 모델 판단을 부탁한 횟수이고, attempt는 전송 재시도까지 포함해 API에 실제로
시도한 횟수다. 후보 생성은 재시도가 없어 둘이 60으로 같다. NIM 설정의 명목 token 비용은
0달러이고 0.01달러는 예상 청구액이 아니라 설정 변경을 막는 안전장치다. 모델 목록을 확인하는
사전 점검은 추론 60회와 별도의 metadata 조회 1회다.

NVIDIA와 Google에 전송되는 자료, 계정별 model·quota·가격과 무료/유료 tier를 각각 확인한다.
API key와 `.env`는 공유하거나 제출하지 않는다. OpenCQA JPEG와 Gemma 답을 Google로 보내는
범위는 2026-08-17에 승인됐다. 다만 Free Tier에서는 보낸 자료가 제품 개선에 사용될 수 있다.
이 조건을 허용할 수 없거나 현재 프로젝트가 Free Tier가 아니면 실행하지 않는다.

RPM은 분당 요청, TPM은 분당 token, RPD는 하루 요청 수다. 승인 때 확인한 공개 Free Tier
한도는 15 RPM, 입력 250,000 TPM, 500 RPD다. 실행 당일에는
Google AI Studio에서 **지금 API key가 속한 프로젝트**의 한도를 다시 확인한다. 수업 코드는
15 RPM, 요청당 입력 5,000 token·출력 500 token, 한 full run당 최대 240요청으로 더 좁게
막는다. 따라서 분당 잠재 상한은 입력 75,000 TPM·출력 7,500 TPM이다. 코드는 하루 누적 요청을
추적하지 않는다. 실행 전에 현재 프로젝트의 당일 잔여 RPD가 240건 이상인지 확인한다. 같은
프로젝트에서 N명이 실행하면 최대 240N 요청이 되어 500 RPD를 넘을 수 있으므로 개인 프로젝트를
쓰거나 실행 시간을 나눈다. 입력·출력 단가는 0달러로 계산하고, 예상치 못한 설정 변경을 막는
비용 안전장치는 0.01달러로 둔다. 전송 승인은 실행 성공을 보장하지 않는다.

## 3. NIM Gemma 실제 답 60개 만들기

같은 `nvidia_nim/google/gemma-4-31b-it` 모델과 같은 30개 입력에 baseline과 improved prompt를
각각 적용한다. OpenCQA 기대 답은 작업 모델에 보내지 않는다. 실행 코드는 prompt 두 파일을
결과 폴더에 snapshot으로 보존하고, 각 case의 두 답을 A/B에 고정된 방식으로 익명 배치한다.

improved prompt는 baseline에 질문의 대상·기간·범주·단위 확인, 범례·축 확인, 비교 방향 재검사,
차트 수치 재확인을 추가한다. 모델·차트·질문은 그대로 두고 이 확인 절차만 바꾸는 실험이다.
development·validation·test의 30개를 고정 prompt로 한 번 평가하고, 어느 분할의 결과를 본 뒤에도
prompt를 고쳐 같은 실행을 반복하지 않는다. 이 작은 공개 분할을 실제 배포 성능의 보장으로
해석하지 않는다.

이 단계에서 확인할 것은 세 가지다.

1. 같은 30개 입력에 baseline·improved prompt를 적용해 답을 30개씩 만든다.
2. 두 결과를 익명 A/B로 고정해 이후 Judge의 비교 조건을 통제한다.
3. 수집 완료 상태와 후보 출력 품질을 구분해 읽는다.

### 검증된 실제 실행 예시

2026-08-17에 같은 명령으로 실행한 결과다. 이 표는 집계만 보여 주며,
어떤 pair와 후보가 출력 계약을 어겼는지는 사람 판단을 잠그기 전에 공개하지 않는다.

| 확인 항목 | 실제 결과 |
| --- | --- |
| 수집 상태 / 품질 상태 | `observed_status=complete` / `status=fail` |
| 후보·호출 | 30쌍, 요청·attempt·call 각 60회, 재시도 0회 |
| 출력 계약 | 통과 58개, `invalid_output` 2개 |
| 실제 처리 모델 | `google/gemma-4-31b-it` |
| 실제 token | 입력 35,732, 출력 9,502 |
| 실제 비용·시간 | `$0.00`, 약 510초 |
| 후보 세트 SHA-256 | `6818f0748262d6c2f042847df960dfdae2028da45cafb2c2717d4cebe84dfd00` |

이 결과는 60회 수집이 완결됐다는 증거이지만 후보 품질이 모두 통과했다는 증거는 아니다.
실제 원본은 튜터가 별도로 보존하며, 학습자는 자신의 고유 실행 폴더에서 같은 항목을 확인한다.

실행 당일 [NVIDIA NIM 이용 안내](https://docs.api.nvidia.com/nim/docs/product)의 가격·이용 조건도
확인하고 날짜를 기록한다. 다음 명령은 최대 60회만 요청하며, 중단된 폴더에 이어 쓰지 않는다.

```bash
unset CATALOG_DATE PRICING_DATE
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/week-03-candidates.yaml &&
  CATALOG_DATE=$(date +%F) &&
  PRICING_DATE="$CATALOG_DATE"
```

모델 목록 확인이 성공했을 때만 다음 명령을 실행한다.

```bash
uv run --locked python scripts/run_open_cqa_candidates.py \
  --live-task --pair-limit 30 \
  --max-requests 60 --max-attempts 60 --max-retries 0 \
  --max-input-tokens 1200000 --max-output-tokens 30000 \
  --max-cost-usd 0.01 --max-wall-seconds 7200 \
  --catalog-verified-on "$CATALOG_DATE" --pricing-verified-on "$PRICING_DATE" \
  --output "$CANDIDATE_RUN_DIR"
```

명령이 끝나면 수집 상태와 후보 품질을 따로 읽는다.

```bash
rg -n '"(status|observed_status|completed_pair_count|actual_request_count|invalid_output_count)"' \
  "$CANDIDATE_RUN_DIR/candidate-summary.json"
```

| 확인 항목 | 완료 기준 |
| --- | --- |
| 수집 상태 | `observed_status=complete` |
| 후보 수 | `completed_pair_count=30` |
| 실제 호출 | `actual_request_count=60` |
| 후보 품질 | `status`와 `invalid_output_count`를 함께 해석 |

60회가 끝났어도 출력 계약을 어긴 답이 있으면 `status=fail`로 남는다. 이 답은 고쳐 쓰거나
다시 호출하지 않고 원문을 Judge 후보로 사용한다. 명령이 중단되면 폴더를 보존하고 같은 폴더에
이어 쓰지 않는다. 실행 스크립트가 Git 상태·결과 폴더·모델·입력·prompt·hash와 상한을 검사한다.
`observed_status=complete`면 `candidate_collection_status=complete`, 요청 전에 중단돼 폴더가 없으면
`not_run`, 폴더가 남았지만 완결되지 않았으면 `partial`로 기록한다. `complete`가 아니면 사람
판단과 Gemini Judge를 시작하지 않는다.

## 4. 결과를 보기 전에 사람이 한 쌍을 판단하기

`candidate-summary.json`의 `observed_status=complete`를 확인한 경우에만 배정된 차트·질문·후보
A/B를 연다. 이때 A/B가 baseline인지 improved인지, 기대 답, Gemini 결과는 보지 않는다.

```bash
CANDIDATE_RESULTS="$CANDIDATE_RUN_DIR/candidate-results.jsonl"
uv run --locked python scripts/inspect_judge_pair.py \
  --candidates "$CANDIDATE_RESULTS" \
  --number "$PAIR_NUMBER"
```

출력의 `[평가표 ID]`를 이후 조회에 쓸 변수로 기록한다.

```bash
PAIR_ID="opencqa-val-..."
```

터미널에 표시된 차트 경로를 편집기의 파일 탐색기에서 열어 이미지를 직접 본다. 판단 순서는
다음과 같다.

1. 질문의 대상·기간·단위와 비교할 것을 찾는다.
2. 차트에서 필요한 값과 증가·감소 방향을 확인한다.
3. A와 B의 수치·대상·기간·단위·핵심 누락·차트 밖 주장을 표시한다.
4. 더 정확하고 완결적인 후보를 고른다. 같은 의미로 맞거나 같은 정도로 틀릴 때만 `tie`를
   고른다.
5. 이유를 `차트 사실 → 후보 차이 → label` 순서로 쓴다.

개인 양식을 한 번만 복사한다.

```bash
test -e "$HUMAN_LABEL" || cp templates/judge-human-label-template.yaml "$HUMAN_LABEL"
```

다음 여섯 필드를 채운다.

| 필드 | 내용 |
| --- | --- |
| `pair_number` | 배정받은 번호 |
| `pair_id` | 검사 결과의 `opencqa-val-...` |
| `candidate_set_sha256` | 검사 결과의 후보 세트 SHA-256 |
| `reviewer_id` | 실명 대신 과정용 별칭 |
| `label` | `candidate_a / tie / candidate_b` 중 하나 |
| `reason` | 차트 사실과 두 후보의 중요한 차이를 연결한 이유 |

“B가 더 길다”, “A가 좋아 보인다”, “Judge도 B를 고를 것 같다”는 근거가 아니다.
[사람 사전 label 양식](../templates/judge-human-label-template.yaml)의 주석에서 작성 규칙을
확인할 수 있다.

SHA-256은 품질 점수가 아니라 내용이 같으면 같은 값이 나오는 파일 지문이다. 후보 세트
SHA-256은 사람 판단이 어느 A/B 묶음에 대한 것인지 고정하고, 아래 사람 label SHA-256은 Judge
결과를 본 뒤 판단 파일을 바꾸지 않았는지 확인하는 데 쓴다.

번호·ID·후보 세트와 파일 형식을 검사한 뒤 `human-label.yaml`의 SHA-256을 기록한다.

```bash
uv run --locked python scripts/inspect_judge_pair.py \
  --candidates "$CANDIDATE_RESULTS" \
  --number "$PAIR_NUMBER" \
  --human-label "$HUMAN_LABEL"
HUMAN_LABEL_SHA256=$(shasum -a 256 "$HUMAN_LABEL" | awk '{print $1}')
printf 'human_label_sha256=%s\n' "$HUMAN_LABEL_SHA256"
```

이 값을 튜터에게 제출한 다음 파일을 수정하지 않는다. 이미지나 입력이 손상돼 판단할 수 없으면
임의로 `tie`를 쓰지 말고 pair ID를 튜터에게 알린다.

## 5. Gemini Judge가 순서와 반복에 흔들리는지 확인하기

사람 label의 SHA-256을 잠근 뒤에만 Judge를 시작한다. 개인 full Judge 명령에는 사람 label을
넣지 않고, 실행이 끝난 뒤 비교 명령에서 연결한다. 후보 생성과 Judge의 Git SHA가 다르면
실행기가 요청 전에 중단한다. Gemini는 OpenCQA JPEG·질문·사람이 쓴
기대 답과 익명 Gemma 후보 A/B를 함께 본다. 사람 label, reviewer ID와 후보의 baseline/improved
출처는 보내지 않는다.

이 단계에서 확인할 것은 다섯 가지다.

1. baseline과 improved 가운데 어느 prompt의 답이 더 자주 선택되는가?
2. 후보의 A/B 위치를 바꾸면 판단이 달라지는가?
3. 같은 평가를 두 번 하면 판단이 달라지는가?
4. Judge 결과가 미리 잠근 사람 판단과 일치하는가?
5. 결과가 안정적이어도 왜 자동 승인 기준으로 쓸 수 없는가?

실행 직전에 [Google AI Studio 한도](https://ai.google.dev/gemini-api/docs/rate-limits)의 현재
프로젝트에서 Free Tier, Gemini 3.5 Flash Lite,
15 RPM·입력 250,000 TPM·500 RPD와 당일 잔여 RPD 240건 이상을 확인한다. 가격표의 입력·출력
0달러와 데이터 이용 조건도 [공식 가격표](https://ai.google.dev/gemini-api/docs/pricing)에서 같은
날 다시 읽는다. 하나라도 다르면 아래 명령을 실행하지 않고
`not_run`으로 남긴다. 이 확인은 공개 표의 숫자만 읽는 것으로 대신할 수 없다.

확인을 마친 오늘 날짜를 다시 기록한다. 후보 생성 다음 날 Judge를 실행한다면 반드시 새 날짜가
된다.

```bash
CATALOG_DATE=$(date +%F)
PRICING_DATE=$(date +%F)
```

아래에서는 세 가지만 한다. 먼저 Gemini가 30쌍을 판단하고, 그 결과에 미리 잠근 사람 판단을
연결한 다음, 해석에 필요한 수치만 화면에 표시한다.

```bash
uv run --locked python scripts/run_open_cqa_judge.py \
  --live-judge --candidate-run "$CANDIDATE_RUN_DIR" --pair-limit 30 \
  --max-requests 240 --max-retries 1 \
  --max-input-tokens 1200000 --max-output-tokens 120000 \
  --max-cost-usd 0.01 --max-wall-seconds 10800 \
  --catalog-verified-on "$CATALOG_DATE" --pricing-verified-on "$PRICING_DATE" \
  --output "$JUDGE_RUN_DIR"
```

이 명령이 중단되면 다음 비교 명령을 실행하지 않는다. 성공했을 때만 잠근 사람 판단을 연결한다.

```bash
uv run --locked python scripts/compare_open_cqa_judge.py \
  --candidate-run "$CANDIDATE_RUN_DIR" \
  --judge-results "$JUDGE_RUN_DIR/judge-results.jsonl" \
  --human-label "$HUMAN_LABEL" --human-label-sha256 "$HUMAN_LABEL_SHA256" \
  --output "$JUDGE_RUN_DIR/comparison.json"
```

두 명령이 모두 끝나면 아래 값만 읽는다. 첫 명령은 실행 완결 여부를, 둘째 명령은 Judge의 선택과
흔들림을 보여 준다.

```bash
rg -n '"(status|observed_status|completed_trial_count|actual_request_count)"' \
  "$JUDGE_RUN_DIR/summary.json"
rg -n '"(baseline_wins|improved_wins|ties|reviews|order_conflicts|repetition_conflicts|individual_human_agreement|blocking_eligible)"' \
  "$JUDGE_RUN_DIR/comparison.json"
```

각 pair를 두 trial로 반복하고, 각 trial에서 A/B와 B/A를 모두 평가한다. 한 평가가 구조화된
선택과 이유를 만들기 위해 한 번 또는 두 번의 요청을 쓰므로 실제 요청 수는 120~240회다.
`completed_trial_count=60`이면 30쌍의 두 trial이 모두 저장된 것이다.
15 RPM pacing만 계산하면 약 8~16분이고 API·네트워크 응답 시간이 더해진다. 실행 스크립트는
요청·attempt 240/240, 입력 1,200,000 token, 출력 120,000 token, 비용 안전장치 0.01달러,
10,800초와 요청당 재시도 1회를 넘기기 전에 중단한다.

여기서 `status=pass`는 Judge 호출과 결과 구조가 완결됐다는 뜻이지 improved prompt가 더
좋다는 뜻이 아니다. 어느 답이 더 많이 선택됐는지는 아래 승패·충돌 값을 따로 읽는다.

| 결과 | 학습자가 해석할 내용 |
| --- | --- |
| `baseline_wins / improved_wins` | 어느 prompt의 답을 더 자주 선택했는가 |
| `ties / reviews` | 동률 결론과 충돌 때문에 사람 검토로 보낸 수가 몇 개인가 |
| `order_conflicts` | 후보 위치가 바뀌자 판단도 달라졌는가 |
| `repetition_conflicts` | 같은 평가를 반복했을 때 판단이 달라졌는가 |
| `individual_human_agreement` | `1`은 일치, `0`은 불일치, `null`은 `review` 등으로 비교할 수 없음 |
| `blocking_eligible=false` | 이 결과만으로 자동 승인할 수 없는 이유 |

같은 release의 수업 리허설에서는 Gemini 요청 234회로 60 trial을 완료했다. baseline 승 3건,
improved 승 5건, tie 1건, review 21건이었고 순서 충돌 21건·반복 충돌 14건이었다. 이 수치는
정답표가 아니라 Judge가 얼마나 흔들리는지 읽는 실제 예시다.

Judge 실행기는 후보·모델·rubric·Git·hash·30쌍·60 trial·상한을 검사하고, 비교기는 잠근 사람
label의 SHA-256까지 검사한다.
두 명령이 끝나고 `status=pass`, `observed_status=complete`이면 `complete`로 기록한다. 어느 명령이든
중단되면 결과 폴더를 보존하고 비교나 품질 해석을 계속하지 않는다. 폴더가 생긴 뒤 중단되면
`partial`, 사전 확인을 통과하지 못해 요청과 폴더가 없으면 `not_run`이다. 원하는 label이 나올
때까지 다시 실행하지 않는다. 과거 NIM Gemma Judge 저장 결과는 새 완료 근거가 아니다.

## 6. 사람 판단과 Judge 결과 비교하기

Judge가 `complete`일 때만 배정된 같은 `pair_id`를 `judge-results.jsonl`과 `comparison.json`에서
찾는다.

```bash
uv run --locked python scripts/inspect_judge_pair.py \
  --candidates "$CANDIDATE_RESULTS" --number "$PAIR_NUMBER" \
  --human-label "$HUMAN_LABEL" --reveal
rg -n "$PAIR_ID" "$JUDGE_RUN_DIR/judge-results.jsonl"
rg -n -A 14 "$PAIR_ID" "$JUDGE_RUN_DIR/comparison.json"
```

| 반복 | A/B에서 고른 실제 후보 | B/A에서 고른 실제 후보 |
| --- | --- | --- |
| trial 1 |  |  |
| trial 2 |  |  |

그다음 다음 항목을 확인한다.

1. 같은 trial의 두 값이 다르면 순서 충돌인가?
2. 같은 순서에서 trial 1과 trial 2가 다르면 반복 충돌인가?
3. 충돌이 있으면 결론을 `review`로 두었는가?
4. 충돌이 없으면 Judge 결론과 잠근 사람 label이 같은가?
5. baseline과 improved 가운데 어느 답이 선택됐고, 그 차이를 기대 답과 차트가 뒷받침하는가?
6. API 오류나 실제 처리 모델 불일치를 품질 차이로 해석하지 않았는가?

사람과 Judge의 판단이 같다고 해서 사람 판단이 정답이 되거나 Judge 보정이 끝난 것은 아니다.
한 사람의 한 pair는 전체 30쌍의 사람 보정 자료도 아니다. 과거 Codex 합성 기준과 그 기준으로
계산한 수치는 다른 후보·Judge로 만든 legacy 결과이므로 새 비교의 기준값으로 쓰지 않는다.

이번 활동은 후보 길이를 의도적으로 바꾸는 실험이 아니다. 장황한 답 선호를 측정했다고
주장하지 않는다. 실제로 확인한 범위는 같은 개인 실행에서의 prompt별 답 차이, 위치 민감도,
반복 변동과 한 사람의 사전 판단이다.

## 7. 제출

실행이 `complete`라면 다음 **두 파일만 제출**한다.

1. `$HUMAN_LABEL`: 실제 파일 `human-label.yaml`
2. `$INTERPRETATION`: 실제 파일 `interpretation.md`

실행 결과 9개와 그 경로는 제출하지 않는다. 수강생은 `$WEEK3_RUN_DIR`을 본인 컴퓨터에
그대로 보존한다.

- `candidates/`: `candidate-calls.jsonl`, `candidate-results.jsonl`,
  `candidate-summary.json`, `open-cqa-answer-baseline.md`, `open-cqa-answer-improved.md`
- `judge/`: `judge-calls.jsonl`, `judge-results.jsonl`, `summary.json`, `comparison.json`

해석문 양식을 한 번만 복사한 뒤 각 항목을 쓴다.

```bash
test -e "$INTERPRETATION" || cp templates/week-03-interpretation.md "$INTERPRETATION"
```

양식에는 pair와 실행 상태, 차트 사실, prompt별 답 차이, 네 Judge 결과와 충돌, 사람 판단과의
일치 여부, 이 실행만으로 주장할 수 없는 범위가 순서대로 적혀 있다.

번호와 ID는 `human-label.yaml`에 잠근 값과 같게 쓴다. 실행을 완료하지 못했을 때 제출 범위는
다음과 같다.

| 실행 상태 | 제출 항목 |
| --- | --- |
| `complete` | `human-label.yaml`, `interpretation.md` |
| `partial` | `interpretation.md`, 이미 만들었다면 `human-label.yaml` |
| `not_run` | 중단 이유와 상태를 적은 `interpretation.md` |

없는 결과를 legacy 저장 결과로 대체하지 않는다. API key, `.env`, OpenCQA 원본은 제출하지
않는다.

## 완료 기준

- 같은 Gemma 모델과 입력에 baseline·improved prompt만 바꿔 실제 답을 30개씩 만들었다.
- OpenCQA `abstractive_answer`를 기대 답으로만 쓰고 후보로 사용하지 않았다.
- 후보 출처·기대 답·Judge 결과를 보기 전에 배정된 한 쌍의 사람 label을 SHA-256으로 잠갔다.
- 승인된 Gemini 3.5 Flash Lite 상한 안에서 30쌍·두 trial·A/B와 B/A를 완료했다.
- 후보와 Judge의 호출·결과·요약·비교, prompt snapshot과 직접 쓴 두 파일을 보존했다.
- 안정성·사람 일치·정답·사람 보정 완료가 서로 다른 개념임을 해석문에 적었다.
- 어느 단계든 `partial` 또는 `not_run`이면 전체 완료로 표시하지 않고 보완 상태를 남겼다.

Google 전송 범위는 승인됐지만 새 full 실행이 성공했다는 뜻은 아니다. 현재 프로젝트 사전
확인이나 실행 완결 검사를 통과하지 못했다면 `not_run` 또는 `partial`로 남긴다. legacy 저장
결과로 완료 기준을 충족했다고 표시하지 않는다.

## 선택 읽기

- [OpenCQA](https://aclanthology.org/2022.emnlp-main.811/): 차트 기반 설명형 답 평가
- [G-Eval](https://aclanthology.org/2023.emnlp-main.153/): 단계가 있는 평가 기준
- [MT-Bench·Chatbot Arena](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html):
  LLM Judge의 위치·장황함 편향
