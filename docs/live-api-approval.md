# 실제 API 외부 전송 확인표

실제 API 실습은 문서 페이지·차트 이미지를 모델 제공사의 서버로 보낸다. 실행 전에는 사람이 아래 항목을
확인한다. 모델·접속 주소·가격 근거·키 이름은 아래 실제 설정 파일을 기준으로 삼으며, 실행 코드도
이 파일을 읽는다.

## Week 1–3 기준 파일

| 실습 | 실행 설정 |
| --- | --- |
| Week 1 Nemotron | `configs/nvidia-nim.yaml` |
| Week 2 Gemma 기준 지시문 | `configs/nvidia-nim-gemma4-baseline.yaml` |
| Week 2 Gemma 개선 지시문 | `configs/nvidia-nim-gemma4.yaml` |
| Week 2 NIM Gemma–AI Studio Gemini 경로 비교 | `configs/week-02-live.yaml` |
| Week 3 NIM Gemma 후보 생성 | `configs/week-03-candidates.yaml` |

요청·토큰·비용·시간·재시도 상한은 [Week 1 실습](week-01-lab.md),
[Week 2 실습](week-02-lab.md), [Week 3 실습](week-03-lab.md)의 실제 명령을 따른다. Week 1은 해당
학습자 문서의 계약대로 진행하고, Week 2는 저장 예시 분석과 학습자별 40건 full live를
함께 한다. 설정과 runbook 명령이 다르면 호출하지 말고 먼저 둘을 맞춘다. 별도 승인 YAML을
실행 코드가 읽는다고 가정하지 않는다.

NVIDIA 설정은 `developer_program_free_endpoint`, 수업용 비용 계산값 0달러와 공식 상품 안내
URL을 기록한다. 이 값은 NVIDIA가 공개한 token 단가가 아니다. 실행 당일 개발 endpoint 이용
조건, 모델 목록과 계정 할당량을 다시 확인한다.

Google AI Studio 경로는 `free_tier`다. 공식 가격표상 무료 입력·출력은 0달러지만 요청이
제품 개선에 사용될 수 있다. 공개·합성·비식별 자료만 보내고 실행 당일 가격표와 계정의
무료 한도를 다시 확인한다.

### Week 2 역할과 합산 승인

Week 2의 실제 NIM 요청은 세 역할을 섞지 않는다.

| 역할 | 호출 범위 | 목적 |
| --- | --- | --- |
| 튜터 수업 전 baseline | 같은 release에서 40건 한 번 | 학습자 후보와 비교할 공통 기준 |
| 각 학습자 | 학생별 prompt로 40건 한 번 | 저장 baseline과 개인 후보의 prompt-only 비교 |
| 강의자 수업 시연 | `r01` 최대 1건 | 원응답·actual model·저장 형식 확인 |

같은 release에서 이미 완결된 baseline이 있으면 튜터는 40건을 다시 호출하지 않는다. 학습자
수가 `N`명이면 새 요청 상한은 baseline 준비가 필요할 때
`40 + 40×N + 1`, 재사용할 수 있을 때 `40×N + 1`이다. 강의자 시연이
승인되지 않으면 마지막 1건은 0건이다. 공유 credential을 쓸 때는 runner별 상한만 보지 말고
계정의 동시 요청·분당 요청·전체 할당량과 학습자별 0.01달러 상한의 합도 승인한다. 필요하면
학습자를 wave로 나눈다.

학습자 prompt는 `local-data/week-02-students/$STUDENT_ALIAS/prompt.md`, 비교 결과는
`reports/week-02/students/$STUDENT_ALIAS/` 아래에 둔다. 기존
`local-data/week-02-prompt.md`는 강의자 `r01` 시연 호환용이며 학습자 full
run에 공동으로 쓰지 않는다.

### 모델·가격 확인 날짜

실행 당일 설정의 공식 URL에서 billing basis와 이용 조건을 확인한다. provider에 공개 단가가
있으면 입력·출력 단가도 설정값과 비교한다. 값이 달라졌으면 설정을 고쳐 commit하고,
같다면 소스 파일은 바꾸지 않는다. 두 항목을 실제로 확인한 날짜만 live 명령에 넘긴다.

```bash
CATALOG_DATE=$(date +%F)
PRICING_DATE=$(date +%F)
```

모든 live 명령은 `--catalog-verified-on "$CATALOG_DATE"`와
`--pricing-verified-on "$PRICING_DATE"`를 모두 요구한다. Week 1–2 실행기는 7일 이내 확인만
받고, Week 3 후보·Judge 실행기는 실행 당일 날짜만 받는다.

## Week 1–2 전송 범위

| 보낸다 | 보내지 않는다 |
| --- | --- |
| AIHub PDF에서 만든 페이지 JPEG | PDF 추출 문장 |
| 한국어 질문 | 기대 정답과 채점 결과 |
| 구조화 답변용 지시문(prompt) | 다른 API 제공자의 응답 |
|  | API 키 |

VLM이 페이지 JPEG를 직접 읽는다. PDF 추출 문장은 모델 입력이나 채점에 넣지 않는다. AIHub
이용정책과 각 API 제공자의 데이터 이용 조건을 확인한 뒤 실행한다.

## Week 1–2 실행 전 확인

1. 실행할 설정 파일에서 API 제공자, 요청 모델, 실제 처리 예상 모델, 접속 주소와 키 환경
   변수 이름을 읽는다.
2. 사전 점검에서 요청 모델이 현재 목록에 있는지 확인한다.
3. 설정의 가격 근거 URL과 단가·billing basis, 계정 할당량을 당일 확인하고 두 확인
   날짜를 live 명령에 넘긴다.
4. `.env`에 필요한 키가 있고 Git과 화면 공유에 포함되지 않는지 확인한다.
5. 실습 명령에 요청·입력 token·출력 token·비용·시간·재시도 상한이 모두 있는지 확인한다.
6. 전체 실행은 `git status --short` 출력이 없을 때만 하고 probe는 새 결과 폴더를 쓴다.
7. Week 2 baseline의 `provenance.git_sha`가 학습자 release commit과 같고,
   비교기가 workflow·dataset·lockfile·schema·scorer hash를 확인할 수 있는지 검사한다.
8. 강의자 probe에서 원응답, 실제 처리 모델, 사용량과 오류를 확인한다. 학습자 full live는
   별도로 승인한 wave와 개인 상한에서만 시작한다.

Google AI Studio 경로의 `free_tier`와 0달러 계산은 해당 API key가 실제 무료 tier일 때만
맞는다. 실행 당일 계정 tier와 가격표의 데이터 이용 항목(`Used to improve our products`)을
확인한다. 제출 데이터의 제품 개선 이용이 허용되지 않거나 유료 tier라면 현재 0달러 설정으로
호출하지 않는다.

probe 일부가 성공해도 전체 품질이 좋다고 결론 내리지 않는다. Git에 포함된 고정·재생
응답은 시험 전용 증거(`test_only`)다. 대체 경로와 재실행을 끈 실제 API 원본은 저장되어
있어도 실제 품질 증거(`live_quality`)다.

Week 2 학습자 한 명의 full live 상한은 요청·attempt 40/40, 입력 800,000 token, 출력
20,000 token, 비용 안전장치 0.01달러, 7,200초, 재시도 0이다. `summary.json`의
`observed_status=complete`와 40개 원응답·결과가 있어야 실행이 완결된다. 품질 실패로
runner가 exit 1을 반환하더라도 이 완전성 조건을 만족할 수 있다. exit 2, 부분 실행, provider
오류나 provenance 불일치는 저장하되 비교 결론은 `inconclusive`로 둔다. 승인이
없으면 안전하게 0건을 호출하고, 개인 full live 완료는 승인된 보충 시간으로 미룬다.

## Week 3 두 단계 전송 명세

Week 3은 **NIM Gemma 답 생성**과 **Google AI Studio Gemini Judge**를 서로 다른 실제 실행으로
승인한다. 첫 단계 승인이 둘째 단계를 자동 승인하지 않는다. 각 학습자는 자기 계정과 자기
API key만 사용하며 key와 `.env`를 공유하거나 제출하지 않는다.

### 1. NIM Gemma 기준·개선 답 60개

| 항목 | 값 |
| --- | --- |
| API 제공자·모델 | NVIDIA NIM, `nvidia_nim/google/gemma-4-31b-it` |
| 키 환경 변수 | `NVIDIA_NIM_API_KEY` |
| 보낸다 | OpenCQA JPEG, 질문, 기준 또는 개선 지시문 |
| 보내지 않는다 | OpenCQA 기대 답, 다른 후보, 사람 label, Judge 결과, API key |
| 범위 | 같은 모델에서 기준 지시문 30개 + 개선 지시문 30개 |
| 상한 | 요청·attempt 60/60, 입력 1,200,000 token, 출력 30,000 token, 비용 안전장치 $0.01, 7,200초, 재시도 0 |
| 결과 | 후보 생성 호출·결과·요약, 기준·개선 지시문 snapshot |

60개 실제 답, 두 지시문 snapshot, requested/actual Gemma model과 입력 hash가 모두 맞아야
후보 생성을 `complete`로 판정한다. 그 뒤에만 기준·개선 출처를 가리고 개인 A/B 30쌍을 만든다.
후보 생성이 `partial` 또는 `not_run`이면 Gemini Judge를 시작하지 않는다.

### 2. Gemini 3.5 Flash Lite Judge 30쌍

| 항목 | 값 |
| --- | --- |
| API 제공자·모델 | Google AI Studio, `gemini/gemini-3.5-flash-lite` |
| 실행 설정 | `configs/google-gemini-3.5-flash-lite-judge.yaml` |
| 키 환경 변수 | `GEMINI_API_KEY` |
| 고정 평가 기준 | `configs/week-03-judge-rubric.yaml` |
| 보낸다 | OpenCQA JPEG, 질문, 사람 작성 기대 답(`abstractive_answer`), 익명 Gemma 후보 A와 B, 고정 Judge rubric |
| 보내지 않는다 | 개인 사람 label, 후보의 기준·개선 출처, article·summary·OCR, API key |
| 범위 | 30쌍 × 2 trial × A/B·B/A |
| 전체 hard cap | 실제 요청 120~240회, 요청·attempt 최대 240/240, 입력 1,200,000 token, 출력 120,000 token, 비용 안전장치 $0.01, 10,800초, 요청당 재시도 1회 |
| 코드 rate cap | 15 RPM, 요청당 입력 5,000 token·출력 500 token, 한 full run당 최대 240요청 |
| 결과 | Judge 호출·60행 trial 결과·요약, `comparison.json` |

개인 30쌍 Judge 명령은 사람 label을 받지 않는다. 실행이 완결된 뒤 비교 명령이 잠근 label을
로컬에서 연결하고 SHA-256이 그대로인지 확인한다. 사람 label은 Gemini 요청에 들어가지 않는다.

Gemini 3.5 Flash Lite는 현재 잠긴 LiteLLM adapter로 요청 모델·actual model·구조화 출력을
확인할 수 있어 Judge로 선택했다.

Google로 보내는 자료에는 OpenCQA JPEG·질문·기대 답뿐 아니라 NIM Gemma가 만든 실제 출력도
포함된다. 실행 당일 계정 tier, quota, 가격과 데이터 이용 조건을 확인하고 이 전송 범위를
명시적으로 승인해야 한다. Free Tier 입력·출력이 제품 개선에 사용될 수 있다는 조건을 허용할
수 없으면 호출하지 않는다.

OpenCQA JPEG·질문·기대 답·익명 Gemma 출력의 Google 전송은 2026-08-17에 승인됐다. 승인 당시
[공식 rate limit 안내](https://ai.google.dev/gemini-api/docs/rate-limits)에서 확인한 Free Tier
근거는 15 RPM, 입력 250,000 TPM, 500 RPD다. 이 수치는 모든 API key에 자동 적용되는 보장이
아니다. 실행 당일 Google AI Studio에서 현재 key가 속한 프로젝트의 tier와 실제 한도가 이보다
낮지 않은지, 당일 잔여 RPD가 240건 이상인지 확인한다. 확인하지 못했거나 값이 낮으면 요청
0건과 `not_run`으로 남긴다.

수업 코드는 15 RPM과 요청당 입력 5,000 token·출력 500 token을 함께 적용한다. 분당 잠재
상한은 입력 75,000 TPM·출력 7,500 TPM으로, 확인한 입력 250,000 TPM보다 낮다. 30쌍은 판정
형식에 따라 120~240회 요청한다. 코드는 한 full run의 240요청만 막고 프로젝트의 하루 누적
요청은 추적하지 않는다. pacing만 약 8~16분이고 API·네트워크 응답 시간이 더해진다. 10,800초는
예상 시간이 아니라 중단을 보장하는 hard cap이다.

[공식 가격표](https://ai.google.dev/gemini-api/docs/pricing)의 Free Tier 입력·출력 단가는
0달러로 설정한다. 0.01달러는 예상 청구액이 아니라 tier나 설정이 달라졌을 때 호출 전에 멈추는
코드 안전장치다. 전송 승인은 full 실행 성공을 뜻하지 않는다. 새 호출·결과·요약·비교가 완결
검사를 통과하기 전에는 `complete`로 표시하지 않는다.

학습자가 `N`명이면 반 전체 최대 상한은 NIM 요청 `60N`·입력
`1,200,000N`·출력 `30,000N` token·비용 `$0.01N`과, Gemini 요청
`240N`·입력 `1,200,000N`·출력 `120,000N` token·비용
안전장치 `$0.01N`을 분리해 승인한다. Gemini의 Free Tier token 단가로 계산한 명목
비용은 0달러다. provider별 분당 요청과 현재 프로젝트의 하루 한도를 함께 확인하고, 여러
학습자가 하나의 프로젝트를 쓰지 않도록 한다. 같은 프로젝트에서 N명이 실행하면 최대 240N
요청으로 500 RPD를 넘을 수 있다. 필요한 경우 학습자를 wave로 나눈다.

과거 `abstractive_answer / extractive_answer` 후보와 NIM Gemma Judge로 만든 Codex 합성 기준,
그 기준으로 계산한 수치와 저장 결과는 legacy다. 새 입력·provider·hash와 다르므로 새 학습자
실행의 fallback이나 완료 근거로 쓰지 않는다.

개인 완료에는 후보 생성과 Judge가 모두 `complete`여야 한다. 어느 단계든 `partial`이면 원본을
지우거나 같은 폴더에 이어 쓰지 않고, `not_run`이면 요청하지 않은 사유를 기록한다. 한 사람의
label과 Judge가 일치했다는 이유만으로 `human_calibrated`나 blocking 품질 증거라고 부르지 않는다.

## Week 4–6에서 새 외부 전송을 추가할 때

현재 Week 1–3 설정은 Week 4–6의 새 API 전송을 승인하지 않는다. 이후 코드가 이미지,
질문, 모델 응답이나 도구 입력을 외부 서비스에 보내게 될 때만 다음 확인표를 작성한다.

1. 실제 실행할 script와 config 경로
2. API 제공자, 요청 모델, 접속 주소와 키 환경 변수 이름
3. 보내는 필드와 보내지 않는 필드, 개인정보·기밀정보 포함 여부
4. 데이터 출처·license와 외부 전송 허용 여부
5. 공식 모델·가격 URL, 확인 날짜와 계정 할당량
6. 요청·token·비용·시간·재시도 상한과 중단 조건
7. 원응답·오류·실제 처리 모델을 남길 결과 파일
8. 승인한 사람과 날짜

이 항목과 실제 실행 코드가 모두 생기기 전에는 외부 전송 기능이나 승인 상태가 구현됐다고
문서에 쓰지 않는다.
