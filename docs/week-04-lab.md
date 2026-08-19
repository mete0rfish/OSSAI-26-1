# Week 4 실습 — 지시문을 고치고 실제로 나아졌는지 확인하기

## 오늘의 수업 목표

**지시문(Prompt)**은 모델에게 “무엇을 보고, 어떻게 답하라”고 알려 주는 업무 설명이다.
**지시문 최적화**는 실제로 실패한 답을 보고 이 설명을 고친 뒤, 다른 문제에서도 답이 좋아졌는지
확인하는 과정이다. 여기서 최적화는 문장을 길게 만들거나 새 문장을 무조건 채택한다는 뜻이
아니다.

오늘 수업은 두 단계로 진행한다. 먼저 NIM Gemma가 답을 만들고 Python 채점기가 점수와 감점
이유를 붙인다. Gemini가 이 기록을 읽고 새 지시문을 제안한다. NIM Gemma가 처음 지시문과 새
지시문으로 검증 문제에 다시 답하면 두 평균을 비교해 하나를 고른다. 그다음 선택한 지시문을
원본·변형 이미지에 적용해, 필요한 근거가 남을 때는 답을 유지하고 근거가 사라질 때는 안전하게
답변을 보류하는지 확인한다.

수업에서는 DeepEval `PromptOptimizer`의 GEPA 기능으로 새 지시문을 만들었다. 도구 이름보다
중요한 원칙은 **실패를 보고 고치기 → 고칠 때 보지 않은 문제로 확인하기 → 결과가 나쁘면
처음 지시문으로 돌아가기**다.

수업을 마치면 다음 다섯 가지를 이해할 수 있다.

1. 지시문 최적화를 “실패한 답을 보고 지시문을 고친 뒤, 다른 문제로 확인하는 일”이라고 설명할 수 있다.
2. NIM Gemma는 답을 만들고, Python은 점수를 계산하며, Gemini는 지시문을 제안한다는 역할 차이를 말한다.
3. 지시문을 고치는 개발 문제와 지시문을 고르는 검증 문제를 왜 나누는지 설명할 수 있다.
4. 실제로 바뀐 문장 하나를 찾고, 그 변화가 모델 답과 점수에 미친 영향을 확인한다.
5. 필요한 근거가 남은 이미지에는 같은 답을, 근거가 사라진 이미지에는 답변 보류를 기대한다.

전체 최적화 과정은 다음과 같다.

```text
Week 3 OpenCQA 사례 30건
→ 처음 지시문으로 NIM Gemma가 개발 문제 18개에 답함
→ 프로그램이 답을 채점하고 빠진 값·불필요한 값을 알려 줌
→ Gemini가 답·점수·감점 이유를 읽고 새 지시문을 제안함
→ 처음·새 지시문으로 NIM Gemma가 검증 문제 6개에 답함
→ 두 평균을 비교해 실제로 사용할 지시문을 선택함
→ 공개 test 6개는 지시문 생성·선택에 사용하지 않음
→ 원본 차트와 이미지 변형 4개를 사람이 확인
→ 근거가 남으면 답 유지, 근거가 사라지면 안전한 답변 보류를 검사
```

수업 시간에는 전체 API 실행을 모두 마치기 어려우므로, 멘토가 사전에 완료한 결과를 바탕으로
실습한다. 준비를 마친 뒤 개발 사례 2건을 실제로 최적화해 후보가 만들어지는 과정을 보고,
이어서 전체 30건의 사전 실행 결과로 검증 평균과 최종 선택을 확인한다. 전체 API 실행 명령은
7절에 있으며, 수업 후 같은 `release`와 본인의 API 키로 실행할 수 있다.

```text
실습 준비
→ 개발 사례 2건 실제 최적화 실습
→ 개발 18건·검증 6건 전체 저장 결과 확인
→ 이미지 변형 판정과 저장 응답 평가
→ 수업 후 전체 API 실행
```

## 1. 지시문을 고칠 때 쓰는 문제와 선택할 때 쓰는 문제를 나누는 이유

### 답 생성·채점·지시문 제안 역할

| 역할 | 담당 | 하는 일 |
| --- | --- | --- |
| 답 생성 모델 | NIM Gemma | 차트와 질문을 읽고 정해 둔 JSON 모양으로 답함 |
| 고정 채점기 | Python 코드 | Gemma 답과 기준 답의 숫자·단어를 비교해 점수와 감점 이유를 만듦 |
| 지시문 제안 모델 | Gemini Flash Lite | 처음 지시문·NIM 답·기대 답·점수와 감점 이유를 읽고 새 지시문을 제안함 |

Gemini에는 OpenCQA 이미지와 사람의 변형 검토표를 보내지 않는다. Gemini가 새 지시문을
제안했다는 사실만으로 더 좋아졌다고 볼 수는 없다. 별도의 검증 문제에서 같은 고정 점수로
처음 지시문과 비교해 하나를 고른다.

### Gemma 답은 누가 어떻게 채점하는가

Gemma의 답을 채점하는 별도 모델은 없다. Python으로 만든 고정 채점기가 OpenCQA의 사람이 쓴
기준 답과 Gemma의 `answer`를 비교한다. 따라서 같은 두 답에는 언제 실행해도 같은 점수가
나온다. Gemini는 이 점수를 만들지 않고, 점수와 감점 이유를 읽어 지시문을 고치는 역할만 한다.

채점 순서는 다음과 같다.

1. Gemma 응답이 `StructuredAnswer` JSON 모양인지 확인한다. JSON이 잘못됐거나 필수 필드가
   빠졌으면 0점이다.
2. 기준 답과 Gemma의 `answer`에서 숫자와 핵심 단어를 각각 뽑는다.
3. 두 답에 함께 있는 항목, 빠진 항목, 불필요하게 추가된 항목으로 숫자 F1과 단어 F1을 구한다.
4. `0.7 × 숫자 F1 + 0.3 × 단어 F1`로 0부터 1 사이의 점수를 만든다.
5. 빠진 숫자와 추가된 숫자를 감점 이유에 남긴다.

개별 문제는 0.8 이상이면 통과로 표시한다. 다만 최종 지시문은 통과 개수로 고르지 않고, 검증
문제 6개의 평균 점수를 비교해 선택한다. 이 채점기는 숫자와 단어가 얼마나 겹치는지 보는 수업용
기준이므로 문장의 모든 의미나 실제 서비스 품질을 판정하는 모델로 해석하지 않는다.

### DeepEval GEPA가 지시문을 고치는 과정

GEPA는 모델의 가중치를 학습하지 않는다. 실패 답과 감점 이유를 이용해 지시문 문장을 바꾸고,
바뀐 지시문으로 Gemma를 다시 실행해 더 나은 후보만 남긴다. 전체 실행에서는 개발 문제 18개를
지시문 수정용 15개와 후보 내부 확인용 3개로 나눈다. 이 분할과 사례 선택은 `random_seed=42`로
고정한다.

1. 현재 지시문으로 Gemma가 개발 문제에 답하고, Python 채점기가 점수와 감점 이유를 만든다.
2. GEPA가 수정용 문제에서 최대 4개를 골라 현재 지시문의 성공과 실패를 모은다.
3. Gemini가 현재 지시문·질문·기준 답·Gemma 답·점수·감점 이유를 읽고 실패 원인을 정리한다.
4. Gemini가 그 분석을 반영한 새 지시문을 제안한다.
5. Gemma가 같은 문제에 새 지시문으로 다시 답한다. 새 평균이 오르지 않으면 후보를 버린다.
6. 평균이 오른 후보는 내부 확인용 3개에서도 다시 채점한다. 기존 후보보다 모든 문제에서
   밀리는 후보는 버리고, 문제별 장점이 남아 있는 후보만 보관한다.
7. 이 과정을 최대 2회 반복하고, 남은 후보 가운데 내부 확인 평균이 가장 높은 지시문을
   GEPA의 후보로 반환한다.

GEPA가 반환한 후보가 곧 최종 지시문은 아니다. GEPA가 지시문을 고칠 때 사용하지 않은 검증
문제 6개에서 처음 지시문과 후보 지시문으로 Gemma를 다시 실행한다. 같은 고정 채점기로 두 평균을
비교해 후보 평균이 더 높을 때만 새 지시문을 선택한다. 공개 test 6개는 지시문 생성과 선택에
사용하지 않는다.

### 데이터를 세 묶음으로 나누기

Week 3에서 준비한 `local-data/opencqa/week-03-cases.jsonl` 30건을 그대로 사용한다.

| 데이터 구분 | 개수 | 이번 주의 역할 |
| --- | ---: | --- |
| 개발(`development`) | 18 | 실패 답을 찾고 새 지시문을 만드는 데 사용 |
| 검증(`validation`) | 6 | 처음·새 지시문 가운데 하나를 선택 |
| 공개 test | 6 | 지시문 생성과 선택에 사용하지 않음 |

공개 test 답도 같은 파일에 들어 있으므로 test를 보지 못했다고 주장하지 않는다.
`test_used_for_generation_or_selection=false`는 test 6건을 후보 생성과 선택 함수에 전달하지
않았음을 나타낸다. 이 작은 공개 test만으로 배포 성능도 주장하지 않는다.

### 이미지 상태 두 가지

| 사람이 확인한 상태 | 쉬운 뜻 | 기대하는 모델 행동 |
| --- | --- | --- |
| `preserved` | 질문에 필요한 수치와 비교 대상이 보임 | 원본과 같은 핵심 답을 근거와 함께 반환 |
| `destroyed` | 필요한 근거가 잘리거나 가려짐 | 추정하지 않고 답변을 보류하며 근거를 비움 |

변형 이름만 보고 상태를 정하지 않는다. 사람이 실제 이미지를 본 결과가 변형의 의도와 다르면
그 변형은 `invalid_variant`다. 성공이나 실패에 넣지 않고 평가에서 제외한다.

## 2. 한 번의 명령으로 실습 준비하기

실행 프로젝트 저장소의 최상위 폴더에서 다음 명령을 한 번 실행한다. `minsu`만 본인의 영문·숫자
별칭으로 바꾼다.

```bash
uv run --locked python scripts/prepare_week_04_lab.py --alias minsu
```

스크립트는 `configs/week-04.yaml`의 `class_materials`에 지정된 공통 수업 자료를 읽는다.
화면에 `수업 자료`와 `저장 응답을 만든 코드 버전`이 나오면 출처를 확인한다. Git 번호나
파일 지문을 명령에 직접 넣지 않는다.

### Git SHA와 SHA-256을 남기는 이유

모델 답은 같아 보여도 코드나 입력 파일이 바뀌면 비교 조건이 달라질 수 있다. 그래서 “어느
코드가 이 답을 만들었는가”와 “그때 쓴 파일 내용이 지금도 같은가”를 짧은 지문으로 남긴다.
이 두 지문은 점수를 높이는 장치가 아니라 서로 다른 실행 결과를 섞지 않게 붙이는 이름표다.
브랜치 이름이나 파일 이름은 같은 이름으로 내용이 바뀔 수 있지만, Git SHA와 SHA-256은 내용이
바뀌면 함께 달라지므로 비교 기준으로 쓴다.

| 값 | 뜻 | 어디서 확인할까 | 확인할 일 |
| --- | --- | --- | --- |
| `minsu` 같은 별칭 | 내 개인 폴더 이름 | 준비 명령의 `--alias` | 본인 별칭으로 바꾼다. |
| `4b53815` 같은 Git SHA | 공통 저장 응답을 만든 코드 버전의 앞 7자리 | 준비·결과 읽기 화면의 `저장 응답을 만든 코드 버전`; 원본 기록은 `summary.json`의 `git_sha` | 자기 Git SHA로 바꾸지 않고 출처만 확인한다. |
| 64자리 SHA-256 | 입력·이미지·채점 파일 내용의 지문 | 자동 검사 결과; 자세한 원본 기록은 `summary.json`과 `evaluation-manifest.json` | 복사하거나 입력하지 않고 자동 일치 검사가 통과했는지 확인한다. |

현재 작업 중인 Git SHA와 공통 저장 응답의 Git SHA는 역할이 다르다. 이 수업에서 결과를 읽을
때 기준으로 삼는 SHA는 **공통 저장 응답의 출처**다. 공통 자료가 바뀌어도 설정 파일이 새
경로를 가리키므로 실행할 명령은 그대로다.

이 스크립트는 API를 호출하지 않는다. 다음 작업을 대신 처리한다.

1. OpenCQA 문제 30개와 이미지가 있는지 확인한다.
2. 개발 18개·검증 6개·공개 test 6개로 정확히 나뉘었는지 확인한다.
3. 지시문 최적화 결과와 이미지 응답 5개가 모두 저장돼 있는지 확인한다.
4. 현재 입력과 저장 실행의 SHA-256이 같은지 확인한다.
5. 개인 작업 폴더, 개인 결과 폴더와 학습 기록 파일을 준비한다.

정상이라면 `4주차 실습 준비 완료`와 함께 사용할 폴더가 화면에 나온다. 실패하면 빠진 파일이나
맞지 않는 조건을 한글로 알려 준다. 과거 `week-03-pairs.jsonl`로 만든 결과는 이번 수업에서
사용하지 않는다. 이 단계에서는 전체 실행의 호출 수와 최종 선택을 아직 보여 주지 않는다.

## 3. 개발 사례 2건으로 지시문 최적화 실습해 보기

전체 30건 결과를 열기 전에 개발 문제 첫 2건으로 지시문 최적화 과정을 실습한다. 2건은
DeepEval GEPA가 요구하는 최소 사례 수다. 실행하기 전에 외부 전송 조건과 모델·가격·남은
요청 수를 확인한다. 조건을 확인했다면 다음 명령을 실행한다.

```bash
uv run --locked python scripts/optimize_open_cqa_prompt.py \
  --live-optimize \
  --demo-samples 2 \
  --max-requests 5 \
  --max-input-tokens 100000 \
  --max-output-tokens 2500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 900 \
  --catalog-verified-on "$(date +%F)" \
  --pricing-verified-on "$(date +%F)" \
  --optimizer-max-requests 2 \
  --optimizer-max-attempts 4 \
  --optimizer-max-input-tokens 20000 \
  --optimizer-max-output-tokens 8000 \
  --optimizer-max-cost-usd 0.01 \
  --optimizer-max-wall-seconds 900 \
  --optimizer-catalog-verified-on "$(date +%F)" \
  --optimizer-pricing-verified-on "$(date +%F)"
```

- `--live-optimize`: 저장 결과를 읽는 대신 실제 NIM·Gemini API를 호출한다.
- `--demo-samples 2`: 개발 문제 가운데 첫 2건만 실행한다.
- NIM은 요청·시도 5회, Gemini는 요청 2회·시도 4회가 최대다.
- 결과 폴더는 실행 시각을 기준으로 자동 생성되며, 저장 경로가 화면에 나온다.

2건 실습에서는 한 사례로 수정 의견을 만들고 다른 한 사례로 후보를 내부 확인한다. GEPA는
지시문 수정을 한 번만 시도한다. 별도의 검증 문제 6개를 사용하지 않으므로 후보 생성 과정까지만
확인하고 최종 지시문은 선택하지 않는다.

실습에서는 다음 흐름을 확인한다.

1. NIM Gemma가 처음 지시문으로 개발 문제에 답한다.
2. 프로그램이 답에 고정 점수와 감점 이유를 붙인다.
3. Gemini가 그 기록을 읽고 새 지시문을 제안한다.
4. 후보가 달라졌다면 바뀐 문장을 찾고, 같다면 바뀐 문장이 없음을 확인한다.
5. `summary.json`의 `run_mode=classroom_demo`와
   `quality_selection_allowed=false`를 확인한다.

이 시연에는 검증 문제 6개가 없으므로 후보를 선택하지 않는다. 시연 폴더의
`selected=null`은 실패가 아니라 의도한 결과다. 다음 4절에서 개발 18건·검증 6건 전체 저장
기록을 처음 열어 최종 선택과 품질을 확인한다. 6절에서는 이미지 저장 응답 5건을 같은 규칙으로
다시 채점한다. 승인이 없거나 API 서비스가 중단되면 시연을 생략하고 전체 저장 기록으로
계속한다.

후보가 처음 지시문과 같을 수도 있다. 이때는 `candidate_changed=false`와 바뀐 문장이 없음을
확인한 뒤, 4절의 전체 저장 결과에서 실제 지시문 변화와 점수 차이를 분석한다.

2026-08-19 리허설에서는 개발 사례 `884`와 `43`으로 이 과정을 끝까지 실행했다. NIM 5회와
Gemini 2회가 모두 요청한 모델에서 응답했고 오류는 없었다. Gemini는 “가장 높거나 낮은 항목을
묻는 질문이면 모든 항목을 나열하지 말라”는 후보 문장을 추가했다. 그러나 이 두 사례만으로
더 좋아졌다고 판단하지 않았고 `quality_selection_allowed=false`, `selected=null`로 남겼다.

## 4. 전체 저장 결과에서 지시문 변화와 최종 선택 확인하기

이제 처음으로 전체 저장 결과를 연다. 실제 결과에서는 새 지시문의 평균이 더 낮았다. 지시문을
자동으로 고쳤지만 결과가 나아지지 않아 처음 지시문을 그대로 사용했다. 이것도 지시문
최적화에서 얻을 수 있는 올바른 결론이다.

이 절에서 **기준 지시문**은 처음 사용한 지시문이다. 저장 파일에서는 `baseline`이라고 쓴다.
**후보 지시문**은 Gemini의 제안을 반영한 새 지시문이다. 저장 파일에서는 `candidate`라고 쓴다.

먼저 저장 결과를 읽기 좋게 정리해 주는 스크립트를 실행한다.

```bash
uv run --locked python scripts/inspect_week_04_prompt_results.py
```

화면에는 지시문 최적화의 뜻, 두 모델의 호출 수, 실제로 바뀐 문장, 검증 문제 6개의 점수,
최종 선택 이유와 대표 답 두 개가 순서대로 나온다. 아래 설명은 이 출력을 한 단계씩 읽는 법이다.

2026-08-19에 새 설정과 실제 API 키로 다시 실행한 순서는 다음과 같다.

1. NIM Gemma가 처음 지시문으로 개발 문제에 답했다.
2. Gemini가 점수가 낮은 답과 그 이유를 읽고 지시문 수정을 제안했다.
3. NIM Gemma가 바뀐 지시문으로 검증 문제 6개에 다시 답했다.
4. 같은 6개 문제에서 처음 지시문과 바뀐 지시문의 점수를 비교했다.
5. 바뀐 지시문의 평균이 더 낮아서 처음 지시문으로 돌아갔다.

이 실행에서는 승인 상한 NIM 45회·Gemini 4회 안에서 실제로 NIM을 42회, Gemini를 4회
호출했다. API 호출 오류는 없었다. 요청한 모델과 실제로 응답한 모델도 같았다. 검증 문제
6개 중 2개는 점수가 올랐고, 1개는 같았으며, 3개는 떨어졌다.

### 4-1. Gemini가 지시문 수정을 제안한 이유

처음 지시문에는 다음 문장이 있었다.

```text
`answer`에는 질문이 요구한 값과 단위만 간결하게 씁니다.
```

이 문장은 “몇 퍼센트인가?”처럼 짧은 값을 묻는 질문에는 잘 맞는다. 하지만 개발 문제에는
“어떻게 변했는가?”, “두 집단은 어떻게 다른가?”처럼 문장으로 설명해야 하는 질문도 있었다.
Gemini는 처음 지시문, NIM의 실제 답, 사람이 작성한 기준 답, 점수와 감점 이유를 읽었다.
그 뒤 다음과 같은 뜻의 제안을 만들었다.

```text
잘된 점: NIM은 정해 둔 JSON 모양과 근거 작성 규칙을 대체로 지켰다.
고칠 점: "값과 단위만" 쓰게 하면 설명이 필요한 질문에서도 답이 너무 짧아진다.
바꿀 점: 질문이 설명을 요구하면 짧은 문장과 필요한 세부 내용도 쓰게 한다.
```

위 내용은 `calls.jsonl`에 저장된 Gemini 답을 쉬운 한국어로 풀어 쓴 요약이다. Gemini는 차트
이미지를 보지 않았다. 후보를 고르는 데 사용할 검증 문제 6개와 마지막 확인용 test 문제
6개도 보지 않았다.

여기서 전체 내용을 보여 주는 `system` 메시지는 NIM Gemma가 받은 메시지다. 현재
`calls.jsonl`에는 Gemini가 받은 전체 요청 문장이 아니라 Gemini의 전체 응답만 저장돼 있다.
Gemini 요청 문장은 저장되지 않았으므로 실제 요청인 것처럼 재구성해 보여 주지 않는다.

Gemini는 `문제 설명 → 새 지시문 작성` 과정을 두 번 거쳤다. 최종 `candidate-prompt.md`에는
질문이 설명을 요구할 때 문장으로 답하도록 바꾼 후보가 저장됐다. 현재 저장 파일에는 두 제안의
점수를 나란히 보여 주는 표가 없으므로, 저장되지 않은 선택 이유를 추측하지 않는다.

### 4-2. NIM Gemma가 받은 전체 메시지를 본다

지시문 파일은 설명용 메모가 아니다. 코드가 파일 내용을 읽어 NIM Gemma의 `system` 메시지로
보낸다. `{question}` 자리에는 현재 문제의 질문이 들어간다. 그다음 `user` 메시지로 같은 질문과
차트 이미지를 보낸다.

아래는 문제 `5978`을 처음 지시문으로 실행했을 때의 전체 `system` 메시지다. 영어 질문은
“2050년 캘리포니아 인구 전망을 설명하라”는 뜻이다.

```text
/no_think

차트 이미지에서 아래 질문의 답을 찾고, JSON 하나만 반환하세요.

질문: Describe the projections of the California population in 2050.

이미지에서 확인한 내용만 사용하세요. 답을 찾는 과정은 출력하지 않습니다.

`answer`에는 질문이 요구한 값과 단위만 간결하게 씁니다. `evidence`의 `quote`에는
답을 직접 확인할 수 있는 차트의 연속된 글자와 수치를 그대로 씁니다. 차트 이미지는
1페이지로 보고 `evidence_id`는 `chart`, `page_number`는 `1`을 사용합니다.

출력 규칙:

- 첫 글자는 `{`, 마지막 글자는 `}`입니다.
- Markdown code fence, 설명, 두 번째 JSON을 출력하지 않습니다.
- 아래 6개 field를 모두 한 번씩 넣습니다.
- `confidence`는 0 이상 1 이하 숫자입니다.
- `tool_requests`는 항상 빈 목록입니다.

답을 확인할 수 있을 때의 JSON 모양은 다음과 같습니다. 예시 값은 복사할 정답이 아닙니다.

{"answer":"값과 단위","evidence":[{"evidence_id":"chart","quote":"답을 포함한 차트 글자와 수치","page_number":1}],"confidence":0.9,"abstained":false,"abstention_reason":null,"tool_requests":[]}

이미지에서 답을 확인할 수 없을 때는 추정하지 말고 아래 모양으로 답변을 보류합니다.

{"answer":"답변 보류","evidence":[],"confidence":0.0,"abstained":true,"abstention_reason":"이미지에서 답을 확인할 수 없음","tool_requests":[]}
```

Gemini의 제안을 반영한 뒤에는 다음 전체 `system` 메시지를 보냈다.

```text
/no_think

차트 이미지에서 아래 질문의 답을 찾고, JSON 하나만 반환하세요.

질문: Describe the projections of the California population in 2050.

이미지에서 확인한 내용만 사용하세요. 답을 찾는 과정은 출력하지 않습니다.

`answer`에는 질문이 요구하는 형태에 맞춰 작성합니다. 질문이 서술형 설명이나 문장 형태의 답을 요구할 경우 그에 맞는 설명적 문장으로 작성하고, 단순 수치를 요구할 경우 값과 단위를 간결하게 씁니다. `evidence`의 `quote`에는 답을 직접 확인할 수 있는 차트의 연속된 글자와 수치를 그대로 씁니다. 차트 이미지는 1페이지로 보고 `evidence_id`는 `chart`, `page_number`는 `1`을 사용합니다.

출력 규칙:

- 첫 글자는 `{`, 마지막 글자는 `}`입니다.
- Markdown code fence, 설명, 두 번째 JSON을 출력하지 않습니다.
- 아래 6개 field를 모두 한 번씩 넣습니다.
- `confidence`는 0 이상 1 이하 숫자입니다.
- `tool_requests`는 항상 빈 목록입니다.

답을 확인할 수 있을 때의 JSON 모양은 다음과 같습니다. 예시 값은 복사할 정답이 아닙니다.

{"answer":"값 또는 서술형 답변","evidence":[{"evidence_id":"chart","quote":"답을 포함한 차트 글자와 수치","page_number":1}],"confidence":0.9,"abstained":false,"abstention_reason":null,"tool_requests":[]}

이미지에서 답을 확인할 수 없을 때는 추정하지 말고 아래 모양으로 답변을 보류합니다.

{"answer":"답변 보류","evidence":[],"confidence":0.0,"abstained":true,"abstention_reason":"이미지에서 답을 확인할 수 없음","tool_requests":[]}
```

두 실행에서 `user` 메시지는 같다. 실제 코드에서는 이미지 파일을 base64 데이터로 바꿔
전송한다. 아래에서는 긴 이미지 데이터를 파일 경로로 줄여 표시했다.

```text
role: user

text:
Describe the projections of the California population in 2050.

image:
local-data/opencqa/images/5978.jpg
```

API 요청에는 Pydantic의 `StructuredAnswer`에서 만든 JSON Schema도 함께 들어간다. 이것은
프롬프트 문장이 아니라 API가 응답 모양을 제한하는 별도 설정이다.

두 메시지를 비교하면 바뀐 곳은 두 군데다.

1. `answer`에 값만 쓰게 하던 문장을, 질문에 필요하면 설명도 쓰게 바꿨다.
2. 정상 답 예시의 `answer`를 `값과 단위`에서 `값 또는 서술형 답변`으로 바꿨다.

그 밖의 질문, 이미지 사용 규칙, 근거 작성법, JSON 6개 항목, `confidence` 범위와 답변 보류
규칙은 그대로다. 우리가 예상한 결과는 다음과 같다.

```text
설명이 필요한 질문에서는 빠졌던 대상·시점·비교 내용이 답에 추가될 것이다.
```

이 예상이 맞는지는 Gemini의 설명만 보고 정하지 않는다. NIM이 새 지시문으로 만든 답을
처음 답과 직접 비교한다. 앞에서 실행한 결과의 `[지시문에서 바뀐 줄]`에는 실제로 빠진 문장과
추가된 문장이 `-`와 `+`로 표시된다.

### 4-3. 검증 문제 6개의 실제 점수를 비교한다

아래 6개 문제는 Gemini가 지시문을 고칠 때 사용하지 않았다. 프로그램은 NIM의 답이 정해 둔
JSON 모양인지 먼저 확인한다. JSON이 맞으면 기준 답과 숫자가 얼마나 같은지 70%, 주요 단어가
얼마나 같은지 30% 비율로 점수를 계산한다. `missing`은 NIM이 빠뜨린 값이고, `extra`는
기준 답에는 없는데 NIM이 추가한 값이다.

| 문제 번호 | 처음 지시문 | 바뀐 지시문 | 점수 변화 | 모델 답에서 실제로 일어난 일 |
| --- | ---: | ---: | ---: | --- |
| `171` | 0.1000 | 0.1525 | +0.0525 | 후보 답이 주요 단어를 더 포함했지만 필요한 두 수치는 여전히 빠뜨림 |
| `699` | 0.5237 | 0.2732 | -0.2505 | G7 모든 나라의 수치를 추가해 `extra` 값이 늘어남 |
| `6447` | 0.0808 | 0.1385 | +0.0577 | 단순 목록을 질문에 맞는 설명 문장으로 바꿈 |
| `2208` | 0.0000 | 0.0000 | 0.0000 | 답을 보류하면서 정해 둔 `answer` 문구를 쓰지 않음 |
| `4327` | 0.3650 | 0.3333 | -0.0317 | 설명을 바꿨지만 기준 답과 겹치는 주요 단어가 줄어듦 |
| `5978` | 0.7004 | 0.5778 | -0.1226 | 전체 인구 외에 집단별 수치 다섯 개를 추가함 |
| **평균** | **0.2950** | **0.2459** | **-0.0491** | **처음 지시문의 평균이 더 높음** |

점수만 보면 왜 좋아지거나 나빠졌는지 알 수 없다. 아래에서 NIM이 실제로 쓴 답을 읽는다.

#### 가장 좋아진 사례 — `6447`

질문은 `Describe the groups that make up the unemployed Americans.`이다. 한국어로는
“미국 실업자를 구성하는 집단을 설명하라”는 뜻이다.

```text
처음 지시문의 answer:
Re-entered labor force (31.6%), Permanent job loss (26.5%), Newly entered
labor force (11.5%), Completed temp. job (10.8%), Temp. layoff (10.1%),
and Quit job (9.5%)

바뀐 지시문의 answer:
The unemployed Americans are made up of those who re-entered the labor force
(31.6%), experienced permanent job loss (26.5%), newly entered the labor force
(11.5%), completed a temporary job (10.8%), had a temporary layoff (10.1%),
or quit their job (9.5%).
```

수치 목록은 같지만 바뀐 지시문으로 만든 답은 “실업자는 이 집단들로 구성된다”는 설명 문장이
됐다. 주요 단어 점수가 올라 전체 점수도 `0.0808 → 0.1385`로 올랐다. 다만 수치 표기 차이로
숫자 점수는 여전히 0이므로, 이 사례 하나만으로 후보가 충분히 좋아졌다고 말하지 않는다.

#### 처음보다 점수가 떨어진 사례 — `699`

질문은 `How is the income inequality among G7 countries?`이다. 한국어로는 “G7 국가의
소득 불평등은 어떠한가?”라는 뜻이다.

```text
처음 지시문의 answer:
The U.S. has the highest level of income inequality among G7 countries
with a Gini coefficient of 0.434, while France has the lowest at 0.326.

바뀐 지시문의 answer:
Among G7 countries, the U.S. has the highest level of income inequality
with a Gini coefficient of 0.434, followed by the UK (0.392), Italy (0.373),
Japan (0.363), Canada (0.352), Germany (0.351), and France, which has the
lowest at 0.326.
```

바뀐 지시문으로 만든 답은 모든 나라의 수치를 길게 나열했다. 프로그램은 기준 답에 없는
수치들을 `extra`로 세었고 점수는 `0.5237 → 0.2732`로 떨어졌다. 이것은 이 수업의 채점
방법으로 계산한 결과다. 사람이 읽었을 때도 무조건 더 나쁜 답이라는 뜻은 아니다. 다만
**답이 길고 자세하다는 이유만으로 더 좋은 답이라고 선택할 수도 없다.**

### 4-4. 처음 지시문을 그대로 사용한 이유

수업을 시작하기 전에 “검증 문제 6개의 평균이 더 높은 지시문을 사용한다”는 기준을 정했다.
처음 지시문의 평균은 `0.2950`, 바뀐 지시문의 평균은 `0.2459`다. 두 평균을 비교해 처음
지시문을 선택했다. `summary.json`에도 같은 결과가 저장돼 있다.

```json
{
  "baseline_mean": 0.2949833333333333,
  "candidate_mean": 0.24588333333333334,
  "candidate_changed": true,
  "selected": "baseline",
  "selection_reason": "validation_not_improved"
}
```

`selected-prompt.md`를 열면 처음 지시문과 같은 내용이 나온다. 지시문을 자동으로 최적화해도
항상 새 지시문을 사용할 필요는 없다. 이번 실행에서 알게 된 내용은 다음 한 문장이다.

```text
설명을 허용하자 일부 답은 좋아졌지만, 6개 전체 평균은 낮아져 처음 지시문을 유지했다.
```

Gemini가 만든 문장이 처음 지시문과 글자까지 완전히 같을 수도 있다. 이때는 새 지시문이
생긴 것이 아니므로 NIM을 다시 호출하지 않는다. 저장 결과에는 `candidate_identical`이라고
쓴다.

### 4-5. 스크립트가 읽는 파일

`inspect_week_04_prompt_results.py`는 다음 다섯 파일을 찾아 순서대로 읽는다.

1. `candidate-prompt.md`: Gemini의 제안을 반영해 바뀐 지시문
2. `validation.jsonl`: 문제 6개의 처음 답·바뀐 답·점수와 감점 이유
3. `selected-prompt.md`: 최종적으로 사용할 처음 지시문
4. `summary.json`: 두 평균과 처음 지시문을 선택한 이유
5. `calls.jsonl`: NIM 42회와 Gemini 4회의 실제 응답, 사용한 토큰 수, 걸린 시간과 오류

준비 스크립트는 저장 결과가 이번 수업 입력으로 만든 것인지 먼저 확인한다. 저장 결과와 현재
파일의 SHA-256을 계산해 같으면 통과하고, 다르면 어느 조건이 맞지 않는지 알려 준다.

화면에 나온 영문 필드는 다음 뜻이다.

- `observed_status=complete`: 예정한 호출이 끝났고 필요한 파일이 모두 생겼다.
- development 18, validation 6, test 6: 30개 문제를 세 묶음으로 나눴다.
- `test_used_for_generation_or_selection=false`: 마지막 6개 문제는 지시문을 만들거나 고를 때
  사용하지 않았다.
- `candidate_changed=true`: 처음 지시문과 다른 후보가 만들어졌다.
- `selection_reason=validation_not_improved`: 후보 평균이 오르지 않아 처음 지시문을 골랐다.
- `provider_error_count=0`: API 호출 실패가 없었다.
- `model_drift_count=0`: 요청한 모델과 실제 응답한 모델이 달랐던 경우가 없었다.
- `git_sha`: 이 저장 응답을 만든 코드 버전이다. 내 현재 Git SHA와 같아야 하는 값은 아니다.
- `artifact_sha256.week-03-cases.jsonl`: 실행에 사용한 입력 파일의 SHA-256이다. 준비 스크립트가
  현재 파일과 자동으로 비교한다.

결과 읽기 스크립트는 바뀐 문장과 문제별 점수를 사람이 읽을 수 있는 표로 바꿔 보여 준다.
이번 실행의 `selected-prompt.md`는 처음 지시문과 같다. `status=pass`는 실행이 끝나고 필요한
파일이 모두 생겼음을 나타낸다. 새 지시문이 선택됐거나 모든 모델 답이 품질 기준을
통과했다는 뜻은 아니다.

이 숫자는 2026-08-19에 실제 API로 다시 저장한 실행의 예시다. 수업 코드나 입력 파일이 바뀌면
현재 `summary.json`의 숫자를 사용한다. 다음 표를 `local-data/learning-progress.md`의 Week 4에
기록한다.

| 확인 항목 | 기록할 값 |
| --- | --- |
| 처음 지시문에서 고치려고 한 문제 |  |
| 후보에서 실제로 바뀐 문장 |  |
| 점수가 오른 검증 문제와 이유 |  |
| 점수가 떨어진 검증 문제와 이유 |  |
| 처음 지시문의 평균 |  |
| 바뀐 지시문의 평균 |  |
| 실제 선택과 선택 이유 |  |

### 4-6. 지시문을 바꾼 뒤 모델 답의 변화를 확인한다

앞에서 실행한 스크립트의 `[점수가 가장 오른 사례: 6447]`과
`[점수가 가장 떨어진 사례: 699]`를 찾는다.

각 문제에는 질문과 사람이 쓴 기준 답 한 줄이 있다. `validation.jsonl`에는 처음 지시문으로
만든 NIM 답과 바뀐 지시문으로 만든 NIM 답이 한 줄씩 있다. 세 줄을 같은 `sample_id`로
연결한 뒤 아래 표를 채운다.

| 확인할 내용 | 점수가 가장 오른 `6447` | 점수가 가장 떨어진 `699` |
| --- | --- | --- |
| 질문이 요구한 것 |  |  |
| 처음 지시문으로 만든 `answer` |  |  |
| 바뀐 지시문으로 만든 `answer` |  |  |
| 새 답에서 추가되거나 빠진 말 |  |  |
| 처음 점수 → 바뀐 점수 |  |  |
| 프로그램이 알려 준 감점 이유 |  |  |
| 처음 예상한 변화가 나타났는가 |  |  |

이 두 문제는 답이 왜 달라졌는지 배우기 위한 예시다. 두 문제만 보고 지시문을 고르지 않는다.
최종 선택에는 검증 문제 6개 전체의 평균을 사용한다.

## 5. 이미지 변형을 직접 보고 판정하기

수업용 저장 응답에 사용한 첫 번째 사례의 이미지 변형을 개인 폴더에 만든다. 이 명령은 API를
호출하지 않는다.

```bash
uv run --locked python scripts/generate_image_variants.py --student-alias minsu
```

`minsu`는 준비 단계에서 사용한 본인 별칭으로 바꾼다. 스크립트 위쪽의 주석에는 실행 목적과
정상적으로 만들어지는 파일이 적혀 있다. 같은 폴더에 결과가 이미 있으면 덮어쓰지 않고 멈춘다.

`case.json`의 `original_image`와 다음 네 파일을 편집기에서 직접 연다.

- `rotate-2.png`: 원본을 2도 회전
- `jpeg-60.jpg`: JPEG 품질 60으로 압축
- `crop-left.png`: 왼쪽 40% 제거
- `occlude-answer.png`: 왼쪽 일부를 회색으로 가림

질문에서 묻는 대상·기간·수치·비교 대상을 먼저 찾는다. 그 근거가 남아 있으면 `preserved`,
찾을 수 없으면 `destroyed`를 개인 `variant-review.csv`의 `grounding_status`에 쓴다. 다른 열은
프로그램이 이미지와 행을 연결하는 정보이므로 바꾸지 않는다. 특히 SHA-256 열은 직접 입력하는
답이 아니다.

```text
preserved 또는 destroyed만 입력
빈칸 금지
변형 이름이 아니라 실제 이미지로 판단
```

개인 변형은 이미지를 보고 판단하는 연습 자료다. 수업용 NIM 답에 사용한 이미지와 개인
이미지가 같은지는 다음 평가 명령이 SHA-256으로 자동 확인한다. 오류가 없다면 두 이미지가
일치한다.

## 6. 저장된 이미지 답을 같은 규칙으로 다시 채점하기

수업용 저장 결과는 같은 지시문으로 원본 이미지 1개와 변형 이미지 4개를 NIM에 실제로 보낸
결과다.

개인 `variant-review.csv`의 네 칸을 모두 채운 뒤 다음 명령을 실행한다. API를 다시 호출하지
않고 수업용 저장 응답을 현재 채점기로 다시 채점한다.

```bash
uv run --locked python scripts/evaluate_image_robustness.py --student-alias minsu
```

`minsu`는 본인 별칭으로 바꾼다. 정상이라면 화면에 `통과`, `실패`, `판정 불가`, `변형 무효`
개수가 나온다. 필요한 파일이 없거나 검토표에 빈칸이 있으면 무엇을 고쳐야 하는지 알려 준다.

다음 순서로 결과를 읽는다.

1. `summary.json`: 응답 5개가 모두 있는지, 어떤 모델과 지시문을 썼는지, 오류가 있었는지
2. `responses.jsonl`: 원본·변형별 원응답과 구조화 답
3. 개인 `evaluation.json`: 원본과 변형 4개의
   `passed / failed / inconclusive / invalid_variant`
4. 개인 `evaluation-manifest.json`: 응답·이미지·검토표·채점 규칙·출력 형식의 SHA-256

`source_git_sha`는 저장된 NIM 답을 만들 때 사용한 코드 버전으로, “이 답은 어느 코드에서
나왔는가”를 추적하는 기록이다. `scorer_sha256`은 그 답을 지금 채점한
Python 파일의 내용 지문이다. NIM을 다시 호출하지 않고 채점 규칙만 고쳤다면 저장 응답의 Git
SHA는 그대로이고 채점기 SHA-256만 바뀔 수 있다.

근거가 남은 변형은 원본과 변형 모두 점수 0.8 이상이고, 근거가 있으며, 원본 답의 숫자를
유지해야 한다. 원본 품질이 0.8 미만이면 변형 답이 같아도 정답 유지를 확인할 수 없어
`inconclusive`다. 근거가 사라진 변형은 `abstained=true`, 빈 근거와 보류 이유를 가져야 한다.
두 상태를 하나의 정답 유지율로 합치지 않는다.

같은 날 NIM 실제 응답 5건은 모두 정해 둔 JSON 모양을 지켰다. API 호출 오류도 없었고 요청한
모델과 다른 모델이 응답한 경우도 없었다. 원본 답은 필요 이상으로 주변 값을 나열해 점수
`0.139`로 실패했다. 회전·JPEG 답은 원본과 같았지만 원본 품질 때문에 `inconclusive`였고,
잘림·가림은 둘 다 안전하게 답변을
보류해 `passed`였다. 결과는 통과 2, 실패 1, 판정 불가 2, 변형 무효 0이다.

## 7. 수업 후 전체 API 실행하기

이 절에서는 수업과 같은 `release`에서 본인의 API 키로 지시문 최적화 전체 과정과 이미지 5건을
실행한다. 7-2의 명령은 개발 18건으로 후보를 만들고 검증 6건으로 지시문을 선택한다. 이어서
7-3의 명령으로 선택한 지시문을 원본 이미지 1개와 변형 이미지 4개에 적용한다.

### 7-1. 실행 전 확인

다음 조건을 모두 확인한 뒤에만 실제 API를 호출한다.

1. 수업에서 사용한 `release`를 checkout한 뒤 `git status --porcelain` 출력이 비어 있다.
2. `local-data/opencqa/week-03-cases.jsonl` 30건과 5절에서 만든 개인 이미지 변형이 있다.
3. 개인 `variant-review.csv`의 네 행을 모두 판정했다.
4. NVIDIA와 Google에 보낼 자료와 실행 당일의 모델·가격·API 사용 한도·데이터 이용 조건을 확인했다.
5. 본인의 NVIDIA·Google API 키를 사용하며 다른 사람과 키를 공유하지 않는다.

```bash
git status --porcelain
uv run --locked python scripts/check_week_04_api_keys.py
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim-gemma4.yaml
```

전체 실행 한 건의 최대 사용량은 다음과 같다. 이 값은 목표 횟수가 아니라 실행 전에 정한 중단
상한이다.

| 모델 역할 | 요청·시도 상한 | 입력·출력 토큰 상한 | 관리용 비용·시간 상한 |
| --- | --- | --- | --- |
| NIM 답 생성과 이미지 5건 | 요청·시도 50/50회 | 입력 1,000,000·출력 25,000 | $0.02·최대 2시간 15분 |
| Gemini 지시문 검토 | 요청 4회·시도 8회 | 입력 40,000·출력 16,000 | $0.01·최대 2시간 |

승인, API 키, API 사용 한도 중 하나라도 준비되지 않으면 호출하지 않는다. 학습 진행표에는
`not_run`과 사유, 다시 실행할 날짜를 적는다.

### 7-2. 개인 폴더에서 전체 지시문 최적화 실행

실행 프로젝트 저장소의 최상위 폴더에서 아래 명령을 실행한다. `minsu`만 본인 별칭으로 바꾼다.
같은 별칭으로 다시 실행해도 실행 시각이 달라 새 폴더가 생긴다.

```bash
STUDENT_ALIAS=minsu
RUN_ID=$(date +%Y%m%d-%H%M%S)
STUDENT_RUN_DIR="reports/week-04/student-full/${STUDENT_ALIAS}-${RUN_ID}"
OPTIMIZATION_RUN_DIR="$STUDENT_RUN_DIR/optimization"
ROBUSTNESS_RUN_DIR="$STUDENT_RUN_DIR/robustness"
VARIANTS_DIR="local-data/week-04-students/$STUDENT_ALIAS/variants"
NIM_CATALOG_DATE=$(date +%F)
NIM_PRICING_DATE=$(date +%F)
GEMINI_MODEL_DATE=$(date +%F)
GEMINI_PRICING_DATE=$(date +%F)

uv run --locked python scripts/optimize_open_cqa_prompt.py \
  --live-optimize \
  --max-requests 45 \
  --max-input-tokens 900000 \
  --max-output-tokens 22500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --catalog-verified-on "$NIM_CATALOG_DATE" \
  --pricing-verified-on "$NIM_PRICING_DATE" \
  --optimizer-max-requests 4 \
  --optimizer-max-attempts 8 \
  --optimizer-max-input-tokens 40000 \
  --optimizer-max-output-tokens 16000 \
  --optimizer-max-cost-usd 0.01 \
  --optimizer-max-wall-seconds 7200 \
  --optimizer-catalog-verified-on "$GEMINI_MODEL_DATE" \
  --optimizer-pricing-verified-on "$GEMINI_PRICING_DATE" \
  --output "$OPTIMIZATION_RUN_DIR"
```

명령이 중단돼도 같은 폴더에 이어 쓰거나 지우고 다시 시작하지 않는다. 그 폴더를 `partial`
상태의 근거로 보존하고 다시 실행할 때 새 `RUN_ID`를 사용한다.

다음 명령으로 결과를 읽는다.

```bash
uv run --locked python scripts/inspect_week_04_prompt_results.py \
  --optimization-dir "$OPTIMIZATION_RUN_DIR"
```

`optimization/`에는 `calls.jsonl`, `candidate-prompt.md`, `validation.jsonl`,
`selected-prompt.md`, `summary.json`이 있어야 한다. `summary.json`에서
`run_mode=full_evaluation`, `observed_status=complete`, 18/6/6 분할, 실제로 응답한 모델,
오류 유무, `candidate_changed`, `selected`, `selection_reason`을 확인한다.

### 7-3. 개인 이미지 5건 실행과 평가

지시문 최적화가 끝나면 같은 터미널에서 아래 명령을 실행한다. `--variants-dir`는 5절에서
본인이 직접 보고 판정한 이미지 폴더를 지정한다.

```bash
uv run --locked python scripts/run_image_robustness.py \
  --live \
  --prompt "$OPTIMIZATION_RUN_DIR/selected-prompt.md" \
  --variants-dir "$VARIANTS_DIR" \
  --max-requests 5 \
  --max-input-tokens 100000 \
  --max-output-tokens 2500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 900 \
  --catalog-verified-on "$NIM_CATALOG_DATE" \
  --pricing-verified-on "$NIM_PRICING_DATE" \
  --output "$ROBUSTNESS_RUN_DIR"

uv run --locked python scripts/evaluate_image_robustness.py \
  --variants "$VARIANTS_DIR/variants.jsonl" \
  --reviews "$VARIANTS_DIR/variant-review.csv" \
  --case "$VARIANTS_DIR/case.json" \
  --responses "$ROBUSTNESS_RUN_DIR/responses.jsonl" \
  --output "$ROBUSTNESS_RUN_DIR/evaluation.json"
```

`robustness/`에는 `calls.jsonl`, `responses.jsonl`, `summary.json`, `evaluation.json`,
`evaluation-manifest.json`이 있어야 한다. `summary.json`에서 `observed_status=complete`,
`record_count=5`, `target_count=5`, 실제로 응답한 모델과 오류 유무를 확인한다. 개인 이미지·검토표의
SHA-256이 응답 실행 기록과 다르면 평가 명령이 멈춘다.

두 `summary.json`에서 다음 세 계보 항목도 비교한다. 하나라도 다르면 서로 다른 코드·지시문·입력을
섞은 것이므로 `complete`가 아니다.

- `optimization/summary.json`과 `robustness/summary.json`의 `git_sha`
- 지시문 최적화의 `selected_prompt_sha256`와 이미지 실행의 `prompt_sha256`
- 두 파일의 `artifact_sha256.week-03-cases.jsonl`

### 7-4. 개인 실행 상태 기록

| 상태 | 기록 기준 | 다음 행동 |
| --- | --- | --- |
| `complete` | 두 `summary.json`이 모두 `observed_status=complete`이고 위 결과 파일이 모두 있으며 세 계보가 일치함 | 품질 `pass / fail / inconclusive`와 선택 이유를 따로 해석 |
| `partial` | 실제 호출을 시작했지만 두 실행 중 하나가 끝나지 않았거나 결과 파일이 빠졌거나 세 계보 항목이 일치하지 않음 | 기존 폴더를 보존하고 새 폴더에서 다시 실행 |
| `not_run` | 승인·API 키·API 사용 한도 문제로 실제 호출을 시작하지 않음 | 사유와 승인받은 재실행 날짜 기록 |

`complete`는 필요한 호출이 끝나고 결과 파일이 모두 생긴 상태다. 품질 통과 여부는 별도로
판단한다.

기존 `local-data/learning-progress.md`의 Week 4에 아래 항목이 없으면 추가해 본인 결과를 적는다.

```text
개인 전체 실행 폴더:
지시문 최적화 상태(complete / partial / not_run):
이미지 5건 상태(complete / partial / not_run):
개인 전체 상태(complete / partial / not_run):
계보 3개 확인(일치 / 불일치):
중단 사유와 재실행 폴더·날짜:
```

## 8. 제출

다음 네 경로를 보존한다.

1. `local-data/week-04-students/minsu/variants/variant-review.csv`: 직접 판정한 네 행
2. `reports/week-04/students/minsu/evaluation.json`: 수업용 저장 응답을 현재 채점기로 다시 채점한 결과
3. `local-data/learning-progress.md`: 데이터 분할, 선택 이유와 주장할 수 없는 범위
4. `reports/week-04/student-full/`: 별칭과 실행 시각으로 구분한 개인 전체 실행의 원응답·요약·평가

위 경로의 `minsu`는 본인 별칭이다.

학습 기록에는 다음 문장을 본인의 결과에 맞게 완성한다.

```text
개발 문제 18개를 참고해 바꾼 지시문을 검증 문제 6개에서 비교한 결과 ______을 선택했다.
질문에 필요한 수치가 남은 이미지에는 ______을 기대했다.
질문에 필요한 수치가 사라진 이미지에는 ______을 기대했다.
이 한 번의 수업 결과만으로 다른 문제나 실제 서비스의 ______은 말할 수 없다.
```

API 키, `.env`, OpenCQA 원본과 공통 수업용 실제 실행 폴더는 제출하지 않는다.

## 완료 기준

- 개발·검증·공개 test의 역할을 설명했다.
- Gemma 답을 Python 고정 채점기가 `0.7 × 숫자 F1 + 0.3 × 단어 F1`로 채점하고 Gemini는
  점수와 감점 이유를 읽어 지시문을 고친다는 역할 차이를 설명했다.
- GEPA가 개발 문제에서 실패 분석·지시문 수정·후보 내부 확인을 반복한 뒤, 별도 검증 문제
  6개의 평균으로 최종 지시문을 고르는 과정을 설명했다.
- 준비 스크립트가 현재 `week-03-cases.jsonl`과 저장 결과의 SHA-256을 자동 비교해 같은 입력임을
  확인했다.
- 검증 평균이 높지 않으면 처음 지시문을 유지하는 코드를 확인했다.
- 처음·새 지시문에서 실제로 달라진 문장을 설명했다.
- 점수가 오른 사례와 떨어진 사례에서 모델 답·점수·감점 이유를 같은 `sample_id`로 연결했다.
- 원본과 변형 네 개를 직접 보고 질문에 필요한 근거가 남았는지 사라졌는지 판정했다.
- 저장 VLM 응답 5개를 다시 채점하고 실패 이유를 한 사례에서 연결했다.
- 잘못 만든 이미지(`invalid_variant`), 모델 답 실패(`fail`), 원본 답 또는 API 문제로 판단할
  수 없는 경우(`inconclusive`)를 구분했다.
- 수업 중 개발 사례 2건은 지시문 생성 과정만 확인하고 후보 선택에 쓰지 않았다.
- 최종 지시문과 품질 판단은 개발 18건·검증 6건 전체 저장 기록으로 확인했다.
- 수업 후 개인 전체 실행 폴더와 `complete / partial / not_run` 상태를 학습 진행표에 기록했다.
- `complete`라면 지시문 최적화와 이미지 5건 각각의 `summary.json`이
  `observed_status=complete`이고 필요한 결과 파일이 모두 있으며 세 계보 항목이 일치한다.
