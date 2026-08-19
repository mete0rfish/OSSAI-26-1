# 검증 가능한 AI 작업 흐름(Workflow) 설계·평가 과정

이 교육용 프로젝트에서는 공개 문서와 차트를 읽는 멀티모달 작업 흐름(workflow) 하나를 6주
동안 발전시킨다. 현재 브랜치에는 Week 1부터 Week 4까지의 실습이 담겨 있다.

- Week 1: PDF를 페이지 이미지로 만든다. VLM을 호출해 구조화된 답을 받고 고정 규칙으로 채점한다.
- Week 2: 동일 release baseline을 기준으로 각 학습자가 자기 prompt 40건을 실행·비교한다.
  저장된 개선·provider 결과는 설명 예시와 실패 fallback으로 쓰고, 강의자는 `r01` 한 건까지만
  별도로 시연한다.
- Week 3: 각 학습자가 같은 NIM Gemma에 기준·개선 지시문을 적용해 OpenCQA 실제 답 30개씩을
  만든다. 두 답을 익명 A/B 30쌍으로 묶고 배정된 1쌍의 사람 판단을 먼저 잠근 뒤, Gemini
  3.5 Flash Lite Judge를 두 번·양방향으로 실행한다.
- Week 4: Prompt 최적화를 배운다. NIM Gemma가 개발 문제에 답하면 Gemini가 낮은 점수의
  원인을 읽고 지시문을 고쳐 쓴다. 검증 문제 6개에서 처음·새 지시문을 비교한 뒤 이미지
  변형을 평가한다. 수업 중에는 개발 사례 2건의 생성 과정을 실제로 먼저 보여 주고, 그다음
  수업 전에 저장한 전체 결과를 열어 지시문 선택과 품질을 판단한다. 수업 후에는 각 학습자가
  개인 폴더에서 전체 최적화와 이미지 5건을 실행한다.
- Week 5 이후: 도구 호출과 CI를 같은 작업 흐름에 추가한다.

처음 실습한다면 [Week 1 실습](docs/week-01-lab.md),
[Week 2 실습](docs/week-02-lab.md), [Week 3 실습](docs/week-03-lab.md),
[Week 4 실습](docs/week-04-lab.md) 순서로 진행한다.
낯선 용어나 도구는 [수업 도구·채점기·용어](docs/terms-tools-and-scoring.md)에서 확인한다.

## 수업에서 먼저 보는 한 사례

Week 1–2는 전체 평균을 보기 전에 대표 사례 한 건을 다음 순서로 읽는다. Week 3는 Judge
결과를 보기 전에 차트·질문·후보 한 쌍만 읽고 사람 사전 label을 먼저 작성한다.

```text
페이지 이미지와 질문
→ 모델 원응답
→ 구조화된 답
→ 기대 답과 근거 페이지
→ 고정 규칙 채점 결과
```

| 주차 | 명령 | 확인할 내용 |
| --- | --- | --- |
| Week 1 | `uv run --locked python scripts/inspect_deterministic_scoring_case.py` | 한 답이 왜 통과하거나 실패하는지 |
| Week 2 | `uv run --locked python scripts/inspect_prompt_comparison_case.py` | 미리 준비한 같은 모델의 기준·후보 응답과 점수 차이 |
| Week 3 | `uv run --locked python scripts/inspect_judge_pair.py --candidates "$CANDIDATE_RESULTS" --number "$PAIR_NUMBER"` | 개인 후보 생성 뒤, 결과 공개 전에 사람이 판단할 차트·질문·후보 한 쌍 |
| Week 4 | `uv run --locked python scripts/generate_image_variants.py --pair-number 1` | 원본과, 질문에 필요한 수치가 남거나 사라진 변형 네 개 |

위 명령들은 외부 API를 호출하지 않는다. Week 1–2의 Git 고정 응답은 코드 학습과 회귀검사용
`test_only`다. Week 3 명령은 [Week 3 실습](docs/week-03-lab.md)에서 만든 개인 후보 경로와 배정
번호를 사용한다. `--human-label`을 넘기면 작성한 사람 사전 label도 검증한다. 실제 API로 수집한
원본은 저장되어 있어도 `live_quality`다. 실행 조건과 완전성을 확인한 뒤에만 모델 품질을
판단한다.

## 6주 학습 경로

| 주차 | 배우는 내용 | 결과물 |
| --- | --- | --- |
| Week 1 | 이미지 입력, 구조화 출력, 고정 규칙 채점기(deterministic scorer) | 질문·답·근거 페이지를 검사하는 첫 작업 흐름 |
| Week 2 | 동일 release baseline과 자기 prompt 40건, 저장된 두 API 예시 비교 | 개인 원본·요약·baseline 비교, 경로 묶음·오류 구분 |
| Week 3 | NIM Gemma 기준·개선 실제 답과 Gemini Judge 30쌍 | 사람 사전 label, Judge trial 결과 60행의 위치·반복 충돌과 사용 한계 |
| Week 4 | 이미지 변형과 지시문 최적화 | 개인 전체 최적화·이미지 5건 실행의 원응답, 지시문 선택과 안전 비교 |
| Week 5 | 도구 호출 기록(trace)과 최종 상태 | 결과뿐 아니라 실행 과정까지 포함한 평가 |
| Week 6 | PR·정기 평가·출시 판단 | 자동 검사 결과와 사람의 최종 결정 |

Week 3–6에 필요한 실행 식별, 비용 상한, 오류 보존 기능은 공통 코드에 남아 있다. 각 주차의
실습 문서에서는 이 기능을 모두 설명하지 않고, 필요한 시점과 이유만 다룬다.

## 환경 준비

필요한 도구는 Python 3.12, `uv`, Git, 결과에서 필요한 줄을 찾는 `rg`(ripgrep)다. Docker는
사용하지 않는다.

```bash
uv python install 3.12
uv sync --locked --dev
uv run --locked python scripts/check_environment.py
```

`uv sync --locked --dev`는 `uv.lock`에 기록된 버전대로 실행 환경과 수업용 개발 도구를
설치한다. `--locked`는 잠금 파일을 임의로 바꾸지 않으며, `--dev`는 pytest와 Ruff도 함께
설치한다.

## 데이터 준비

Week 1은 AIHub `멀티모달 정보검색 데이터_Sample`의 보고서 PDF 1개와 보도자료 PDF
1개를 사용한다. 두 문서를 바탕으로 답을 찾는 질문 36건과 답이 없어 보류하는 질문 4건을
구성한다.

AIHub 원본과 라벨은 Git에 올리지 않는다. 내려받은 폴더를 다음 위치에 둔다.

```text
local-data/aihub/source/
├── 01.원천데이터/
└── 02.라벨링데이터/
```

전처리 결과는 `local-data/aihub/prepared/`, 실행 결과는 `reports/`에 생성되며 둘 다 Git에서
제외된다. 자세한 경로는 [AIHub 데이터 준비](docs/aihub-data.md)를 따른다.

Week 3 OpenCQA 원본도 Git에 넣지 않는다. [OpenCQA 데이터 준비](docs/open-cqa-data.md)를
따라 선택한 차트·질문·사람 작성 기준 답 30개를 `local-data/opencqa/`에 만든다. 사람 작성
`abstractive_answer`는 기대 답이며 후보가 아니다. 후보 A/B는 각 학습자가 같은 NIM Gemma에
기준·개선 지시문을 적용해 만든 실제 답이다.

Week 2 개인 prompt는 `local-data/week-02-students/<alias>/prompt.md`에 두고, 개인 원본·요약은
`reports/week-02-gemma-baseline/runs/`, baseline 비교는 `reports/week-02/students/<alias>/`에
저장한다. Week 3의 `human-label.yaml`·`interpretation.md`는
`local-data/week-03-student-judges/<alias>/`에 둔다. 개인 후보 생성의 호출·결과·요약·두
지시문 snapshot과 Gemini Judge 호출·결과·요약·비교는
`reports/week-03/student-full/<alias-시각>/candidates/`와 `judge/`에 나눠 보존한다. Week 4
개인 전체 실행 결과는 `reports/week-04/student-full/<alias-시각>/optimization/`과 `robustness/`에
나눠 보존한다. 전체 실행 명령은 아래 주차 실습서에만 둔다.

모델에는 PDF 문장을 보내지 않는다. PDF를 페이지 JPEG로 바꿔 VLM이 이미지에서 직접 읽도록
한다. 전처리할 때는 원본·라벨 확인용 텍스트도 저장하지만, 모델 입력이나 채점에는 사용하지
않는다.

## 실제 API 모델

| 주차와 역할 | API 제공자 | 요청 모델 |
| --- | --- | --- |
| Week 1 기준 | NVIDIA NIM | `nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| Week 2 기준·개선 | NVIDIA NIM | `nvidia_nim/google/gemma-4-31b-it` |
| Week 2 비교 후보 | Google AI Studio | `gemini/gemini-3.5-flash-lite` |
| Week 3 기준·개선 답 생성 | NVIDIA NIM | `nvidia_nim/google/gemma-4-31b-it` |
| Week 3 Judge | Google AI Studio | `gemini/gemini-3.5-flash-lite` |
| Week 4 최적화 타깃·견고성 | NVIDIA NIM | `nvidia_nim/google/gemma-4-31b-it` |
| Week 4 GEPA 검토 | Google AI Studio | `gemini/gemini-3.5-flash-lite` |

Gemini 3.5 Flash Lite는 현재 잠긴 LiteLLM adapter로 요청 모델·실제 처리 모델과 구조화 출력을
확인하므로 Week 2·3 Judge와 Week 4 지시문 검토에 쓴다. 2026-08-17에 OpenCQA JPEG·질문·
기대 답·Gemma 실제 출력을 Google로 보내는 범위를 승인했다. Week 4 검토에는 JPEG 대신
지시문·질문·기대 답·NIM 출력·고정 점수와 이유를 보낸다. Free Tier 자료가 제품 개선에
사용될 수 있다는 조건도 실행 전에 다시 확인한다.

승인 때 확인한 공개 한도는 15 RPM, 입력 250,000 TPM, 500 RPD다. 실제 실행은 현재 프로젝트의
할당량과 당일 잔여 RPD가 240건 이상인지 사전 점검한다. 코드는 15 RPM·입력 75,000 TPM·
출력 7,500 TPM과 한 full run당 최대 240요청을 적용한다. 하루 누적 요청은 추적하지 않으므로,
같은 프로젝트에서 여러 명이 실행하면 240N 요청이 500 RPD를 넘을 수 있다. Free Tier 입력·출력
단가는 0달러로 계산하되 비용 안전장치는 0.01달러로 둔다. 30쌍은 실제로 120~240회 요청하며
pacing에만 약 8~16분, 여기에 API 응답 시간이 더 걸린다. 전송 승인은 전체 실행 성공을 뜻하지
않는다. 새 결과가 완결 검사를 통과하기 전에는 `not_run` 또는 `partial`로 기록한다.

`.env.example`을 복사한 뒤 사용할 API 키만 입력한다.

```bash
cp .env.example .env
```

```dotenv
NVIDIA_NIM_API_KEY="Week-1~4-NIM을-호출할-때-입력"
GEMINI_API_KEY="Week-2-비교·Week-3-Judge·Week-4-GEPA-검토를-호출할-때-입력"
DEEPEVAL_DISABLE_DOTENV=1
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

`.env`는 Git에서 제외된다. 외부 전송 자료와 실행 상한은
[실제 API 실행 승인 범위](docs/live-api-approval.md), NIM 호출 방법은
[NVIDIA NIM 실행 안내](docs/nvidia-nim.md)를 확인한다.

## 학습 자료

- [Week 1 실습](docs/week-01-lab.md): 환경 준비부터 한 사례·40건 실행과 고정 규칙 채점까지
- [Week 2 실습](docs/week-02-lab.md): 자기 prompt 40건과 동일 release baseline, 저장된 Gemma–Gemini 예시 비교
- [Week 3 실습](docs/week-03-lab.md): 개인 Gemma 실제 답 60개, 사람 사전 label과 Gemini Judge 30쌍
- [Week 4 실습](docs/week-04-lab.md): GEPA 지시문 최적화와 원본·변형 이미지 견고성 평가
- [수업 도구·채점기·용어](docs/terms-tools-and-scoring.md): 라이브러리, 지표, 실행 용어의 뜻
- [코드 구조](docs/architecture.md): 실행 파일과 내부 코드의 연결
- [AIHub 데이터 준비](docs/aihub-data.md): 원본 위치와 전처리 결과
- [OpenCQA 데이터 준비](docs/open-cqa-data.md): 공식 원본 revision과 로컬 30쌍 준비
- [NVIDIA NIM 실행 안내](docs/nvidia-nim.md): 사전 점검과 실제 호출 안전장치
- [실제 API 실행 승인 범위](docs/live-api-approval.md): 외부 전송 자료와 호출 상한
