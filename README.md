# 검증 가능한 AI 작업 흐름(Workflow) 설계·평가 과정

공개 문서와 차트를 읽는 하나의 멀티모달 작업 흐름(workflow)을 6주 동안 발전시키는
교육용 프로젝트다. 현재 브랜치는 Week 1과 Week 2까지의 실습을 담는다.

- Week 1: PDF를 페이지 이미지로 만들고, VLM을 호출하고, 구조화된 답을 고정 규칙으로 채점한다.
- Week 2: 모델과 지시문(prompt)을 한 번에 하나씩 바꾸며 결과를 비교한다.
- Week 3 이후: 모델 기반 채점, 견고성, 도구 호출과 CI를 같은 작업 흐름에 추가한다.

처음 실습한다면 [Week 1 실습](docs/week-01-lab.md),
[Week 2 실습](docs/week-02-lab.md) 순서로 진행한다. 낯선 용어와 도구는
[수업 도구·채점기·용어](docs/terms-tools-and-scoring.md)에서 한 번에 확인할 수 있다.

## 수업에서 먼저 보는 한 사례

전체 평균을 보기 전에 대표 사례 한 건을 다음 순서로 읽는다.

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
| Week 2 | `uv run --locked python scripts/inspect_prompt_comparison_case.py` | 같은 모델에서 지시문만 바꿨을 때 무엇이 달라졌는지 |

두 명령은 저장 응답(fixture)을 사용하므로 코드 학습과 회귀검사용이다. 현재 모델 품질은
실제 API로 새로 얻은 응답에서만 판단한다.

## 6주 학습 경로

| 주차 | 배우는 내용 | 결과물 |
| --- | --- | --- |
| Week 1 | 이미지 입력, 구조화 출력, 고정 규칙 채점기(deterministic scorer) | 질문·답·근거 페이지를 검사하는 첫 작업 흐름 |
| Week 2 | 지시문 A/B와 두 API 제공자(provider) 비교 | 사례별 변화와 공통 실패 목록 |
| Week 3 | 모델 기반 채점기(LLM judge)와 사람 평가 보정 | 채점기가 맡을 범위와 신뢰 기준 |
| Week 4 | 이미지 변형과 지시문 최적화 | 원본·변형 입력의 품질 및 안전 비교 |
| Week 5 | 도구 호출 기록(trace)과 최종 상태 | 결과뿐 아니라 실행 과정까지 포함한 평가 |
| Week 6 | PR·정기 평가·출시 판단 | 자동 검사 결과와 사람의 최종 결정 |

Week 3–6에 필요한 실행 식별, 비용 상한, 오류 보존 기능은 공통 코드에 남아 있다. 현재
주차의 실습 문서는 그 기능을 모두 다루지 않고, 사용하는 시점과 이유만 설명한다.

## 환경 준비

필요한 도구는 Python 3.12, `uv`, Git이다. Docker는 사용하지 않는다.

```bash
uv python install 3.12
uv sync --locked --dev
uv run --locked python scripts/check_environment.py
```

`uv sync --locked --dev`는 `uv.lock`에 기록된 버전대로 실행 환경과 수업용 개발 도구를
설치한다. `--locked`는 잠금 파일을 임의로 바꾸지 않고, `--dev`는 pytest와 Ruff도 함께
설치한다.

## 데이터 준비

Week 1은 AIHub `멀티모달 정보검색 데이터_Sample`의 보고서 PDF 1개와 보도자료 PDF
1개를 사용한다. 두 문서에서 답을 찾는 질문 36건과 답이 없어 보류해야 하는 질문 4건을
구성한다.

AIHub 원본과 라벨은 Git에 올리지 않는다. 내려받은 폴더를 다음 위치에 둔다.

```text
local-data/aihub/source/
├── 01.원천데이터/
└── 02.라벨링데이터/
```

전처리 결과는 `local-data/aihub/prepared/`, 실행 결과는 `reports/`에 생성되며 둘 다 Git에서
제외된다. 자세한 경로는 [AIHub 데이터 준비](docs/aihub-data.md)를 따른다.

모델에는 PDF 문장을 보내지 않는다. PDF를 페이지 JPEG로 바꾸고 VLM이 이미지에서 직접
읽게 한다. 전처리 때 저장되는 텍스트는 원본·라벨 확인용이며 모델 입력과 채점에 사용하지
않는다.

## 실제 API 모델

| 주차와 역할 | API 제공자 | 요청 모델 |
| --- | --- | --- |
| Week 1 기준 | NVIDIA NIM | `nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| Week 2 기준·개선 | NVIDIA NIM | `nvidia_nim/google/gemma-4-31b-it` |
| Week 2 비교 후보 | Google AI Studio | `gemini/gemini-3.5-flash-lite` |

`.env.example`을 복사한 뒤 사용할 API 키만 입력한다.

```bash
cp .env.example .env
```

```dotenv
NVIDIA_NIM_API_KEY="발급받은-키를-여기에-입력"
GEMINI_API_KEY="Week-2-Gemini를-호출할-때만-입력"
DEEPEVAL_DISABLE_DOTENV=1
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

`.env`는 Git에서 제외된다. 외부로 보내는 자료와 실행 상한은
[실제 API 실행 승인 범위](docs/live-api-approval.md), NIM 호출 방법은
[NVIDIA NIM 실행 안내](docs/nvidia-nim.md)를 확인한다.

## 학습 자료

- [Week 1 실습](docs/week-01-lab.md): 환경 준비부터 한 사례·40건 실행과 고정 규칙 채점까지
- [Week 2 실습](docs/week-02-lab.md): Gemma 지시문 개선과 Gemma–Gemini 비교까지
- [수업 도구·채점기·용어](docs/terms-tools-and-scoring.md): 라이브러리, 지표, 실행 용어의 뜻
- [Week 2 비교 결과](docs/week-02-gemma-gemini-comparison-report-2026-08-05.md): 실제 결과를 읽는 예시
- [코드 구조](docs/architecture.md): 실행 파일과 내부 코드의 연결
- [AIHub 데이터 준비](docs/aihub-data.md): 원본 위치와 전처리 결과
- [NVIDIA NIM 실행 안내](docs/nvidia-nim.md): 사전 점검과 실제 호출 안전장치
- [실제 API 실행 승인 범위](docs/live-api-approval.md): 외부 전송 자료와 호출 상한
