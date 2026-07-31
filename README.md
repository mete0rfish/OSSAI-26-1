# 검증 가능한 AI Workflow 설계·평가 과정

공개 문서와 차트를 읽는 AI workflow 하나를 6주 동안 같은 코드베이스에서 발전시키는
교육용 프로젝트다. 실제 모델 API 응답을 구조화하고, 정답과 근거를 평가한 뒤, 모델 비교,
LLM-as-a-Judge, 견고성, 도구 실행과 CI까지 단계적으로 연결한다.

현재 저장소에는 독립적으로 실행할 수 있는 Week 1 코드가 구현되어 있다. Week 2~6은 이
코드를 복사하지 않고 전주 완료 상태에서 누적 개발한다. Week 1의 상세 명령과 수업 진행은
[Week 1 실습 안내](docs/week-01-lab.md)를 따른다.

## 전체 실행 전에 사례 한 건 보기

전체 평균이나 운영 설정부터 설명하지 않는다. 먼저 대표 사례 한 건을 실행하고
`input → model_output → expected → evaluation_design → evaluation_result` 순서로 읽는다.
스크립트는 기존 데이터 읽기 코드와 채점 함수를 그대로 호출하는 직선형 실습 코드다.
새 평가 엔진이나 별도 추상 계층은 없다. 아래 명령은 환경 설치와 AIHub 문서 전처리를
먼저 끝낸 뒤 실행한다.

| 주차 | 실행 명령 | 한 화면에서 확인하는 내용 |
| --- | --- | --- |
| 1 | `uv run --locked python scripts/show_week01_case.py` | 현재 준비된 모델 입력 JPEG 9장, 질문, 저장 원응답, 기대 답, PDF 문장에 근거한 점수 |

이 명령은 수업에서 계산을 반복할 수 있도록 저장 응답을 사용하므로 `test_only`다.
과거 실제 모델 원응답을 보여 주지만, 현재 준비된 이미지와 당시 응답을 같은 요청으로
묶는 입력 hash는 없다. 현재 모델 품질은 승인된 실제 API 실행 결과에서 확인해야 한다.

## 교육과정

| 주차 | 학습 주제 | 누적 결과 |
| --- | --- | --- |
| Week 1 | PDF 전처리, 실제 API 호출, 구조화 응답, 결정적 평가 | 질문·답변·근거 페이지를 검증하는 첫 workflow |
| Week 2 | 같은 조건에서 두 provider 비교 | 품질과 가용성을 분리한 비교 결과 |
| Week 3 | LLM-as-a-Judge와 사람 평가 보정 | Judge를 사용할 수 있는 범위와 기준 |
| Week 4 | prompt 최적화와 멀티모달 견고성 | 원본·변형 입력의 품질 및 안전 비교 |
| Week 5 | 도구 호출, trace와 최종 상태 | 실행 과정과 side effect까지 포함한 평가 |
| Week 6 | PR·nightly·weekly 평가와 release 판단 | 재현 가능한 CI 결과와 사람의 최종 결정 |

Week 1은 현재 구현되어 있으며 Week 2~6의 PDF workflow 연동은 후속 과정에서 구현한다.
저장 응답은 코드 회귀검사용이고, 모델 품질은 fallback을 끈 실제 API 응답으로만 판단한다.

## 필요한 환경

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package·가상환경 관리자
- Git
- NVIDIA 계정과 NIM API key
- 수업용 AIHub 샘플 데이터

Docker는 사용하지 않는다. 저장소를 내려받은 뒤 이 README가 있는 폴더에서 다음을
실행하면 같은 `uv.lock`을 기준으로 환경이 준비된다.

```bash
uv python install 3.12
uv sync --locked --dev
uv run python scripts/check_environment.py
```

JupyterLab도 개발 의존성에 포함되어 있다. Notebook으로 학습하려면 다음 명령을
실행하고 `notebooks/week-01-aihub-pdf-workflow.ipynb`을 연다.

```bash
uv run jupyter lab
```

## 필수 라이브러리

| 라이브러리 | 이 과정에서 하는 일 |
| --- | --- |
| LiteLLM | NVIDIA NIM을 포함한 모델 API를 같은 호출 방식으로 연결 |
| Pydantic v2 | 모델의 JSON 응답과 답변·근거 사이의 규칙 검증 |
| DeepEval OSS | 실제 응답과 저장 응답의 평가, TestRun 저장과 실패 탐색 |
| pypdfium2, Pillow | PDF를 페이지 이미지와 근거 확인용 text로 전처리 |
| PyYAML | 문서·모델·실행 설정 읽기 |
| python-dotenv | Git에 넣지 않는 로컬 API key 읽기 |
| pytest, Ruff | API 없이 구성요소와 코드 품질 검사 |
| JupyterLab, nbconvert | 수업용 Notebook 실행과 검증 |

정확한 버전 범위는 `pyproject.toml`, 실제 설치 버전은 `uv.lock`이 기준이다.

## 데이터셋

Week 1은 AIHub `멀티모달 정보검색 데이터_Sample`에서 보고서 PDF 1개와 보도자료 PDF
1개를 사용한다. 두 문서로 질문 40건을 구성하며, 문서에서 답을 찾는 36건과 답이 없을 때
보류하는 4건을 함께 평가한다.

AIHub에서 받은 원본과 라벨은 Git에 올리지 않는다. 다운로드한 샘플의 폴더 구조를
유지해 다음 위치에 넣는다.

```text
local-data/aihub/source/
├── 01.원천데이터/
└── 02.라벨링데이터/
```

전처리 결과와 실행 결과도 각각 `local-data/aihub/prepared/`와 `reports/`에 생성되며
Git에서 제외된다. 문서 이름, 질문 구성과 다른 경로를 지정하는 방법은
[AIHub 데이터 준비](docs/aihub-data.md)에서 확인한다.

## NVIDIA NIM API

Week 1의 task model은 NVIDIA hosted NIM endpoint를 LiteLLM adapter로 호출한다.

| 항목 | 설정 |
| --- | --- |
| API base | `https://integrate.api.nvidia.com/v1` |
| 환경 변수 | `NVIDIA_NIM_API_KEY` |
| 수업 기준 모델 | `google/gemma-4-31b-it` |
| LiteLLM model ID | `nvidia_nim/google/gemma-4-31b-it` |

Gemma 4는 이미지와 한국어를 함께 처리하는 다국어 모델이다. 이 실습은 한국어 문서와
질문을 사용하므로 영어 중심 모델 대신 Gemma 4를 고정한다.

`.env.example`을 `.env`로 복사한 뒤 비어 있는 `NVIDIA_NIM_API_KEY`에 발급받은 key만
입력한다.

```bash
cp .env.example .env
```

```dotenv
NVIDIA_NIM_API_KEY="<발급받은 key>"
DEEPEVAL_DISABLE_DOTENV=1
DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

이 파일에서 secret은 `NVIDIA_NIM_API_KEY` 하나다. 나머지 두 값은 DeepEval의 로컬 실행
방식을 정하는 설정이며 그대로 둔다. `.env`는 Git에서 제외된다.

NIM 모델 제공 상태는 바뀔 수 있으므로 실제 호출 전 preflight로 확인한다. 모델 선택과
지원 입력·언어 정보는 [NVIDIA NIM 모델 카탈로그](docs/nvidia-model-catalog.md), API
실행 규칙은 [NVIDIA NIM 안내](docs/nvidia-nim.md)를 참고한다.

## 학습 자료

- [Week 1 실습 안내](docs/week-01-lab.md): 환경 설치부터 실제 40건 호출과 회귀평가까지
- [Notebook](notebooks/week-01-aihub-pdf-workflow.ipynb): 같은 workflow를 셀 단위로 실행
- [폴더와 실행 흐름](docs/architecture.md): 전처리·호출·검증·평가 코드의 경계
- [AIHub 데이터 준비](docs/aihub-data.md): 원본 위치와 전처리 결과
- [NVIDIA NIM 안내](docs/nvidia-nim.md): key, model preflight와 호출 제한
- [NVIDIA NIM 모델 카탈로그](docs/nvidia-model-catalog.md): 확인된 수업용 모델 후보

처음 실습한다면 환경과 데이터를 준비한 뒤 [Week 1 실습 안내](docs/week-01-lab.md)의
순서를 그대로 진행한다.
