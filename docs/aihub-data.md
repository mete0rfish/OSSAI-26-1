# AIHub 데이터 준비

이 실습은 AIHub `멀티모달 정보검색 데이터_Sample`에 포함된 PDF 두 개를 사용한다.

| 문서 | 종류 | 페이지 | 질문 |
| --- | --- | ---: | ---: |
| `MI2_240819_TY1_0012.pdf` | 경제전망 보고서 | 9 | 32 |
| `MI2_240725_TY2_0002.pdf` | 감염병전문병원 보도자료 | 3 | 8 |

질문 40건 중 36건은 문서에서 답을 찾는 문제이고, 4건은 문서에 답이 없을 때 답변을
보류하는 문제다.

## 데이터 넣기

다운로드한 샘플 폴더의 내용물을 프로젝트 안의 `local-data/aihub/source/`에 넣는다.

```text
local-data/aihub/source/
├── 01.원천데이터/
│   ├── 01.보고서/
│   └── 02.보도자료/
└── 02.라벨링데이터/
    ├── 01.보고서/
    └── 02.보도자료/
```

폴더 안쪽의 파일 이름과 구조는 다운로드한 상태 그대로 사용한다.

다른 위치에 데이터를 두었다면 경로를 직접 지정할 수 있다.

```bash
uv run python scripts/prepare_documents.py \
  --source-dir "<멀티모달 정보검색 데이터_Sample 경로>"
```

## PDF 준비하기

```bash
uv run python scripts/prepare_documents.py
```

명령을 실행하면 PDF의 각 페이지가 사람이 확인하는 PNG, API용 JPEG와 라벨 작성·점검용
텍스트로 변환되고 문서 정보가 목록 파일(`manifest.json`)에 저장된다. 이 텍스트는 작업
모델(task model) 입력이나 고정 규칙 채점에 사용하지 않는다.

```text
local-data/aihub/prepared/
├── MI2_240819_TY1_0012/
│   ├── manifest.json
│   ├── pages/
│   ├── model-pages/
│   └── text/
└── MI2_240725_TY2_0002/
    ├── manifest.json
    ├── pages/
    ├── model-pages/
    └── text/
```

## 질문 준비하기

사람이 읽고 수정하는 질문은 `data/cases/week-01-aihub.yaml`에 있다.

```bash
uv run python scripts/prepare_cases.py
```

명령을 실행하면 작업 흐름(workflow)이 읽을 `local-data/aihub/cases.jsonl`이 만들어진다.
각 질문에는 문서 ID, 질문, 기대 답과 가능한 근거 페이지가 들어 있다. 같은 답이 여러
페이지에 반복되면 `pages`에 모두 적고 모델은 그중 하나를 인용하면 된다.

원문 대조 과정에서 보도자료 P07의 근거를 2쪽에서 3쪽으로 바로잡았고, P08은 동일 표가
1쪽과 3쪽에 있어 두 페이지를 모두 가능한 근거로 기록했다. 나머지 표·차트 숫자 중
PDF 텍스트 계층(text layer)에 나오지 않는 값은 탐색 단계(EDA)에서 별도 확인하고 원본
페이지 이미지로 사람이 검토한다.

현재 자동 점검 결과는 정답 30건 일치, 텍스트 계층으로 확인할 수 없는 표·복합 날짜 6건,
답변 보류 수동 검토 4건, 페이지 라벨 불일치 0건이다. 자동 점검 통과는 두 사람 검토를
대체하지 않는다.
