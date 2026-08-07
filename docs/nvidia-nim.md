# NVIDIA NIM 실행 안내

이 문서는 실제 NVIDIA NIM 호출에 공통으로 쓰는 두 스크립트의 역할을 설명한다. 수업에서
입력할 전체 명령은 [Week 1 실습](week-01-lab.md)과 [Week 2 실습](week-02-lab.md)을 따른다.

## 주차별 설정

| 설정 파일 | 모델 | 수업에서 바꾸는 것 |
| --- | --- | --- |
| `configs/nvidia-nim.yaml` | Nemotron | Week 1 기준 작업 흐름 |
| `configs/nvidia-nim-gemma4-baseline.yaml` | Gemma 4 | Week 2 모델 변경 뒤 기준 지시문 |
| `configs/nvidia-nim-gemma4.yaml` | Gemma 4 | 같은 모델에서 개선 지시문만 적용 |

Week 2의 두 Gemma 설정은 데이터와 모델이 같고 지시문·결과 경로만 다르다. 따라서 두 결과의
차이를 지시문 변경과 연결해 살펴볼 수 있다.

## 1. 모델 사전 점검

`preflight_nvidia.py`는 모델 추론을 하지 않고 NVIDIA의 현재 모델 목록만 조회한다.

```bash
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim.yaml
```

Week 2에서는 `--config` 값을 사용할 Gemma 설정으로 바꾼다. 출력은 두 줄이다.

```text
configured model: 설정에 적힌 모델 ID
available now: True
```

`False`면 실제 호출을 진행하지 않는다. 제공 모델 목록은 바뀔 수 있어 과거 카탈로그를 코드에
복사해 두지 않고 현재 설정의 모델 하나만 확인한다.

## 2. 실제 실행

`run_nvidia_nim.py`는 다음 순서로 한 실행(run)을 처리한다.

```text
설정·Git 상태·상한 확인
→ 페이지 JPEG와 질문 준비
→ NIM 호출
→ 원응답 즉시 저장
→ JSON과 Pydantic 출력 형식 검사
→ 고정 규칙 채점
→ DeepEval 결과 저장
```

다음 옵션이 모두 있어야 네트워크 요청을 시작한다.

| 옵션 | 뜻 | 필요한 이유 |
| --- | --- | --- |
| `--live` | 실제 API 호출 허용 | 저장 응답 실행과 혼동하지 않게 한다. |
| `--max-requests` | 최대 요청 수 | 예상보다 많은 호출을 막는다. |
| `--max-input-tokens` | 최대 입력 토큰 | 큰 이미지 요청의 사용량을 제한한다. |
| `--max-output-tokens` | 최대 출력 토큰 | 응답 사용량을 제한한다. |
| `--max-cost-usd` | 최대 추정 비용 | 승인한 비용을 넘기 전에 중단한다. |
| `--max-wall-seconds` | 최대 전체 시간 | 멈추지 않는 실행을 종료한다. |
| `--max-retries` | 최대 재시도 | 오류 뒤 중복 호출 수를 고정한다. |
| `--catalog-verified-on` | 모델 목록을 확인한 날짜 | 오래된 확인 결과로 호출하지 않게 한다. |

`--sample-id`를 주면 한 사례만 실행한다. 전체 실행 전에 이 소규모 사전 실행(probe)으로
원응답, 실제 처리 모델(actual model), 사용량과 오류를 먼저 확인한다.

실제 실행은 변경 사항이 없는 Git 커밋에서만 허용한다. 다음 출력이 없어야 한다.

```bash
git status --short
```

이 제한은 나중에 어떤 코드와 지시문이 응답을 만들었는지 찾기 위한 것이다. 파일을 수정했다면
커밋한 뒤 호출한다.

## 3. 중단된 실행 재개

전체 실행이 중간에 중단됐을 때만 출력에 표시된 실행 식별자(`run_id`)로 재개한다.

```bash
RUN_ID=week01-터미널에-출력된-식별자

uv run --locked python scripts/run_nvidia_nim.py \
  --live --resume --run-id "$RUN_ID" \
  --max-requests 40 --max-input-tokens 800000 \
  --max-output-tokens 20000 --max-cost-usd 0.01 \
  --max-wall-seconds 7200 --max-retries 0 \
  --catalog-verified-on 2026-08-06
```

재개는 처음 실행과 같은 데이터·설정·상한에서만 가능하다. 한 사례 사전 실행을 40건 전체
실행으로 확장하는 기능이 아니다.

## 4. 결과 파일

```text
reports/{설정별 폴더}/runs/{run-id}/
├── run-manifest.json   실행 조건과 입력 식별값
├── budget.json         요청·토큰·비용·시간 누적값
├── observations.jsonl  모델 원응답과 호출 정보
├── records.jsonl       조건·응답·평가 결과를 묶은 기록
├── results.jsonl       사례별 점수와 통과 상태
├── summary.json        전체 요약
└── deepeval/           DeepEval 탐색 결과
```

수업에서는 먼저 `observations.jsonl`에서 모델이 실제로 무엇을 반환했는지 보고,
`results.jsonl`에서 같은 사례의 실패 지표를 확인한 뒤, 마지막으로 `summary.json`의 전체
개수를 읽는다. 지표 뜻은 [수업 도구·채점기·용어](terms-tools-and-scoring.md)에 있다.
