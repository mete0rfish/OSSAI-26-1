# NVIDIA NIM 실행 안내

`preflight_nvidia.py`로 현재 모델 목록을 확인한 뒤 Week 1–2는 `run_nvidia_nim.py`, Week 3은
`run_open_cqa_candidates.py`로 실제 NVIDIA NIM을 호출한다. 수업에서 입력할 전체 명령은
[Week 1 실습](week-01-lab.md), [Week 2 실습](week-02-lab.md), [Week 3 실습](week-03-lab.md)을 따른다.

## 주차별 설정

| 설정 파일 | 모델 | 수업에서 바꾸는 것 |
| --- | --- | --- |
| `configs/nvidia-nim.yaml` | Nemotron | Week 1 기준 작업 흐름 |
| `configs/nvidia-nim-gemma4-baseline.yaml` | Gemma 4 | Week 2 모델 변경 뒤 기준 지시문 |
| `configs/nvidia-nim-gemma4.yaml` | Gemma 4 | 같은 모델에서 개선 지시문만 적용 |
| `configs/week-03-candidates.yaml` | Gemma 4 | OpenCQA 기준·개선 답 30개씩 생성 |

Week 2의 두 Gemma 설정은 데이터와 모델이 같고 지시문·결과 경로만 다르다. 저장된 수업
예시는 이 두 설정을 쓴다. 학습자 개인 비교는 기준 설정
`configs/nvidia-nim-gemma4-baseline.yaml`에
`local-data/week-02-students/$STUDENT_ALIAS/prompt.md`를 `--prompt`로
덮어써서 prompt만 바꾼다.

## 1. 모델 사전 점검

`preflight_nvidia.py`는 모델 추론을 하지 않고 NVIDIA의 현재 모델 목록만 조회한다.

```bash
uv run --locked python scripts/preflight_nvidia.py \
  --config configs/nvidia-nim.yaml
```

Week 2에서는 `--config` 값을 사용할 Gemma 설정으로, Week 3에서는
`configs/week-03-candidates.yaml`로 바꾼다. 출력은 두 줄이다.

```text
configured model: 설정에 적힌 모델 ID
available now: True
```

`False`면 실제 호출을 진행하지 않는다. 제공 모델 목록은 바뀔 수 있으므로 과거 카탈로그를 코드에
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

네트워크 요청을 시작하려면 다음 옵션이 모두 있어야 한다.

| 옵션 | 뜻 | 필요한 이유 |
| --- | --- | --- |
| `--live` | 실제 API 호출 허용 | 저장 응답 실행과 혼동하지 않게 한다. |
| `--max-requests` | 최대 요청 수 | 예상보다 많은 호출을 막는다. |
| `--max-input-tokens` | 최대 입력 토큰 | 큰 이미지 요청의 사용량을 제한한다. |
| `--max-output-tokens` | 최대 출력 토큰 | 응답 사용량을 제한한다. |
| `--max-cost-usd` | 비용 안전 상한 | 설정으로 계산한 사용량이 승인 범위를 넘기 전에 중단한다. |
| `--max-wall-seconds` | 최대 전체 시간 | 멈추지 않는 실행을 종료한다. |
| `--max-retries` | 최대 재시도 | 오류 뒤 중복 호출 수를 고정한다. |
| `--catalog-verified-on` | 모델 목록 확인 날짜 | 오래된 확인 결과로 호출하지 않게 한다. |
| `--pricing-verified-on` | 공식 가격표 확인 날짜 | 설정의 단가를 실행 전에 다시 확인하게 한다. |

`--sample-id aihub-report-r01`을 주면 한 사례만 실행한다. Week 2 수업에서는
강의자가 이 한 건만 화면에 시연한다. 학습자는 `--sample-id` 없이 자기 prompt로
40건 전체를 실행한다. `--prompt`는 `local-data` 아래에 이미 있는 파일만
허용한다.

전체 40건을 실행할 때는 Git 커밋에 변경 사항이 없어야 한다. 한 사례 probe는 지시문을
수정하며 관찰할 수 있으므로 이 검사를 요구하지 않는다. 전체 실행 전에는 다음 출력이
없어야 한다.

```bash
git status --short
```

이렇게 제한하면 어떤 코드와 지시문이 응답을 만들었는지 나중에 찾을 수 있다. 파일을 수정했다면
커밋한 뒤 호출한다.

### Week 2 학습자 full live 계약

튜터가 같은 release의 clean commit에서 만든 `WEEK2_BASELINE_RUN`을 먼저 배포한다.
학습자는 그 baseline의 `provenance.git_sha`와 현재 `git rev-parse HEAD`가
같고 `git status --short` 출력이 없을 때만 실행한다. 전체 명령은
[Week 2 실습](week-02-lab.md#5-내-지시문으로-같은-40건-전체-실행하기)에 있다.

승인된 학습자 한 명당 상한은 다음과 같다.

```text
target 40건 · max requests/attempts 40/40 · input/output token 800000/20000
max cost $0.01 · max wall 7200초 · retry 0
```

`--max-requests 40`, `--max-input-tokens 800000`,
`--max-output-tokens 20000`, `--max-cost-usd 0.01`,
`--max-wall-seconds 7200`, `--max-retries 0` 중 하나라도 다르면 runner가
호출 전에 차단한다. 모델 목록과 가격·이용 조건을 실제로 확인한 날짜도 각각
`--catalog-verified-on`, `--pricing-verified-on`에 넣어야 한다.

새 실행 식별자는 자동 생성된다. 학습자 실습에서는 고유 `--trial-id`와 학생별 log를
쓰고, `tee`로 보존한 터미널의 `run directory:` 줄을 그대로 읽어
`LEARNER_RUN_DIR`로 잡는다. 여러 학생이 같은 장비를 써도 “가장 최근 폴더”를
고르거나 경로를 추측하지 않는다. 저장 baseline과의 비교 결과는
`reports/week-02/students/$STUDENT_ALIAS/` 아래의 고유 파일에 쓴다.

## 3. 중단된 실행 재개

전체 실행이 중간에 멈췄을 때만 출력에 표시된 실행 식별자(`run_id`)로 재개한다.

```bash
RUN_ID=week01-터미널에-출력된-식별자
RUN_CONFIG=configs/nvidia-nim.yaml
CATALOG_DATE=2026-08-08
PRICING_DATE=2026-08-08

uv run --locked python scripts/run_nvidia_nim.py \
  --config "$RUN_CONFIG" \
  --live --resume --run-id "$RUN_ID" \
  --max-requests 40 --max-input-tokens 800000 \
  --max-output-tokens 20000 --max-cost-usd 0.01 \
  --max-wall-seconds 7200 --max-retries 0 \
  --catalog-verified-on "$CATALOG_DATE" \
  --pricing-verified-on "$PRICING_DATE"
```

`RUN_CONFIG`, `CATALOG_DATE`, `PRICING_DATE`에는 새 값을 넣지 않는다. 처음 실행 폴더의
`run-manifest.json`에서 `contract.provenance.config_path`,
`contract.provenance.catalog_verified_on`, `contract.provenance.pricing_verified_on`을 읽어 그대로
복사한다. 재개는 처음 실행과 같은 데이터·설정·상한에서만 가능하다. 한 사례 사전 실행을 40건 전체 실행으로
확장하는 기능이 아니다.

Week 2 학습자가 첫 실행에서 개인 prompt를 썼다면 재개 명령에도
`--prompt "local-data/week-02-students/$STUDENT_ALIAS/prompt.md"`를 추가하고 파일을
수정하지 않는다. 처음 저장한 `prompt.md`와 hash가 다르면 재개가 차단된다.

## 4. 결과 파일

```text
reports/{설정별 폴더}/runs/{run-id}/
├── run-manifest.json   실행 조건과 입력 식별값
├── prompt.md           이번 실행에 실제로 사용한 지시문
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

`observed_status`는 실행 기록의 완성 여부다. `complete`면 목표한 사례의 응답이 오류와
실제 처리 모델 불일치 없이 모두 저장된 것이다. `status`는 품질 자동 상태다. 유효한 전체
실행에서 모든 사례가 성공하면 `pass`, 하나라도 실패하면 `fail`이다. probe 또는 실행이
불완전하면 `inconclusive`다.

전체 실행은 `status=fail`이면 exit 1을 반환할 수 있다. 이때도
`observed_status=complete`, `record_count=40`, `target_count=40`이면
원본 실행은 완결됐다. Week 2에서는 `prompt.md`, `observations.jsonl`,
`results.jsonl`, `summary.json`을 확인하고 저장 baseline과 비교한다. exit 2나
`observed_status`가 `complete`가 아닌 결과는 보존하되 품질 비교에 쓰지 않는다.
