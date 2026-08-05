# NVIDIA NIM 실제 실행

> 채점기 변경: 아래 기존 API 원응답은 보존한다. 현재 실습에서는 PDF 추출 문장을
> 채점에 사용하지 않으며 `aihub-vqa-deterministic-v2`로 다시 계산한 값을 함께 적는다.

## 모델

Week 1은 NVIDIA 호스팅 접속 주소(endpoint)에서 `google/gemma-4-31b-it`을 사용한다.
한국어 문서와 질문을 이미지와 함께 읽어야 하므로 다국어 멀티모달 모델인 Gemma 4를
수업 기준으로 고정한다.

```yaml
provider:
  model: nvidia_nim/google/gemma-4-31b-it
  api_base: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_NIM_API_KEY
```

Gemma 4 31B IT는 텍스트, 이미지, 영상과 256K 문맥 길이(context)를 지원한다. 공식 카드는
35개가 넘는 언어의 즉시 지원과 140개가 넘는 언어의 사전학습을 구분한다. 2026-08-01 실제 NVIDIA
`/v1/models` 응답에서 사용 가능함을 확인했다. 모델 상태는 변경될 수 있으므로
`preflight_nvidia.py`를 매 수업 전에 실행한다.

## API 키

```bash
cp .env.example .env
```

```dotenv
NVIDIA_NIM_API_KEY="<발급받은-키>"
```

`.env`는 Git에서 제외되며 실행 코드가 명시적으로 읽는다.

## 실행

먼저 모델 목록(catalog)만 조회한다. 이 명령은 모델 추론을 호출하지 않는다.

```bash
uv run python scripts/preflight_nvidia.py
```

승인된 요청 내용(payload) 한 건만 보내는 소규모 사전 확인(probe)은 다음처럼 실행한다.
`--catalog-verified-on`은 위 조회가 성공한 실제 날짜로 바꾼다.

```bash
uv run python scripts/run_nvidia_nim.py \
  --live \
  --sample-id aihub-report-r01 \
  --max-requests 1 \
  --max-input-tokens 20000 \
  --max-output-tokens 500 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 120 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

40건 전체 실행은 소규모 사전 확인과 별도 실행(run)이다.
실행기는 Git에 기록된 `data/cases/week-01-aihub.yaml`에서 다시 만든 40건과
`local-data/aihub/cases.jsonl`을 순서와 모든 필드까지 비교한다. 하나라도 다르거나
봉인 평가 데이터(`sealed_test`)가 섞이면 네트워크 요청 전에 중단한다.

```bash
uv run python scripts/run_nvidia_nim.py \
  --live \
  --max-requests 40 \
  --max-input-tokens 800000 \
  --max-output-tokens 20000 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

중단된 전체 실행만 출력에 표시된 `run_id`와 같은 상한(cap)으로 재개한다. 한 건짜리
소규모 사전 확인을 40건으로 확장해 재개할 수는 없다. 재개 직후에도 저장된 마지막 시도
(attempt) 시각을 기준으로
20 RPM의 남은 대기 시간을 먼저 지킨다.

```bash
uv run python scripts/run_nvidia_nim.py \
  --live \
  --resume \
  --run-id <중단된-run-id> \
  --max-requests 40 \
  --max-input-tokens 800000 \
  --max-output-tokens 20000 \
  --max-cost-usd 0.01 \
  --max-wall-seconds 7200 \
  --max-retries 0 \
  --catalog-verified-on 2026-08-01
```

## 429 제어

- 순차 실행
- 설정 20 RPM
- 호출 사이 최소 3초
- 위 예시는 호출 횟수를 명확히 제한하기 위해 재시도 0회
- 재시도를 허용하려면 그 횟수까지 포함해 호출·토큰·시간 상한을 다시 승인
- 각 응답 즉시 저장
- 같은 `run_id`와 최초 실행 조건으로만 중단 후 `--resume`

무료 접속 주소(Free Endpoint)는 서비스 수준 보장(SLA)이 아니므로 429가 절대 발생하지
않는다고 보장하지 않는다. 재시도 후에도 응답을 받지 못한 사례는 판단 보류
(`inconclusive`)로 기록한다.

## 결과

```text
reports/week-01-nvidia/
└── runs/
    └── <run-id>/
        ├── run-manifest.json
        ├── budget.json
        ├── observations.jsonl
        ├── records.jsonl
        ├── results.jsonl
        ├── summary.json
        └── deepeval/
```

`observations.jsonl`은 원본 모델 응답과 호출 정보를 담는다. `records.jsonl`은
데이터셋·지시문(prompt)·출력 형식(schema)·Git 해시와 평가 결과를 한 행에 묶는다.
`budget.json`은 중단 시에도 누적 호출·토큰·비용·시간 예약을 보존한다.

### 2026-08-01 Gemma 4 전체 실행

같은 AIHub 질문 40건을 `google/gemma-4-31b-it`에 순차 전달했다.

- 실행 ID: `week01-20260801T083955Z-5f75374b`
- 결과: 40건 응답. 당시 채점기는 6건 통과, 현재 채점기는 같은 원응답에서 8건 통과
- 모델 확인: 요청 `nvidia_nim/google/gemma-4-31b-it`, 실제
  `google/gemma-4-31b-it`, 모델 변경(model drift)과 provider 오류 0건
- 사용량: 입력 101,113 토큰, 출력 6,052 토큰, 기록 비용 USD 0
- 실행 시간: 약 560.9초
- 주요 평균: `schema_validity=0.9000`, `answer_correct=0.2000`,
  `evidence_page_f1=0.8917`, 현재 전체 성공(`task_success`) 0.2000

모델은 근거 페이지를 비교적 잘 찾았지만 짧은 기대 답 대신 설명 문장을 반환하거나 JSON
앞뒤에 Markdown 코드 블록 표시(fence)를 붙이는 사례가 많았다. 따라서 근거 점수보다
정답 허용 기준과 전체 성공률이 낮았다.

같은 모델에서 지시문만 바꾸는 후보 설정, 별도 결과 경로와 비교 명령은
[Week 1 실습 안내의 Gemma 4 지시문 후보 평가](week-01-lab.md#gemma-4-지시문prompt-후보를-따로-평가하기)에
정리했다. 기준·후보를 같은 코드 상태에서 새로 실행하지 않으면 지시문 단독 효과로
판정하지 않는다.

### 2026-08-03 Gemma 4 지시문 A/B 실행

같은 코드 상태에서 기준 지시문과 짧은 답 전용 후보 지시문을 각각 40건 비교했다. 두 실행
모두 최대 20 RPM, 재시도 0회였다.

- 기준 실행 `week01-20260803T135626Z-0c8a3821`: 8건 통과, 32건 실패,
  전체 성공 0.2000
- 후보 실행 `week01-20260803T141015Z-9485bcce`: 27건 통과, 12건 실패,
  provider 오류 1건, 전체 성공 0.6750
- 후보 변화: `answer_correct` +47.5%p, `numeric_match` +42.5%p,
  `json_object_only` +27.5%p
- 사용량: 기준 입력 101,113/출력 6,099 토큰, 후보 입력 108,590/출력
  3,400 토큰, 기록 비용 USD 0

긴 설명 문장 대신 값·단위·기관명만 쓰라는 지시가 정답 채점과 잘 맞아 20건이 새로
통과했다. 반면 후보는 깨진 JSON 3건, 잘못된 필드 1건과 NVIDIA NIM HTTP 500 1건이
있었다. 이 provider 오류 때문에 자동 비교는 판단 보류(`inconclusive`)이며, 점수
상승만으로 후보를 자동 채택하지 않는다.

모델 정보는 [NVIDIA NIM 모델 카탈로그](nvidia-model-catalog.md)를 참고한다.
