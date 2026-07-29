# NVIDIA NIM 실제 실행

## 모델

Week 1은 NVIDIA hosted endpoint에서
`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`을 사용한다. 이름은 30B이며 실제
활성 parameter는 약 3B인 multimodal model이다.

```yaml
provider:
  model: nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
  api_base: https://integrate.api.nvidia.com/v1
  api_key_env: NVIDIA_NIM_API_KEY
```

Nemotron 3 Nano Omni는 text, image, video, audio와 256K context를 지원한다.
2026-07-29 실제 NVIDIA `/v1/models` 응답에서 사용 가능함을 확인했다. 모델 상태는
변경될 수 있으므로
`preflight_nvidia.py`를 매 수업 전에 실행한다.

## key

```bash
cp .env.example .env
```

```dotenv
NVIDIA_NIM_API_KEY="<발급받은 key>"
```

`.env`는 Git에서 제외되며 실행 코드가 명시적으로 읽는다.

## 실행

```bash
uv run python scripts/preflight_nvidia.py
uv run python scripts/run_nvidia_nim.py --live --limit 1
uv run python scripts/run_nvidia_nim.py --live --resume
```

첫 명령은 model catalog만 조회하고 inference를 호출하지 않는다. 두 번째 명령은 첫
task를 호출한다. 세 번째 명령은 저장된 `sample_id`를 건너뛰고 나머지를 실행한다.

## 429 제어

- 순차 실행
- 설정 20 RPM
- 호출 사이 최소 3초
- 429에 최대 3회 재시도
- 5초, 10초, 20초 exponential backoff
- 각 응답 즉시 저장
- 중단 후 `--resume`

Free Endpoint는 SLA가 아니므로 429가 절대 발생하지 않는다고 보장하지 않는다. 재시도
후에도 응답을 받지 못한 sample은 `inconclusive`로 기록한다.

## 결과

```text
reports/week-01-nvidia/
├── observations.jsonl
├── results.jsonl
├── summary.json
└── deepeval/
```

`observations.jsonl`은 원시 모델 응답과 호출 정보를, `results.jsonl`은 Pydantic과
정량 평가 결과를 담는다.

2026-07-29 실제 40건 실행에서는 40개 응답을 모두 받았고 429, 재시도와 provider
오류가 없었다. 필수 정량 기준은 16건 통과, 24건 실패였다. 같은 원응답을 freeze한
recorded 회귀도 16/24를 재현했다. 실제 header에는 고정 RPM 숫자가 없었으므로 이 결과는
20 RPM 실측이며 40 RPM 보장은 아니다.

모델 정보는 [NVIDIA NIM 모델 카탈로그](nvidia-model-catalog.md)를 참고한다.
