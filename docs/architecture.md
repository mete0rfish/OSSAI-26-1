# 코드 구조

각 주차 문서에서는 필요한 `scripts/` 명령과 실행 주체를 구분한다. 아래 표에는 핵심 산출물을
만드는 주요 명령만 정리했다. 모든 명령은 `src/verifiable_ai_workflow/`의 공통 구현을 호출하며,
별도 평가 엔진은 두지 않는다.

## Week 1–4 주요 실행 파일

| 순서 | 실행 파일 | 하는 일 | 주요 결과 |
| ---: | --- | --- | --- |
| 1 | `prepare_documents.py` | PDF를 페이지 이미지로 준비 | 문서별 `manifest.json`, JPEG |
| 2 | `prepare_cases.py` | 사람이 편집하는 YAML을 실행용 JSONL로 변환 | `cases.jsonl` |
| 3 | `inspect_inputs.py` | 문서·질문·이미지 제한 검사 | `eda.json` |
| 4 | `inspect_deterministic_scoring_case.py` | 저장 응답 한 건의 채점 과정 확인 | 터미널 JSON |
| 5 | `preflight_nvidia.py` | 현재 설정 모델이 NVIDIA 목록에 있는지 확인 | 사용 가능 여부 |
| 6 | `run_nvidia_nim.py` | 튜터 baseline 또는 학습자 Week 2 prompt의 NIM 실제 호출·즉시 채점 | 원응답·점수·요약 |
| 7 | `evaluate_workflow.py` | 저장 응답 재실행과 고정 규칙·DeepEval 채점 | 원응답·사례별 점수·요약 |
| 8 | `evaluate_failures.py` | 의도적으로 깨진 네 응답 채점 | 실패 주입 결과 |
| 9 | `compare_gemma_prompts.py` | 같은 release Gemma baseline과 저장·개인 지시문 결과 비교 | 사례별 변화·계보 검사 |
| 10 | `compare_live_provider_routes.py` | Gemma와 Gemini 실제 비교 | 호출 경로별 결과·비교 |
| 11 | `rescore_provider_comparison.py` | 저장 원응답을 현재 채점기로 재계산 | 원본과 분리된 파생 비교 |
| 12 | `rehearse_provider_faults.py` | 저장된 API 장애 상황 확인 | 장애 상황 결과 |
| 13 | `prepare_opencqa.py` | OpenCQA 질문·사람 기대 답과 검토용 JPEG 30개 준비 | `week-03-cases.jsonl`, 이미지 |
| 14 | `run_open_cqa_candidates.py` | 같은 NIM Gemma에 baseline·improved prompt를 적용해 개인 후보 30쌍 생성 | 후보 호출·결과·요약, prompt snapshot |
| 15 | `inspect_judge_pair.py` | 개인 후보 한 쌍을 출처·기대 답 없이 확인하고 사람 사전 label 검증 | 터미널 출력 |
| 16 | `run_open_cqa_judge.py` | Gemini 3.5 Flash Lite로 개인 30쌍 또는 대표 1쌍을 두 trial·두 순서로 판단 | Judge 호출·60 trial·요약 |
| 17 | `compare_open_cqa_judge.py` | Judge 판단을 baseline·improved 출처, 잠근 사람 label과 연결하고 충돌 계산 | 승·무승부·review·순서·반복 비교 |
| 18 | `prepare_week_04_lab.py` | 4주차 입력·저장 결과를 확인하고 개인 폴더 준비 | 준비 상태와 개인 경로 |
| 19 | `inspect_week_04_prompt_results.py` | 바뀐 지시문·검증 점수·선택 이유를 쉬운 문장으로 표시 | 터미널 수업 설명 |
| 20 | `check_week_04_api_keys.py` | 두 API key가 있는지만 값 노출 없이 확인 | `present / missing` |
| 21 | `optimize_open_cqa_prompt.py` | 개발 문제의 실패 답으로 새 지시문을 만들고 검증 문제에서 처음 지시문과 비교 | 역할별 호출, 후보·선택 지시문, 검증 결과 |
| 22 | `generate_image_variants.py` | OpenCQA 첫 차트의 이미지 변형 생성 | 변형 이미지·사람 검토표 |
| 23 | `run_image_robustness.py` | 선택 지시문으로 원본 1개와 변형 4개 실행 | 구조화 답·호출 요약 |
| 24 | `evaluate_image_robustness.py` | 근거가 남았는지에 따라 이미지 답을 서로 다른 규칙으로 판정 | 이미지별 결과·검증 manifest |

`inspect_*.py`는 사람 판단 전에 후보의 prompt 출처·기대 답·Judge 결과를 숨긴다. 각 학습자는
Week 2 자기 prompt 40건과 Week 3 NIM 답 60개·Gemini Judge 30쌍을 실행한다. 강의자의 별도
시연은 Week 2 `r01` 한 건과 Week 3 대표 한 쌍이 최대다. Google 전송 범위는 2026-08-17에
승인됐지만, 실행 당일 현재 프로젝트의 Free Tier·model·quota·가격·데이터 이용 조건을 먼저
확인한다. 확인하지 못하면 `not_run`으로 남긴다. 실제 실행의 식별값·비용·오류는 결과 파일에
기록하고, 첫 개념 설명에서는 다루지 않는다.

Judge 설정은 공개 한도 15 RPM·입력 250,000 TPM·500 RPD보다 낮은 15 RPM·입력 75,000 TPM·
출력 7,500 TPM과 한 full run당 최대 240요청을 적용한다. 프로젝트의 하루 누적 요청은 추적하지
않으므로, 실행 전 당일 잔여 RPD가 240건 이상인지 확인한다. Free Tier token 단가는 0달러이며
비용 안전장치는 0.01달러다. 120~240회 요청의 pacing에는 약 8~16분이 걸리고 API 응답 시간이
더해진다. Free Tier로 보낸 자료가 제품 개선에 사용될 수 있다는 조건도 확인한다. 승인은 실행
성공 기록이 아니므로 새 결과의 완결 검사는 따로 통과해야 한다.

## Week 2–4 개인 결과 위치

| 주차 | 작성 입력 | 개인 실행 결과 | 비교·해석 |
| --- | --- | --- | --- |
| Week 2 | `local-data/week-02-students/<alias>/prompt.md` | `reports/week-02-gemma-baseline/runs/`의 고유 폴더 | `reports/week-02/students/<alias>/` 비교 JSON |
| Week 3 | `local-data/week-03-student-judges/<alias>/human-label.yaml` | `reports/week-03/student-full/<alias-시각>/candidates/`의 NIM 후보, `judge/`의 Gemini 판단·비교 | `local-data/week-03-student-judges/<alias>/interpretation.md` |
| Week 4 | `local-data/week-04-students/<alias>/variants/variant-review.csv` | `reports/week-04/student-full/<alias-시각>/optimization/`의 지시문 결과, `robustness/`의 이미지 응답·평가 | `local-data/learning-progress.md`의 실행 상태·지시문 선택 결과·실패 원인 |

저장된 Week 2 개선·provider 결과는 설명 예시와 실패 fallback으로만 쓴다. 과거 OpenCQA
`abstractive_answer / extractive_answer` 후보·NIM Judge·Codex 합성 기준 결과는 legacy이며,
새 Week 3 입력이나 fallback이 아니다. 개인 실행이 `partial / not_run`이어도 완료로 바꾸지
않는다. 전체 명령은 [Week 2 실습](week-02-lab.md), [Week 3 실습](week-03-lab.md)과
[Week 4 실습](week-04-lab.md)에만 둔다.

## 내부 코드

```text
src/verifiable_ai_workflow/
├── config/          YAML 설정과 .env 읽기
├── schemas/         질문·모델 답·평가 결과의 출력 형식(schema)
├── preprocessing/   PDF 페이지 이미지 준비
├── data/            수업 데이터 읽기와 JSONL 생성
├── providers/       LiteLLM 실제 호출과 저장 응답 제공자(provider)
├── workflow/        질문·페이지 이미지·모델 호출 연결
├── evaluation/      고정 규칙 점수 계산과 DeepEval 저장
├── open_cqa_candidates.py  Gemma 후보 생성·익명 배치·후보 세트 hash
├── prompt_optimization.py  개발·검증 분할, 점수 계산, 지시문 선택
├── image_robustness.py     이미지 변형, 사람 판정표, 결과 계산
├── week4_materials.py      튜터가 지정한 4주차 공통 저장 결과 경로 읽기
├── course_live.py          주차별 LiteLLM 예산·모델 설정 연결
└── judge_*.py      Gemini Judge 호출, 지표와 순서·반복 충돌 계산
```

작업 모델에는 질문·지시문(prompt)·페이지 JPEG만 보낸다. OpenCQA 사람이 쓴 기대 답은 보내지
않는다. Judge 모델에는 질문·페이지 JPEG·기대 답·익명 Gemma 후보 A/B와 고정 rubric을 보낸다.
PDF 추출 문장은 원본·라벨을 점검할 때만 쓰며, 모델 입력이나 채점에는 넣지 않는다.

Week 4에서 NIM Gemma에는 JPEG·질문·지시문을 보낸다. Gemini에는 JPEG와 사람의 이미지
검토표를 보내지 않는다. 대신 지시문·질문·기대 답·NIM 출력·고정 점수와 이유를 보낸다.
`calls.jsonl`에서는 NIM 호출을 `provider_role=target`, Gemini 호출을
`provider_role=optimizer`로 구분한다.

Gemini가 처음 지시문과 같은 문장을 제안하면, NIM을 다시 호출해 생긴 점수 차이를 개선으로
세지 않는다. 원본 답이 기준에 못 미치면 회전·압축 이미지의 답 유지 여부는
`inconclusive`로 둔다. 잘림·가림 이미지에서 안전하게 답변을 보류했는지는 별도로 판정한다.

## 한 사례의 실제 흐름

```text
평가 사례
→ 문서의 페이지 JPEG 읽기
→ 지시문 + 질문 + 이미지 구성
→ LiteLLM을 통해 작업 모델 호출
→ 원응답 저장
→ Pydantic 출력 형식 검사
→ 고정 규칙 채점
→ DeepEval 결과 저장
```

Week 3에서는 이 흐름을 다음처럼 확장한다.

```text
같은 OpenCQA 입력
→ NIM Gemma baseline·improved 답 생성
→ 출처를 숨긴 A/B pair와 candidate-set SHA-256 저장
→ 사람 label과 파일 SHA-256 잠금
→ Gemini 3.5 Flash Lite에 이미지·질문·기대 답·A/B 전달
→ 두 trial의 A/B·B/A 결과와 baseline·improved 승패 계산
```

채점기의 필수 지표와 진단 지표는
[수업 도구·채점기·용어](terms-tools-and-scoring.md#고정-규칙-채점기와-평가지표)에 있다.

## Week 6까지 유지하는 공통 기능

다음 코드는 Week 1의 핵심 개념은 아니지만, 실제 호출 결과를 Week 6까지 비교하는 데 필요하므로
유지한다.

| 파일 | 기능 명세 | 사용하는 시점 |
| --- | --- | --- |
| `live_execution.py` | 요청·토큰·비용·시간 상한을 호출 전에 검사하고 중단 시 누적값을 보존한다. 같은 실행을 동시에 쓰지 않게 잠그고 JSON을 원자적으로 저장한다. | Week 1–6 실제 호출 |
| `model_identity.py` | 요청 모델과 API가 반환한 실제 처리 모델(actual model)이 같은지 확인한다. | Week 1–6 실제 호출 |
| `comparison.py` | 두 실행의 데이터·지시문·출력 형식·채점 조건이 같은지 확인하고 사례별 변화를 계산한다. | Week 2 비교 |
| `prompt_comparison.py` | 같은 모델에서 지시문만 다른 두 전체 실행을 비교한다. | Week 2 |
| `live_provider_comparison.py` | 서로 다른 두 API 제공자를 같은 입력과 상한으로 실행한다. | Week 2 |
| `open_cqa_candidates.py` | 같은 작업 모델의 baseline·improved 답을 만들고 출처를 가린 A/B와 candidate-set SHA-256으로 묶는다. | Week 3 |
| `judge_model.py` | Gemini 3.5 Flash Lite 실제 호출을 DeepEval Judge 인터페이스에 연결한다. | Week 3 |
| `judge_metrics.py` | 이미지·질문·기대 답·Gemma 후보를 과정의 고정 4단계 기준으로 비교한다. | Week 3 |
| `judge_comparison.py` | 잠근 사람 label·후보 출처·Judge 결과를 연결해 baseline·improved 승패와 순서·반복 충돌을 계산한다. | Week 3 |
| `course_live.py` | Week 4 실제 호출이 기존 LiteLLM 예산·모델 검사를 재사용하게 한다. | Week 4 |
| `prompt_optimization.py` | 개발 문제 18개로 지시문 후보를 만들고 검증 문제 6개로 처음·후보 지시문 중 하나를 고른다. 공개 test 6개는 이 과정에 쓰지 않는다. | Week 4 |
| `image_robustness.py` | 원본·변형 파일의 SHA-256과 사람 판정표를 확인하고 각 이미지의 결과를 계산한다. | Week 4 |

이 기능들은 학습자가 직접 다시 구현하지 않는다. 해당 주차에서는 결과 파일을 보고 기능이
지켜졌는지만 확인한다.

`answer_correct`는 기준 답 또는 명시한 `accepted_answers`와 정규화한 값이 완전히 일치할 때만
통과한다. `answer_similarity`와 `numeric_match`는 진단용으로만 저장한다.
