# 실습 데이터 폴더

AIHub `멀티모달 정보검색 데이터_Sample`에서 받은 파일을 `aihub/source/`에 넣는다.

```text
local-data/aihub/
├── source/       다운로드한 AIHub 파일
├── prepared/     PNG, API용 JPEG, page text와 manifest
└── cases.jsonl   workflow가 읽는 질문 40건
```

`prepared/`와 `cases.jsonl`은 실습 명령을 실행하면 생성된다.

Week 2 개인 지시문은 다음 위치에 둔다.

```text
local-data/week-02-students/<과정-별칭>/prompt.md
```

`local-data/week-02-full-runs/gemma-baseline/`과 `gemma-improved/`는 저장 예시 A/B다.
튜터는 수업 release와 같은 clean commit에서 만든 baseline 40건을
`local-data/week-02-full-runs/gemma-release-baseline-<short-sha>/`에 별도로 배포한다.
각 학습자의 실제 40건 원본·요약은 `reports/week-02-gemma-baseline/runs/`에 저장되고,
baseline 비교 JSON은 `reports/week-02/students/<과정-별칭>/`에 생성된다. 전체 실행과 완결 검사는
[Week 2 실습](../docs/week-02-lab.md)을 따른다.

Week 3 OpenCQA 준비 명령은 차트·질문·사람 작성 기대 답을 만든다.

```text
local-data/opencqa/
├── images/                       API와 blind 비교용 차트 JPEG 30개
└── week-03-cases.jsonl           질문·abstractive 기대 답·출처·이미지 hash 30개
```

OpenCQA `abstractive_answer`는 기대 답이며 후보 A나 B가 아니다. 각 학습자는 같은 NIM Gemma에
기준·개선 지시문을 적용해 30개씩 실제 답을 만든다. 60개 답과 두 지시문 snapshot이 완결된
뒤에만 출처를 가린 A/B 30쌍을 만든다.

과거 `abstractive_answer / extractive_answer` 후보와 NIM Gemma Judge를 사용한 Codex 합성
기준과 `local-data/week-03-full-runs/judge-30/`은 legacy 자료다. 새 개인 실행의 입력·fallback·
완료 근거가 아니다.

개인 과제로 직접 작성하는 두 파일은 다음 폴더에 둔다.

```text
local-data/week-03-student-judges/<과정-별칭>/human-label.yaml
local-data/week-03-student-judges/<과정-별칭>/interpretation.md
```

`human-label.yaml`은 A/B의 기준·개선 출처와 Gemini Judge 결과를 보기 전에 작성한다.
`candidate_set_sha256`으로 개인 후보 세트에 묶은 뒤 파일 SHA-256을 동결한다.
과정에서는 `configs/week-03-judge-rubric.yaml`을 고정 평가 기준으로 사용하며, 개인 폴더에
복사하지 않는다. 사람 판단을 잠근 다음, 각 학습자가 자기 Google AI Studio 계정으로
Gemini 3.5 Flash Lite Judge 30쌍을 full 실행한다. 개인 후보 생성·판정·요약·비교는 다음 고유 폴더에
서로 구분해 보존한다.

```text
reports/week-03/student-full/<과정-별칭-시각>/
├── candidates/
│   ├── candidate-calls.jsonl, candidate-results.jsonl, candidate-summary.json
│   └── open-cqa-answer-baseline.md, open-cqa-answer-improved.md
└── judge/
    ├── judge-calls.jsonl, judge-results.jsonl, summary.json
    └── comparison.json
```

Gemini에는 OpenCQA JPEG·질문·기대 답과 Gemma 후보 A/B를 보내지만 개인 사람 label은 보내지
않는다. 개인 30쌍 Judge 명령에도 label을 넣지 않고, 실행 뒤 비교 명령에서 로컬 파일을
연결한다. 완결된 60행 Judge 결과에서 같은 pair를 찾아 사람 판단과 직접 비교한 뒤
`interpretation.md`에 task model 오류, Judge 오류, 순서·반복 충돌과 한계를 적는다. 전체 실행과
완결 검사는 [Week 3 실습](../docs/week-03-lab.md)을 따른다.

Google 전송 범위는 2026-08-17에 승인됐다. 공식 Free Tier 근거는 15 RPM·입력 250,000 TPM·
500 RPD이고, 수업 코드는 15 RPM·입력 75,000 TPM·출력 7,500 TPM과 한 full run당 최대
240요청을 적용한다. 프로젝트의 하루 누적 요청은 추적하지 않으므로 실행 전 당일 잔여 RPD가
240건 이상인지 확인한다. Free Tier token 단가는 0달러이며 비용 안전장치는 0.01달러다.
120~240회 요청은 pacing에만 약 8~16분이 걸리고 API 응답 시간이 더해진다.

실행 당일 현재 프로젝트의 실제 tier·model·quota·가격·데이터 이용 조건을 확인하지 못하면
Judge 상태를 `not_run`으로 둔다. Free Tier 자료가 제품 개선에 사용될 수 있다는 조건도 다시
확인한다. 전송 승인만으로 새 full 실행이 성공한 것은 아니며, legacy 저장 결과로 완료를 대신하지
않는다. 개인 사람 label 한 건만으로는 `human_calibrated`가 아니다. 두 사람·최소 30쌍으로 만든
별도 보정 자료와 같은 것으로 보지 않는다.

수업 전에 만든 실제 API 원본은 다음 위치에 둔다.

```text
local-data/week-01-full-runs/nemotron/
local-data/week-02-full-runs/gemma-baseline/
local-data/week-02-full-runs/gemma-improved/
local-data/week-02-full-runs/gemma-release-baseline-<short-sha>/
local-data/week-02-full-runs/provider-comparison/
local-data/week-03-full-runs/judge-30/   legacy Week 3 결과
```

Week 2 저장 개선·provider 결과는 설명 예시와 분석 fallback으로 쓴다. Week 3 legacy 결과는
이전 계약을 설명할 때만 쓰며, 새 Gemma 후보·Gemini Judge의 입력이나 fallback이 아니다. 개인
40건·새 Week 3 실행 완료를 대신하지 않는다.

`local-data/`는 Git에서 제외된다. 튜터는 배포가 허용된 과정 저장소를 통해 위 폴더를 따로
전달한다. 학습자는 해당 주차 실습 문서의 파일 확인 명령이 모두 통과한 것을 확인한 뒤
실습한다.
