# OpenCQA 데이터 준비

Week 3은 [OpenCQA](https://github.com/vis-nlp/OpenCQA)를 사용한다. OpenCQA는 설명형 차트
질의응답 데이터다. 저장소에는 원본 이미지나 답을 복사하지 않고, 사용할 30개 ID와 원본
revision만 기록한다.

OpenCQA 사람이 작성한 `abstractive_answer`는 **기대 답(reference)**이다. 후보 A나 B로
사용하지 않는다. `extractive_answer`도 Week 3 후보가 아니다. 각 학습자는 같은 NVIDIA NIM
Gemma 모델에 기준 지시문과 개선 지시문을 각각 적용해 질문당 실제 답 두 개를 만든다. 실행이
완결된 뒤 두 답의 출처를 가리고 A/B 위치를 고정해 개인 평가쌍 30개를 만든다. 과정 분할은
`week-03-selection.yaml`의 `course_splits`에 development 18쌍, validation 6쌍, 공개 test
6쌍으로 명시한다. 파일 순서만 보고 분할을 추측하지 않는다. test는 생성·최적화·후보 선택에
쓰지 않는다.

## 1. 원본 받기

프로젝트 밖에서 공식 저장소를 받는다.

```bash
git clone https://github.com/vis-nlp/OpenCQA.git ../OpenCQA
git -C ../OpenCQA checkout 28db0fd26a12fd376f6c30b7feb8a4db32313424
```

이 실습이 확인한 원본은 GPL-3.0이며 선택한 ID와 revision은
`data/opencqa/week-03-selection.yaml`에 있다. 다른 revision을 쓰면 준비 명령이 중단된다.

## 2. 실습 자료 만들기

```bash
uv run --locked python scripts/prepare_opencqa.py --source-root ../OpenCQA
```

이 단계에서는 차트·질문·기대 답과 출처 정보를 준비한다. 실제 Gemma 후보는 아직 만들지
않는다. 준비한 파일은 `local-data/opencqa/`에 생기며 Git에 올라가지 않는다.

```text
images/                       175KB 이하 JPEG 차트 이미지 30개
week-03-cases.jsonl           ID·질문·abstractive 기대 답·출처·이미지 hash 30개
```

과거 `abstractive_answer / extractive_answer` 후보, Codex 합성 기준과 NIM Gemma Judge로 만든
결과는 **legacy 자료**다. 새 Gemma 후보와 입력 hash가 다르므로 새 사람 판단·Gemini Judge
비교나 개인 완료 근거로 쓰지 않는다. 파일을 새 결과로 덮어쓰지도 않는다.

## 3. 개인 Gemma 실제 답과 익명 평가쌍 만들기

각 학습자는 [Week 3 실습](week-03-lab.md)의 승인된 절차로 같은 NIM Gemma 모델에 기준·개선
지시문을 적용한다. 질문 30개에 지시문별 30개, 모두 60개 실제 답을 만든다. 상한은 요청
60회, 입력 1,200,000 token, 출력 30,000 token, 비용 안전장치 0.01달러, 7,200초와 재시도
0회다.

개인 실행 폴더에는 다음 근거가 함께 있어야 한다.

- 후보 생성 API 호출 기록
- 30개 기준 답과 30개 개선 답
- 후보 생성 요약과 `complete / partial / not_run` 상태
- 실제 사용한 기준·개선 지시문 snapshot
- Git SHA, OpenCQA revision·입력 hash, requested/actual Gemma model

60개 응답과 두 지시문 snapshot이 모두 맞고 provider 오류·actual model 불일치가 없을 때만
후보 생성을 `complete`로 판정한다. 그 뒤 기준·개선 출처를 숨기고 A/B 위치를 고정해 30쌍을
만든다. `partial`이나 `not_run`이면 pair를 추측하거나 과거 OpenCQA 저자 답으로 채우지 않는다.

## 4. Judge 결과를 보기 전에 사람 판단 잠그기

개인 과제에서는 1~30 중 번호 하나를 배정받는다. 후보 생성은 끝났지만 기준·개선 출처와
Gemini Judge 결과는 공개하지 않은 상태에서
다음 경로에 사람 판단을 기록한다.

```text
local-data/week-03-student-judges/<과정-별칭>/human-label.yaml
```

[사람 사전 label 양식](../templates/judge-human-label-template.yaml)을 한 번만 복사하고
`pair_number`, `pair_id`, `candidate_set_sha256`, `reviewer_id`, `label`, `reason`을 채운다. `label`은
`candidate_a`, `tie`, `candidate_b` 중 하나다. `reason`에는 차트에서 확인한 수치·대상·비교를
적는다. 수업 release가 제공하는 검사 명령으로 배정 pair와 YAML을 확인한 뒤 SHA-256을
동결한다. 사람 판단이 잠긴 뒤에는 파일과 개인 A/B 평가쌍을 수정하지 않는다.

모든 학습자는 고정된 `configs/week-03-judge-rubric.yaml`을 사용하며 이를 개인 폴더에
복사하거나 수정하지 않는다. 각 학습자는 사람 판단을 잠근 뒤 자기 Google AI Studio 계정으로
Gemini 3.5 Flash Lite Judge 30쌍을 full 실행한다. Gemini는 OpenCQA JPEG·질문·기대 답과 Gemma
후보 A/B를 본다. 개인 30쌍 Judge 명령에는 사람 label을 넣지 않으며, 완료 뒤 비교 명령에서
로컬 파일을 연결한다. 개인 사람 label은 Gemini 요청에 보내지 않는다. 전체 명령과 완결 검사는
[Week 3 실습](week-03-lab.md)에만 둔다.

개인 후보 생성과 Judge 근거는 같은 고유 실행 폴더에서 서로 구분해 보존한다.

```text
reports/week-03/student-full/<과정-별칭-시각>/
├── candidates/
│   ├── candidate-calls.jsonl, candidate-results.jsonl, candidate-summary.json
│   └── open-cqa-answer-baseline.md, open-cqa-answer-improved.md
└── judge/
    ├── judge-calls.jsonl, judge-results.jsonl, summary.json
    └── comparison.json
```

개인 완료에는 후보 생성 `complete`와 Judge `complete`가 모두 필요하다. Judge는 30쌍을 두
trial·A/B·B/A로 평가하므로 trial 결과가 60행이어야 한다. 배정 pair의 네 결과를 잠근 사람
label과 수동으로 비교하고 `interpretation.md`에 task model 오류, Judge 오류, 순서·반복 충돌과
한계를 적는다.

이 전송 범위는 2026-08-17에 승인됐다. 승인 때 확인한 Free Tier 근거는 15 RPM, 입력
250,000 TPM, 500 RPD지만, 실제로 쓸 API key의 현재 프로젝트 한도는 실행 당일 다시 확인한다.
당일 잔여 RPD도 240건 이상이어야 한다. 수업 코드는 15 RPM·입력 75,000 TPM·출력 7,500 TPM과
한 full run당 최대 240요청을 적용하지만, 프로젝트의 하루 누적 요청은 추적하지 않는다.
입력·출력 단가는 0달러이며 비용 안전장치는 0.01달러다. 120~240회 요청의 pacing에는 약
8~16분이 들고 API 응답 시간이 더해진다.

Free Tier에서는 보낸 자료가 제품 개선에 사용될 수 있다. 이 조건을 허용할 수 없거나 현재
프로젝트 사전 확인을 마치지 못하면 Judge 상태를 `not_run`으로 두고 요청하지 않는다. 전송
승인만으로 full 실행이 성공한 것은 아니다. 과거 NIM Judge 저장 결과를 대신 넣어 새 개인
완료로 표시하지 않는다.

개인 판단 한 건만으로는 `human_calibrated`가 아니다. Judge를 실제 업무에 사용하려면 최소
30개 실제 업무 출력 pair에 두 사람이 blind label을 작성하고 불일치를 조정하도록 권장한다.
현재 저장소에는 이를 실행할 명령이나 완료된 사람 보정 결과가 없다.

개인 평가쌍에는 OpenCQA의 article, summary, OCR을 넣지 않는다. 각 pair에는
`family_id`, 과정 분할(`course_split`), 원본 분할·revision·license, 이미지 SHA-256을 함께
기록한다. 작업 모델과 Week 3 Judge는 차트 이미지를 직접 본다. 기대 답은 Gemini Judge에는
보내지만 사람의 blind A/B 판단에는 공개하지 않는다. 원본 차트는 기존 공통
전처리를 재사용해 너비 1024px, 175KB 이하 JPEG로 만든다.

준비 명령은 질문과 기대 답에서 `chrome-extension`, `class=`, HTML tag 잔재, 공백이 낀
`& amp ;`, 닫히지 않은 ASCII 따옴표를 발견하면 어떤 파일도 만들기 전에 중단한다. 원본을
임의로 고치지 않고 선택 ID를 다시 검토한다.

개인 후보쌍 hash, 두 지시문 hash, Judge rubric hash 중 하나라도 실행 중 바뀌면 비교를
중단한다. 입력을 바꿨다면 새 고유 폴더에서 다시 시작하며 기존 결과를 이어 쓰지 않는다.
