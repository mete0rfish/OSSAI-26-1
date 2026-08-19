/no_think

차트 이미지에서 아래 질문의 답을 찾고, JSON 하나만 반환하세요.

질문: {question}

이미지에서 확인한 내용만 사용하세요. 답을 찾는 과정은 출력하지 않습니다.

`answer`에는 질문이 요구한 값과 단위만 간결하게 씁니다. `evidence`의 `quote`에는
답을 직접 확인할 수 있는 차트의 연속된 글자와 수치를 그대로 씁니다. 차트 이미지는
1페이지로 보고 `evidence_id`는 `chart`, `page_number`는 `1`을 사용합니다.

출력 규칙:

- 첫 글자는 `{`, 마지막 글자는 `}`입니다.
- Markdown code fence, 설명, 두 번째 JSON을 출력하지 않습니다.
- 아래 6개 field를 모두 한 번씩 넣습니다.
- `confidence`는 0 이상 1 이하 숫자입니다.
- `tool_requests`는 항상 빈 목록입니다.

답을 확인할 수 있을 때의 JSON 모양은 다음과 같습니다. 예시 값은 복사할 정답이 아닙니다.

{"answer":"값과 단위","evidence":[{"evidence_id":"chart","quote":"답을 포함한 차트 글자와 수치","page_number":1}],"confidence":0.9,"abstained":false,"abstention_reason":null,"tool_requests":[]}

이미지에서 답을 확인할 수 없을 때는 추정하지 말고 아래 모양으로 답변을 보류합니다.

{"answer":"답변 보류","evidence":[],"confidence":0.0,"abstained":true,"abstention_reason":"이미지에서 답을 확인할 수 없음","tool_requests":[]}
