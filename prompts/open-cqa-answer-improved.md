OpenCQA 차트 이미지를 보고 질문에 직접 답하세요. 답을 찾는 과정은 출력하지 마세요.

답하기 전에 다음을 내부적으로 확인하세요.

1. 질문이 묻는 대상, 기간, 범주와 단위를 구분합니다.
2. 범례와 축의 단위를 확인하고 인접한 항목의 값을 섞지 않습니다.
3. 비교를 묻는 질문은 대상별 값과 차이를 계산해 비교 방향까지 확인합니다.
4. 답의 핵심 수치를 차트에서 한 번 더 확인합니다.

`answer`에는 질문에 필요한 대상·값·단위·비교 방향을 빠짐없이 적으세요. 차트에 없는
원인이나 추가 설명은 추측하지 마세요.

출력은 아래 6개 field가 모두 있는 JSON object 하나여야 합니다. Markdown code
fence, `json` 표식, 설명과 두 번째 JSON을 출력하지 마세요. `tool_requests`는 항상 빈
목록입니다. `evidence.quote`에는 답을 확인한 차트의 대상과 값을 적고, `page_number`는
1로 적으세요.

{"answer":"대상과 값·단위를 포함한 답","evidence":[{"evidence_id":"chart-1","quote":"대상: 값과 단위","page_number":1}],"confidence":0.9,"abstained":false,"abstention_reason":null,"tool_requests":[]}

차트만으로 답을 확인할 수 없을 때만 다음 형식을 사용하세요.

{"answer":"답변 보류","evidence":[],"confidence":0.0,"abstained":true,"abstention_reason":"차트에서 답을 확인할 수 없음","tool_requests":[]}
