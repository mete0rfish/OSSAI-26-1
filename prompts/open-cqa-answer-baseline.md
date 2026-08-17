OpenCQA 차트 이미지를 보고 질문에 답하세요. 답을 찾는 과정은 출력하지 마세요.

출력은 아래 6개 field가 모두 있는 JSON object 하나여야 합니다. Markdown code
fence, `json` 표식, 설명과 두 번째 JSON을 출력하지 마세요. `tool_requests`는 항상 빈
목록입니다. 차트에서 답을 확인했으면 `evidence`의 `page_number`는 1입니다.

{"answer":"질문에 대한 답","evidence":[{"evidence_id":"chart-1","quote":"차트에서 읽은 값과 대상","page_number":1}],"confidence":0.9,"abstained":false,"abstention_reason":null,"tool_requests":[]}

차트만으로 답을 확인할 수 없을 때만 다음 형식을 사용하세요.

{"answer":"답변 보류","evidence":[],"confidence":0.0,"abstained":true,"abstention_reason":"차트에서 답을 확인할 수 없음","tool_requests":[]}
