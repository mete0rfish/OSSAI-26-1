"""의도적으로 깨뜨린 응답이 예상 metric을 실패시키는지 확인한다."""

from __future__ import annotations

import json
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import score_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    settings = load_settings(PROJECT_ROOT / "configs/week-01.yaml")
    case = build_cases(project_path(PROJECT_ROOT, settings.paths.case_authoring))[0]
    valid = {
        "answer": case.expected.answer,
        "evidence": [
            {
                "evidence_id": "page-1",
                "quote": case.expected.answer,
                "page_number": case.expected.pages[0],
            }
        ],
        "confidence": 0.9,
        "abstained": False,
        "abstention_reason": None,
        "tool_requests": [],
    }
    failures = {
        "broken_json": "{",
        "confidence_out_of_range": {**valid, "confidence": 1.5},
        "wrong_answer": {**valid, "answer": "0%"},
        "wrong_page": {
            **valid,
            "evidence": [{**valid["evidence"][0], "page_number": 2}],
        },
    }
    report = {name: score_output(response, case)[1] for name, response in failures.items()}
    output = PROJECT_ROOT / "reports/week-01-failures/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(scores["task_success"] == 0.0 for scores in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
