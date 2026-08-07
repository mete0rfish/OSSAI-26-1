"""같은 Gemma 4의 기준·후보 지시문(prompt) 실제 실행을 비교한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from verifiable_ai_workflow.config import project_path
from verifiable_ai_workflow.prompt_comparison import compare_prompt_runs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma 지시문(prompt) A/B 비교")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument(
        "--output",
        default="reports/week-02/gemma-prompt-comparison.json",
    )
    parser.add_argument(
        "--rescore-current",
        action="store_true",
        help="저장 점수 대신 원응답을 현재 고정 규칙 채점기로 다시 계산합니다",
    )
    parser.add_argument(
        "--case-authoring",
        default="data/cases/week-01-aihub.yaml",
    )
    args = parser.parse_args()

    report = compare_prompt_runs(
        project_path(PROJECT_ROOT, args.baseline_run),
        project_path(PROJECT_ROOT, args.candidate_run),
        case_authoring_path=(
            project_path(PROJECT_ROOT, args.case_authoring) if args.rescore_current else None
        ),
    )
    output = project_path(PROJECT_ROOT, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0 if report.automated_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
