"""편집용 YAML을 공통 EvaluationCase JSONL로 바꾼다."""

from __future__ import annotations

import argparse
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import build_cases, write_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="질문 case 준비")
    parser.add_argument("--config", default="configs/week-01.yaml")
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    cases = build_cases(project_path(PROJECT_ROOT, settings.paths.case_authoring))
    output_path = project_path(PROJECT_ROOT, settings.paths.cases)
    write_cases(cases, output_path)
    print(f"{output_path}: {len(cases)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
