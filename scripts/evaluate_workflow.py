"""저장된 관찰값을 채점하고 DeepEval 결과를 만든다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import load_cases
from verifiable_ai_workflow.evaluation.deepeval_runner import evaluate_results
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.schemas import ModelObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_observations(path: Path) -> list[ModelObservation]:
    return [
        ModelObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 1 결정적 평가")
    parser.add_argument("--config", default="configs/week-01.yaml")
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    output_dir = project_path(PROJECT_ROOT, settings.paths.output)
    cases = load_cases(project_path(PROJECT_ROOT, settings.paths.cases))
    observations = _load_observations(output_dir / "observations.jsonl")
    results = score_observations(
        cases,
        observations,
        prepared_documents=project_path(
            PROJECT_ROOT,
            settings.paths.prepared_documents,
        ),
    )

    (output_dir / "results.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results),
        encoding="utf-8",
    )
    evaluate_results(results, cases, output_dir / "deepeval")
    score_names = tuple(results[0].scores) if results else ()
    summary = {
        "record_count": len(results),
        "target_count": len(cases),
        "status_counts": dict(Counter(result.status for result in results)),
        "score_averages": {
            name: round(
                sum(result.scores[name] for result in results) / len(results),
                4,
            )
            for name in score_names
        },
        "evidence_kind": "test_only",
        "judge_status": "not_requested",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
