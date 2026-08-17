"""저장 응답으로 workflow를 실행하고 고정 규칙과 DeepEval로 채점한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import load_cases
from verifiable_ai_workflow.evaluation.deepeval_runner import evaluate_results
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.providers.recorded import RecordedProvider
from verifiable_ai_workflow.workflow import run_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 1 고정 규칙 평가")
    parser.add_argument("--config", default="configs/week-01.yaml")
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    if settings.provider.kind != "recorded" or not settings.paths.recorded_responses:
        raise ValueError("이 명령은 API를 호출하지 않는 recorded 실습용입니다")
    output_dir = project_path(PROJECT_ROOT, settings.paths.output)
    cases = load_cases(project_path(PROJECT_ROOT, settings.paths.cases))
    observations = run_cases(
        cases=cases,
        prepared_documents=project_path(PROJECT_ROOT, settings.paths.prepared_documents),
        prompt_path=project_path(PROJECT_ROOT, settings.paths.prompt),
        provider=RecordedProvider(
            project_path(PROJECT_ROOT, settings.paths.recorded_responses)
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "observations.jsonl").write_text(
        "".join(observation.model_dump_json() + "\n" for observation in observations),
        encoding="utf-8",
    )
    results = score_observations(cases, observations)

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
