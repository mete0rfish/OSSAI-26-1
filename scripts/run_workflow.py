"""저장 응답 제공자(provider)로 Week 1 작업 흐름(workflow)을 실행한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import load_cases
from verifiable_ai_workflow.providers.recorded import RecordedProvider
from verifiable_ai_workflow.workflow import run_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 1 저장 응답 workflow")
    parser.add_argument("--config", default="configs/week-01.yaml")
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    if settings.provider.kind != "recorded":
        raise ValueError("이 명령은 API를 호출하지 않는 recorded 실습용입니다")

    cases = load_cases(project_path(PROJECT_ROOT, settings.paths.cases))
    provider = RecordedProvider(project_path(PROJECT_ROOT, settings.paths.recorded_responses))
    observations = run_cases(
        cases=cases,
        prepared_documents=project_path(
            PROJECT_ROOT,
            settings.paths.prepared_documents,
        ),
        prompt_path=project_path(PROJECT_ROOT, settings.paths.prompt),
        provider=provider,
    )
    output_path = project_path(PROJECT_ROOT, settings.paths.output) / "observations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(observation.model_dump_json() + "\n" for observation in observations),
        encoding="utf-8",
    )
    print(f"{output_path}: {len(observations)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
