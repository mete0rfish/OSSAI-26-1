"""NVIDIA NIM VLM으로 AIHub 질문을 순차 실행하고 즉시 평가한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from verifiable_ai_workflow.config import (
    load_project_env,
    load_settings,
    project_path,
)
from verifiable_ai_workflow.data.dataset import load_cases
from verifiable_ai_workflow.evaluation.deepeval_runner import evaluate_results
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.providers.litellm_provider import LiteLLMProvider
from verifiable_ai_workflow.schemas import ModelObservation
from verifiable_ai_workflow.workflow import run_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_observations(path: Path) -> list[ModelObservation]:
    if not path.is_file():
        return []
    return [
        ModelObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_observation(path: Path, observation: ModelObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(observation.model_dump_json() + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA NIM Week 1 live batch")
    parser.add_argument("--config", default="configs/nvidia-nim.yaml")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("실제 API 호출에는 --live가 필요합니다")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit은 양수여야 합니다")

    load_project_env(PROJECT_ROOT)
    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    if settings.provider.kind != "litellm":
        raise ValueError("NVIDIA NIM 설정의 provider.kind는 litellm이어야 합니다")
    if not settings.provider.api_key_env:
        raise ValueError("NVIDIA NIM 설정에 api_key_env가 필요합니다")

    cases = load_cases(project_path(PROJECT_ROOT, settings.paths.cases))
    if args.sample_id:
        cases = [case for case in cases if case.sample_id == args.sample_id]
        if len(cases) != 1:
            raise ValueError(f"sample_id를 찾을 수 없습니다: {args.sample_id}")

    output_dir = project_path(PROJECT_ROOT, settings.paths.output)
    observations_path = output_dir / "observations.jsonl"
    existing = _load_observations(observations_path)
    if existing and not args.resume:
        parser.error("기존 실행이 있습니다. 이어서 실행하려면 --resume을 사용하세요")
    completed_ids = {observation.sample_id for observation in existing}
    pending = [case for case in cases if case.sample_id not in completed_ids]
    if args.limit is not None:
        pending = pending[: args.limit]
    if len(existing) + len(pending) > settings.limits.max_requests:
        parser.error(f"설정된 task 상한은 {settings.limits.max_requests}건입니다")

    if pending:
        provider = LiteLLMProvider(
            model=settings.provider.model,
            api_key_env=settings.provider.api_key_env,
            api_base=settings.provider.api_base,
            structured_output=settings.provider.structured_output,
            max_requests=len(pending),
            requests_per_minute=settings.limits.requests_per_minute,
            max_retries=settings.limits.max_retries,
            retry_initial_seconds=settings.limits.retry_initial_seconds,
            max_cost_usd=settings.limits.max_cost_usd,
            max_input_tokens=settings.limits.max_input_tokens,
            max_output_tokens=settings.limits.max_output_tokens,
            max_wall_seconds=settings.limits.max_wall_seconds,
            input_cost_per_token_usd=settings.provider.input_cost_per_token_usd,
            output_cost_per_token_usd=settings.provider.output_cost_per_token_usd,
        )
        for index, case in enumerate(pending, start=1):
            observation = run_cases(
                cases=[case],
                prepared_documents=project_path(
                    PROJECT_ROOT,
                    settings.paths.prepared_documents,
                ),
                prompt_path=project_path(PROJECT_ROOT, settings.paths.prompt),
                provider=provider,
            )[0]
            _append_observation(observations_path, observation)
            result = score_observations(
                [case],
                [observation],
                prepared_documents=project_path(
                    PROJECT_ROOT,
                    settings.paths.prepared_documents,
                ),
            )[0]
            print(f"[{len(existing) + index}/{len(cases)}] {case.sample_id}: {result.status}")

    all_observations = _load_observations(observations_path)
    case_by_id = {
        case.sample_id: case
        for case in load_cases(project_path(PROJECT_ROOT, settings.paths.cases))
    }
    evaluated_cases = [case_by_id[observation.sample_id] for observation in all_observations]
    results = score_observations(
        evaluated_cases,
        all_observations,
        prepared_documents=project_path(
            PROJECT_ROOT,
            settings.paths.prepared_documents,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.jsonl").write_text(
        "".join(result.model_dump_json() + "\n" for result in results),
        encoding="utf-8",
    )
    evaluate_results(results, evaluated_cases, output_dir / "deepeval")

    score_names = tuple(results[0].scores) if results else ()
    summary = {
        "record_count": len(results),
        "target_count": len(case_by_id),
        "status_counts": dict(Counter(result.status for result in results)),
        "score_averages": {
            name: round(
                sum(result.scores[name] for result in results) / len(results),
                4,
            )
            for name in score_names
        },
        "evidence_kind": "live_quality",
        "judge_status": "not_requested",
        "requested_model": settings.provider.model,
        "requests_per_minute": settings.limits.requests_per_minute,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["status_counts"].get("inconclusive", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
