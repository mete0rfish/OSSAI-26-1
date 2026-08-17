"""저장된 Week 2 provider 원응답을 현재 고정 규칙 채점기로 다시 계산한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifiable_ai_workflow.comparison import ComparisonContract, compare_routes, sha256_file
from verifiable_ai_workflow.config import project_path
from verifiable_ai_workflow.data.dataset import build_cases
from verifiable_ai_workflow.evaluation.scoring import score_observations
from verifiable_ai_workflow.live_provider_comparison import (
    EXPECTED_WEEK2_SAMPLE_IDS,
    build_live_comparison_contract,
    enforce_live_comparison_requirements,
    load_week2_live_config,
)
from verifiable_ai_workflow.schemas import ModelObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object가 필요합니다: {path}")
    return value


def _load_observations(path: Path) -> list[ModelObservation]:
    observations = [
        ModelObservation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if tuple(item.sample_id for item in observations) != EXPECTED_WEEK2_SAMPLE_IDS:
        raise ValueError(f"canonical 40건 원응답이 필요합니다: {path}")
    return observations


def _keep_source_prompt(
    contract: ComparisonContract,
    source_manifest: dict,
) -> ComparisonContract:
    source_prompt = source_manifest.get("prompt_sha256")
    if not isinstance(source_prompt, str) or len(source_prompt) != 64:
        raise ValueError("저장 실행의 prompt hash가 없습니다")
    return contract.model_copy(update={"prompt_sha256": source_prompt})


def _provenance_context(
    source_manifest: dict,
    run_manifest: dict,
    contract: ComparisonContract,
) -> dict:
    source_contract = run_manifest.get("comparison_contract_sha256")
    if not isinstance(source_contract, str) or len(source_contract) != 64:
        raise ValueError("저장 실행의 comparison contract hash가 없습니다")
    return {
        "source_execution": {
            "run_id": run_manifest.get("run_id"),
            "comparison_contract_sha256": source_contract,
            "input_manifest_sha256": run_manifest.get("input_manifest_sha256"),
            "scoring_profile": source_manifest.get("scoring_profile"),
            "dataset_sha256": source_manifest.get("case_authoring_sha256"),
            "prompt_sha256": source_manifest.get("prompt_sha256"),
            "output_schema_sha256": source_manifest.get("output_schema_sha256"),
            "scorer_sha256": source_manifest.get("scorer_sha256"),
            "lockfile_sha256": None,
            "limitations": [
                "원 실행 계약은 source lockfile hash를 개별 field로 저장하지 않았습니다."
            ],
        },
        "effective_rescoring": contract.model_dump(mode="json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="local-data/week-02-full-runs/provider-comparison",
    )
    parser.add_argument(
        "--output",
        default="reports/week-02/provider-comparison-rescored.json",
    )
    args = parser.parse_args()

    run_dir = project_path(PROJECT_ROOT, args.run)
    source_manifest = _load_json(run_dir / "input-manifest.json")
    run_manifest = _load_json(run_dir / "run-manifest.json")
    if source_manifest.get("sample_ids") != list(EXPECTED_WEEK2_SAMPLE_IDS):
        raise ValueError("저장 입력 manifest가 canonical 40건과 다릅니다")
    if run_manifest.get("input_manifest_sha256") != source_manifest.get("sha256"):
        raise ValueError("저장 실행과 입력 manifest hash가 다릅니다")

    config = load_week2_live_config(PROJECT_ROOT / "configs/week-02-live.yaml")
    cases = build_cases(PROJECT_ROOT / config.paths.case_authoring)
    baseline_path = run_dir / "baseline-observations.jsonl"
    candidate_path = run_dir / "candidate-observations.jsonl"
    baseline_results = score_observations(cases, _load_observations(baseline_path))
    candidate_results = score_observations(cases, _load_observations(candidate_path))
    contract = _keep_source_prompt(
        build_live_comparison_contract(PROJECT_ROOT, config),
        source_manifest,
    )
    comparison = compare_routes(
        baseline_results,
        candidate_results,
        baseline_route=config.baseline_route.descriptor(),
        candidate_route=config.candidate_route.descriptor(),
        baseline_contract=contract,
        candidate_contract=contract,
    )
    comparison = enforce_live_comparison_requirements(
        comparison,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
        baseline_route=config.baseline_route,
        candidate_route=config.candidate_route,
    )
    payload = {
        "artifact_schema_version": 2,
        "score_source": "rescored_observations",
        **_provenance_context(source_manifest, run_manifest, contract),
        "source_observation_sha256": {
            "baseline": sha256_file(baseline_path),
            "candidate": sha256_file(candidate_path),
        },
        "comparison": comparison.model_dump(mode="json"),
    }
    output = project_path(PROJECT_ROOT, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if comparison.automated_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
