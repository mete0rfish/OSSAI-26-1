"""저장된 40건 호출 경로(route)와 여섯 API 장애 상황을 비교한다."""

from __future__ import annotations

import json
from pathlib import Path

from verifiable_ai_workflow.evaluation.scoring import SCORING_PROFILE
from verifiable_ai_workflow.provider_evaluation import rehearse_faults, run_offline_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output_dir = PROJECT_ROOT / "reports/week-02"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = run_offline_comparison(PROJECT_ROOT)
    faults = rehearse_faults(PROJECT_ROOT / "data/scenarios/week-02-provider-faults.yaml")
    (output_dir / "comparison.json").write_text(
        comparison.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "faults.json").write_text(
        json.dumps(faults, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "offline_rehearsal_complete",
        "evidence_kind": "test_only",
        "input_modality": "recorded_image_vlm_responses",
        "scoring_profile": SCORING_PROFILE,
        "sample_count": comparison.baseline.record_count,
        "classification_counts": comparison.classification_counts,
        "automated_status": comparison.automated_status,
        "fault_scenario_count": faults["scenario_count"],
        "live_quality_claim": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    expected = {"new_success": 2, "unchanged": 38}
    return 0 if comparison.classification_counts == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
