"""승인된 실제 NIM raw response를 API 없는 회귀 fixture로 고정한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.data.dataset import load_cases
from verifiable_ai_workflow.schemas import ModelObservation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="실제 NIM 응답 fixture 고정")
    parser.add_argument("--config", default="configs/nvidia-nim.yaml")
    parser.add_argument(
        "--output",
        default="data/recorded/week-01-nvidia-responses.jsonl",
    )
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    observations_path = project_path(PROJECT_ROOT, settings.paths.output) / "observations.jsonl"
    observations = [
        ModelObservation.model_validate_json(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = load_cases(project_path(PROJECT_ROOT, settings.paths.cases))
    expected_ids = {case.sample_id for case in cases}
    actual_ids = {observation.sample_id for observation in observations}
    if actual_ids != expected_ids:
        raise ValueError(
            "준비된 전체 질문의 실행 결과가 필요합니다: "
            f"missing={sorted(expected_ids - actual_ids)}"
        )
    if any(observation.model_error for observation in observations):
        raise ValueError("provider 오류가 있는 실행은 fixture로 고정할 수 없습니다")

    target = project_path(PROJECT_ROOT, args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": observation.sample_id,
                    "response": observation.raw_output,
                    "model_call": observation.model_call,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for observation in observations
        ),
        encoding="utf-8",
    )
    print(f"{target}: {len(observations)}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
