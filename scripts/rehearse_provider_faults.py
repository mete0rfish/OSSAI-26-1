"""저장된 여섯 API 장애 상황을 품질 실패와 분리해 확인한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifiable_ai_workflow.provider_faults import rehearse_faults

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    faults = rehearse_faults(PROJECT_ROOT / "data/scenarios/week-02-provider-faults.yaml")
    output = PROJECT_ROOT / "reports/week-02/faults.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(faults, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(faults, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
