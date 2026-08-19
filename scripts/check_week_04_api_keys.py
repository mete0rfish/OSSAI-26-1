# 목적: 4주차 실제 실행에 필요한 두 API key가 있는지만 안전하게 확인한다.
# 기대 결과: key 값은 보이지 않고 NVIDIA_NIM_API_KEY와 GEMINI_API_KEY가 present로 표시된다.

from __future__ import annotations

import os
from pathlib import Path

from verifiable_ai_workflow.config.secrets import load_project_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ("NVIDIA_NIM_API_KEY", "GEMINI_API_KEY")


def main() -> int:
    load_project_env(PROJECT_ROOT)
    missing = []
    for name in KEYS:
        present = bool(os.getenv(name))
        print(f"{name}: {'present' if present else 'missing'}")
        if not present:
            missing.append(name)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
