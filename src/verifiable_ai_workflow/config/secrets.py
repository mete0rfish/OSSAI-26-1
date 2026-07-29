"""API key 값은 환경 변수에서만 읽는다."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(project_root: str | Path) -> Path:
    """프로젝트의 .env를 명시적으로 읽는다."""

    env_path = Path(project_root) / ".env"
    load_dotenv(env_path, override=False)
    return env_path


def require_api_key(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"{env_name} 환경 변수가 필요합니다")
    return value
