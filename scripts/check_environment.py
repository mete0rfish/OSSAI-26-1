"""Python과 Week 1 필수 패키지 설치를 확인한다."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12가 필요합니다: {sys.version.split()[0]}")

    packages = (
        "pydantic",
        "pyyaml",
        "pillow",
        "pypdfium2",
        "python-dotenv",
        "litellm",
        "deepeval",
    )
    versions = {name: importlib.metadata.version(name) for name in packages}
    print(f"Python: {sys.version.split()[0]}")
    for name, version in versions.items():
        print(f"{name}: {version}")
    print("Python·패키지 확인 완료; API 키는 provider 사전 점검에서 확인합니다")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
