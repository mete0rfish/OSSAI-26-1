"""튜터가 지정한 4주차 수업용 저장 결과 경로를 읽는다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Week4ClassMaterials:
    label: str
    prompt_optimization_dir: Path
    image_response_dir: Path


def _project_path(project_root: Path, value: str) -> Path:
    path = (project_root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(project_root.resolve()):
        raise ValueError("4주차 수업 자료 경로는 project root 안의 상대 경로여야 합니다")
    return path


def load_week4_class_materials(project_root: Path) -> Week4ClassMaterials:
    config = yaml.safe_load((project_root / "configs/week-04.yaml").read_text(encoding="utf-8"))
    materials = config["class_materials"]
    return Week4ClassMaterials(
        label=str(materials["label"]),
        prompt_optimization_dir=_project_path(
            project_root, str(materials["prompt_optimization_dir"])
        ),
        image_response_dir=_project_path(project_root, str(materials["image_response_dir"])),
    )
