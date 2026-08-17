"""공식 OpenCQA clone에서 Week 3 로컬 실습 자료를 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml
from PIL import Image

from verifiable_ai_workflow.preprocessing import save_model_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION = PROJECT_ROOT / "data/opencqa/week-03-selection.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "local-data/opencqa"
COURSE_SPLITS = ("development", "validation", "test")
FORBIDDEN_MARKERS = ("chrome-extension", "class=", "& amp ;")
HTML_TAG = re.compile(r"<[^>]*>")


def _git_revision(source_root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if changes.strip():
        raise ValueError("OpenCQA source clone에 commit되지 않은 변경이 있습니다")
    return revision


def _validate_text(sample_id: str, question: str, *answers: str) -> None:
    values = (question, *answers)
    if any(marker in value for value in values for marker in FORBIDDEN_MARKERS) or any(
        HTML_TAG.search(value) for value in values
    ):
        raise ValueError(f"OpenCQA sample {sample_id}에 금지된 원본 marker가 있습니다")
    if any(answer.count('"') % 2 for answer in answers):
        raise ValueError(f"OpenCQA sample {sample_id}의 답에 닫히지 않은 ASCII 따옴표가 있습니다")


def prepare(source_root: Path, output_root: Path = OUTPUT_ROOT) -> list[dict[str, object]]:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    if _git_revision(source_root) != selection["revision"]:
        raise ValueError("OpenCQA revision이 week-03-selection.yaml과 다릅니다")
    license_path = source_root / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError("OpenCQA LICENSE를 찾을 수 없습니다")
    license_text = license_path.read_text(encoding="utf-8")
    if selection["license"] == "GPL-3.0" and not all(
        marker in license_text for marker in ("GNU GENERAL PUBLIC LICENSE", "Version 3")
    ):
        raise ValueError("OpenCQA LICENSE가 selection의 GPL-3.0과 다릅니다")

    source_split = selection["source_split"]
    split_ids = selection["course_splits"]
    if set(split_ids) != set(COURSE_SPLITS):
        raise ValueError("course_splits에는 development, validation, test가 모두 필요합니다")
    selected = [
        (sample_id, course_split)
        for course_split in COURSE_SPLITS
        for sample_id in split_ids[course_split]
    ]
    if len(selected) != 30 or len({sample_id for sample_id, _ in selected}) != 30:
        raise ValueError("OpenCQA 선택은 중복 없는 30개 ID여야 합니다")

    annotation = source_root / f"etc/data(full_summary_article)/{source_split}_extended.json"
    chart_root = source_root / "chart_images"
    if not annotation.is_file() or not chart_root.is_dir():
        raise FileNotFoundError("OpenCQA annotation 또는 chart_images를 찾을 수 없습니다")
    source = json.loads(annotation.read_text(encoding="utf-8"))

    cases: list[dict[str, object]] = []
    source_images: list[tuple[Path, str]] = []
    for sample_id, course_split in selected:
        try:
            image_name, _title, _article, _summary, question, abstractive, _extractive = source[
                sample_id
            ]
        except KeyError as exc:
            raise ValueError(f"OpenCQA val split에 sample {sample_id}가 없습니다") from exc
        source_image = chart_root / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"OpenCQA chart image가 없습니다: {source_image}")
        _validate_text(sample_id, question, abstractive)
        output_name = f"{Path(image_name).stem}.jpg"
        cases.append(
            {
                "artifact_schema_version": 1,
                "pair_id": f"opencqa-{source_split}-{sample_id}",
                "sample_id": sample_id,
                "family_id": f"opencqa-{source_split}-{Path(image_name).stem}",
                "course_split": course_split,
                "source_split": source_split,
                "source_revision": selection["revision"],
                "source_license": selection["license"],
                "image_path": f"local-data/opencqa/images/{output_name}",
                "question": question,
                "reference_answer": abstractive,
            }
        )
        source_images.append((source_image, output_name))

    image_output = output_root / "images"
    image_output.mkdir(parents=True, exist_ok=True)
    for case, (source_image, output_name) in zip(cases, source_images, strict=True):
        output_path = image_output / output_name
        with Image.open(source_image) as image:
            save_model_image(image, output_path, max_bytes=175_000, max_width=1024)
        case["image_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()

    cases_path = output_root / "week-03-cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    cases = prepare(args.source_root.resolve())
    print(f"OpenCQA Week 3 case {len(cases)}개를 local-data/opencqa에 준비했습니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
