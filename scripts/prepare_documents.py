"""지정한 폴더의 PDF를 전체 페이지 이미지와 manifest로 준비한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from verifiable_ai_workflow.config import load_settings, project_path
from verifiable_ai_workflow.preprocessing import DocumentPreparationError, prepare_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF 별도 전처리")
    parser.add_argument("--config", default="configs/week-01.yaml")
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    settings = load_settings(project_path(PROJECT_ROOT, args.config))
    source_dir = project_path(
        PROJECT_ROOT,
        args.source_dir or settings.paths.raw_documents,
    )
    output_dir = project_path(
        PROJECT_ROOT,
        args.output_dir or settings.paths.prepared_documents,
    )
    try:
        manifests = prepare_directory(
            source_dir,
            output_dir,
            render_dpi=settings.documents.render_dpi,
            model_image_max_bytes=settings.documents.model_image_max_bytes,
            model_image_max_width=settings.documents.model_image_max_width,
        )
    except DocumentPreparationError as exc:
        raise SystemExit(str(exc)) from None
    for manifest in manifests:
        print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
