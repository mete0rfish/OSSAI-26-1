# 목적: 원본 차트에서 회전·압축·잘림·가림 이미지와 사람 검토표를 만든다.
# 기대 결과: 개인 variants 폴더에 이미지 4개, case.json, variants.jsonl, 검토표가 생긴다.

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from verifiable_ai_workflow.image_robustness import generate_variants
from verifiable_ai_workflow.open_cqa_candidates import load_open_cqa_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def _student_alias(value: str) -> str:
    if not _ALIAS.fullmatch(value):
        raise argparse.ArgumentTypeError("별칭은 영문·숫자로 시작하고 -와 _만 쓸 수 있습니다")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-number", type=int, default=1)
    parser.add_argument("--student-alias", type=_student_alias, help="예: minsu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.student_alias and args.output:
        parser.error("--student-alias와 --output은 함께 쓸 수 없습니다")
    output = (
        PROJECT_ROOT / "local-data/week-04-students" / args.student_alias / "variants"
        if args.student_alias
        else args.output or PROJECT_ROOT / "local-data/opencqa/week-04-variants"
    )
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"비어 있지 않은 출력 폴더입니다: {output}")
    cases = load_open_cqa_cases(PROJECT_ROOT / "local-data/opencqa/week-03-cases.jsonl")
    if not 1 <= args.pair_number <= len(cases):
        raise SystemExit(f"--pair-number는 1부터 {len(cases)}까지입니다")
    case = cases[args.pair_number - 1]
    source = PROJECT_ROOT / case.image_path
    artifacts = generate_variants(
        source_path=source,
        sample_id=case.sample_id,
        output_dir=output,
        config_path=PROJECT_ROOT / "configs/week-04.yaml",
        project_root=PROJECT_ROOT,
        expected_source_sha256=case.image_sha256,
    )
    (output / "variants.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in artifacts), encoding="utf-8"
    )
    review = output / "variant-review.csv"
    if not review.exists():
        with review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_id",
                    "variant_id",
                    "image_sha256",
                    "intended_behavior",
                    "grounding_status",
                ]
            )
            for item in artifacts:
                writer.writerow(
                    [
                        item.sample_id,
                        item.variant_id,
                        item.image_sha256,
                        item.intended_behavior,
                        "",
                    ]
                )
    (output / "case.json").write_text(
        json.dumps(
            {
                "pair_id": case.pair_id,
                "sample_id": case.sample_id,
                "family_id": case.family_id,
                "course_split": case.course_split,
                "source_split": case.source_split,
                "source_revision": case.source_revision,
                "source_license": case.source_license,
                "question": case.question,
                "reference_answer": case.reference_answer,
                "original_image": case.image_path,
                "original_image_sha256": case.image_sha256,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"이미지 변형 {len(artifacts)}개와 검토표를 {output}에 만들었습니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
