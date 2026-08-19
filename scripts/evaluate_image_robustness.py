# 목적: 저장된 원본·변형 이미지 답을 사람이 표시한 근거 상태에 맞춰 다시 채점한다.
# 기대 결과: 개인 reports 폴더에 이미지 5개의 판정과 SHA-256 기록이 생긴다.

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from verifiable_ai_workflow.image_robustness import (
    VariantArtifact,
    load_response_map,
    load_reviews,
    score_original,
    score_variant,
)
from verifiable_ai_workflow.week4_materials import load_week4_class_materials

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "local-data/opencqa/week-04-variants"
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def _student_alias(value: str) -> str:
    if not _ALIAS.fullmatch(value):
        raise argparse.ArgumentTypeError("별칭은 영문·숫자로 시작하고 -와 _만 쓸 수 있습니다")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _variant_inputs(artifacts: list[VariantArtifact]) -> dict[str, tuple[str, str, str, str]]:
    return {
        item.variant_id: (
            item.sample_id,
            item.intended_behavior,
            item.source_sha256,
            item.image_sha256,
        )
        for item in artifacts
    }


def _same_variant_inputs(
    artifacts: list[VariantArtifact],
    case: dict,
    canonical_artifacts: list[VariantArtifact],
    canonical_case: dict,
) -> bool:
    return (
        _variant_inputs(artifacts) == _variant_inputs(canonical_artifacts)
        and case == canonical_case
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-alias", type=_student_alias, help="예: minsu")
    parser.add_argument("--variants", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    materials = load_week4_class_materials(PROJECT_ROOT) if args.student_alias else None
    student_root = (
        PROJECT_ROOT / "local-data/week-04-students" / args.student_alias / "variants"
        if args.student_alias
        else DEFAULT_ROOT
    )
    variants_path = args.variants or student_root / "variants.jsonl"
    reviews_path = args.reviews or student_root / "variant-review.csv"
    case_path = args.case or student_root / "case.json"
    responses_path = args.responses or (
        materials.image_response_dir / "responses.jsonl" if materials else None
    )
    output_path = args.output or (
        PROJECT_ROOT / "reports/week-04/students" / args.student_alias / "evaluation.json"
        if args.student_alias
        else PROJECT_ROOT / "reports/week-04/robustness.json"
    )
    if responses_path is None:
        parser.error("--responses 또는 --student-alias가 필요합니다")
    artifacts = [
        VariantArtifact.model_validate_json(line)
        for line in variants_path.read_text().splitlines()
        if line.strip()
    ]
    case = json.loads(case_path.read_text(encoding="utf-8"))
    source_summary_path = responses_path.parent / "summary.json"
    source_summary = (
        json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary_path.is_file()
        else {}
    )
    stored_hashes = source_summary.get("artifact_sha256", {})
    response_inputs = {
        "responses.jsonl": responses_path,
        "case.json": case_path,
        "variants.jsonl": variants_path,
        "variant-review.csv": reviews_path,
    }
    if stored_hashes.get("responses.jsonl") != _sha256(responses_path):
        raise SystemExit(
            "이미지 응답을 만들 때 기록한 SHA-256과 현재 응답 파일의 SHA-256이 다릅니다"
        )
    if not args.student_alias and (
        not stored_hashes
        or any(stored_hashes.get(name) != _sha256(path) for name, path in response_inputs.items())
    ):
        raise SystemExit(
            "이미지 응답을 만들 때 기록한 SHA-256과 현재 평가 입력의 SHA-256이 다릅니다"
        )
    if args.student_alias:
        canonical_artifacts = [
            VariantArtifact.model_validate_json(line)
            for line in (DEFAULT_ROOT / "variants.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if (
            not _same_variant_inputs(
                artifacts,
                case,
                canonical_artifacts,
                json.loads((DEFAULT_ROOT / "case.json").read_text(encoding="utf-8")),
            )
            or stored_hashes.get("variants.jsonl") != _sha256(DEFAULT_ROOT / "variants.jsonl")
            or stored_hashes.get("case.json") != _sha256(DEFAULT_ROOT / "case.json")
        ):
            raise SystemExit("개인 이미지와 저장 응답에 사용한 이미지가 다릅니다")
    reviews = load_reviews(
        reviews_path,
        artifacts,
        project_root=PROJECT_ROOT,
        source_path=PROJECT_ROOT / case["original_image"],
    )
    responses = load_response_map(responses_path)
    required = {"original", *(item.variant_id for item in artifacts)}
    if set(responses) != required:
        raise SystemExit(f"응답 ID가 다릅니다: required={sorted(required)}")
    scores = [score_original(case["reference_answer"], responses["original"])] + [
        score_variant(
            item,
            reviews[item.variant_id],
            case["reference_answer"],
            responses["original"],
            responses[item.variant_id],
        )
        for item in artifacts
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([item.model_dump() for item in scores], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "source_git_sha": source_summary.get("git_sha"),
        "evaluation_sha256": _sha256(output_path),
        "responses_sha256": _sha256(responses_path),
        "case_sha256": _sha256(case_path),
        "variants_sha256": _sha256(variants_path),
        "reviews_sha256": _sha256(reviews_path),
        "scorer_sha256": _sha256(PROJECT_ROOT / "src/verifiable_ai_workflow/image_robustness.py"),
        "metric_sha256": _sha256(
            PROJECT_ROOT / "src/verifiable_ai_workflow/prompt_optimization.py"
        ),
        "schema_sha256": _sha256(PROJECT_ROOT / "src/verifiable_ai_workflow/schemas/models.py"),
    }
    (output_path.parent / "evaluation-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        (f"수업 자료={materials.label}, " if materials else "")
        + f"통과={sum(item.status == 'passed' for item in scores)}, "
        f"실패={sum(item.status == 'failed' for item in scores)}, "
        f"판정 불가={sum(item.status == 'inconclusive' for item in scores)}, "
        f"변형 무효={sum(item.status == 'invalid_variant' for item in scores)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
