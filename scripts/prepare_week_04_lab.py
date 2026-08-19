# 목적: 4주차 실습에 필요한 입력과 저장 결과를 한 번에 확인하고 개인 폴더를 준비한다.
# 기대 결과: 18/6/6 데이터 분할과 수업용 결과 경로가 보이고, 개인 기록 폴더가 만들어진다.

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from verifiable_ai_workflow.image_robustness import VariantArtifact, load_reviews
from verifiable_ai_workflow.open_cqa_candidates import load_open_cqa_cases
from verifiable_ai_workflow.prompt_optimization import split_goldens
from verifiable_ai_workflow.week4_materials import load_week4_class_materials

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def _student_alias(value: str) -> str:
    if not _ALIAS.fullmatch(value):
        raise argparse.ArgumentTypeError("별칭은 영문·숫자로 시작하고 -와 _만 쓸 수 있습니다")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stored_hashes_match(stored: dict, paths: dict[str, Path]) -> bool:
    return bool(stored) and all(stored.get(name) == _sha256(path) for name, path in paths.items())


def _require_paths(paths: list[Path], project_root: Path) -> None:
    missing = [str(path.relative_to(project_root)) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("준비되지 않은 파일이 있습니다:\n- " + "\n- ".join(missing))


def prepare(
    project_root: Path,
    alias: str,
    *,
    create_student_files: bool = True,
) -> dict[str, object]:
    materials = load_week4_class_materials(project_root)
    cases_path = project_root / "local-data/opencqa/week-03-cases.jsonl"
    variant_root = project_root / "local-data/opencqa/week-04-variants"
    optimization = materials.prompt_optimization_dir
    robustness = materials.image_response_dir
    required = [
        cases_path,
        project_root / "local-data/opencqa/images",
        project_root / "prompts/week-04-baseline.md",
        project_root / "configs/nvidia-nim-gemma4.yaml",
        project_root / "configs/google-gemini-3.5-flash-lite-judge.yaml",
        variant_root / "case.json",
        variant_root / "variants.jsonl",
        variant_root / "variant-review.csv",
        *(
            optimization / name
            for name in (
                "candidate-prompt.md",
                "selected-prompt.md",
                "validation.jsonl",
                "summary.json",
                "calls.jsonl",
            )
        ),
        *(
            robustness / name
            for name in (
                "calls.jsonl",
                "responses.jsonl",
                "summary.json",
                "evaluation.json",
                "evaluation-manifest.json",
            )
        ),
    ]
    _require_paths(required, project_root)

    variant_case = json.loads((variant_root / "case.json").read_text(encoding="utf-8"))
    variants = [
        VariantArtifact.model_validate_json(line)
        for line in (variant_root / "variants.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        load_reviews(
            variant_root / "variant-review.csv",
            variants,
            project_root=project_root,
            source_path=project_root / variant_case["original_image"],
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"공통 이미지 변형 확인 실패: {exc}") from exc

    splits = split_goldens(load_open_cqa_cases(cases_path))
    counts = {name: len(rows) for name, rows in splits.items()}
    summary = json.loads((optimization / "summary.json").read_text(encoding="utf-8"))
    robustness_summary = json.loads((robustness / "summary.json").read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        (robustness / "evaluation-manifest.json").read_text(encoding="utf-8")
    )
    optimization_paths = {
        "calls.jsonl": optimization / "calls.jsonl",
        "validation.jsonl": optimization / "validation.jsonl",
        "candidate-prompt.md": optimization / "candidate-prompt.md",
        "selected-prompt.md": optimization / "selected-prompt.md",
        "week-03-cases.jsonl": cases_path,
    }
    robustness_paths = {
        "calls.jsonl": robustness / "calls.jsonl",
        "responses.jsonl": robustness / "responses.jsonl",
        "week-03-cases.jsonl": cases_path,
        "case.json": variant_root / "case.json",
        "variants.jsonl": variant_root / "variants.jsonl",
        "variant-review.csv": variant_root / "variant-review.csv",
    }
    evaluation_paths = {
        "evaluation_sha256": robustness / "evaluation.json",
        "responses_sha256": robustness / "responses.jsonl",
        "case_sha256": variant_root / "case.json",
        "variants_sha256": variant_root / "variants.jsonl",
        "reviews_sha256": variant_root / "variant-review.csv",
    }
    expected_input_hash = summary.get("artifact_sha256", {}).get("week-03-cases.jsonl")
    source_git_sha = summary.get("git_sha")
    target_budget = summary.get("target_provider", {}).get("budget", {})
    optimizer_budget = summary.get("optimizer_provider", {}).get("budget", {})
    call_counts = (
        target_budget.get("request_count"),
        target_budget.get("attempt_count"),
        optimizer_budget.get("request_count"),
        optimizer_budget.get("attempt_count"),
    )
    same_git_sha = bool(source_git_sha) and source_git_sha == robustness_summary.get("git_sha")
    checks = {
        "지시문 실행 완료": summary.get("observed_status") == "complete",
        "지시문 결과 파일 SHA-256 일치": _stored_hashes_match(
            summary.get("artifact_sha256", {}), optimization_paths
        ),
        "두 저장 결과의 코드 버전 일치": same_git_sha,
        "두 저장 결과의 선택 지시문 일치": summary.get("selected_prompt_sha256")
        == summary.get("artifact_sha256", {}).get("selected-prompt.md")
        == robustness_summary.get("prompt_sha256"),
        "두 저장 결과의 입력 일치": summary.get("artifact_sha256", {}).get("week-03-cases.jsonl")
        == robustness_summary.get("artifact_sha256", {}).get("week-03-cases.jsonl"),
        "데이터 분할 18/6/6": counts
        == {
            "development": 18,
            "validation": 6,
            "test": 6,
        },
        "공개 test 미사용": summary.get("test_used_for_generation_or_selection") is False,
        "현재 입력과 저장 입력 일치": expected_input_hash == _sha256(cases_path),
        "지시문 호출 승인 상한 이내": all(isinstance(value, int) for value in call_counts)
        and 0 < call_counts[0] <= 45
        and 0 < call_counts[1] <= 45
        and 0 < call_counts[2] <= 4
        and 0 < call_counts[3] <= 8,
        "지시문 실행 오류·모델 불일치 0건": summary.get("provider_error_count") == 0
        and summary.get("model_drift_count") == 0,
        "이미지 응답 5개 완료": robustness_summary.get("observed_status") == "complete"
        and robustness_summary.get("record_count") == 5
        and robustness_summary.get("target_count") == 5
        and robustness_summary.get("invalid_output_count") == 0,
        "이미지 결과 파일 SHA-256 일치": _stored_hashes_match(
            robustness_summary.get("artifact_sha256", {}), robustness_paths
        )
        and _stored_hashes_match(evaluation_manifest, evaluation_paths)
        and evaluation_manifest.get("source_git_sha") == robustness_summary.get("git_sha"),
        "이미지 실행 오류·모델 불일치 0건": robustness_summary.get("provider_error_count") == 0
        and robustness_summary.get("model_drift_count") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("수업용 저장 결과 확인 실패:\n- " + "\n- ".join(failed))

    student_root = project_root / "local-data/week-04-students" / alias
    report_root = project_root / "reports/week-04/students" / alias
    if create_student_files:
        student_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
        progress = project_root / "local-data/learning-progress.md"
        if not progress.exists():
            shutil.copyfile(
                project_root.parents[1] / "templates/learner-progress-template.md",
                progress,
            )

    return {
        "materials_label": materials.label,
        "source_git_sha": str(source_git_sha)[:7],
        "target_requests": call_counts[0],
        "optimizer_requests": call_counts[2],
        "counts": counts,
        "optimization": optimization.relative_to(project_root),
        "robustness": robustness.relative_to(project_root),
        "student_root": student_root.relative_to(project_root),
        "report_root": report_root.relative_to(project_root),
        "selected": summary["selected"],
        "selection_reason": summary["selection_reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="4주차 입력·저장 결과를 확인하고 개인 실습 폴더를 준비합니다"
    )
    parser.add_argument("--alias", type=_student_alias, help="멘티 별칭. 예: minsu")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="멘토 점검용: 개인 폴더를 만들지 않고 준비 상태만 확인합니다",
    )
    args = parser.parse_args()
    if not args.check_only and not args.alias:
        parser.error("개인 폴더를 만들 때는 --alias가 필요합니다")
    result = prepare(
        PROJECT_ROOT,
        args.alias or "tutor-check",
        create_student_files=not args.check_only,
    )

    print("4주차 준비 상태 확인 완료" if args.check_only else "4주차 실습 준비 완료")
    print(f"- 수업 자료: {result['materials_label']}")
    print(
        f"- 저장 응답을 만든 코드 버전: {result['source_git_sha']} "
        "(멘티가 입력하거나 바꾸는 값 아님)"
    )
    print("- 데이터: 개발 18개, 검증 6개, 공개 test 6개")
    if args.check_only:
        print(
            f"- 실제 API 호출: NIM {result['target_requests']}회, "
            f"Gemini {result['optimizer_requests']}회 (승인 상한 45/4회)"
        )
        selection = {
            "validation_improved": "새 지시문의 검증 평균이 높아 새 지시문을 선택함",
            "validation_not_improved": "새 지시문의 검증 평균이 높지 않아 처음 지시문을 유지함",
            "candidate_identical": "제안이 처음 지시문과 같아 처음 지시문을 유지함",
        }.get(str(result["selection_reason"]), str(result["selection_reason"]))
        print(f"- 선택 결과: {selection}")
        print(f"- 지시문 비교 자료: {result['optimization']}")
        print(f"- 이미지 응답 자료: {result['robustness']}")
    else:
        print("- 다음 순서: 개발 사례 2건 시연 후 전체 저장 결과 확인")
        print(f"- 개인 작업 폴더: {result['student_root']}")
        print(f"- 개인 결과 폴더: {result['report_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
