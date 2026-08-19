# 목적: 처음 지시문과 Gemini가 제안한 지시문의 차이와 검증 결과를 한 화면에서 읽는다.
# 기대 결과: 바뀐 문장, 검증 문제 6개의 점수 변화, 최종 선택 이유를 쉬운 문장으로 확인한다.

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path

from verifiable_ai_workflow.open_cqa_candidates import load_open_cqa_cases
from verifiable_ai_workflow.week4_materials import load_week4_class_materials

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _changed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            lineterm="",
        )
        if line.startswith(("-", "+")) and not line.startswith(("---", "+++"))
    ]


def _comparisons(rows: list[dict]) -> list[dict]:
    by_sample: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), {})[str(row["prompt"])] = row
    comparisons = []
    for sample_id, prompts in by_sample.items():
        if set(prompts) != {"baseline", "candidate"}:
            raise ValueError(f"{sample_id}: 처음·새 지시문 결과가 모두 필요합니다")
        baseline = prompts["baseline"]
        candidate = prompts["candidate"]
        comparisons.append(
            {
                "sample_id": sample_id,
                "baseline": baseline,
                "candidate": candidate,
                "delta": float(candidate["score"]) - float(baseline["score"]),
            }
        )
    return comparisons


def _representatives(comparisons: list[dict]) -> tuple[dict, dict]:
    return (
        max(comparisons, key=lambda item: item["delta"]),
        min(comparisons, key=lambda item: item["delta"]),
    )


def _answer(row: dict) -> str:
    try:
        return str(json.loads(row["output"])["answer"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return "정해 둔 JSON 형식으로 읽을 수 없는 답"


def inspect(project_root: Path, optimization_dir: Path | None = None) -> str:
    materials = load_week4_class_materials(project_root)
    result_dir = (
        (project_root / optimization_dir).resolve()
        if optimization_dir
        else materials.prompt_optimization_dir
    )
    if not result_dir.is_relative_to(project_root.resolve()):
        raise SystemExit("결과 폴더는 project root 안에 있어야 합니다")
    paths = {
        "summary": result_dir / "summary.json",
        "candidate": result_dir / "candidate-prompt.md",
        "selected": result_dir / "selected-prompt.md",
        "validation": result_dir / "validation.jsonl",
        "calls": result_dir / "calls.jsonl",
        "baseline": project_root / "prompts/week-04-baseline.md",
        "cases": project_root / "local-data/opencqa/week-03-cases.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit("읽을 수 없는 결과 파일: " + ", ".join(missing))

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    stored_hashes = summary.get("artifact_sha256", {})
    artifacts = {
        "calls.jsonl": paths["calls"],
        "validation.jsonl": paths["validation"],
        "candidate-prompt.md": paths["candidate"],
        "selected-prompt.md": paths["selected"],
        "week-03-cases.jsonl": paths["cases"],
    }
    if any(stored_hashes.get(name) != _sha256(path) for name, path in artifacts.items()):
        raise SystemExit("summary.json에 기록된 SHA-256이 현재 결과 파일의 SHA-256과 다릅니다")
    expected_prompt = (
        paths["baseline"] if summary.get("selected") == "baseline" else paths["candidate"]
    )
    if paths["selected"].read_bytes() != expected_prompt.read_bytes():
        raise SystemExit("summary.json의 선택과 selected-prompt.md가 다릅니다")
    changes = _changed_lines(
        paths["baseline"].read_text(encoding="utf-8"),
        paths["candidate"].read_text(encoding="utf-8"),
    )
    validation_rows = _read_jsonl(paths["validation"])
    candidate_changed = bool(summary["candidate_changed"])
    if candidate_changed:
        comparisons = _comparisons(validation_rows)
    else:
        if any(row.get("prompt") != "baseline" for row in validation_rows):
            raise ValueError("지시문이 같다면 검증 결과에는 처음 지시문 결과만 있어야 합니다")
        comparisons = [
            {"sample_id": str(row["sample_id"]), "baseline": row}
            for row in validation_rows
        ]
    cases = {case.sample_id: case for case in load_open_cqa_cases(paths["cases"])}
    roles = Counter(str(row.get("provider_role")) for row in _read_jsonl(paths["calls"]))
    reason = {
        "validation_improved": "새 지시문의 검증 평균이 높아 새 지시문을 선택함",
        "validation_not_improved": "새 지시문의 검증 평균이 높지 않아 처음 지시문을 유지함",
        "candidate_identical": "제안이 처음 지시문과 같아 처음 지시문을 유지함",
    }.get(str(summary.get("selection_reason")), str(summary.get("selection_reason")))

    lines = [
        "[어느 저장 결과를 읽었나]",
        f"- 수업 자료: {materials.label if optimization_dir is None else '직접 지정한 결과'}",
        f"- 저장 위치: {result_dir.relative_to(project_root.resolve())}",
        f"- 저장 응답을 만든 코드 버전: {str(summary.get('git_sha', '알 수 없음'))[:7]}",
        "- 이 Git 번호는 결과의 출처 표시이며 멘티가 입력하거나 바꾸는 값이 아닙니다.",
        "",
        "[Prompt 최적화가 하는 일]",
        "실제 실패 답을 보고 지시문을 고친 뒤, 별도 검증 문제에서 더 나아졌는지 확인합니다.",
        "'최적화'는 새 문장을 무조건 채택한다는 뜻이 아닙니다.",
        "",
        "[이번 실행의 조건]",
        f"- 실행 상태: {summary['observed_status']}",
        f"- 개발/검증/공개 test: {summary['development_count']}/"
        f"{summary['validation_count']}/{summary['test_count']}",
        "- 공개 test를 지시문 생성·선택에 사용함: "
        f"{'예' if summary['test_used_for_generation_or_selection'] else '아니오'}",
        f"- NIM 답 생성: {roles['target']}회",
        f"- Gemini 지시문 검토: {roles['optimizer']}회",
        f"- NIM 요청/실제 모델: {summary['target_provider']['requested_model']} / "
        f"{', '.join(summary['target_provider']['actual_models'])}",
        f"- Gemini 요청/실제 모델: {summary['optimizer_provider']['requested_model']} / "
        f"{', '.join(summary['optimizer_provider']['actual_models'])}",
        f"- API 오류/모델 불일치: {summary['provider_error_count']}/{summary['model_drift_count']}",
        "",
        "[지시문에서 바뀐 줄]",
        *(changes or ["바뀐 줄 없음"]),
        "",
        "[검증 문제 6개]",
    ]
    for item in comparisons:
        if candidate_changed:
            lines.append(
                f"- {item['sample_id']}: {item['baseline']['score']:.4f} → "
                f"{item['candidate']['score']:.4f} ({item['delta']:+.4f})"
            )
        else:
            lines.append(
                f"- {item['sample_id']}: 처음 지시문 {item['baseline']['score']:.4f} "
                "(새 지시문은 실행하지 않음)"
            )
    lines.extend(
        [
            "",
            "[최종 선택]",
            f"- 처음 지시문 평균: {summary['baseline_mean']:.4f}",
            (
                f"- 새 지시문 평균: {summary['candidate_mean']:.4f}"
                if summary["candidate_mean"] is not None
                else "- 새 지시문 평균: 지시문이 같아 실행하지 않음"
            ),
            f"- 결론: {reason}",
        ]
    )
    if not candidate_changed:
        lines.extend(
            ["", "[대표 변화 사례 없음]", "지시문이 같아 새 답을 만들지 않았습니다."]
        )
        return "\n".join(lines)

    best, worst = _representatives(comparisons)
    for item, label in (
        (best, "점수가 가장 오른 사례"),
        (worst, "점수가 가장 떨어진 사례"),
    ):
        sample_id = item["sample_id"]
        lines.extend(
            [
                "",
                f"[{label}: {sample_id}]",
                f"질문: {cases[sample_id].question}",
                f"처음 답: {_answer(item['baseline'])}",
                f"새 답: {_answer(item['candidate'])}",
                f"감점 이유: {item['candidate']['reason']}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="4주차 지시문 최적화 저장 결과를 쉽게 읽습니다")
    parser.add_argument(
        "--optimization-dir",
        type=Path,
        help="직접 지정한 전체 실행 결과 폴더를 점검합니다",
    )
    args = parser.parse_args()
    print(inspect(PROJECT_ROOT, args.optimization_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
