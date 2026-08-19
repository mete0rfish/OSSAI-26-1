# 목적: 선택한 지시문으로 원본과 변형 이미지 4개를 같은 NIM Gemma에 보낸다.
# 기대 결과: 이미지 5개의 원응답, 구조화 답, 모델·사용량·오류 기록이 저장된다.

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

from deepeval.prompt import Prompt

from verifiable_ai_workflow.config.secrets import load_project_env
from verifiable_ai_workflow.config.settings import load_settings
from verifiable_ai_workflow.course_live import build_course_provider, summarize_call_failures
from verifiable_ai_workflow.image_robustness import VariantArtifact, load_reviews
from verifiable_ai_workflow.live_execution import LiveBudgetCaps
from verifiable_ai_workflow.open_cqa_candidates import load_open_cqa_cases
from verifiable_ai_workflow.schemas import StructuredAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_ROOT = PROJECT_ROOT / "local-data/opencqa/week-04-variants"
CASES = PROJECT_ROOT / "local-data/opencqa/week-03-cases.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise SystemExit("이미지 견고성 실제 실행은 변경사항이 없는 Git commit에서만 허용합니다")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _image_message(path: Path, question: str, expected_sha256: str) -> dict:
    image_bytes = path.read_bytes()
    if hashlib.sha256(image_bytes).hexdigest() != expected_sha256:
        raise ValueError("전송할 이미지 bytes가 manifest SHA-256과 다릅니다")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    suffix = "jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "png"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{suffix};base64,{encoded}"},
            },
        ],
    }


def _require_current_pair(case: dict) -> None:
    current = next(
        (item for item in load_open_cqa_cases(CASES) if item.pair_id == case.get("pair_id")),
        None,
    )
    if current is None:
        raise SystemExit("Week 4 case의 pair_id가 현재 OpenCQA 30쌍에 없습니다")
    expected = {
        "pair_id": current.pair_id,
        "sample_id": current.sample_id,
        "family_id": current.family_id,
        "course_split": current.course_split,
        "source_split": current.source_split,
        "source_revision": current.source_revision,
        "source_license": current.source_license,
        "question": current.question,
        "reference_answer": current.reference_answer,
        "original_image": current.image_path,
        "original_image_sha256": current.image_sha256,
    }
    if case != expected:
        raise SystemExit("Week 4 case가 현재 OpenCQA pair identity와 다릅니다")


def _artifact_sha256(output: Path, variants_dir: Path) -> dict[str, str | None]:
    paths = {
        "calls.jsonl": output / "calls.jsonl",
        "responses.jsonl": output / "responses.jsonl",
        "week-03-cases.jsonl": CASES,
        "case.json": variants_dir / "case.json",
        "variants.jsonl": variants_dir / "variants.jsonl",
        "variant-review.csv": variants_dir / "variant-review.csv",
    }
    return {name: _sha256(path) if path.is_file() else None for name, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prompt", type=Path, default=PROJECT_ROOT / "prompts/week-04-baseline.md")
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--catalog-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--pricing-verified-on", type=date.fromisoformat, required=True)
    parser.add_argument("--variants-dir", type=Path, default=VARIANT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        raise SystemExit("실제 VLM 호출에는 --live가 필요합니다")
    catalog_age = (date.today() - args.catalog_verified_on).days
    if catalog_age < 0 or catalog_age > 7:
        raise SystemExit("--catalog-verified-on은 오늘부터 7일 이내여야 합니다")
    pricing_age = (date.today() - args.pricing_verified_on).days
    if pricing_age < 0 or pricing_age > 7:
        raise SystemExit("--pricing-verified-on은 오늘부터 7일 이내여야 합니다")
    caps = LiveBudgetCaps(
        max_requests=args.max_requests,
        max_attempts=args.max_requests,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_cost_usd=args.max_cost_usd,
        max_wall_seconds=args.max_wall_seconds,
    )
    approved_caps = LiveBudgetCaps(
        max_requests=5,
        max_attempts=5,
        max_input_tokens=100_000,
        max_output_tokens=2_500,
        max_cost_usd=0.01,
        max_wall_seconds=900,
    )
    if caps != approved_caps:
        raise SystemExit("원본 1개와 변형 4개 실행은 문서의 승인 cap과 정확히 같아야 합니다")
    git_sha = _git_sha()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"비어 있지 않은 출력 폴더입니다: {args.output}")

    variants_dir = args.variants_dir
    case = json.loads((variants_dir / "case.json").read_text(encoding="utf-8"))
    _require_current_pair(case)
    variants = [
        VariantArtifact.model_validate_json(line)
        for line in (variants_dir / "variants.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(variants) != 4:
        raise SystemExit("이미지 견고성 실행에는 변형 4개가 필요합니다")
    if {item.sample_id for item in variants} != {case["sample_id"]}:
        raise SystemExit("case와 이미지 변형의 sample_id가 다릅니다")
    original_image = PROJECT_ROOT / case["original_image"]
    if _sha256(original_image) != case["original_image_sha256"]:
        raise SystemExit("원본 이미지 bytes가 Week 4 case와 다릅니다")
    load_reviews(
        variants_dir / "variant-review.csv",
        variants,
        project_root=PROJECT_ROOT,
        source_path=original_image,
    )
    source_hashes = _artifact_sha256(args.output, variants_dir)
    source_names = (
        "week-03-cases.jsonl",
        "case.json",
        "variants.jsonl",
        "variant-review.csv",
    )
    images = [("original", original_image, case["original_image_sha256"])] + [
        (item.variant_id, PROJECT_ROOT / item.image_path, item.image_sha256) for item in variants
    ]
    prompt_bytes = args.prompt.read_bytes()
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    prompt = Prompt(text_template=prompt_bytes.decode("utf-8"))
    instruction = prompt.interpolate(question=case["question"])
    settings = load_settings(PROJECT_ROOT / "configs/nvidia-nim-gemma4.yaml")
    load_project_env(PROJECT_ROOT)
    calls_path = args.output / "calls.jsonl"
    call_records: list[dict] = []
    last_journal_call: dict | None = None

    def record_call(call: dict) -> None:
        nonlocal last_journal_call
        call_records.append(call)
        with calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(call, ensure_ascii=False) + "\n")
        last_journal_call = call

    provider = build_course_provider(
        settings,
        caps,
        structured_output="json_schema",
        on_response=record_call,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    responses_path = args.output / "responses.jsonl"
    record_count = 0
    invalid_output_count = 0
    run_error: Exception | None = None
    try:
        for variant_id, image_path, image_sha256 in images:
            raw = provider.generate(
                f"{case['sample_id']}:{variant_id}",
                [
                    {"role": "system", "content": instruction},
                    _image_message(image_path, case["question"], image_sha256),
                ],
            )
            try:
                parsed = StructuredAnswer.model_validate_json(raw)
                response_record = {
                    "variant_id": variant_id,
                    "raw_output": raw,
                    "output": parsed.model_dump(mode="json"),
                    "parse_error": None,
                }
            except Exception as exc:
                invalid_output_count += 1
                response_record = {
                    "variant_id": variant_id,
                    "raw_output": raw,
                    "output": None,
                    "parse_error": {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                }
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(response_record, ensure_ascii=False) + "\n")
            record_count += 1
    except Exception as exc:
        run_error = exc
        terminal_call = dict(provider.last_call or {})
        if terminal_call and terminal_call != last_journal_call:
            record_call(terminal_call)
    actual_models = sorted(
        {str(call["actual_model"]) for call in call_records if call.get("actual_model") is not None}
    )
    provider_error_count, model_drift_count = summarize_call_failures(
        call_records, provider.expected_actual_model
    )
    artifact_hashes = _artifact_sha256(args.output, variants_dir)
    input_changed = (
        (_sha256(args.prompt) if args.prompt.is_file() else None) != prompt_sha256
        or any(artifact_hashes[name] != source_hashes[name] for name in source_names)
        or any(
            not image_path.is_file() or _sha256(image_path) != image_sha256
            for _, image_path, image_sha256 in images
        )
    )
    artifact_hashes.update({name: source_hashes[name] for name in source_names})
    complete = record_count == 5 and run_error is None and not input_changed
    budget_summary = provider.budget.summary()
    summary = {
        "status": (
            "inconclusive"
            if not complete
            or provider_error_count
            or model_drift_count
            or actual_models != [provider.expected_actual_model]
            else "fail"
            if invalid_output_count
            else "pass"
        ),
        "observed_status": (
            "complete"
            if complete
            else "partial"
            if budget_summary.get("attempt_count", 0)
            else "not_run"
        ),
        "evidence_kind": "live_quality",
        "git_sha": git_sha,
        "catalog_verified_on": args.catalog_verified_on.isoformat(),
        "billing_basis": settings.provider.billing_basis,
        "structured_output": provider.structured_output,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": _sha256(PROJECT_ROOT / "src/verifiable_ai_workflow/schemas/models.py"),
        "pricing_source_url": settings.provider.pricing_source_url,
        "pricing_verified_on": args.pricing_verified_on.isoformat(),
        "input_cost_per_token_usd": settings.provider.input_cost_per_token_usd,
        "output_cost_per_token_usd": settings.provider.output_cost_per_token_usd,
        "requested_model": provider.model,
        "expected_actual_model": provider.expected_actual_model,
        "actual_models": actual_models,
        "model_drift_count": model_drift_count,
        "provider_error_count": provider_error_count,
        "sample_id": case["sample_id"],
        "pair_id": case["pair_id"],
        "family_id": case["family_id"],
        "course_split": case["course_split"],
        "source_split": case["source_split"],
        "source_revision": case["source_revision"],
        "source_license": case["source_license"],
        "original_image_sha256": case["original_image_sha256"],
        "record_count": record_count,
        "target_count": 5,
        "invalid_output_count": invalid_output_count,
        "target_variant_ids": [variant_id for variant_id, _, _ in images],
        "completed_variant_ids": [
            json.loads(line)["variant_id"]
            for line in responses_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if responses_path.is_file()
        else [],
        "budget": budget_summary,
        "input_changed_during_run": input_changed,
        "artifact_sha256": artifact_hashes,
    }
    if run_error is not None:
        summary["error_type"] = type(run_error).__name__
        summary["error_message"] = str(run_error)
    elif input_changed:
        summary["error_type"] = "InputChangedDuringRun"
        summary["error_message"] = "실행 중 지시문 또는 입력 파일이 바뀌었습니다"
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if run_error:
        print(f"이미지 견고성 실행이 중단됐습니다: 완료={record_count}/5")
        return 2
    if input_changed:
        print("실행 중 입력이 바뀌어 결과를 partial로 저장했습니다")
        return 2
    print("원본 1개와 변형 4개의 VLM 응답을 저장했습니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
