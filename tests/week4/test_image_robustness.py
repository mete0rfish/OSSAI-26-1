import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import evaluate_image_robustness, generate_image_variants, run_image_robustness
from verifiable_ai_workflow.image_robustness import (
    VariantArtifact,
    generate_variants,
    load_response_map,
    load_reviews,
    score_original,
    score_variant,
)
from verifiable_ai_workflow.open_cqa_candidates import OpenCQACase
from verifiable_ai_workflow.schemas import Evidence, StructuredAnswer


def _answer(value: str = "47%", *, abstained: bool = False) -> StructuredAnswer:
    return StructuredAnswer(
        answer="답변 보류" if abstained else value,
        evidence=(
            [] if abstained else [Evidence(evidence_id="chart#page=1", quote=value, page_number=1)]
        ),
        confidence=0,
        abstained=abstained,
        abstention_reason="근거가 가려짐" if abstained else None,
    )


def test_generate_four_variants_and_score_by_human_review(
    tmp_path: Path,
    project_root: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(source)
    artifacts = generate_variants(
        source_path=source,
        sample_id="19",
        output_dir=tmp_path / "variants",
        config_path=project_root / "configs/week-04.yaml",
        project_root=tmp_path,
    )

    assert len(artifacts) == 4
    assert len({item.image_sha256 for item in artifacts}) == 4
    assert all(not Path(item.image_path).is_absolute() for item in artifacts)
    preserved = next(item for item in artifacts if item.intended_behavior == "invariance")
    destroyed = next(item for item in artifacts if item.intended_behavior == "graceful_degradation")
    assert score_variant(preserved, "preserved", "47%", _answer(), _answer()).status == "passed"
    destroyed_result = score_variant(
        destroyed,
        "destroyed",
        "47%",
        _answer(),
        _answer(abstained=True),
    )
    assert destroyed_result.status == "passed"
    assert (
        score_variant(destroyed, "preserved", "47%", _answer(), _answer()).status
        == "invalid_variant"
    )


def test_student_alias_cannot_escape_week_04_output_folder() -> None:
    for parser in (
        generate_image_variants._student_alias,
        evaluate_image_robustness._student_alias,
    ):
        with pytest.raises(argparse.ArgumentTypeError, match="별칭"):
            parser("../../other")


def test_student_variants_must_match_images_used_for_saved_responses() -> None:
    artifact = VariantArtifact(
        sample_id="884",
        variant_id="rotate-2",
        intended_behavior="invariance",
        image_path="variant.png",
        source_sha256="a" * 64,
        image_sha256="b" * 64,
    )
    assert evaluate_image_robustness._same_variant_inputs(
        [artifact], {"sample_id": "884"}, [artifact], {"sample_id": "884"}
    )
    assert not evaluate_image_robustness._same_variant_inputs(
        [artifact.model_copy(update={"image_sha256": "c" * 64})],
        {"sample_id": "884"},
        [artifact],
        {"sample_id": "884"},
    )
    for update in (
        {"sample_id": "885"},
        {"intended_behavior": "graceful_degradation"},
        {"source_sha256": "c" * 64},
    ):
        assert not evaluate_image_robustness._same_variant_inputs(
            [artifact.model_copy(update=update)],
            {"sample_id": "884"},
            [artifact],
            {"sample_id": "884"},
        )


def test_image_message_rejects_changed_bytes(tmp_path: Path) -> None:
    image = _image(tmp_path / "image.png")

    with pytest.raises(ValueError, match="SHA-256"):
        run_image_robustness._image_message(image, "question", "0" * 64)


def test_degradation_variants_remove_the_left_answer_region(
    tmp_path: Path,
    project_root: Path,
) -> None:
    source = tmp_path / "source.png"
    image = Image.new("RGB", (200, 100), "blue")
    image.paste("red", (0, 0, 80, 100))
    image.save(source)
    artifacts = generate_variants(
        source_path=source,
        sample_id="884",
        output_dir=tmp_path / "variants",
        config_path=project_root / "configs/week-04.yaml",
        project_root=tmp_path,
    )
    by_id = {item.variant_id: tmp_path / item.image_path for item in artifacts}

    with Image.open(by_id["crop-left"]) as cropped:
        assert cropped.size == (120, 100)
        assert cropped.getpixel((0, 50)) == (0, 0, 255)
    with Image.open(by_id["occlude-answer"]) as occluded:
        assert occluded.getpixel((10, 50)) == (119, 119, 119)
        assert occluded.getpixel((190, 50)) == (0, 0, 255)


def test_preserved_variant_must_match_reference_not_only_numbers(tmp_path: Path) -> None:
    artifact = generate_variants(
        source_path=_image(tmp_path / "source.png"),
        sample_id="19",
        output_dir=tmp_path / "variants",
        config_path=Path(__file__).parents[2] / "configs/week-04.yaml",
        project_root=tmp_path,
    )[0]
    reference = "Alpha revenue climbed sharply from 31% toward 47%."
    original = _answer(reference)
    wrong_variant = _answer("Beta costs collapsed unexpectedly between 31% and 47%.")

    result = score_variant(artifact, "preserved", reference, original, wrong_variant)

    assert result.status == "failed"
    assert score_original(reference, wrong_variant).status == "failed"


def test_low_quality_original_does_not_hide_safe_abstention(tmp_path: Path) -> None:
    artifacts = generate_variants(
        source_path=_image(tmp_path / "source.png"),
        sample_id="19",
        output_dir=tmp_path / "variants",
        config_path=Path(__file__).parents[2] / "configs/week-04.yaml",
        project_root=tmp_path,
    )
    preserved = next(item for item in artifacts if item.intended_behavior == "invariance")
    destroyed = next(item for item in artifacts if item.intended_behavior == "graceful_degradation")
    low_quality = _answer("10%")

    assert (
        score_variant(preserved, "preserved", "47%", low_quality, low_quality).status
        == "inconclusive"
    )
    assert (
        score_variant(
            destroyed,
            "destroyed",
            "47%",
            low_quality,
            _answer(abstained=True),
        ).status
        == "passed"
    )


def test_review_must_be_completed(tmp_path: Path) -> None:
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerow(["19", "crop", "abc", "graceful_degradation", ""])

    try:
        load_reviews(path)
    except ValueError as exc:
        assert "grounding_status" in str(exc)
    else:
        raise AssertionError("빈 사람 검토가 통과하면 안 됩니다")


def test_review_must_match_current_variant(tmp_path: Path) -> None:
    source = _image(tmp_path / "source.png")
    artifact = generate_variants(
        source_path=source,
        sample_id="19",
        output_dir=tmp_path / "variants",
        config_path=Path(__file__).parents[2] / "configs/week-04.yaml",
        project_root=tmp_path,
    )[0]
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerow(["19", artifact.variant_id, "stale-hash", "invariance", "preserved"])

    with pytest.raises(ValueError, match="현재 sample과 이미지"):
        load_reviews(
            path,
            [artifact],
            project_root=tmp_path,
            source_path=source,
        )


def test_moved_manifest_rechecks_original_and_variant_bytes(
    tmp_path: Path,
    project_root: Path,
) -> None:
    first = tmp_path / "first"
    source = _image(first / "source.png")
    artifacts = generate_variants(
        source_path=source,
        sample_id="19",
        output_dir=first / "variants",
        config_path=project_root / "configs/week-04.yaml",
        project_root=first,
    )
    (first / "variants.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in artifacts),
        encoding="utf-8",
    )
    _write_reviews(first / "review.csv", artifacts)

    moved = tmp_path / "moved"
    shutil.copytree(first, moved)
    moved_artifacts = [
        VariantArtifact.model_validate_json(line)
        for line in (moved / "variants.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    load_reviews(
        moved / "review.csv",
        moved_artifacts,
        project_root=moved,
        source_path=moved / "source.png",
    )

    original = moved / "source.png"
    original_bytes = original.read_bytes()
    original.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="원본 이미지 bytes"):
        load_reviews(
            moved / "review.csv",
            moved_artifacts,
            project_root=moved,
            source_path=original,
        )
    original.write_bytes(original_bytes)

    (moved / moved_artifacts[0].image_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="변형 이미지 bytes"):
        load_reviews(
            moved / "review.csv",
            moved_artifacts,
            project_root=moved,
            source_path=moved / "source.png",
        )


def test_manifest_rejects_duplicate_variant_and_mixed_sample(
    tmp_path: Path,
    project_root: Path,
) -> None:
    source = _image(tmp_path / "source.png")
    artifacts = generate_variants(
        source_path=source,
        sample_id="19",
        output_dir=tmp_path / "variants",
        config_path=project_root / "configs/week-04.yaml",
        project_root=tmp_path,
    )
    review = _write_reviews(tmp_path / "review.csv", artifacts)

    with pytest.raises(ValueError, match="variant_id가 중복"):
        load_reviews(
            review,
            [artifacts[0], artifacts[0]],
            project_root=tmp_path,
            source_path=source,
        )
    with pytest.raises(ValueError, match="sample_id가 일치"):
        load_reviews(
            review,
            [artifacts[0], artifacts[1].model_copy(update={"sample_id": "20"})],
            project_root=tmp_path,
            source_path=source,
        )


def test_source_must_match_pair_hash_and_stay_inside_project(
    tmp_path: Path,
    project_root: Path,
) -> None:
    source = _image(tmp_path / "root/source.png")
    with pytest.raises(ValueError, match="OpenCQA pair"):
        generate_variants(
            source_path=source,
            sample_id="19",
            output_dir=tmp_path / "root/variants",
            config_path=project_root / "configs/week-04.yaml",
            project_root=tmp_path / "root",
            expected_source_sha256="0" * 64,
        )
    assert not (tmp_path / "root/variants").exists()

    artifacts = generate_variants(
        source_path=source,
        sample_id="19",
        output_dir=tmp_path / "root/variants",
        config_path=project_root / "configs/week-04.yaml",
        project_root=tmp_path / "root",
    )
    review = _write_reviews(tmp_path / "root/review.csv", artifacts)
    outside = _image(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="project root"):
        load_reviews(
            review,
            artifacts,
            project_root=tmp_path / "root",
            source_path=outside,
        )


def test_response_ids_must_be_unique(tmp_path: Path) -> None:
    row = {"variant_id": "rotate-2", "output": _answer().model_dump(mode="json")}
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="variant_id가 중복"):
        load_response_map(path)


def test_robustness_case_must_match_current_pair(monkeypatch, tmp_path) -> None:
    case_record = OpenCQACase(
        pair_id="opencqa-val-884",
        sample_id="884",
        family_id="opencqa-val-884",
        course_split="development",
        source_split="val",
        source_revision="a" * 40,
        source_license="GPL-3.0",
        image_sha256="b" * 64,
        image_path="local-data/opencqa/images/884.png",
        question="What changed?",
        reference_answer="47%",
    )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(case_record.model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(run_image_robustness, "CASES", cases_path)
    case = {
        "pair_id": case_record.pair_id,
        "sample_id": case_record.sample_id,
        "family_id": case_record.family_id,
        "course_split": case_record.course_split,
        "source_split": case_record.source_split,
        "source_revision": case_record.source_revision,
        "source_license": case_record.source_license,
        "question": case_record.question,
        "reference_answer": case_record.reference_answer,
        "original_image": case_record.image_path,
        "original_image_sha256": case_record.image_sha256,
    }

    run_image_robustness._require_current_pair(case)
    for field in ("question", "original_image_sha256"):
        with pytest.raises(SystemExit, match="pair identity"):
            run_image_robustness._require_current_pair({**case, field: "changed"})


def test_evaluation_manifest_binds_inputs_and_scores(monkeypatch, tmp_path) -> None:
    source = _image(tmp_path / "source.png")
    artifacts = generate_variants(
        source_path=source,
        sample_id="884",
        output_dir=tmp_path / "variants",
        config_path=Path(__file__).parents[2] / "configs/week-04.yaml",
        project_root=tmp_path,
    )
    variants = tmp_path / "variants.jsonl"
    variants.write_text(
        "".join(item.model_dump_json() + "\n" for item in artifacts), encoding="utf-8"
    )
    reviews = _write_reviews(tmp_path / "reviews.csv", artifacts)
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "reference_answer": "47%",
                "original_image": str(source.relative_to(tmp_path)),
            }
        ),
        encoding="utf-8",
    )
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "".join(
            json.dumps(
                {
                    "variant_id": variant_id,
                    "output": _answer().model_dump(mode="json"),
                    "parse_error": None,
                }
            )
            + "\n"
            for variant_id in ["original", *(item.variant_id for item in artifacts)]
        ),
        encoding="utf-8",
    )
    scorer = tmp_path / "src/verifiable_ai_workflow/image_robustness.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# scorer\n", encoding="utf-8")
    metric = tmp_path / "src/verifiable_ai_workflow/prompt_optimization.py"
    metric.write_text("# metric\n", encoding="utf-8")
    schema = tmp_path / "src/verifiable_ai_workflow/schemas/models.py"
    schema.parent.mkdir(parents=True)
    schema.write_text("# schema\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "git_sha": "a" * 40,
                "artifact_sha256": {
                    "responses.jsonl": hashlib.sha256(responses.read_bytes()).hexdigest(),
                    "case.json": hashlib.sha256(case.read_bytes()).hexdigest(),
                    "variants.jsonl": hashlib.sha256(variants.read_bytes()).hexdigest(),
                    "variant-review.csv": hashlib.sha256(reviews.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(evaluate_image_robustness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_image_robustness.py",
            "--variants",
            str(variants),
            "--reviews",
            str(reviews),
            "--case",
            str(case),
            "--responses",
            str(responses),
            "--output",
            str(output),
        ],
    )

    assert evaluate_image_robustness.main() == 0
    manifest = json.loads((tmp_path / "evaluation-manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["responses_sha256"] == hashlib.sha256(responses.read_bytes()).hexdigest()
    assert manifest["source_git_sha"] == "a" * 40
    assert manifest["scorer_sha256"] == hashlib.sha256(scorer.read_bytes()).hexdigest()
    assert manifest["metric_sha256"] == hashlib.sha256(metric.read_bytes()).hexdigest()
    assert manifest["schema_sha256"] == hashlib.sha256(schema.read_bytes()).hexdigest()

    reviews.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256"):
        evaluate_image_robustness.main()


def test_student_evaluation_rejects_changed_saved_responses(monkeypatch, tmp_path) -> None:
    source = _image(tmp_path / "source.png")
    artifacts = generate_variants(
        source_path=source,
        sample_id="884",
        output_dir=tmp_path / "generated",
        config_path=Path(__file__).parents[2] / "configs/week-04.yaml",
        project_root=tmp_path,
    )
    canonical = tmp_path / "canonical"
    student = tmp_path / "local-data/week-04-students/minsu/variants"
    canonical.mkdir()
    student.mkdir(parents=True)
    variants_text = "".join(item.model_dump_json() + "\n" for item in artifacts)
    case_text = json.dumps(
        {"reference_answer": "47%", "original_image": str(source.relative_to(tmp_path))}
    )
    for root in (canonical, student):
        (root / "variants.jsonl").write_text(variants_text, encoding="utf-8")
        (root / "case.json").write_text(case_text, encoding="utf-8")
    _write_reviews(student / "variant-review.csv", artifacts)
    responses = tmp_path / "saved/responses.jsonl"
    responses.parent.mkdir()
    responses.write_text(
        "".join(
            json.dumps(
                {
                    "variant_id": variant_id,
                    "output": _answer().model_dump(mode="json"),
                    "parse_error": None,
                }
            )
            + "\n"
            for variant_id in ["original", *(item.variant_id for item in artifacts)]
        ),
        encoding="utf-8",
    )
    (responses.parent / "summary.json").write_text(
        json.dumps(
            {
                "git_sha": "a" * 40,
                "artifact_sha256": {
                    "responses.jsonl": hashlib.sha256(responses.read_bytes()).hexdigest(),
                    "case.json": hashlib.sha256((canonical / "case.json").read_bytes()).hexdigest(),
                    "variants.jsonl": hashlib.sha256(
                        (canonical / "variants.jsonl").read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    scorer = tmp_path / "src/verifiable_ai_workflow/image_robustness.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# scorer\n", encoding="utf-8")
    metric = tmp_path / "src/verifiable_ai_workflow/prompt_optimization.py"
    metric.write_text("# metric\n", encoding="utf-8")
    schema = tmp_path / "src/verifiable_ai_workflow/schemas/models.py"
    schema.parent.mkdir(parents=True)
    schema.write_text("# schema\n", encoding="utf-8")
    monkeypatch.setattr(evaluate_image_robustness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate_image_robustness, "DEFAULT_ROOT", canonical)
    monkeypatch.setattr(
        evaluate_image_robustness,
        "load_week4_class_materials",
        lambda project_root: SimpleNamespace(label="test", image_response_dir=responses.parent),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_image_robustness.py", "--student-alias", "minsu"],
    )

    assert evaluate_image_robustness.main() == 0
    responses.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="응답 파일"):
        evaluate_image_robustness.main()


def test_invalid_structured_response_is_a_failed_variant(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps(
            {
                "variant_id": "rotate-2",
                "raw_output": "not-json",
                "output": None,
                "parse_error": {"error_type": "ValidationError"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response = load_response_map(path)["rotate-2"]
    artifact = VariantArtifact(
        sample_id="19",
        variant_id="rotate-2",
        intended_behavior="invariance",
        image_path="rotate-2.png",
        source_sha256="a" * 64,
        image_sha256="b" * 64,
    )

    assert response is None
    assert score_variant(artifact, "preserved", "47%", _answer(), response).status == "failed"


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def _write_reviews(path: Path, artifacts: list[VariantArtifact]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
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
                    "preserved",
                ]
            )
    return path


@pytest.mark.parametrize(
    ("catalog_date", "pricing_date"),
    [
        ("2000-01-01", date.today().isoformat()),
        (date.today().isoformat(), "2000-01-01"),
    ],
)
def test_robustness_runner_rejects_stale_preflight_before_live_work(
    monkeypatch,
    tmp_path,
    catalog_date,
    pricing_date,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_image_robustness.py",
            "--live",
            "--max-requests",
            "5",
            "--max-input-tokens",
            "100000",
            "--max-output-tokens",
            "2500",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "900",
            "--catalog-verified-on",
            catalog_date,
            "--pricing-verified-on",
            pricing_date,
            "--output",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="7일 이내"):
        run_image_robustness.main()


def test_robustness_runner_rejects_larger_than_approved_caps(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_image_robustness.py",
            "--live",
            "--max-requests",
            "5",
            "--max-input-tokens",
            "100001",
            "--max-output-tokens",
            "2500",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "900",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--output",
            str(tmp_path / "oversized"),
        ],
    )

    with pytest.raises(SystemExit, match="승인 cap"):
        run_image_robustness.main()


@pytest.mark.parametrize(
    (
        "failure_on",
        "change_input",
        "change_image",
        "expected_status",
        "expected_quality",
        "expected_count",
        "expected_return",
    ),
    [
        (0, False, False, "not_run", "inconclusive", 0, 2),
        (2, False, False, "partial", "inconclusive", 1, 2),
        (1, False, False, "partial", "inconclusive", 0, 2),
        (None, True, False, "partial", "inconclusive", 5, 2),
        (None, False, True, "partial", "inconclusive", 5, 2),
        (None, False, False, "complete", "fail", 5, 0),
    ],
)
def test_robustness_runner_records_completion_state_and_input_hashes(
    monkeypatch,
    tmp_path: Path,
    failure_on: int | None,
    change_input: bool,
    change_image: bool,
    expected_status: str,
    expected_quality: str,
    expected_count: int,
    expected_return: int,
) -> None:
    variant_root = tmp_path / "local-data/week-04-students/minsu/variants"
    variant_root.mkdir(parents=True)
    prompt = tmp_path / "prompts/week-04-baseline.md"
    prompt.parent.mkdir()
    prompt.write_text("질문: {question}", encoding="utf-8")
    schema = tmp_path / "src/verifiable_ai_workflow/schemas/models.py"
    schema.parent.mkdir(parents=True)
    schema.write_text("# schema\n", encoding="utf-8")
    original = _image(tmp_path / "local-data/opencqa/images/original.png")
    artifacts = []
    for index in range(4):
        image = _image(variant_root / f"variant-{index}.png")
        artifacts.append(
            VariantArtifact(
                sample_id="884",
                variant_id=f"variant-{index}",
                intended_behavior="invariance",
                image_path=str(image.relative_to(tmp_path)),
                source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            )
        )
    (variant_root / "variants.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in artifacts),
        encoding="utf-8",
    )
    reviews = _write_reviews(variant_root / "variant-review.csv", artifacts)
    reviews_hash = hashlib.sha256(reviews.read_bytes()).hexdigest()
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    (variant_root / "case.json").write_text(
        json.dumps(
            {
                "pair_id": "opencqa-val-884",
                "sample_id": "884",
                "family_id": "opencqa-val-884",
                "course_split": "development",
                "source_split": "val",
                "source_revision": "a" * 40,
                "source_license": "GPL-3.0",
                "question": "What changed?",
                "reference_answer": "47%",
                "original_image": str(original.relative_to(tmp_path)),
                "original_image_sha256": original_hash,
            }
        ),
        encoding="utf-8",
    )

    class FailingProvider:
        model = "nvidia_nim/google/gemma-4-31b-it"
        expected_actual_model = "google/gemma-4-31b-it"
        structured_output = "json_schema"
        last_call = None

        def __init__(self, on_response) -> None:
            self.on_response = on_response
            self.calls = 0
            self.budget = SimpleNamespace(
                summary=lambda: {
                    "request_count": self.calls,
                    "attempt_count": self.calls,
                }
            )

        def generate(self, *args, **kwargs):
            del args, kwargs
            if failure_on == 0:
                self.last_call = {
                    "provider_status": "blocked",
                    "error_type": "LiveBudgetExceeded",
                }
                raise RuntimeError("provider blocked")
            self.calls += 1
            if change_input and self.calls == 1:
                reviews.write_text("changed\n", encoding="utf-8")
            if change_image and self.calls == 5:
                (variant_root / "variant-3.png").write_bytes(b"changed")
            if self.calls == failure_on:
                self.last_call = {
                    "provider_status": "provider_error",
                    "error_type": "APIConnectionError",
                }
                raise RuntimeError("provider unavailable")
            self.last_call = {
                "provider_status": "success",
                "actual_model": self.expected_actual_model,
                "error_type": None,
            }
            self.on_response(dict(self.last_call))
            return "not-json"

    provider_settings = SimpleNamespace(
        billing_basis="developer_program_free_endpoint",
        pricing_source_url="https://example.invalid",
        input_cost_per_token_usd=0.0,
        output_cost_per_token_usd=0.0,
    )
    monkeypatch.setattr(run_image_robustness, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_image_robustness, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(run_image_robustness, "_require_current_pair", lambda case: None)
    monkeypatch.setattr(run_image_robustness, "load_project_env", lambda *args: None)
    monkeypatch.setattr(
        run_image_robustness,
        "load_settings",
        lambda *args: SimpleNamespace(provider=provider_settings),
    )
    monkeypatch.setattr(
        run_image_robustness,
        "build_course_provider",
        lambda *args, **kwargs: FailingProvider(kwargs["on_response"]),
    )
    output = tmp_path / "reports/robustness"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_image_robustness.py",
            "--live",
            "--max-requests",
            "5",
            "--max-input-tokens",
            "100000",
            "--max-output-tokens",
            "2500",
            "--max-cost-usd",
            "0.01",
            "--max-wall-seconds",
            "900",
            "--catalog-verified-on",
            date.today().isoformat(),
            "--pricing-verified-on",
            date.today().isoformat(),
            "--variants-dir",
            str(variant_root),
            "--output",
            str(output),
        ],
    )

    assert run_image_robustness.main() == expected_return
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == expected_quality
    assert summary["observed_status"] == expected_status
    assert summary["record_count"] == expected_count
    assert summary["invalid_output_count"] == expected_count
    assert summary["input_changed_during_run"] is (change_input or change_image)
    assert summary["pricing_verified_on"] == date.today().isoformat()
    assert summary["schema_sha256"]
    assert summary["artifact_sha256"]["calls.jsonl"]
    assert bool(summary["artifact_sha256"]["responses.jsonl"]) is bool(expected_count)
    assert (
        summary["artifact_sha256"]["variants.jsonl"]
        == hashlib.sha256((variant_root / "variants.jsonl").read_bytes()).hexdigest()
    )
    assert summary["artifact_sha256"]["variant-review.csv"] == reviews_hash
    if expected_count == 1:
        response = json.loads((output / "responses.jsonl").read_text(encoding="utf-8"))
        assert response["output"] is None
        assert response["raw_output"] == "not-json"
    calls = [json.loads(line) for line in (output / "calls.jsonl").read_text().splitlines()]
    if failure_on is not None:
        assert calls[-1]["error_type"] == (
            "LiveBudgetExceeded" if failure_on == 0 else "APIConnectionError"
        )


def test_original_compares_reference_numbers() -> None:
    assert score_original("It rose from 10% to 20%.", _answer("10% and 20%")).status == "passed"
    assert score_original("It rose from 10% to 20%.", _answer("20%")).status == "failed"
    assert score_original("It rose from 10% to 20%.", None).status == "failed"
