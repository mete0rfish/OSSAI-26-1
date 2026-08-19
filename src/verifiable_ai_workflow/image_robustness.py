"""OpenCQA 차트 변형과 사람이 확인한 근거 보존 여부를 평가한다."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from deepeval.dataset import Golden
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from .prompt_optimization import OpenCqaDeterministicMetric, score_output
from .schemas import StructuredAnswer

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?%?")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VariantArtifact(StrictModel):
    sample_id: str
    variant_id: str
    intended_behavior: Literal["invariance", "graceful_degradation"]
    image_path: str
    source_sha256: str
    image_sha256: str


class VariantScore(StrictModel):
    variant_id: str
    grounding_status: Literal["preserved", "destroyed"]
    status: Literal["passed", "failed", "inconclusive", "invalid_variant"]
    reason: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transform(image: Image.Image, name: str) -> Image.Image:
    if name == "rotate_2":
        return image.rotate(2, expand=True, fillcolor="white")
    if name == "jpeg_60":
        return image.copy()
    if name == "crop_left":
        return image.crop((round(image.width * 0.4), 0, image.width, image.height))
    if name == "occlude_answer":
        result = image.copy()
        draw = ImageDraw.Draw(result)
        draw.rectangle(
            (
                0,
                round(image.height * 0.25),
                round(image.width * 0.2),
                round(image.height * 0.8),
            ),
            fill="#777777",
        )
        return result
    raise ValueError(f"지원하지 않는 이미지 변형: {name}")


def generate_variants(
    *,
    source_path: str | Path,
    sample_id: str,
    output_dir: str | Path,
    config_path: str | Path,
    project_root: str | Path,
    expected_source_sha256: str | None = None,
) -> list[VariantArtifact]:
    project_root = Path(project_root).resolve()
    source_path, output_dir = Path(source_path).resolve(), Path(output_dir).resolve()
    if not source_path.is_relative_to(project_root) or not output_dir.is_relative_to(
        project_root
    ):
        raise ValueError("원본과 변형 출력 경로는 project root 안에 있어야 합니다")
    source_hash = _sha256(source_path)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256:
        raise ValueError("원본 이미지 bytes가 OpenCQA pair와 다릅니다")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["robustness"]
    artifacts: list[VariantArtifact] = []
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        for spec in specs:
            suffix = ".jpg" if spec["transformation"] == "jpeg_60" else ".png"
            target = output_dir / f"{spec['variant_id']}{suffix}"
            transformed = _transform(image, spec["transformation"])
            if suffix == ".jpg":
                transformed.save(target, format="JPEG", quality=60)
            else:
                transformed.save(target, format="PNG")
            artifacts.append(
                VariantArtifact(
                    sample_id=sample_id,
                    variant_id=spec["variant_id"],
                    intended_behavior=spec["intended_behavior"],
                    image_path=str(target.resolve().relative_to(project_root)),
                    source_sha256=source_hash,
                    image_sha256=_sha256(target),
                )
            )
    return artifacts


def load_reviews(
    path: str | Path,
    artifacts: list[VariantArtifact] | None = None,
    *,
    project_root: str | Path | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Literal["preserved", "destroyed"]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed = {"preserved", "destroyed"}
    if not rows or any(row["grounding_status"] not in allowed for row in rows):
        raise ValueError("모든 이미지의 grounding_status를 preserved 또는 destroyed로 입력하세요")
    reviews = {row["variant_id"]: row["grounding_status"] for row in rows}
    if len(reviews) != len(rows):
        raise ValueError("variant_id가 중복되었습니다")
    if artifacts is not None:
        if project_root is None or source_path is None:
            raise ValueError("이미지 bytes 확인에는 project_root와 source_path가 필요합니다")
        if len({item.variant_id for item in artifacts}) != len(artifacts):
            raise ValueError("변형 manifest의 variant_id가 중복되었습니다")
        if len({item.sample_id for item in artifacts}) != 1:
            raise ValueError("변형 manifest의 sample_id가 일치하지 않습니다")
        expected = {item.variant_id: item for item in artifacts}
        if set(reviews) != set(expected) or any(
            row.get("sample_id") != expected[row["variant_id"]].sample_id
            or row.get("image_sha256") != expected[row["variant_id"]].image_sha256
            or row.get("intended_behavior")
            != expected[row["variant_id"]].intended_behavior
            for row in rows
            if row["variant_id"] in expected
        ):
            raise ValueError("사람 검토표가 현재 sample과 이미지 변형에 해당하지 않습니다")
        root = Path(project_root).resolve()
        source = Path(source_path).resolve()
        if not source.is_relative_to(root):
            raise ValueError("원본 이미지 경로는 project root 안에 있어야 합니다")
        source_hash = _sha256(source)
        if any(item.source_sha256 != source_hash for item in artifacts):
            raise ValueError("원본 이미지 bytes가 변형 manifest와 다릅니다")
        for item in artifacts:
            relative = Path(item.image_path)
            if relative.is_absolute() or not (root / relative).resolve().is_relative_to(root):
                raise ValueError("변형 이미지 경로는 project-relative여야 합니다")
            if _sha256(root / relative) != item.image_sha256:
                raise ValueError(f"변형 이미지 bytes가 manifest와 다릅니다: {item.variant_id}")
    return reviews


def score_variant(
    artifact: VariantArtifact,
    grounding_status: Literal["preserved", "destroyed"],
    reference_answer: str,
    original: StructuredAnswer | None,
    variant: StructuredAnswer | None,
) -> VariantScore:
    intended_status = "preserved" if artifact.intended_behavior == "invariance" else "destroyed"
    if grounding_status != intended_status:
        return VariantScore(
            variant_id=artifact.variant_id,
            grounding_status=grounding_status,
            status="invalid_variant",
            reason="의도한 변형과 사람이 확인한 근거 상태가 달라 평가에서 제외",
        )
    if original is None or variant is None:
        failed_output = "원본" if original is None else "변형"
        return VariantScore(
            variant_id=artifact.variant_id,
            grounding_status=grounding_status,
            status="failed",
            reason=f"{failed_output} 모델 출력 형식이 유효하지 않음",
        )
    golden = Golden(name=artifact.sample_id, input="차트 질문", expected_output=reference_answer)
    metric = OpenCqaDeterministicMetric()
    original_result = score_output(metric, golden, original.model_dump_json())
    variant_result = score_output(metric, golden, variant.model_dump_json())
    original_ready = (
        not original.abstained
        and bool(original.evidence)
        and original_result["score"] >= metric.threshold
    )
    if grounding_status == "preserved":
        original_numbers = set(_NUMBER.findall(original.answer))
        variant_numbers = set(_NUMBER.findall(variant.answer))
        if not original_ready:
            return VariantScore(
                variant_id=artifact.variant_id,
                grounding_status=grounding_status,
                status="inconclusive",
                reason=(
                    f"원본 점수={original_result['score']:.3f}로 invariance 판정 불가, "
                    f"원본 숫자={sorted(original_numbers)}, "
                    f"변형 숫자={sorted(variant_numbers)}"
                ),
            )
        variant_ready = (
            not variant.abstained
            and bool(variant.evidence)
            and variant_result["score"] >= metric.threshold
        )
        passed = original_ready and variant_ready and original_numbers == variant_numbers
        reason = (
            f"원본 점수={original_result['score']:.3f}, 변형 점수={variant_result['score']:.3f}, "
            f"원본 숫자={sorted(original_numbers)}, 변형 숫자={sorted(variant_numbers)}"
        )
    else:
        passed = (
            variant.abstained
            and not variant.evidence
            and bool(variant.abstention_reason)
        )
        reason = (
            f"원본 점수={original_result['score']:.3f}, "
            f"abstained={variant.abstained}, "
            f"evidence={len(variant.evidence)}, "
            f"reason={bool(variant.abstention_reason)}"
        )
    return VariantScore(
        variant_id=artifact.variant_id,
        grounding_status=grounding_status,
        status="passed" if passed else "failed",
        reason=reason,
    )


def score_original(
    reference_answer: str, answer: StructuredAnswer | None
) -> VariantScore:
    if answer is None:
        return VariantScore(
            variant_id="original",
            grounding_status="preserved",
            status="failed",
            reason="원본 모델 출력 형식이 유효하지 않음",
        )
    golden = Golden(name="original", input="차트 질문", expected_output=reference_answer)
    metric = OpenCqaDeterministicMetric()
    result = score_output(metric, golden, answer.model_dump_json())
    passed = not answer.abstained and bool(answer.evidence) and result["score"] >= metric.threshold
    return VariantScore(
        variant_id="original",
        grounding_status="preserved",
        status="passed" if passed else "failed",
        reason=f"원본 점수={result['score']:.3f}, evidence={len(answer.evidence)}",
    )


def load_response_map(path: str | Path) -> dict[str, StructuredAnswer | None]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != len({row["variant_id"] for row in rows}):
        raise ValueError("응답의 variant_id가 중복되었습니다")
    responses: dict[str, StructuredAnswer | None] = {}
    for row in rows:
        if row.get("output") is None:
            if not row.get("parse_error"):
                raise ValueError("output이 없는 응답에는 parse_error가 필요합니다")
            responses[row["variant_id"]] = None
        else:
            responses[row["variant_id"]] = StructuredAnswer.model_validate(
                row["output"], strict=True
            )
    return responses
