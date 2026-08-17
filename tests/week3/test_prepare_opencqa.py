import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.prepare_opencqa import SELECTION, prepare


def _clean_git_result(args, revision: str) -> subprocess.CompletedProcess:
    output = revision if args[1:3] == ["rev-parse", "HEAD"] else ""
    return subprocess.CompletedProcess(args, 0, output, "")


def test_selection_has_30_unique_pairs() -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    splits = selection["course_splits"]
    assert set(splits) == {"development", "validation", "test"}
    assert {name: len(ids) for name, ids in splits.items()} == {
        "development": 18,
        "validation": 6,
        "test": 6,
    }
    assert splits["development"][0] == "884"
    sample_ids = [sample_id for ids in splits.values() for sample_id in ids]
    assert len(sample_ids) == len(set(sample_ids)) == 30
    assert not {"7446", "4435", "5070", "7151", "2746", "1876"} & set(sample_ids)
    assert {"43", "101", "106", "131", "171", "183"} <= set(sample_ids)


def _source(tmp_path: Path, selection: dict, first_answer: str = "abstractive answer"):
    source = tmp_path / "OpenCQA"
    source.mkdir()
    (source / "LICENSE").write_text(
        "GNU GENERAL PUBLIC LICENSE\nVersion 3\n",
        encoding="utf-8",
    )
    annotation = source / "etc/data(full_summary_article)"
    images = source / "chart_images"
    annotation.mkdir(parents=True)
    images.mkdir()
    sample_ids = [
        sample_id
        for split in ("development", "validation", "test")
        for sample_id in selection["course_splits"][split]
    ]
    rows = {}
    for index, sample_id in enumerate(sample_ids):
        image_name = f"{sample_id}.png"
        image = (
            Image.effect_noise((1600, 900), 100).convert("RGB")
            if index == 0
            else Image.new("RGB", (100, 100), "white")
        )
        image.save(images / image_name, format="PNG")
        image.close()
        rows[sample_id] = [
            image_name,
            "title must not be copied",
            "article must not be copied",
            "summary must not be copied",
            "question",
            first_answer if index == 0 else "abstractive answer",
            "extractive answer",
        ]
    annotation_path = annotation / "val_extended.json"
    annotation_path.write_text(json.dumps(rows), encoding="utf-8")
    return source, annotation_path, sample_ids


def test_prepare_uses_images_and_questions_only(tmp_path: Path, monkeypatch) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    source, annotation_path, sample_ids = _source(tmp_path, selection)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _clean_git_result(args, selection["revision"]),
    )

    output = tmp_path / "output"
    stale_image = output / "images/stale.jpg"
    stale_image.parent.mkdir(parents=True)
    Image.new("RGB", (10, 10), "black").save(stale_image, format="JPEG")
    cases = prepare(source, output)

    assert cases[0]["question"] == "question"
    assert cases[0]["reference_answer"] == "abstractive answer"
    assert cases[0]["sample_id"] == "884"
    assert cases[0]["course_split"] == "development"
    assert cases[0]["source_split"] == "val"
    assert cases[0]["source_revision"] == selection["revision"]
    assert cases[0]["source_license"] == "GPL-3.0"
    assert "article" not in cases[0]
    assert "summary" not in cases[0]
    assert "extractive_answer" not in cases[0]
    assert "candidate_a" not in cases[0]
    assert "candidate_b" not in cases[0]
    prepared_image = output / "images/884.jpg"
    assert cases[0]["image_path"] == "local-data/opencqa/images/884.jpg"
    assert cases[0]["image_sha256"] == hashlib.sha256(prepared_image.read_bytes()).hexdigest()
    assert prepared_image.stat().st_size <= 175_000
    with Image.open(prepared_image) as image:
        assert image.format == "JPEG"
        assert image.width <= 1024
    cases_path = output / "week-03-cases.jsonl"
    original_cases = cases_path.read_text(encoding="utf-8")
    assert len(original_cases.splitlines()) == 30
    assert not (output / "week-03-pairs.jsonl").exists()
    assert not (output / "week-03-review-pairs.jsonl").exists()
    assert not (output / "week-03-review-sheet.md").exists()
    assert "title must not be copied" not in original_cases
    assert "article must not be copied" not in original_cases
    assert "summary must not be copied" not in original_cases
    assert "extractive answer" not in original_cases
    source_rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    source_rows[sample_ids[0]][5] = "changed answer"
    annotation_path.write_text(json.dumps(source_rows), encoding="utf-8")
    prepare(source, output)
    assert cases_path.read_text(encoding="utf-8") != original_cases


def test_prepare_does_not_copy_or_validate_extractives(tmp_path: Path, monkeypatch) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    source, annotation_path, sample_ids = _source(tmp_path, selection)
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows[sample_ids[0]][6] = "leftover <span> from an unused extractive"
    annotation_path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _clean_git_result(args, selection["revision"]),
    )

    output = tmp_path / "output"
    prepare(source, output)

    payload = (output / "week-03-cases.jsonl").read_text(encoding="utf-8")
    assert "extractive" not in payload


@pytest.mark.parametrize(
    "bad_answer",
    [
        "chrome-extension injected",
        "class=mutihighlight",
        "leftover <span>",
        "broken & amp ; entity",
        'unbalanced " quote',
    ],
)
def test_prepare_fails_before_writing_on_source_markers(
    tmp_path: Path,
    monkeypatch,
    bad_answer: str,
) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    source, _annotation_path, _sample_ids = _source(tmp_path, selection, bad_answer)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _clean_git_result(args, selection["revision"]),
    )
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="marker|따옴표"):
        prepare(source, output)

    assert not output.exists()


def test_prepare_rejects_modified_source_clone_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    source, _annotation_path, _sample_ids = _source(tmp_path, selection)

    def git_result(args, **kwargs):
        del kwargs
        output = selection["revision"] if args[1] == "rev-parse" else " M chart_images/884.png\n"
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(subprocess, "run", git_result)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="commit되지 않은 변경"):
        prepare(source, output)

    assert not output.exists()


def test_prepare_requires_source_license_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    source, _annotation_path, _sample_ids = _source(tmp_path, selection)
    (source / "LICENSE").unlink()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _clean_git_result(args, selection["revision"]),
    )
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="LICENSE"):
        prepare(source, output)

    assert not output.exists()


def test_prepare_rejects_wrong_source_license_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = yaml.safe_load(SELECTION.read_text(encoding="utf-8"))
    source, _annotation_path, _sample_ids = _source(tmp_path, selection)
    (source / "LICENSE").write_text("another license\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: _clean_git_result(args, selection["revision"]),
    )
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="GPL-3.0"):
        prepare(source, output)

    assert not output.exists()
