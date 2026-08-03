from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_week01_case_walkthrough_shows_input_output_expected_and_score(
    project_root: Path,
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = project_root / "scripts/show_week01_case.py"
    spec = importlib.util.spec_from_file_location("show_week01_case_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepared_root = tmp_path / "prepared"
    monkeypatch.setattr(module, "PREPARED_ROOT", prepared_root)
    assert module.main() == 2
    assert "scripts/prepare_documents.py" in capsys.readouterr().err

    document_root = prepared_root / "MI2_240819_TY1_0012"
    (document_root / "model-pages").mkdir(parents=True)
    (document_root / "text").mkdir()
    (document_root / "model-pages/page-0001.jpg").write_bytes(b"jpeg-bytes")
    (document_root / "text/page-0001.txt").write_text(
        "은행 가계대출 중 변동금리 비중은 71.6%(2016년말 기준)",
        encoding="utf-8",
    )
    (document_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": "fixture-source-sha256",
                "total_pages": 1,
                "pages": [
                    {
                        "page_number": 1,
                        "model_image_path": "model-pages/page-0001.jpg",
                        "text_path": "text/page-0001.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert list(payload) == [
        "week",
        "case_id",
        "evidence_boundary",
        "input",
        "model_output",
        "expected",
        "evaluation_design",
        "evaluation_result",
    ]
    assert payload["week"] == 1
    assert payload["case_id"] == "aihub-report-r01"
    assert payload["input"]["question"] == (
        "2016년 말 은행 가계대출 중 변동금리 비중은 얼마인가요?"
    )
    prepared = payload["input"]["prepared_document"]
    assert prepared["source_sha256"] == "fixture-source-sha256"
    assert prepared["sent_page_images"] == [
        {
            "page_number": 1,
            "path": ("local-data/aihub/prepared/MI2_240819_TY1_0012/model-pages/page-0001.jpg"),
            "bytes": 10,
        }
    ]
    assert "71.6%" in prepared["expected_page_preview"]["text_excerpt"]
    assert isinstance(payload["model_output"]["raw_response"], str)
    assert payload["model_output"]["parsed_answer"]["answer"] == "71.6%"
    assert payload["expected"] == {
        "answer": "71.6%",
        "pages": [1],
        "abstained": False,
    }
    assert payload["evaluation_result"]["scores"]["task_success"] == 1.0
    assert payload["evaluation_result"]["status"] == "passed"
    assert payload["evidence_boundary"]["evidence_kind"] == "test_only"
    assert payload["evidence_boundary"]["live_quality_claim"] is False
