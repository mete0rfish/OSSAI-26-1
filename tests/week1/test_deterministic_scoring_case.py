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
    script = project_root / "scripts/inspect_deterministic_scoring_case.py"
    spec = importlib.util.spec_from_file_location(
        "inspect_deterministic_scoring_case_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepared_root = tmp_path / "prepared"
    monkeypatch.setattr(module, "PREPARED_ROOT", prepared_root)
    assert module.main() == 0
    unprepared = json.loads(capsys.readouterr().out)
    assert unprepared["input"]["prepared_input_status"] == "not_prepared"
    assert unprepared["input"]["page_image_count"] is None
    assert unprepared["scoring"]["task_success"] == 0.0

    document_root = prepared_root / "MI2_240819_TY1_0012"
    (document_root / "model-pages").mkdir(parents=True)
    (document_root / "model-pages/page-0001.jpg").write_bytes(b"jpeg-bytes")
    (document_root / "manifest.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 1,
                        "model_image_path": "model-pages/page-0001.jpg",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert list(payload) == [
        "sample_id",
        "input",
        "model_output",
        "expected",
        "scoring",
        "evidence_kind",
    ]
    assert payload["sample_id"] == "aihub-report-r04"
    assert payload["input"]["question"] == (
        "2017년 상반기 민간소비 증가율은 얼마인가요?"
    )
    assert payload["input"]["prepared_input_status"] == "prepared"
    assert payload["input"]["page_image_count"] == 1
    assert payload["input"]["expected_page_image"] == (
        "local-data/aihub/prepared/MI2_240819_TY1_0012/model-pages/page-0001.jpg"
    )
    assert isinstance(payload["model_output"]["raw_response"], str)
    assert payload["model_output"]["parsed_answer"]["answer"] == "2.7"
    assert payload["expected"] == {
        "answer": "2.0%",
        "accepted_answers": [],
        "pages": [1],
        "abstained": False,
    }
    assert payload["scoring"]["task_success"] == 0.0
    assert payload["scoring"]["failed_requirements"]
    assert payload["evidence_kind"] == "test_only"
