from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_week02_case_walkthrough_explains_new_success(
    project_root: Path,
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    script = project_root / "scripts/inspect_prompt_comparison_case.py"
    spec = importlib.util.spec_from_file_location(
        "inspect_prompt_comparison_case_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prepared_root = tmp_path / "prepared"
    monkeypatch.setattr(module, "PREPARED_ROOT", prepared_root)
    assert module.main() == 2
    assert "scripts/prepare_documents.py" in capsys.readouterr().err

    document_root = prepared_root / "MI2_240819_TY1_0012"
    (document_root / "model-pages").mkdir(parents=True)
    (document_root / "model-pages/page-0001.jpg").write_bytes(b"jpeg-bytes")
    (document_root / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": "fixture-source-sha256",
                "total_pages": 1,
                "pages": [
                    {
                        "page_number": 1,
                        "model_image_path": "model-pages/page-0001.jpg",
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
        "experiment_type",
        "case_id",
        "evidence_boundary",
        "input",
        "model_output",
        "expected",
        "evaluation_design",
        "evaluation_result",
    ]
    assert payload["week"] == 2
    assert payload["experiment_type"] == "prompt_only_case_replay"
    assert payload["case_id"] == "aihub-report-r01"
    prepared = payload["input"]["prepared_document"]
    assert prepared["source_sha256"] == "fixture-source-sha256"
    assert prepared["page_images_for_live_request"][0]["bytes"] == 10
    assert prepared["page_images_for_live_request"][0]["sha256"]
    assert prepared["expected_page_reference"]["page_number"] == 1
    assert "text" not in json.dumps(prepared)
    assert payload["model_output"]["baseline"]["fixture"] == (
        "data/recorded/week-02-gemma-baseline-responses.jsonl"
    )
    assert payload["model_output"]["candidate"]["fixture"] == (
        "data/scenarios/week-02-route-b-overrides.jsonl"
    )
    assert payload["model_output"]["baseline"]["parsed_answer"]["answer"] == (
        "2016년 말 기준 은행 가계대출 중 변동금리 비중은 71.6%입니다."
    )
    assert payload["model_output"]["candidate"]["parsed_answer"]["answer"] == "71.6%"
    assert payload["model_output"]["candidate"]["call_metadata"]["actual_model"] == (
        "google/gemma-4-31b-it"
    )
    assert payload["model_output"]["baseline"]["prompt_sha256"] != (
        payload["model_output"]["candidate"]["prompt_sha256"]
    )
    assert payload["expected"]["answer"] == "71.6%"
    assert payload["evaluation_result"]["baseline"]["scores"]["task_success"] == 0.0
    assert payload["evaluation_result"]["candidate"]["scores"]["task_success"] == 1.0
    assert payload["evaluation_result"]["case_comparison"]["classification"] == "new_success"
    assert payload["evaluation_result"]["all_cases_context"]["automated_status"] == "inconclusive"
    assert payload["evidence_boundary"]["evidence_kind"] == "test_only"
    assert payload["evidence_boundary"]["live_quality_claim"] is False
