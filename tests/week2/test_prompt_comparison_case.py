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
        "changed",
        "same_conditions",
        "input",
        "expected",
        "baseline",
        "candidate",
        "classification",
        "evidence_kind",
    ]
    assert payload["sample_id"] == "aihub-report-r01"
    assert payload["changed"] == "prompt_only"
    assert payload["input"]["page_image_count"] == 1
    assert payload["baseline"]["parsed_answer"]["answer"] == (
        "2016년 말 기준 은행 가계대출 중 변동금리 비중은 71.6%입니다."
    )
    assert payload["candidate"]["parsed_answer"]["answer"] == "71.6%"
    assert payload["expected"]["answer"] == "71.6%"
    assert payload["baseline"]["task_success"] == 0.0
    assert payload["candidate"]["task_success"] == 1.0
    assert payload["classification"] == "new_success"
    assert payload["evidence_kind"] == "test_only"
