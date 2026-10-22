"""Unavailable CARLA clients must produce failures rather than live-test passes."""

from __future__ import annotations

import pytest

from myosuite_carla import validation


def test_missing_client_retains_failed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_client(name: str) -> None:
        raise ModuleNotFoundError("Matching UE5.8 client is unavailable")

    monkeypatch.setattr(validation.importlib, "import_module", unavailable_client)
    with pytest.raises(validation.LiveValidationError) as failure:
        validation.run_live_validation(build_ref="ue58-dev")
    report = failure.value.report
    assert report["status"] == "failed"
    assert report["scenarios"] == []
    assert report["error"]["type"] == "ModuleNotFoundError"
    assert report["provenance"]["source_verified"] is False
    assert report["finished_utc"] >= report["started_utc"]
