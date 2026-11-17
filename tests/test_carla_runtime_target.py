"""Release labels must agree with the runtime used for live validation."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from myosuite_carla import fullbody_validation, validation


@pytest.mark.parametrize(
    ("target", "runtime", "expected"),
    [
        ("1.0", "1.0", True),
        ("1.0", "1.0-82c52a7c2", True),
        ("1.0", "1.0-82c52a7c2-dirty", True),
        ("1.0", "1.0-dirty", True),
        ("v1.0", "1.0", True),
        (" 1.0 ", "0.10.0", False),
        ("0.10.0", "0.10.0", True),
        ("0.10.0", "0.10.0-dev", True),
        ("ue58-dev", "1.0-82c52a7c2", None),
        ("custom-build", "custom-version", None),
        ("1.0", "0.10.0", False),
        ("1.0", "1.0.1", False),
        ("1.0", "1.01", False),
        ("1.0", "1.0invalid", False),
        ("0.10.0", "1.0", False),
    ],
)
def test_release_target_matches_complete_numeric_version(
    target: str, runtime: str, expected: bool | None
) -> None:
    report: dict[str, Any] = {"server_version": runtime, "provenance": {}}
    if expected is False:
        with pytest.raises(RuntimeError, match="does not match release target"):
            validation._check_runtime_target(report, target)
    else:
        validation._check_runtime_target(report, target)
    assert report["provenance"]["runtime_matches_target"] is expected


@pytest.mark.parametrize(
    "run", [validation.run_live_validation, fullbody_validation.run_fullbody_validation]
)
def test_matched_old_runtime_fails_before_world_access(
    run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_world() -> None:
        raise AssertionError("Version rejection must precede world access")

    client = SimpleNamespace(
        set_timeout=lambda timeout: None,
        get_client_version=lambda: "0.10.0",
        get_server_version=lambda: "0.10.0",
        get_world=unexpected_world,
    )
    api = SimpleNamespace(Client=lambda host, port: client)
    monkeypatch.setattr(validation.importlib, "import_module", lambda name: api)
    with pytest.raises(
        validation.LiveValidationError, match="release target '1.0'"
    ) as e:
        run(build_ref="1.0")
    report = e.value.report
    assert report["status"] == "failed"
    assert report["client_version"] == report["server_version"] == "0.10.0"
    assert report["provenance"]["runtime_matches_target"] is False
    assert report["provenance"]["source_verified"] is False
    assert report.get("scenarios", report.get("phases")) == []


@pytest.mark.parametrize(
    "run", [validation.run_live_validation, fullbody_validation.run_fullbody_validation]
)
def test_distinct_source_commits_fail_even_with_same_release_family(
    run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = SimpleNamespace(
        set_timeout=lambda timeout: None,
        get_client_version=lambda: "1.0-aaaaaaaaa",
        get_server_version=lambda: "1.0-bbbbbbbbb",
    )
    api = SimpleNamespace(Client=lambda host, port: client)
    monkeypatch.setattr(validation.importlib, "import_module", lambda name: api)
    with pytest.raises(validation.LiveValidationError, match="versions differ"):
        run()


@pytest.mark.parametrize(
    "run", [validation.run_live_validation, fullbody_validation.run_fullbody_validation]
)
def test_empty_target_fails_before_client_import(
    run: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_import(name: str) -> None:
        raise AssertionError("Target rejection must precede CARLA import")

    monkeypatch.setattr(validation.importlib, "import_module", unexpected_import)
    with pytest.raises(
        validation.LiveValidationError, match="target ref must not be empty"
    ):
        run(target_ref=" ")


@pytest.mark.parametrize("script", ["validate_carla.py", "validate_fullbody.py"])
def test_cli_default_records_carla_1_target_on_failure(
    script: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "failure.json"
    monkeypatch.delenv("CARLA_TARGET_REF", raising=False)
    monkeypatch.delenv("CARLA_BUILD_REF", raising=False)
    monkeypatch.setattr(sys, "argv", [script, "--output", str(output)])
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / script)
    )

    def missing_client(name: str) -> None:
        raise ModuleNotFoundError("Matching CARLA client is unavailable")

    monkeypatch.setattr(validation.importlib, "import_module", missing_client)
    assert namespace["main"]() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["provenance"]["expected_ref"] == "1.0"
    assert report["provenance"]["runtime_matches_target"] is None
