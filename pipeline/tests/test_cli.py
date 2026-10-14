from __future__ import annotations

import json

import pytest

from movierec import cli, config


def test_report_prints_table_from_results(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out
    results = json.loads(config.RESULTS_PATH.read_text())
    assert "| Hybrid |" in out
    assert f"{results['profiles']['recent-5']['hybrid']['hit_rate']:.3f}" in out


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2
