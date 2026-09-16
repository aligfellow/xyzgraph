"""Tests for the command-line interface."""

import re
import sys

import pytest

from xyzgraph import cli

VARIABLE_TRAJECTORY = (
    "2\nhydrogen\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n"
)


@pytest.fixture
def trajectory(tmp_path):
    xyz_file = tmp_path / "variable.xyz"
    xyz_file.write_text(VARIABLE_TRAJECTORY, encoding="utf-8")
    return str(xyz_file)


def run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["xyzgraph", *args])
    cli.main()


def formulas(output: str) -> list[str]:
    return re.findall(r"Constructed graph with chemical formula: (\S+)", output)


def test_cli_frame_selects_variable_size_frame(monkeypatch, capsys, trajectory):
    run_cli(monkeypatch, trajectory, "--quick", "--frame", "1")

    output = capsys.readouterr().out
    assert "variable.xyz (frame 1)" in output
    assert formulas(output) == ["H2O"]


@pytest.mark.parametrize("frame", ["2", "-1"])
def test_cli_frame_out_of_range(monkeypatch, trajectory, frame):
    with pytest.raises(ValueError, match=re.escape(f"Frame {frame} out of range. File has 2 frame(s).")):
        run_cli(monkeypatch, trajectory, "--quick", "--frame", frame)


def test_cli_all_frames_processes_each_frame(monkeypatch, capsys, trajectory):
    run_cli(monkeypatch, trajectory, "--quick", "--all-frames")

    output = capsys.readouterr().out
    assert "variable.xyz (frame 0-1)" in output
    assert "Processing all 2 frames" in output
    assert formulas(output) == ["H2", "H2O"]
