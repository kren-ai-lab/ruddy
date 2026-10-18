"""Conformance of the shared ecosystem CLI contract: ``--config``, ``validate`` and the provenance sidecar."""

import json
from pathlib import Path

import jsonschema
import pytest
import typer

from ruddy.cli._config_file import check_config
from ruddy.cli.main import main

HERE = Path(__file__).parent


def _schema(name: str) -> dict:
    return json.loads((HERE / name).read_text())


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    rows = ["id,x,y"] + [f"r{i},{i},{(i * 7) % 11}" for i in range(12)]
    path.write_text("\n".join(rows) + "\n")
    return path


def _summary(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def test_config_precedence(data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("random_state: 5\nid_column: id\n")
    args = ["-q", "pipeline", str(data), "--output-dir", str(tmp_path / "o")]

    def params(*extra: str) -> dict:
        assert main([*args, *extra]) == 0
        capsys.readouterr()
        return json.loads((tmp_path / "o.provenance.json").read_text())["parameters"]

    assert params()["random_state"] == 0
    assert params("--config", str(cfg))["random_state"] == 5
    assert params("--config", str(cfg), "--random-state", "9")["random_state"] == 9


@pytest.mark.parametrize(
    "text",
    ["nope: 1\n", "random_state: three\n", "output_dir: x\n", "config: x\n", "- a\n"],
)
def test_invalid_config_exits_2(data: Path, tmp_path: Path, text: str) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text)
    assert main(["pipeline", str(data), "--config", str(cfg)]) == 2


def test_validate_json_conforms(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("random_state: three\nnope: 1\n")
    assert main(["validate", str(cfg), "--json"]) == 2
    issues = json.loads(capsys.readouterr().out)
    jsonschema.validate(issues, _schema("validate-errors.schema.json"))
    assert {i["path"] for i in issues} == {"random_state", "nope"}
    cfg.write_text("random_state: 3\n")
    assert main(["validate", str(cfg), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_validate_runs_semantic_checks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("confidence_level: 1.5\n")
    assert main(["validate", str(cfg), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)[0]["path"] == ""


def test_provenance_sidecar(data: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    assert main(["-q", "pipeline", str(data), "--id-column", "id", "--output-dir", str(out)]) == 0
    doc = json.loads((tmp_path / "report.provenance.json").read_text())
    jsonschema.validate(doc, _schema("provenance.schema.json"))
    assert doc["tool"]["name"] == "ruddy"
    assert [i["path"] for i in doc["inputs"]] == [str(data)]
    assert doc["extensions"]["executed_blocks"]


def test_no_sidecar_in_stdout_mode(data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-q", "pipeline", str(data), "--id-column", "id"]) == 0
    capsys.readouterr()
    assert not list(tmp_path.glob("*.provenance.json"))


def test_check_config_flags_each_kind_of_problem() -> None:
    app = typer.Typer()
    size_option = typer.Option(8, min=1)
    layer_option = typer.Option([], "--layer")

    @app.command()
    def run(size: int = size_option, layers: list[int] = layer_option) -> None: ...

    cmd = typer.main.get_command(app)
    found = {i.path for i in check_config(cmd, {"size": 0, "layers": 3, "nope": 1}, exclude=frozenset())}
    assert found == {"size", "layers", "nope"}
    assert check_config(cmd, {"size": "4", "layers": [1, 2]}, exclude=frozenset()) == []
