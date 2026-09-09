import json

import numpy as np
import pandas as pd

from ruddy.cli.main import main


def test_project_pca_cli_writes_structured_outputs(tmp_path):
    rng = np.random.default_rng(8)
    source = tmp_path / "features.csv"
    frame = pd.DataFrame(rng.normal(size=(14, 4)), columns=["f1", "f2", "f3", "f4"])
    frame.insert(0, "id", [f"s{i}" for i in range(14)])
    frame.to_csv(source, index=False)
    output = tmp_path / "out"
    assert main([
        "project", str(source), "--method", "pca", "--id-column", "id",
        "--n-components", "3", "--scaling", "standard", "--output-dir", str(output),
    ]) == 0
    assert (output / "pca_scores.csv").is_file()
    assert (output / "pca_loadings.csv").is_file()
    assert (output / "pca_variance.csv").is_file()
    payload = json.loads((output / "pca_provenance.json").read_text())
    assert payload["inferential_allowed"] is True
    assert payload["preprocessing"]["scaling"]["method"] == "standard"


def test_project_tsne_cli_marks_output_noninferential(tmp_path, capsys):
    rng = np.random.default_rng(9)
    source = tmp_path / "features.npy"
    np.save(source, rng.normal(size=(16, 4)))
    assert main([
        "project", str(source), "--method", "tsne", "--perplexity", "4",
        "--max-iter", "250", "--random-state", "2",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "tsne"
    assert payload["exploratory"] is True
    assert payload["inferential_allowed"] is False
