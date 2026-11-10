"""Validate schema, syntax and embedded source snapshot; no Colab/Drive calls locally."""
import ast
import argparse
import base64
import hashlib
import io
import zipfile
from pathlib import Path

import nbformat


def validate(notebook_path=None):
    root = Path(__file__).resolve().parents[1]
    path = Path(notebook_path) if notebook_path else root / "notebooks/keyword_spotting_colab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = None
    code_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        code_cells += 1
        tree = ast.parse(cell.source, filename=f"notebook-cell-{index}")
        compile(tree, f"notebook-cell-{index}", "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCE_PAYLOAD" for target in node.targets):
                source = ast.literal_eval(node.value)
        assert cell.execution_count is None and not cell.outputs, "Notebook must not contain fabricated run outputs"
    assert source, "Missing standalone source payload"
    bundle = base64.b64decode(source)
    assert hashlib.sha256(bundle).hexdigest() == notebook.metadata["source_bundle_sha256"]
    file_count = 0
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.testzip() is None
        for entry in archive.infolist():
            path = root / entry.filename
            assert path.exists() and archive.read(entry.filename) == path.read_bytes(), f"Stale embedded source: {entry.filename}"
            if entry.filename.endswith(".py"):
                compile(archive.read(entry.filename), entry.filename, "exec")
            file_count += 1
    print(f"Notebook valid: {code_cells} code cells, {file_count} embedded files, source matches project.")
    return {"schema_valid": True, "syntax_valid": True, "source_matches_project": True,
            "code_cells": code_cells, "embedded_files": file_count, "colab_runtime_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", help="Notebook path; defaults to the training notebook.")
    validate(parser.parse_args().notebook)
