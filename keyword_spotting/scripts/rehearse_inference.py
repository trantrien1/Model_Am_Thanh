"""Exercise the inference notebook with the downloaded real checkpoint; mock Colab UI only."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    notebook = nbformat.read(ROOT / "notebooks/keyword_spotting_inference_colab.ipynb", as_version=4)
    old_directory, old_check_call = Path.cwd(), subprocess.check_call
    with tempfile.TemporaryDirectory(prefix="kws-inference-notebook-") as temporary:
        content = Path(temporary)
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda path: None)
        colab.files = types.SimpleNamespace(upload=lambda: {
            "test_positive.wav": (ROOT / "runs/baseline/examples/test_positive.wav").read_bytes(),
            "test_negative.wav": (ROOT / "runs/baseline/examples/test_negative.wav").read_bytes()},
            download=lambda path: None)
        display = types.ModuleType("IPython.display")
        display.Audio = lambda *a, **k: None
        display.display = lambda *a, **k: None
        package = types.ModuleType("IPython")
        package.get_ipython = lambda: None
        package.version_info = (9, 0, 0)
        replacements = {"google": google, "google.colab": colab, "IPython": package, "IPython.display": display}
        prior = {name: sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements)
        subprocess.check_call = lambda *a, **k: 0  # Dependencies already verified in this environment.

        class LocalPaths(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value.startswith("/content"):
                    node.value = content.as_posix() + node.value[len("/content"):]
                return node

            def visit_Assign(self, node):
                if any(isinstance(target, ast.Name) and target.id == "RUN_DIR" for target in node.targets):
                    node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(ROOT / "runs/baseline"))], keywords=[])
                    return node
                return self.generic_visit(node)

        namespace = {"__name__": "__main__"}
        executed = 0
        try:
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                tree = ast.fix_missing_locations(LocalPaths().visit(ast.parse(cell.source)))
                exec(compile(tree, "inference-notebook-cell", "exec"), namespace)
                executed += 1
            rows = namespace["results"]
            assert [row["prediction"] for row in rows] == [1, 0]
            assert namespace["output"].is_file()
            info = namespace["spotter"].describe()
            report = {"passed": True, "code_cells_executed": executed, "checkpoint_sha256": info["checkpoint_sha256"],
                      "selected_epoch": info["selected_epoch"], "device": info["device"],
                      "colab_runtime_executed": False, "drive_mounted": False,
                      "note": "Actual trained weights and test examples used on CPU; Drive/upload/display/download UI simulated.",
                      "predictions": [{key: row[key] for key in ("file", "prediction", "probability", "threshold")} for row in rows]}
            (ROOT / "reports/inference_notebook_rehearsal.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("Inference notebook rehearsal PASSED with the downloaded trained checkpoint.")
        finally:
            os.chdir(old_directory)
            subprocess.check_call = old_check_call
            for name, previous in prior.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous


if __name__ == "__main__":
    main()
