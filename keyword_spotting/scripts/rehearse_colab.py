"""Run notebook logic locally with artificial test fixtures and no Google services.

This validates packaging/CLI flow only. It does not train on ESC-50/Speech Commands,
test a Colab GPU, mount Drive, or generate official experiment metrics.
"""
import ast
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    os.environ["PYTHONUTF8"] = "1"
    notebook = nbformat.read(ROOT / "notebooks/keyword_spotting_colab.ipynb", as_version=4)
    old_directory = Path.cwd()
    old_run, old_check = subprocess.run, subprocess.check_call
    previous_modules = {key: sys.modules.get(key) for key in ("IPython", "IPython.display")}
    with tempfile.TemporaryDirectory(prefix="kws-notebook-rehearsal-") as temporary:
        temporary = Path(temporary)
        content = temporary / "content"
        content.mkdir()
        fixture_factory = runpy.run_path(str(ROOT / "tests/conftest.py"))["toy_project"].__wrapped__
        fixture_config, _ = fixture_factory(temporary)
        fixture_root = Path(fixture_config["_root"])
        # Notebook rendering is replaced by no-op display objects; numerical code runs normally.
        package = types.ModuleType("IPython")
        package.get_ipython = lambda: None
        package.version_info = (9, 0, 0)
        display_module = types.ModuleType("IPython.display")
        for name in ("display", "Audio", "Image", "Markdown"):
            setattr(display_module, name, lambda *args, **kwargs: None)
        sys.modules["IPython"] = package
        sys.modules["IPython.display"] = display_module

        def check_call(command, *args, **kwargs):
            if "pip" in command:
                print("Rehearsal: use already installed test dependencies; skip pip mutation.")
                return 0
            return old_check(command, *args, **kwargs)

        def run(command, *args, **kwargs):
            if "pytest" in command:
                print("Rehearsal: pytest is verified separately; skip nested suite.")
                return subprocess.CompletedProcess(command, 0)
            return old_run(command, *args, **kwargs)

        subprocess.run, subprocess.check_call = run, check_call

        class AdaptToLocalFixture(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and (node.value == "/content" or node.value.startswith("/content/")):
                    node.value = content.as_posix() + node.value[len("/content"):]
                return node

            def visit_Assign(self, node):
                node = self.generic_visit(node)
                replacements = {"USE_DRIVE": False, "REQUIRE_GPU": False, "BATCH_SIZE": 4, "MAX_EPOCHS": 2,
                                "DOWNLOAD_RESULTS": False, "ESC_DIR": str(fixture_root / "data/raw/esc"),
                                "SPEECH_DIR": str(fixture_root / "data/raw/speech_commands")}
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in replacements:
                        node.value = ast.Constant(replacements[target.id])
                return node

            def visit_keyword(self, node):
                if node.arg == "num_workers":
                    node.value = ast.Constant(0)
                return self.generic_visit(node)

        namespace = {"__name__": "__main__"}
        executed = 0
        try:
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                print(f"Rehearsal notebook cell {index}", flush=True)
                tree = ast.fix_missing_locations(AdaptToLocalFixture().visit(ast.parse(cell.source)))
                exec(compile(tree, f"notebook-cell-{index}", "exec"), namespace)
                executed += 1
                if "PROJECT" in namespace:
                    project = str(namespace["PROJECT"])
                    if project not in sys.path:
                        sys.path.insert(0, project)
            result_zip = namespace["RESULT_ZIP"]
            import zipfile
            with zipfile.ZipFile(result_zip) as archive:
                assert "keyword_spotting/runs/baseline/best.pt" in archive.namelist()
                assert "keyword_spotting/runs/baseline/evaluation/test/metrics.json" in archive.namelist()
                assert archive.testzip() is None
            report = {"passed": True, "executed_code_cells": executed, "data": "artificial software-test fixtures only",
                      "drive_mounted": False, "gpu_used": False, "colab_runtime_executed": False,
                      "full_official_dataset_training_executed": False,
                      "verified": ["embedded source extraction", "config", "existing dataset overrides", "source preparation",
                                   "mixture generation", "split/audio audit", "smoke", "two-epoch fixture training",
                                   "threshold", "main and supplemental fixture evaluation", "predict/evaluate agreement", "result ZIP"]}
            (ROOT / "reports").mkdir(exist_ok=True)
            (ROOT / "reports/notebook_rehearsal.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("Notebook rehearsal PASSED (artificial fixtures, CPU, no Drive/Colab).", flush=True)
        finally:
            os.chdir(old_directory)
            subprocess.run, subprocess.check_call = old_run, old_check
            for key, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = previous


if __name__ == "__main__":
    main()
