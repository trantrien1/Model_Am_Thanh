"""Create a standalone, post-training Colab notebook for readable WAV inference."""
import base64
import hashlib
import json
import textwrap
from pathlib import Path

from create_colab_notebook import ROOT, source_bundle


def main():
    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    cells = []

    def add(kind, text):
        text = textwrap.dedent(text).strip() + "\n"
        cell = {"cell_type": kind, "id": hashlib.sha256(f"{len(cells)}:{text}".encode()).hexdigest()[:12],
                "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", '''
    # Inference model đã train — phát hiện “yes”

    Notebook tự chứa mã nguồn, dùng `best.pt` và `threshold.json` có sẵn trên Drive.
    **Không train lại. CPU chạy được; không cần bật GPU.**

    1. Mount Drive và kiểm tra đường dẫn run ở cell đầu.
    2. Chạy các cell, upload WAV tối đa 5 giây.
    3. Xem PHÁT HIỆN/KHÔNG PHÁT HIỆN, score, ngưỡng; tải CSV kết quả.

    Run mặc định: `MyDrive/keyword_spotting_yes/runs/baseline`. File `best.pt` chứa weights,
    config log-mel và normalization train. `threshold.json` chứa ngưỡng validation;
    không tính lại stats hoặc chọn lại ngưỡng từ WAV mới.
    ''')
    add("code", '''
    from pathlib import Path
    from google.colab import drive
    drive.mount("/content/drive")
    RUN_DIR = Path("/content/drive/MyDrive/keyword_spotting_yes/runs/baseline")
    DEVICE = "cpu"  # Có thể đổi sang "auto" để dùng GPU nếu runtime có GPU.
    for filename in ("best.pt", "threshold.json"):
        path = RUN_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {path}. Sửa RUN_DIR theo thư mục model trên Drive.")
        print(filename, path, path.stat().st_size, "bytes")
    ''')
    add("markdown", '''
    ## Khôi phục helper và cài thư viện

    Helper được giải nén ở thư mục riêng `/content/keyword_spotting_inference`.
    Giữ nguyên thư mục dữ liệu của notebook train; không ghi lên checkpoint trên Drive.
    ''')
    add("code", '''
    import base64, hashlib, io, os, stat, subprocess, sys, zipfile
    PROJECT = Path("/content/keyword_spotting_inference")
    PROJECT.mkdir(parents=True, exist_ok=True)
    SOURCE_PAYLOAD = "__PAYLOAD__"
    packed = base64.b64decode(SOURCE_PAYLOAD)
    assert hashlib.sha256(packed).hexdigest() == "__DIGEST__"
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        for item in archive.infolist():
            name = item.filename.replace(chr(92), "/")
            destination = (PROJECT / name).resolve()
            if name.startswith("/") or ".." in Path(name).parts or ":" in name or PROJECT.resolve() not in destination.parents:
                raise ValueError("Unsafe embedded source path")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Embedded source links are forbidden")
        archive.extractall(PROJECT)
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"])
    from inference import KeywordSpotter
    spotter = KeywordSpotter(RUN_DIR, device=DEVICE)
    import json
    print(json.dumps(spotter.describe(), indent=2, ensure_ascii=False))
    '''.replace("__PAYLOAD__", base64.b64encode(payload).decode()).replace("__DIGEST__", digest))
    add("markdown", '''
    ## Upload WAV và xem kết quả

    Score ≥ threshold → prediction 1: PHÁT HIỆN từ khóa.
    Score < threshold → prediction 0: KHÔNG PHÁT HIỆN từ khóa.
    Score là điểm model chưa hiệu chuẩn thống kê. File ngắn được mono/resample/pad;
    file dài hơn 5 giây trả lỗi rõ ràng, không cắt mất phần cuối.
    ''')
    add("code", '''
    import pandas as pd
    from google.colab import files
    from IPython.display import Audio, display
    uploaded = files.upload()
    inputs = PROJECT / "inputs"
    inputs.mkdir(exist_ok=True)
    results = []
    for name, content in uploaded.items():
        path = inputs / Path(name).name
        path.write_bytes(content)
        try:
            result = spotter.predict(path)
            result["file"] = Path(name).name
            result["ket_qua"] = "PHÁT HIỆN" if result["prediction"] else "KHÔNG PHÁT HIỆN"
            print(f"{result['file']}: {result['ket_qua']} từ '{result['target_word']}' | score={result['probability']:.6f} | ngưỡng={result['threshold']:.2f}")
            display(Audio(filename=str(path)))
        except (ValueError, RuntimeError) as error:
            result = {"file": Path(name).name, "ket_qua": "LỖI", "error": str(error)}
            print(name, "LỖI:", error)
        results.append(result)
    predictions = pd.DataFrame(results)
    if len(predictions):
        columns = [c for c in ("file", "ket_qua", "target_word", "probability", "threshold", "prediction", "duration_seconds", "error") if c in predictions]
        display(predictions[columns])
    else:
        print("Chưa chọn WAV để dự đoán.")
    ''')
    add("markdown", '''
    ## Tải CSV

    Bảng ghi từng file và lỗi nếu có. Đây là inference WAV mới; để tính accuracy/F1 cần nhãn do người nghe xác nhận.
    Test đã trộn của pipeline có nhãn trong `data/manifests/test.csv` ở project đã khôi phục trên máy.
    ''')
    add("code", '''
    if len(predictions):
        output = PROJECT / "inference_predictions.csv"
        predictions.to_csv(output, index=False, encoding="utf-8-sig")
        files.download(str(output))
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": "keyword_spotting_inference_colab.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
        "source_bundle_sha256": digest}, "cells": cells}
    output = ROOT / "notebooks/keyword_spotting_inference_colab.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {output.name}: {len(cells)} cells.")


if __name__ == "__main__":
    main()
