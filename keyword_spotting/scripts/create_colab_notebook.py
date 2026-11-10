"""Generate a standalone Colab notebook carrying a compressed source snapshot."""
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_bundle():
    files = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("scripts/*.py"), ROOT / "pytest.ini",
                    ROOT / "configs/default.yaml", ROOT / "README.md", ROOT / ".gitignore",
                    ROOT / "inference.py", ROOT / "infer.ps1", ROOT / "INFERENCE.md",
                    ROOT / "requirements.txt", ROOT / "requirements-colab.txt", ROOT / "requirements-dev.txt",
                    ROOT / "data/make_dataset.py", ROOT / "data/real_recordings/README.md"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            entry = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    return buffer.getvalue()


def main():
    payload = source_bundle()
    encoded = base64.b64encode(payload).decode("ascii")
    payload_hash = hashlib.sha256(payload).hexdigest()
    cells = []

    def cell(kind, source):
        source = textwrap.dedent(source).strip() + "\n"
        value = {"cell_type": kind, "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:12],
                 "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    def markdown(source):
        cell("markdown", source)

    def code(source):
        cell("code", source)

    markdown('''
    # Phát hiện “yes” trong âm thanh — train trên Colab

    **File này tự chứa code.** Không cần upload project hay kết nối GitHub.

    1. Chọn **Runtime → Change runtime type → GPU**.
    2. Chạy các cell từ trên xuống. Cho phép Google Drive khi cell mount yêu cầu.
    3. Checkpoint và báo cáo lưu vào `MyDrive/keyword_spotting_yes/runs/baseline`.
    4. Nếu phiên bị ngắt, dùng cùng cấu hình/RUN_NAME và chạy lại; train tiếp tục từ `last.pt`.

    Pipeline: tải ESC-50/Speech Commands v0.02 → QC/split/hash → trộn 4 biến thể/nền → audit →
    smoke → full train → ngưỡng validation → test → kiểm tra bổ sung → predict → tải artifact.
    **Chưa có metrics chính thức trước khi bạn chạy notebook trên hai dataset thật.** Smoke/fixture chỉ kiểm tra phần mềm.
    Giọng nói/nền trong synthetic test chưa thay thế bản ghi thực tế. Không có ASR hoặc streaming.
    ''')
    code('''
    # Thiết lập. Đổi RUN_NAME nếu đổi cấu hình sau khi đã có checkpoint.
    USE_DRIVE = True
    DRIVE_PROJECT_NAME = "keyword_spotting_yes"
    RUN_NAME = "baseline"
    TARGET_WORD = "yes"
    BATCH_SIZE = 32
    MAX_EPOCHS = 30
    REQUIRE_GPU = True
    PERSIST_ARCHIVES = False  # True: giữ archive trên Drive để không tải lại ở phiên mới.
    RUN_SUPPLEMENTAL = True
    DOWNLOAD_RESULTS = True
    EXPORT_GENERATED_DATASET = True  # Lưu cả train/validation/test WAV trong ZIP kết quả.
    ESC_DIR = None           # Ví dụ /content/datasets/ESC-50 (đã giải nén)
    SPEECH_DIR = None        # Phải có validation_list.txt và testing_list.txt

    import sys, platform, shutil
    from pathlib import Path
    import torch
    print("Python:", platform.python_version(), "PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    elif REQUIRE_GPU:
        raise RuntimeError("Bật GPU tại Runtime > Change runtime type rồi chạy lại. Nếu muốn CPU, đặt REQUIRE_GPU=False.")
    print("Dung lượng /content:", shutil.disk_usage("/content"))
    if not RUN_NAME or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in RUN_NAME):
        raise ValueError("RUN_NAME chỉ dùng chữ, số, _ và -.")
    if not DRIVE_PROJECT_NAME or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in DRIVE_PROJECT_NAME):
        raise ValueError("DRIVE_PROJECT_NAME chỉ dùng chữ, số, _ và -.")
    ''')
    markdown('''
    ## Mount Drive và khôi phục mã nguồn

    Dữ liệu/cache xử lý trên `/content` để tránh nhiều lần đọc file nhỏ qua Drive.
    Drive giữ checkpoint mỗi epoch, source snapshot và báo cáo. Tắt `USE_DRIVE` nếu chỉ thử một phiên,
    nhưng khi runtime mất thì run đó cũng mất.
    ''')
    code('''
    import base64, hashlib, io, os, stat, zipfile
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        PERSISTENT = Path("/content/drive/MyDrive") / DRIVE_PROJECT_NAME
    else:
        PERSISTENT = Path("/content") / (DRIVE_PROJECT_NAME + "_artifacts")
    PERSISTENT.mkdir(parents=True, exist_ok=True)
    PROJECT = Path("/content/keyword_spotting")
    PROJECT.mkdir(parents=True, exist_ok=True)
    RUN_DIR = PERSISTENT / "runs" / RUN_NAME
    # Source snapshot do script create_colab_notebook.py tạo; không chứa audio/checkpoint.
    SOURCE_PAYLOAD = "__PAYLOAD__"
    packed = base64.b64decode(SOURCE_PAYLOAD)
    assert hashlib.sha256(packed).hexdigest() == "__PAYLOAD_HASH__", "Source snapshot checksum mismatch"
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        for entry in archive.infolist():
            name = entry.filename.replace("\\\\", "/")
            destination = (PROJECT / name).resolve()
            if name.startswith("/") or ".." in Path(name).parts or ":" in name or PROJECT.resolve() not in destination.parents:
                raise ValueError("Unsafe source archive path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Source archive links are forbidden")
        archive.extractall(PROJECT)
    (PERSISTENT / "source_bundle.zip").write_bytes(packed)
    os.chdir(PROJECT)
    print("Project:", PROJECT, "Run:", RUN_DIR)
    '''.replace("__PAYLOAD__", encoded).replace("__PAYLOAD_HASH__", payload_hash))
    markdown('''
    ## Cài thư viện và tạo config

    Giữ PyTorch CUDA có sẵn của Colab. Phiên bản thực dùng sẽ được ghi trong run.
    Lưu cấu hình giống nhau khi resume; thay target/batch/epoch thì tạo RUN_NAME mới.
    ''')
    code('''
    import subprocess, json, time
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt", "pytest"])
    import yaml
    cfg = yaml.safe_load((PROJECT / "configs/default.yaml").read_text())
    cfg["target_word"] = TARGET_WORD
    cfg["train"].update(batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, num_workers=2,
                        device="auto", amp=True)
    cfg["paths"]["run_dir"] = str(RUN_DIR)
    SOURCE_ARGS = []
    if ESC_DIR:
        cfg["paths"]["esc_dir"] = str(Path(ESC_DIR).resolve())
        SOURCE_ARGS += ["--esc-dir", str(Path(ESC_DIR).resolve())]
    if SPEECH_DIR:
        cfg["paths"]["speech_dir"] = str(Path(SPEECH_DIR).resolve())
        SOURCE_ARGS += ["--speech-dir", str(Path(SPEECH_DIR).resolve())]
    CONFIG = PROJECT / "configs/colab_runtime.yaml"
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    shutil.copy2(CONFIG, PERSISTENT / "config_used.yaml")

    def sync_small():
        for folder in ("data/manifests", "reports"):
            source = PROJECT / folder
            if source.exists():
                shutil.copytree(source, PERSISTENT / folder, dirs_exist_ok=True)

    def run_module(module, *arguments):
        command = [sys.executable, "-u", "-m", module, *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        try:
            result = subprocess.run(command, cwd=PROJECT, check=True)
        except subprocess.CalledProcessError as error:
            failure = {"module": module, "returncode": error.returncode, "command": command,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (PERSISTENT / "last_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            raise
        finally:
            sync_small()

    BASE_ARGS = ["--config", CONFIG]
    run_module("pytest", "-q")  # Fixtures nhân tạo; không phải đánh giá model thật.
    ''')
    markdown('''
    ## Tải và xác thực hai dataset

    Timeout/retry hữu hạn, SHA-256, kiểm tra archive và giải nén an toàn. Archive đã xác thực sẽ được tái sử dụng.
    Nếu bật `PERSIST_ARCHIVES`, cần đủ dung lượng Drive; số byte thực sẽ in và lưu sau tải.
    Không cần Hugging Face API/token. Có thể dùng ESC_DIR/SPEECH_DIR ở cell đầu để bỏ qua tải nguồn tương ứng.
    ''')
    code('''
    ARCHIVES = PROJECT / "data/raw/archives"
    if PERSIST_ARCHIVES and (PERSISTENT / "archives").exists():
        shutil.copytree(PERSISTENT / "archives", ARCHIVES, dirs_exist_ok=True)
    run_module("src.download", *BASE_ARGS, *SOURCE_ARGS)
    if PERSIST_ARCHIVES and ARCHIVES.exists():
        shutil.copytree(ARCHIVES, PERSISTENT / "archives", dirs_exist_ok=True)
    print(json.loads((PROJECT / "data/manifests/download_provenance.json").read_text()))
    ''')
    markdown('''
    ## Kiểm tra nguồn và khóa split

    ESC folds 1/2/3 → train, 4 → validation, 5 → test; giọng đọc theo split lists chính thức.
    Kiểm tra hash PCM, speech/source/speaker IDs. Nhóm trùng hoặc xung đột được cách ly khỏi mọi split;
    không chia lại ngẫu nhiên. Source manifests được lưu trước khi trộn.
    ''')
    code('''
    run_module("src.prepare_sources", *BASE_ARGS, *SOURCE_ARGS, "--workers", "4")
    sources = json.loads((PROJECT / "data/manifests/sources_metadata.json").read_text())
    print("Nguồn giữ lại:", sources["counts"])
    print("QC loại:", sources["qc_rejected_count"], "Cách ly:", sources["quarantined_count"])
    print("Audit split:", sources["split_audit"])
    ''')
    markdown('''
    ## Trộn, audit và nghe kiểm tra

    Mỗi nền: hai target positive, hai negative; 20% N2 là nền sạch. Mỗi WAV mono/16 kHz/5 giây/PCM16.
    Nhãn âm tính vẫn có giới hạn: **chưa có người nghe xác nhận toàn bộ ESC không chứa từ khóa**.
    `background_listening_checklist.csv` ưu tiên clip có âm thanh con người; đây là danh sách kiểm tra, không phải ground truth đã xác minh.
    ''')
    code('''
    run_module("src.build_dataset", *BASE_ARGS)
    run_module("src.audit_dataset", *BASE_ARGS)
    import pandas as pd
    from IPython.display import display, Audio
    manifest = pd.read_csv(PROJECT / "data/manifests/dataset.csv", keep_default_na=False)
    display(manifest.groupby(["split", "label", "variant_type"]).size().rename("files").reset_index())
    checklist = pd.read_csv(PROJECT / "data/manifests/background_listening_checklist.csv", keep_default_na=False)
    display(checklist.head(10))
    if len(checklist):
        first = checklist.iloc[0]
        print("Nền ưu tiên cần nghe:", first["background_id"], first["category"])
        display(Audio(filename=str(PROJECT / first["path"])))
    for label in (1, 0):
        row = manifest[(manifest["split"] == "train") & (manifest["label"] == label)].iloc[0]
        print("Mẫu nghe train:", row["sample_id"], "nhãn", label, "từ", row["inserted_word"])
        display(Audio(filename=str(PROJECT / row["path"])))
    ''')
    markdown('''
    ## Smoke test trước full train

    Ít batch, một epoch, subset được đánh dấu rõ và lưu ở run riêng. Kiểm tra shapes/loss/checkpoint.
    Kết quả smoke không được báo như kết quả dataset đầy đủ.
    ''')
    code('''
    SMOKE_DIR = RUN_DIR.with_name(RUN_DIR.name + "_smoke")
    smoke_arguments = [*BASE_ARGS, "--run-dir", SMOKE_DIR, "--smoke"]
    if (SMOKE_DIR / "last.pt").exists():
        smoke_arguments += ["--resume"]
    run_module("src.train", *smoke_arguments)
    print(json.loads((SMOKE_DIR / "status.json").read_text()))
    ''')
    markdown('''
    ## Full train và chọn ngưỡng validation

    Train trên toàn bộ mixture train giữ lại sau QC; không đọc test để chọn model.
    Best epoch theo validation loss, early stopping patience 5. Ngưỡng F1 chọn trên validation sau khi khóa best checkpoint.
    Checkpoint được ghi trực tiếp lên Drive **mỗi epoch**. Nếu CUDA OOM, giảm batch và chạy lại epoch từ checkpoint trước đó.
    ''')
    code('''
    training_arguments = [*BASE_ARGS, "--run-dir", RUN_DIR]
    if (RUN_DIR / "last.pt").exists():
        training_arguments += ["--resume"]
    run_module("src.train", *training_arguments)
    print("Trạng thái:", json.loads((RUN_DIR / "status.json").read_text()))
    print("Ngưỡng:", json.loads((RUN_DIR / "threshold.json").read_text()))
    print("Môi trường thực:", json.loads((RUN_DIR / "environment.json").read_text()))
    ''')
    markdown('''
    ## Test chính với model và ngưỡng đã khóa

    Accuracy, precision, recall, F1, AUROC, average precision, confusion matrix, FPR, FNR;
    nhóm thiếu điều kiện tính metric trả null với lý do. Báo cáo theo từ negative/nền sạch/loại nền/SNR và fallback.
    Variants cùng nền không phải mẫu độc lập hoàn toàn; báo cả số nền và số bản ghi ESC gốc.
    ''')
    code('''
    run_module("src.evaluate", "--run-dir", RUN_DIR, "--config", CONFIG, "--split", "test")
    TEST_OUTPUT = RUN_DIR / "evaluation/test"
    metrics = json.loads((TEST_OUTPUT / "metrics.json").read_text())
    display(pd.DataFrame([metrics["overall"]]).drop(columns=["confusion_matrix", "undefined_reasons"]))
    print("Files:", metrics["file_count"], "Nền:", metrics["background_count"],
          "Nguồn ESC độc lập:", metrics["independent_esc_source_count"], "Speaker:", metrics["speaker_count"])
    from IPython.display import Image, Markdown
    display(Image(filename=str(RUN_DIR / "loss_curve.png")))
    display(Image(filename=str(TEST_OUTPUT / "confusion_matrix.png")))
    display(Markdown((TEST_OUTPUT / "report.md").read_text()))
    for error_type in ("false_positives", "false_negatives"):
        errors = pd.read_csv(TEST_OUTPUT / (error_type + ".csv"), keep_default_na=False)
        print(error_type, "count:", len(errors))
        display(errors.head(5))
    ''')
    markdown('''
    ## Kiểm tra bổ sung, tách khỏi test chính

    Chỉ nguồn thuộc test: các nền sạch còn lại và mức giọng thấp hơn (SNR -10–0 dB).
    Chỉ khảo sát; không thay đổi model/ngưỡng từ kết quả này.
    ''')
    code('''
    if RUN_SUPPLEMENTAL:
        run_module("src.build_dataset", *BASE_ARGS, "--supplemental")
        run_module("src.audit_dataset", *BASE_ARGS, "--supplemental")
        run_module("src.evaluate", "--run-dir", RUN_DIR, "--config", CONFIG, "--supplemental")
        extra = json.loads((RUN_DIR / "evaluation/supplemental_test/metrics.json").read_text())
        print("Supplemental test only:", extra["overall"])
    ''')
    markdown('''
    ## Predict WAV ví dụ và kiểm tra khớp evaluate

    Hai ví dụ dương/âm lấy từ test chính. Probability là score chưa hiệu chuẩn.
    File ngắn được pad, file dài hơn 5 giây bị từ chối.
    ''')
    code('''
    from src.predict import predict
    example_rows = json.loads((RUN_DIR / "examples/predictions.json").read_text())
    for row in example_rows:
        example = RUN_DIR / row["example_file"]
        result = predict(RUN_DIR, example)
        print(row["example_file"], "nhãn thật:", row["label"], result)
        assert abs(result["probability"] - row["probability"]) < 1e-4, "Predict/evaluate preprocessing mismatch"
        assert result["prediction"] == row["prediction"]
        display(Audio(filename=str(example)))
    print("chưa đánh giá trên bản ghi thực tế")
    ''')
    markdown('''
    ## Tùy chọn: upload WAV tự thu để dự đoán

    Đặt RUN_CUSTOM_PREDICTION=True rồi chạy cell này. Mỗi WAV tối đa 5 giây.
    Nghe/xác nhận nhãn bằng người để đánh giá thực tế; một vài file dự đoán chưa phải đánh giá có hệ thống.
    ''')
    code('''
    RUN_CUSTOM_PREDICTION = False
    if RUN_CUSTOM_PREDICTION:
        from google.colab import files
        uploaded = files.upload()
        real_dir = PROJECT / "data/real_recordings"
        real_dir.mkdir(parents=True, exist_ok=True)
        for name, content in uploaded.items():
            audio_path = real_dir / Path(name).name
            audio_path.write_bytes(content)
            try:
                print(name, predict(RUN_DIR, audio_path))
            except ValueError as error:
                print(name, "Không dự đoán:", str(error))
    ''')
    markdown('''
    ## Lưu số đo dung lượng và tải kết quả về máy

    Gói kết quả có mã nguồn, checkpoint, threshold, thống kê features, manifests, báo cáo và WAV ví dụ.
    `EXPORT_GENERATED_DATASET=True` lưu thêm toàn bộ WAV đã trộn, bao gồm train/validation/test và supplemental.
    Raw nguồn/cache features không được đưa vào ZIP này. Đặt False nếu chỉ cần gói model nhẹ.
    Trên máy yếu chỉ cần cài PyTorch CPU để predict; dùng inference.py hoặc infer.ps1.
    ESC-50: CC BY-NC; Speech Commands audio: CC BY 4.0. Giữ attribution/giấy phép nguồn khi dùng dữ liệu mixture.
    [ESC-50](https://github.com/karolpiczak/ESC-50) ·
    [Speech Commands](https://huggingface.co/datasets/google/speech_commands) ·
    [Paper Speech Commands](https://arxiv.org/abs/1804.03209) ·
    [Colab FAQ](https://research.google.com/colaboratory/faq.html).
    ''')
    code('''
    def folder_bytes(path):
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0
    storage = {"raw_bytes": folder_bytes(PROJECT / "data/raw"),
               "generated_bytes": folder_bytes(PROJECT / "data/generated"),
               "feature_cache_bytes": folder_bytes(PROJECT / "data/features"),
               "run_bytes": folder_bytes(RUN_DIR),
               "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (PROJECT / "reports").mkdir(exist_ok=True)
    (PROJECT / "reports/storage.json").write_text(json.dumps(storage, indent=2), encoding="utf-8")
    sync_small()
    print("Dung lượng thực (GiB):", {key: round(value / 1024**3, 3) for key, value in storage.items() if key.endswith("bytes")})
    RESULT_ZIP = PERSISTENT / (RUN_NAME + "_results.zip")
    with zipfile.ZipFile(RESULT_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        with zipfile.ZipFile(io.BytesIO(packed)) as sources:
            for entry in sources.infolist():
                archive.writestr("keyword_spotting/" + entry.filename, sources.read(entry.filename))
        for folder, prefix in ((RUN_DIR, "runs/" + RUN_NAME),
                               (PROJECT / "reports", "reports"),
                               (PROJECT / "data/manifests", "data/manifests")):
            for path in sorted(folder.rglob("*")):
                if path.is_file() and not path.name.endswith(".tmp"):
                    archive.write(path, "keyword_spotting/" + prefix + "/" + path.relative_to(folder).as_posix())
        if EXPORT_GENERATED_DATASET:
            for path in sorted((PROJECT / "data/generated").rglob("*.wav")):
                archive.write(path, "keyword_spotting/data/generated/" + path.relative_to(PROJECT / "data/generated").as_posix())
    print("Đã lưu:", RESULT_ZIP, "size MiB:", round(RESULT_ZIP.stat().st_size / 1024**2, 2))
    if DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(RESULT_ZIP))
    ''')
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "colab": {"name": "keyword_spotting_colab.ipynb", "provenance": []}, "accelerator": "GPU",
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}, "source_bundle_sha256": payload_hash}, "cells": cells}
    destination = ROOT / "notebooks/keyword_spotting_colab.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / "notebooks/source_bundle.sha256").write_text(payload_hash + "\n", encoding="ascii")
    print(f"Created {destination.name}: {len(cells)} cells, {destination.stat().st_size} bytes")


if __name__ == "__main__":
    main()
