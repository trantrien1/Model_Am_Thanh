# Phát hiện từ “yes” trong WAV tối đa 5 giây

Project triển khai `PLAN_keyword_spotting_codex.md`: ESC-50 + Google Speech Commands v0.02, trộn mono 16 kHz/5 giây, log-mel 64 bins và CNN PyTorch. `target_word: "yes"` cấu hình trong `configs/default.yaml`.

Run bạn đã train trên Colab được tải về `runs/baseline`: **best epoch 11, threshold 0.50**. Bắt đầu sử dụng model tại [INFERENCE.md](INFERENCE.md): `infer.ps1` tự chọn môi trường Python CPU của workspace, `inference.py` in rõ phát hiện/không phát hiện, score và ngưỡng, hỗ trợ cả thư mục WAV. Báo cáo model và dataset hiện tại nằm trong `reports/trained_model_delivery.md`; kết quả đối chiếu CPU nằm trong `reports/cpu_test/`.

## Chạy trên Google Colab (cách khuyến nghị)

1. Mở [Google Colab](https://colab.research.google.com/), chọn **File → Upload notebook**, mở `notebooks/keyword_spotting_colab.ipynb`.
2. Chọn **Runtime → Change runtime type → GPU**.
3. Chạy các cell theo thứ tự. Notebook **chứa mã nguồn bên trong**, không cần GitHub hay upload thêm ZIP. Nó cài thư viện, tải hai archive WAV chính thức, kiểm tra nguồn, trộn, audit, smoke test, full train, chọn ngưỡng trên validation, đánh giá test và dự đoán WAV ví dụ.
4. Cho phép mount Google Drive khi Colab yêu cầu. Mặc định lưu checkpoint, manifest và báo cáo trong `MyDrive/keyword_spotting_yes/runs/baseline`. Audio/cache làm việc ở `/content/keyword_spotting` để giảm số lần đọc Drive.
5. Cell cuối tạo ZIP có code, checkpoint, báo cáo, WAV ví dụ và dataset WAV đã trộn (bao gồm test). Đặt `EXPORT_GENERATED_DATASET=False` nếu chỉ cần gói model nhẹ để predict bằng CPU.

Các biến đầu notebook: `RUN_NAME`, `TARGET_WORD`, `BATCH_SIZE`, `MAX_EPOCHS`, `ESC_DIR`, `SPEECH_DIR`, `PERSIST_ARCHIVES`, `EXPORT_GENERATED_DATASET`. Mặc định batch 32, tối đa 30 epoch, Adam 0.001, early stopping patience 5. Giảm batch khi thiếu VRAM; code cũng tự rollback epoch và giảm batch nếu CUDA OOM. Muốn thí nghiệm cấu hình khác, đổi `RUN_NAME` để giữ run cũ.

Nếu Colab ngắt: mở lại notebook, dùng cùng `RUN_NAME` và cấu hình, chạy từ đầu. Source manifests và mixture sẽ được tái tạo theo seed; train nạp `last.pt`, optimizer, scaler và RNG, tiếp tục từ epoch gần nhất. Nếu run đã hoàn thành, bước train bỏ qua và chuyển evaluate. Mặc định không giữ audio/archive trên Drive; bật `PERSIST_ARCHIVES=True` để lưu hai archive đã xác thực và giảm tải mạng khi khởi động lại (đổi lại tốn dung lượng Drive). Runtime mất thì cache/audio phải tái tạo; checkpoint đã lưu không phụ thuộc việc cache còn tồn tại.

Colab có tài nguyên/quota thay đổi, GPU không được bảo đảm. [FAQ chính thức](https://research.google.com/colaboratory/faq.html) khuyến nghị hạn chế nhiều lần đọc file nhỏ qua Drive; notebook xử lý audio tại VM và lưu các artifact nhỏ lên Drive. Dataset đã khôi phục tại máy có 9.092 WAV trộn, khoảng 1,46 GB; nguồn đã giải nén khoảng 4,22 GB. Archive tải tạm, cache và bản import trùng đã được dọn. Notebook xuất số đo của mỗi lần chạy vào `reports/storage.json` và `runs/baseline/environment.json`.

## Tạo dataset bằng file Python

Từ thư mục `keyword_spotting`, dùng môi trường CPU đã có:

```powershell
..\.venv\Scripts\python.exe data/make_dataset.py
# Xem các bước trước khi chạy:
..\.venv\Scripts\python.exe data/make_dataset.py --dry-run
# Tạo bộ mới ở thư mục riêng:
..\.venv\Scripts\python.exe data/make_dataset.py --output-dir data/new_dataset
```

File gọi đúng các bước của notebook: tải hoặc dùng nguồn đã giải nén → kiểm tra nguồn/split → trộn → audit, gồm cả supplemental test. WAV mono 16 kHz, 5 giây, seed 42, target `yes` theo config. Khi đã có manifest dataset, script kiểm tra config, tái sử dụng WAV đúng hash và khôi phục file thiếu từ manifest đã khóa. Nhãn riêng từng split nằm trong `train.csv`, `validation.csv`, `test.csv`. `--workers 2` là mặc định phù hợp máy yếu; `--no-supplemental` chỉ tạo/kiểm tra dataset chính. Có thể truyền `--config`, `--esc-dir`, `--speech-dir`; chạy `--help` để xem tùy chọn.

## Chạy CLI tại máy hoặc môi trường khác

Python 3.10–3.12 là lựa chọn đã chuẩn bị; máy kiểm thử dùng Python 3.10.0 và PyTorch 2.5.1 CPU. Từ thư mục `keyword_spotting`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Với máy chỉ cần CPU hoặc predict:
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-colab.txt

python -m src.download --config configs/default.yaml
python -m src.prepare_sources --config configs/default.yaml
python -m src.build_dataset --config configs/default.yaml
python -m src.audit_dataset --config configs/default.yaml
python -m src.train --config configs/default.yaml --smoke
python -m src.train --config configs/default.yaml
python -m src.evaluate --run-dir runs/baseline --split test
python -m src.predict --run-dir runs/baseline --audio runs/baseline/examples/test_positive.wav --device cpu
```

GPU ngoài Colab: cài PyTorch phù hợp theo [trang chính thức](https://pytorch.org/get-started/locally/), sau đó `pip install -r requirements-colab.txt`. Notebook giữ PyTorch CUDA có sẵn của Colab, không thay bằng bản CPU. `requirements.txt` dành cho môi trường thông thường; `reports/local_requirements.txt` ghi phiên bản thực dùng ở máy kiểm thử, không dùng file này để ghi đè môi trường CUDA của Colab.

Để tiếp tục run bị ngắt:

```powershell
python -m src.train --config configs/default.yaml --run-dir runs/baseline --resume
```

Để dùng nguồn đã tải, sửa `paths.esc_dir` và `paths.speech_dir` trong YAML hoặc truyền đường dẫn cho hai bước đầu. Ví dụ:

```powershell
python -m src.download --config configs/default.yaml --esc-dir "D:/datasets/ESC-50" --speech-dir "D:/datasets/speech_commands_v0.02"
python -m src.prepare_sources --config configs/default.yaml --esc-dir "D:/datasets/ESC-50" --speech-dir "D:/datasets/speech_commands_v0.02"
python -m src.build_dataset --config configs/default.yaml
```

Các bước sau lấy đường dẫn nguồn từ source manifest. Không cần Hugging Face API/token. Thư mục ESC phải có `audio/` và `meta/esc50.csv`; Speech Commands phải có các thư mục từ và cả `validation_list.txt`, `testing_list.txt`.

Sau khi giải nén ZIP kết quả từ Colab trên máy, vào thư mục `keyword_spotting`, cài bản CPU và dùng `src.predict --run-dir runs/baseline --audio ...`. Predict chỉ cần code, `best.pt`, `threshold.json` và WAV; không cần tải lại dataset. Evaluate lại toàn bộ test cần dựng lại đúng dataset và truyền `--config` để định vị data khi artifact được chuyển sang nơi khác.

## Các ràng buộc thực nghiệm

- Split nền đọc từ CSV: train fold 1/2/3, validation 4, test 5; split giọng từ danh sách chính thức. Không chia lại ngẫu nhiên.
- Canonical hash: decode → mono → resample 16 kHz → PCM16 little-endian (chưa trim). Audit source ID, original ESC source, speech ID, speaker ID và PCM hash, kể cả giữa hai corpora.
- Bản ghi trùng/xung đột bị loại cả thành phần liên thông của các speaker/nguồn ESC liên quan, khỏi mọi split; lưu `quarantine.csv`. Nguồn gốc vẫn nguyên vẹn. Mọi bước sau dùng manifest đã khóa.
- Mỗi nền giữ 4 biến thể P1/P2/N1/N2; 20% N2 là nền sạch, seed riêng theo split. Hai positive dùng bản ghi khác nhau và ưu tiên speaker khác; từ negative chọn gần đều. Lỗi một biến thể loại cả nhóm nền khỏi manifest. Số lượng thật và chênh lệch với 4.800/1.600/1.600 ghi trong `dataset_metadata.json`.
- Mixer đo speech/background trên cùng active-speech mask; SNR 0–15 dB. Nền gần im lặng dùng -25 dBFS và không báo requested SNR như measured SNR. Global gain chống clipping áp dụng cho mọi sample; tất cả WAV đều PCM_16 80.000 mẫu.
- Log-mel: `n_fft=512`, Hann 400, hop 160, Slaney 64 bins 20–8.000 Hz, `center=False`, log power với epsilon `1e-10`; mean/std mỗi bin chỉ tính trên train. Cache dùng preprocessing + hash audio.
- CNN 32/64/128 channels → global max → linear 64/dropout 0.3 → logit. Train BCEWithLogitsLoss; best checkpoint theo validation loss; quét ngưỡng validation 0.05–0.95 bước 0.01, hòa thì ưu tiên FPR thấp và ngưỡng cao.
- Test được chạy sau khi khóa model/ngưỡng; checkpoint và ngưỡng liên kết bằng SHA-256. Không dùng test chọn epoch/cấu hình. Run smoke nằm riêng và bị từ chối evaluate chính thức nếu không chỉ định `--allow-smoke`.

`reports/dataset_audit.json` gồm counts, phân bố label, từ negative, speaker, vị trí, SNR, gain và kết quả chống rò rỉ. `background_listening_checklist.csv` liệt kê tất cả nền để nghe, ưu tiên lớp có âm thanh con người. **Chưa có người nghe xác nhận toàn bộ nền không chứa từ khóa**; nhãn âm tính là nhãn tổng hợp và có giới hạn.

`runs/baseline` có `best.pt`, `last.pt`, `normalization.json`, `threshold.json`, `threshold_sweep.json`, `history.csv`, `loss_curve.png`, môi trường thực dùng và bản sao manifest. `evaluation/test/` có `metrics.json`, `predictions.csv`, `false_positives.csv`, `false_negatives.csv`, `confusion_matrix.png`, `report.md`. Báo cáo theo từ negative, nền sạch, loại nền và SNR; fallback tách riêng. Metric không xác định trả `null` với lý do. Các biến thể của cùng nền/nguồn không phải mẫu độc lập hoàn toàn.

Kiểm tra bổ sung dùng riêng nguồn test, giữ tách khỏi main test:

```powershell
python -m src.build_dataset --config configs/default.yaml --supplemental
python -m src.audit_dataset --config configs/default.yaml --supplemental
python -m src.evaluate --run-dir runs/baseline --supplemental --config configs/default.yaml
```

Đây là các nền sạch còn lại và mixture SNR -10–0 dB, chỉ khảo sát; không dùng điều chỉnh model.

WAV mới tối đa 5 giây được resample và pad; file dài hơn bị từ chối, không âm thầm cắt đầu. Output JSON: `target_word, probability, threshold, prediction, duration_seconds`. Probability là score chưa hiệu chuẩn thống kê. **Chưa đánh giá trên bản ghi thực tế**; xem `data/real_recordings/README.md`. Baseline không có ASR, streaming, API hay website.

## Kiểm thử và tái tạo notebook

```powershell
python -m pip install pytest nbformat
python -m pytest -q
python scripts/create_colab_notebook.py
python scripts/validate_notebook.py
```

Notebook mang snapshot code nên sau khi sửa code phải chạy lại script tạo notebook. Chỉ các test dùng fixture âm thanh nhân tạo; không có audio/cache/checkpoint thật trong gói mã nguồn. `.gitignore` loại những artifact lớn. File plan ban đầu được giữ nguyên.

## Nguồn, giấy phép và trích dẫn

- [ESC-50 chính thức](https://github.com/karolpiczak/ESC-50): CC BY-NC, subset ESC-10 có CC BY; giữ file `LICENSE` trong dataset cho attribution từng clip. Trích dẫn Karol J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015, [DOI](https://doi.org/10.1145/2733373.2806390). Archive URL là nhánh `master`, có thể thay đổi; hash archive và metadata ghi trong provenance để nhận biết phiên bản.
- [Google Speech Commands](https://huggingface.co/datasets/google/speech_commands), [paper của Pete Warden (2018)](https://arxiv.org/abs/1804.03209): CC BY 4.0 cho audio; giữ `LICENSE`/`README.md` của archive v0.02. Archive WAV dùng từ `storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz`.
- [Tutorial TensorFlow](https://www.tensorflow.org/tutorials/audio/simple_audio) tham khảo phương pháp; implementation ở đây dùng PyTorch.

Dữ liệu mixture và các WAV ví dụ kế thừa yêu cầu attribution/phi thương mại có liên quan của nguồn; không tự gán giấy phép mới để loại bỏ những điều kiện đó. SHA-256 ghi sau tải dùng để kiểm tra tính nhất quán/tái sử dụng archive; không được coi là chữ ký xác thực từ nhà phát hành khi không có checksum upstream.
