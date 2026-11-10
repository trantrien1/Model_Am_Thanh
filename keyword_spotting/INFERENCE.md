# Dùng model đã train

Run đã tải từ thư mục Google Drive bạn cung cấp vào `runs/baseline`. Thông tin đọc trực tiếp từ checkpoint: từ khóa **yes**, checkpoint tốt nhất **epoch 11**, ngưỡng validation **0.50**. Các lệnh dưới đây chạy CPU, không train lại.

## Weight nằm ở đâu?

| File | Dùng để làm gì |
| --- | --- |
| `runs/baseline/best.pt` | Checkpoint inference: chứa `model_state_dict`, config model/features, mean/std từng mel bin và metadata nguồn |
| `runs/baseline/threshold.json` | Ngưỡng validation 0.50 và SHA-256 liên kết với đúng `best.pt` |
| `runs/baseline/last.pt` | Trạng thái epoch cuối, optimizer/scaler/RNG để tiếp tục train |
| `runs/baseline/examples/` | Hai WAV dương/âm được lấy từ test cùng kết quả model |

Trên Drive, đường dẫn tương ứng là `MyDrive/keyword_spotting_yes/runs/baseline/`. Bạn cần giữ **best.pt và threshold.json cùng nhau**. `best.pt` đã chứa normalization/preprocessing; không tính lại mean/std từ file mới.

## Chạy ngay trên máy hiện tại

Mở PowerShell tại `E:\Workspace\Learn\Ky_1_Nam_4\PTMD\Model_Am_Thanh\keyword_spotting`:

```powershell
# Xem đường dẫn weights, epoch, ngưỡng, thiết bị thực dùng:
.\infer.ps1 -Inspect

# Hai WAV ví dụ đã tải về:
.\infer.ps1 -Audio runs/baseline/examples/test_positive.wav
.\infer.ps1 -Audio runs/baseline/examples/test_negative.wav

# WAV của bạn, tối đa 5 giây:
.\infer.ps1 -Audio "D:/recordings/my_voice.wav"

# Toàn bộ tập test, có CSV kết quả rõ ràng từng file:
.\infer.ps1 -Folder data/generated/test -Csv reports/my_test_predictions.csv
```

Script tự chọn môi trường `.venv` đã tạo trong workspace; không cần activate và không dùng nhầm Python 3.14 mặc định. Nếu PowerShell của máy chặn chạy script, dùng Python trực tiếp:

```powershell
..\.venv\Scripts\python.exe inference.py --audio "D:/recordings/my_voice.wav"
..\.venv\Scripts\python.exe inference.py --folder data/generated/test --csv reports/my_test_predictions.csv
```

Kết quả của hai file ví dụ đã kiểm tra trên CPU:

```text
test_positive.wav: PHÁT HIỆN từ 'yes' | score=0.999999 | ngưỡng=0.50
test_negative.wav: KHÔNG PHÁT HIỆN từ 'yes' | score=0.000002 | ngưỡng=0.50
```

Model đưa ra score trong [0, 1]. **Score ≥ 0.50 → prediction 1, phát hiện “yes”; score < 0.50 → prediction 0.** Score chưa được hiệu chuẩn thống kê. File ≤5 giây được mono/resample 16 kHz và pad, không kéo giãn; file dài hơn bị từ chối. Script nạp weights một lần khi dự đoán cả thư mục; lỗi từng WAV được ghi CSV, không bị bỏ qua ngầm.

Để lấy JSON:

```powershell
..\.venv\Scripts\python.exe inference.py --audio runs/baseline/examples/test_positive.wav --json
```

Trong Python:

```python
from inference import KeywordSpotter

model = KeywordSpotter("runs/baseline", device="cpu")
print(model.describe())
print(model.predict("path/to/voice.wav"))
```

Trên máy khác, giải nén project, dùng môi trường Python có PyTorch CPU và `requirements-colab.txt`, rồi chạy `python inference.py --run-dir runs/baseline --audio ...`. Inference chỉ cần code, hai artifact model và WAV, không cần dataset gốc.

## Inference trên Colab

Mở `notebooks/keyword_spotting_inference_colab.ipynb`, chạy các cell, cho phép mount Drive rồi upload WAV. Notebook tự chứa code, chỉ nạp model đã có; không train lại và không cần GPU. Nó hiển thị kết quả bằng bảng, có nghe lại WAV và tải CSV. Sửa `RUN_DIR` ở cell đầu nếu bạn dùng run khác.

## Dataset đã train và test

Drive trước đây chỉ lưu manifest/checkpoint/báo cáo; ZIP kết quả không chứa WAV dataset. Pipeline khôi phục tại máy dùng archive gốc có SHA-256 khớp bản đã train, và **vị trí chèn, speech gain, output gain, speech ID, trim đã ghi trong manifest**, thay vì sinh lại một thí nghiệm mới.

| Tập | Thư mục WAV | Số mẫu trong run |
| --- | --- | ---: |
| Train | `data/generated/train/` | 4.800 |
| Validation | `data/generated/validation/` | 1.592 |
| Test chính | `data/generated/test/` | 1.588 |
| Test bổ sung | `data/generated/supplemental/` | 1.112, giữ riêng khỏi test chính |

`data/manifests/dataset.csv` là nhãn và provenance đầy đủ; `data/manifests/test.csv` là bản lọc riêng split test để tiện dùng. Nguồn ESC/Speech Commands nằm trong `data/raw/esc/` và `data/raw/speech_commands/`. Archive tải tạm đã được xóa sau khi giải nén; URL và SHA-256 nguồn vẫn được giữ trong `data/manifests/download_provenance.json`. `reports/dataset_restoration_all.json` ghi số file và kiểm tra SHA-256 từng WAV khôi phục.

Đã khôi phục **9.092 WAV**, gồm 7.980 mẫu chính và 1.112 mẫu supplemental; mọi SHA-256 khớp manifest gốc. Dung lượng WAV đã trộn đo tại máy khoảng **1,46 GB**; nguồn đã giải nén khoảng **4,22 GB** sau khi dọn archive (GB thập phân).

Khôi phục từ nguồn đã giải nén ở máy hiện tại, không tải lại archive:

```powershell
..\.venv\Scripts\python.exe scripts/restore_dataset.py --config configs/trained_run.yaml --skip-download
```

Chỉ khôi phục test khi đã có raw sources:

```powershell
..\.venv\Scripts\python.exe scripts/restore_dataset.py --config configs/trained_run.yaml --skip-download --split test --no-supplemental
```

Trên máy chưa có nguồn đã giải nén, bỏ `--skip-download` để tải và xác thực archive gốc. Những file tồn tại có hash đúng được tái sử dụng. File khác hash hoặc archive khác SHA-256 bị từ chối; script giữ bản gốc và ghi lỗi, không coi mixture gần giống là dữ liệu đã train.

## Đánh giá với nhãn test

`inference.py --folder` chỉ dự đoán WAV, không đọc nhãn để tính accuracy. Để đánh giá đúng với nhãn trong manifest và model/ngưỡng đã khóa:

```powershell
..\.venv\Scripts\python.exe -m src.evaluate --run-dir runs/baseline --config configs/trained_run.yaml --split test --device cpu
```

Kết quả test từ run Colab đã tải: accuracy **93.77%**, precision **95.07%**, recall **92.32%**, F1 **93.67%**, trên 1.588 WAV tổng hợp. Bản gốc báo cáo nằm trong `runs/baseline/evaluation/test/metrics.json` và `report.md`. Đây là test được trộn của pipeline, không phải bản ghi âm ngoài đời; các biến thể cùng nền có tương quan. Chưa có đánh giá có hệ thống trên bản ghi thực tế.

Đã chạy lại toàn bộ 1.588 WAV test trên CPU tại máy: **0 dự đoán nhãn khác Colab**, accuracy/precision/recall/F1 và confusion matrix trùng bản gốc. Score có sai khác số học tối đa khoảng 0,00105 giữa hai môi trường; không dùng sai khác này để đổi threshold. Bản kiểm tra CPU nằm riêng ở `reports/cpu_test/`; CSV dễ đọc đã tạo tại `reports/test_inference.csv`. Báo cáo bàn giao và đối chiếu đầy đủ ở `reports/trained_model_delivery.json`.
