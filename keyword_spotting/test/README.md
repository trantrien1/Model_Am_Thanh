# Chạy model phát hiện từ “yes”

Model nhận một file WAV và trả lời trong file có từ tiếng Anh **yes** hay không. Gói này có sẵn weights đã train, chạy bằng CPU.

## Cài đặt lần đầu trên Windows

Cài Python **3.10, 3.11 hoặc 3.12**. Giải nén toàn bộ gói, mở PowerShell trong thư mục `test` rồi chạy:

```powershell
.\setup.ps1
```

Script chọn Python phù hợp, tạo `.venv` và cài thư viện cùng PyTorch CPU. Cần Internet trong lần cài đầu; sau đó inference chạy tại máy.

Nếu bạn dùng các lệnh Python trực tiếp, ví dụ với Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy thử

```powershell
# Xem thông tin model:
.\infer.ps1 -Inspect

# Mẫu có từ yes:
.\infer.ps1 -Audio examples/test_positive.wav

# Mẫu có từ five, nhãn không có yes:
.\infer.ps1 -Audio examples/test_negative.wav
```

Kết quả đã kiểm tra trên CPU:

```text
test_positive.wav: PHÁT HIỆN từ 'yes' | score=0.999999 | ngưỡng=0.50
test_negative.wav: KHÔNG PHÁT HIỆN từ 'yes' | score=0.000002 | ngưỡng=0.50
```

## WAV của bạn

Chép WAV vào `audio_input/`, rồi chạy một file hoặc cả thư mục:

```powershell
.\infer.ps1 -Audio audio_input/my_voice.wav
.\infer.ps1 -Folder audio_input -Csv output/predictions.csv
```

Cũng có thể truyền đường dẫn tuyệt đối. File cần dài **tối đa 5 giây**; âm thanh stereo và sample rate khác sẽ được đổi thành mono 16 kHz, file ngắn được pad. File dài hơn 5 giây bị từ chối. Lỗi của từng file được ghi trong CSV.

Để chạy bằng Python và lấy JSON:

```powershell
.\.venv\Scripts\python.exe inference.py --audio examples/test_positive.wav --json
.\.venv\Scripts\python.exe inference.py --folder audio_input --csv output/predictions.csv
```

Để gọi trong chương trình Python của bạn:

```python
from inference import KeywordSpotter

spotter = KeywordSpotter(device="cpu")
result = spotter.predict("audio_input/my_voice.wav")
print(result["prediction"], result["probability"])
```

Nạp `KeywordSpotter` một lần rồi dùng lại cho nhiều WAV.

## Hiểu kết quả và model

- `prediction = 1`: phát hiện **yes**; `prediction = 0`: không phát hiện.
- `probability` là score trong [0, 1]. Score **≥ 0.50** cho prediction 1. Score chưa được hiệu chuẩn thành xác suất thống kê.
- Tiền xử lý: mono 16 kHz, 5 giây → log-mel 64 bins → normalization đã lưu trong checkpoint → CNN → sigmoid.
- CNN có ba khối convolution với 32/64/128 channels, global max pooling và đầu phân loại 64 units. Checkpoint được chọn ở epoch 11; ngưỡng 0.50 được chọn trên validation.
- Kết quả test tổng hợp của pipeline trên 1.588 WAV: accuracy **93,77%**, precision **95,07%**, recall **92,32%**, F1 **93,67%**. Chưa đánh giá có hệ thống trên bản ghi âm ngoài đời; các biến thể cùng nền có tương quan.

## Các file cần giữ

| Đường dẫn | Nội dung |
| --- | --- |
| `model/best.pt` | Weights, cấu hình audio/features/model, mean/std đã tính từ train |
| `model/threshold.json` | Ngưỡng và SHA-256 liên kết với đúng weights |
| `model/model_info.json` | Thông tin model và kết quả test để tham khảo |
| `inference.py`, `src/` | Mã inference và các module cần thiết |
| `setup.ps1`, `infer.ps1`, `requirements.txt` | Cài thư viện và chạy CPU |
| `examples/` | Hai WAV mẫu, nguồn gốc và kết quả mong đợi |
| `licenses/` | Giấy phép và attribution cho nguồn audio mẫu |

Giữ `best.pt` và `threshold.json` cùng nhau. Không tính lại normalization hay sửa ngưỡng cho từng WAV. Hai WAV ví dụ được trộn từ ESC-50 và Google Speech Commands; attribution nằm trong `examples/provenance.json` và các file giấy phép được giữ trong `licenses/`.
