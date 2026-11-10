# Nhận diện âm schwa /ə/ trên CPU

Gói có đúng 4 file: `schwa.py`, `best.pt`, `requirements.txt`, `README.md`. Không cần `src`, dataset, notebook hay train. Checkpoint chứa cả bộ trích đặc trưng và bộ phân loại âm vị; không tải model khi chạy inference.

Test: 3,074 WAV; accuracy 87.48%, precision 52.73%, recall 40.80%, F1 0.4600. Speaker test: ksp. Ngưỡng 0.72 chọn trên validation.

Kết quả đo trên nhãn âm vị căn chỉnh tự động của CMU ARCTIC. Các cửa sổ trong cùng bản ghi có tương quan. Chưa kiểm chứng độ chính xác trên micro/giọng nói của bạn; chỉ số trên corpus không bảo đảm cùng độ chính xác trong mọi môi trường.

## Cài một lần

Mở PowerShell trong thư mục đã giải nén. Dùng Python 3.10 hoặc 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Inference

```powershell
.\.venv\Scripts\python.exe schwa.py predict --audio "E:\audio\recording.wav"
.\.venv\Scripts\python.exe schwa.py predict --audio "E:\audio\recording.wav" --output result.json
.\.venv\Scripts\python.exe schwa.py predict --inspect
```

`best.pt` tự nạp từ cùng thư mục với `schwa.py`. WAV được đổi sang mono 16 kHz, quét cửa sổ 0,5 giây với bước 50 ms. WAV ngắn được pad; WAV dài được quét hết. Model trả `detected`, `score`, `threshold`, `windows`; score cao nhất vượt ngưỡng nghĩa là phát hiện schwa. Những cửa sổ trong JSON là vùng dự đoán, chưa phải ranh giới chính xác của âm vị. WAV dài có nhiều cơ hội báo nhầm hơn. Model nhận diện âm /ə/; không bắt từ “yes” hay từ viết “schwa”.

Checkpoint dùng WAV2VEC2_BASE giữ 6 lớp transformer và bộ phân loại âm vị đã train mới. Model chạy CPU với 2 luồng; weights 99.3 MiB. Chế độ tiền xử lý mặc định dùng WAV gốc.

Thử xuất WAV lọc nhiễu nhẹ:

```powershell
.\.venv\Scripts\python.exe schwa.py denoise --audio input.wav --output cleaned.wav
```

Đây là soft spectral gating, giữ thời lượng. Lọc riêng thay đổi đầu vào; chưa có bảo đảm tăng độ chính xác của checkpoint.

## Nguồn model và dữ liệu

Nguồn corpus: [CMU ARCTIC](http://festvox.org/cmu_arctic/), John Kominek và Alan W. Black. Train dùng BDL/CLB/RMS/JMK/SLT. Copyright Carnegie Mellon University: BDL/JMK/SLT 2003, RMS 2004, CLB 2005. Gói không kèm WAV nguồn hoặc mã Festival; README/COPYING nguyên bản giữ tại dữ liệu dự án.

Backbone: [TorchAudio WAV2VEC2_BASE](https://docs.pytorch.org/audio/2.5.0/pipelines.html), nguồn LibriSpeech. Thay đổi: giữ 6 lớp transformer, lưu weights float16 và tính toán float32; thêm bộ phân loại riêng và đặc trưng phổ nếu checkpoint có cấu hình đó. Code inference này được viết mới, không phải sản phẩm chính thức của CMU hay Meta.

Giấy phép backbone được giữ trong README để không cần thư mục `licenses`:

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
