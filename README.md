# Model Âm Thanh: Schwa /ə/ và từ khóa “yes”

Hai pipeline PyTorch chạy trên CPU: `schwa_detection` phát hiện âm /ə/ trong lời nói tiếng Anh; `keyword_spotting` phát hiện từ “yes” trong WAV tối đa 5 giây. Mã nguồn, notebook và hướng dẫn được lưu trên [GitHub](https://github.com/trantrien1/Model_Am_Thanh). Audio, nhãn, manifest, cache, báo cáo và weights được lưu riêng trên Hugging Face:

| Nội dung | Kho Hugging Face |
| --- | --- |
| Dataset gốc/đã xử lý, cache đặc trưng, báo cáo thực nghiệm | [anonymos111/Model_Am_Thanh-data](https://huggingface.co/datasets/anonymos111/Model_Am_Thanh-data) |
| Pretrained, checkpoint, gói inference và bản chụp source của các run | [anonymos111/Model_Am_Thanh-models](https://huggingface.co/anonymos111/Model_Am_Thanh-models) |

`git clone` chỉ tải code. Các đường dẫn dữ liệu trong tài liệu con hoạt động sau khi khôi phục artifact tương ứng. Script tải xuống kiểm tra SHA-256 từng gói và khôi phục đúng cấu trúc thư mục; revision Hugging Face được khóa trong `hub-artifacts.json` sau khi xuất bản hoàn tất. Kho Hugging Face là bộ lưu trữ dự án, không dùng trực tiếp bằng `transformers.from_pretrained()`.

**Trạng thái lần xuất bản đầu:** kiểm tra trường `status` trong [hub-artifacts.json](hub-artifacts.json). `uploading` nghĩa là đang chuyển dữ liệu; `inference-ready` cho phép tải/chạy hai model nhưng bản sao lưu đầy đủ còn đang tải; `complete` nghĩa là đã tải đủ artifact. Tiến trình xuất bản sẽ cập nhật file này trên GitHub. Chạy `git pull` để nhận revision mới trước khi tải.

## 1. Clone và cài môi trường

Dùng **Python 3.10 hoặc 3.11**, khoảng 2 GB RAM trở lên cho inference Schwa (training cần nhiều hơn). Các lệnh dưới đây chạy trong PowerShell trên Windows:

```powershell
git clone https://github.com/trantrien1/Model_Am_Thanh.git
cd Model_Am_Thanh
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-hub.txt
```

Linux/macOS: tạo môi trường bằng `python3.10 -m venv .venv`, thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python` và dùng dấu `/` trong đường dẫn. Trên macOS cài PyTorch từ PyPI bằng `python -m pip install torch==2.5.1 torchaudio==2.5.1` thay cho CPU index của Windows/Linux.

Tải artifact từ kho công khai không cần HF token. Khi xuất bản artifact mới cần đăng nhập tài khoản có quyền ghi bằng `hf auth login`; không ghi token vào file hay commit lên Git.

## 2. Chạy model Schwa có sẵn

```powershell
.\.venv\Scripts\python.exe -m pip install -r schwa_detection/test/requirements.txt
.\.venv\Scripts\python.exe scripts/restore_hub.py --profile schwa
.\.venv\Scripts\python.exe schwa_detection/test/schwa.py predict --inspect
.\.venv\Scripts\python.exe schwa_detection/test/schwa.py predict --audio "D:/audio/recording.wav" --output result.json
```

Profile `schwa` chỉ tải checkpoint inference, khôi phục tại `schwa_detection/test/best.pt` (khoảng 339 MiB khi giải nén); code và requirements đã có từ GitHub. Không cần dataset hoặc train lại. Sau khi cài thư viện và tải weights, inference chạy offline trên CPU. WAV được đổi sang mono 16 kHz và quét hết timeline. Có thể xuất WAV lọc nhiễu bằng:

```powershell
.\.venv\Scripts\python.exe schwa_detection/test/schwa.py denoise --audio input.wav --output cleaned.wav
```

Gói triển khai dùng model V4 đã kiểm chứng: accuracy 90,69%, precision 71,80%, recall 72,62% trên 4.735 vùng 0,5 giây của 170 bản ghi L2-ARCTIC. Đây là kết quả theo vùng có ngữ cảnh bản ghi, chưa chứng minh chất lượng trên micro/giọng nói mới. Các checkpoint head V5 là thí nghiệm, cần encoder đi kèm và **chưa đạt mục tiêu precision/recall 88%**. Lượt cuối full-prior GRU đạt validation precision/recall 72,90%; test V5 chưa chấm. Xem [hướng dẫn Schwa](schwa_detection/README.md) và [gói inference](schwa_detection/test/README.md).

## 3. Chạy model phát hiện “yes” có sẵn

```powershell
.\.venv\Scripts\python.exe -m pip install -r keyword_spotting/test/requirements.txt
.\.venv\Scripts\python.exe scripts/restore_hub.py --profile keyword
.\.venv\Scripts\python.exe keyword_spotting/test/inference.py --inspect
.\.venv\Scripts\python.exe keyword_spotting/test/inference.py --audio keyword_spotting/test/examples/test_positive.wav
.\.venv\Scripts\python.exe keyword_spotting/test/inference.py --audio "D:/audio/my_voice.wav"
```

Profile này khôi phục `keyword_spotting/test/model/` (weights, ngưỡng, metadata) và `test/examples/` (audio cùng nguồn gốc mẫu). WAV dài tối đa 5 giây; file dài hơn bị từ chối. Xem [hướng dẫn gói yes](keyword_spotting/test/README.md).

## 4. Khôi phục dữ liệu để train/evaluate

```powershell
# Xem danh sách/dung lượng trước khi tải:
.\.venv\Scripts\python.exe scripts/restore_hub.py --profile all --list
# Hoặc chỉ tải hai model inference:
.\.venv\Scripts\python.exe scripts/restore_hub.py --profile inference
# Khôi phục toàn bộ dữ liệu, cache, báo cáo, pretrained và checkpoint:
.\.venv\Scripts\python.exe scripts/restore_hub.py --profile all
```

Toàn bộ artifact gốc khoảng **31 GiB (33 GB)**; nên có ít nhất **40 GB trống** khi khôi phục vào máy mới. Script xử lý từng gói, cần thêm dung lượng tạm cho gói lớn nhất. File có sẵn đúng nội dung được giữ; nếu file khác nội dung, script dừng để bạn di chuyển bản cũ trước khi tải lại. Môi trường `.venv`, file tạm và token không được sao lưu. Dữ liệu và checkpoint tải về được `.gitignore` loại khỏi Git.

### Train lại keyword spotting

Sau khi khôi phục `all`, từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m pip install -r keyword_spotting/requirements-colab.txt
Set-Location keyword_spotting
..\.venv\Scripts\python.exe -m src.train --config configs/default.yaml --run-dir runs/reproduce_yes_smoke --smoke
..\.venv\Scripts\python.exe -m src.train --config configs/default.yaml --run-dir runs/reproduce_yes
..\.venv\Scripts\python.exe -m src.evaluate --run-dir runs/reproduce_yes --split test --config configs/default.yaml
Set-Location ..
```

Dùng thư mục run mới để giữ checkpoint gốc. Nếu cần dựng dataset mới từ nguồn chính thức: chạy `python data/make_dataset.py` khi đang ở `keyword_spotting` (dùng Python của `.venv`). Chi tiết nguồn, split, resume và Colab tại [README keyword spotting](keyword_spotting/README.md). Notebook Colab giữ snapshot source bên trong và không tự đồng bộ các sửa đổi code ngoài notebook.

### Train lại Schwa trên V5 đã khóa

Sau khi khôi phục `all`, dùng dataset `schwa_detection/v5/data/` đã lưu và pretrained đi kèm:

```powershell
.\.venv\Scripts\python.exe -m pip install -r schwa_detection/requirements.txt
.\.venv\Scripts\python.exe schwa_detection/finetune_cpu.py --backbone schwa_detection/data/pretrained/phoneme_xlsr53_fp16_linear.pt --threads 2 prepare
.\.venv\Scripts\python.exe schwa_detection/finetune_cpu.py --backbone schwa_detection/data/pretrained/phoneme_xlsr53_fp16_linear.pt --threads 2 train --run-dir schwa_detection/runs/reproduce_finetune
```

Đây là một run fine-tune mới; không cam kết tái tạo metric của model V4 đã triển khai. Giữ validation/test và nhãn gốc, không chọn cấu hình theo test. Các script thí nghiệm có protocol/hash source đăng ký trước; xem báo cáo đã khôi phục và [PLAN Schwa](PLAN_schwa_codex.md) trước khi resume.

**Không chạy `l2_dataset.py --version v5` để tái tạo từ đầu trong bản clone này:** script đó phụ thuộc manifest V4 đã bị xóa khỏi workspace trước khi xuất bản. Bản V5 đầy đủ được lưu trên Hugging Face để phục hồi đúng split, nhãn và cache. Các lệnh tái dựng V4/V5 ở tài liệu lịch sử chỉ dùng khi có đủ đầu vào lịch sử.

## 5. Xuất bản cập nhật

```powershell
hf auth login
python scripts/publish_hub.py --dry-run
python scripts/publish_hub.py --namespace anonymos111
```

Script tạo gói theo nhóm tối đa khoảng 384 MiB dữ liệu đầu vào (file đơn lớn hơn nằm riêng), upload từng gói, xóa file tạm và lưu tiến độ trong `.hf-publish/`. Chạy lại để tiếp tục nếu bị ngắt. Khi hoàn tất, `hub-artifacts.json` khóa hai revision chứa manifest và SHA-256 mới; commit file này cùng thay đổi code lên GitHub. Giữ máy không train/ghi dữ liệu trong quá trình xuất bản.

Trên máy chủ dự án, `python scripts/finish_publication.py` thực hiện upload có retry, ưu tiên hai model inference, kiểm tra tải/chạy từ bản clone sạch rồi tự commit/push revision vào GitHub. Script này dành cho người quản lý kho có quyền ghi ở cả hai dịch vụ. Trạng thái và log lần xuất bản đầu nằm tại `.hf-publish/status.json`, `.hf-publish/publication.log`, `.hf-publish/publication-error.log`; đây là file cục bộ, không commit. Nếu máy tắt hoặc tiến trình dừng, chạy lại lệnh trên để tiếp tục từ các gói đã hoàn tất.

API upload/download dùng thư viện chính thức [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Nguồn và quyền sử dụng

Audio có nguồn ESC-50, Google Speech Commands v0.02, CMU ARCTIC và L2-ARCTIC; pretrained gồm wav2vec2/XLSR, WavLM và Whisper. Giữ các notice/README gốc trong artifact. ESC-50 và L2-ARCTIC có điều kiện phi thương mại; các bản trộn kế thừa điều kiện liên quan. Kho dự án không tự cấp giấy phép mới thay thế giấy phép nguồn. Chi tiết attribution, revision và giới hạn tại README của từng pipeline.
