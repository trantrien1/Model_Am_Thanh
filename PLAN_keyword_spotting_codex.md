# Kế hoạch cũ: phát hiện từ “yes” trong âm thanh ngắn

> **Plan hiện hành từ 09/10/2026:** [PLAN_schwa_codex.md](PLAN_schwa_codex.md) — nhận diện âm /ə/ (schwa), hỗ trợ lọc nhiễu và train trên CPU. Nội dung dưới đây được giữ làm tham khảo cho baseline `yes`; các notebook hiện có giữ nguyên.

## 1. Nhiệm vụ và phạm vi

Hãy triển khai và chạy một pipeline Python hoàn chỉnh: tải dữ liệu, trộn giọng đọc vào âm thanh nền, tạo nhãn nhị phân, huấn luyện, đánh giá và dự đoán trên WAV mới.

- Từ khóa mặc định: `yes`, cấu hình được bằng tham số `target_word`.
- Nhãn `1`: file chứa từ khóa; nhãn `0`: không chứa từ khóa, có thể có giọng nói từ khác.
- Dataset nền: ESC-50. Dataset giọng đọc: Google Speech Commands v0.02.
- File đầu vào cho model: mono, 16 kHz, 5 giây.
- Baseline: log-mel spectrogram + CNN nhỏ, huấn luyện bằng PyTorch.
- Chỉ làm phát hiện từ khóa ở cấp file. Không triển khai ASR, website, API, streaming hoặc deploy trong phạm vi này.
- Thực hiện công việc đến hết train và đánh giá nếu môi trường cho phép; không chỉ tạo code rồi dừng.
- Nếu mạng/tài nguyên chặn một giai đoạn: hoàn thành phần độc lập, ghi lỗi thật và cung cấp lệnh tiếp tục. Không bịa dữ liệu, checkpoint hay số liệu.

Đọc `AGENTS.md` nếu có. Giữ nguyên dữ liệu gốc và những thay đổi có sẵn của người dùng. Tự quyết định chi tiết triển khai thông thường; không yêu cầu xác nhận lại các bước đã nằm trong kế hoạch.

## 2. Nguồn dữ liệu và cách tải

### ESC-50

- Trang chính thức: https://github.com/karolpiczak/ESC-50
- ZIP: https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip
- Dùng `audio/*.wav` và `meta/esc50.csv`.
- Bộ gốc có 2.000 đoạn, mỗi đoạn 5 giây, 50 lớp, 5 fold.
- Ghi lại URL, thời điểm tải, SHA-256 archive và metadata.

### Google Speech Commands v0.02

- Trang dữ liệu: https://huggingface.co/datasets/google/speech_commands
- Archive WAV: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Dùng thư mục `yes/` làm nguồn từ khóa; các thư mục từ khác làm nguồn âm tính.
- Dùng `validation_list.txt`, `testing_list.txt` trong archive đầy đủ; còn lại là train.
- Không coi `_background_noise_` là một từ và không dùng nó làm giọng đọc.
- Tải bằng Python có timeout, retry hữu hạn, file tạm và kiểm tra archive. Không tải lại archive đã được xác thực.
- Giải nén an toàn: từ chối đường dẫn vượt khỏi thư mục đích và các liên kết nguy hiểm.
- Giữ thông tin giấy phép và cách trích dẫn trong README; dữ liệu sinh ra không mặc nhiên có giấy phép khác dữ liệu nguồn.

Cho phép `--esc-dir` và `--speech-dir` để dùng dữ liệu người dùng đã tải. Không buộc phải dùng Hugging Face API khi archive WAV đã đủ.

## 3. Cấu trúc project cần tạo

```text
keyword_spotting/
  README.md
  requirements.txt
  configs/default.yaml
  src/
    download.py
    prepare_sources.py
    build_dataset.py
    audit_dataset.py
    features.py
    dataset.py
    model.py
    train.py
    evaluate.py
    predict.py
  tests/
  data/
    raw/
    manifests/
    generated/
    features/
  runs/
  reports/
```

Được điều chỉnh cách tổ chức nếu repository hiện có yêu cầu khác. Không commit audio, cache hay checkpoint dung lượng lớn. Giữ các entry point CLI rõ ràng và dùng đường dẫn tương đối cấu hình được, không hardcode đường dẫn máy.

## 4. Khóa split trước khi sinh dữ liệu

| Tập | Nền ESC-50 | Giọng đọc Speech Commands |
| --- | --- | --- |
| Train | Fold 1, 2, 3 | Ngoài hai danh sách validation/test |
| Validation | Fold 4 | `validation_list.txt` |
| Test | Fold 5 | `testing_list.txt` |

Yêu cầu bắt buộc:

1. Đọc fold từ `esc50.csv`, không đoán từ tên file.
2. Mọi phiên bản của cùng nền nằm trong một split.
3. Không dùng giọng đọc validation/test để tạo train.
4. Lấy `speaker_id` từ phần tên trước `_nohash_`; kiểm tra giao nhau giữa các split.
5. Kiểm tra `background_id`, nguồn ESC, `speech_id`, `speaker_id` và hash nội dung nguồn không giao nhau giữa các split. Hash phải dựa trên biểu diễn PCM chuẩn hóa nhất quán nếu kiểm tra trùng nội dung audio.
6. Nếu phát hiện bản ghi trùng hoặc split mâu thuẫn, không tiếp tục train: cách ly bản ghi/nhóm liên quan theo quy tắc xác định, ghi báo cáo và audit lại. Không tự chia lại ngẫu nhiên toàn bộ dataset.
7. Sinh và lưu các source manifest trước khi trộn; mọi bước sau dùng chúng.

Số liệu mặc định sau khi tạo 4 phiên bản/nền, nếu không có file bị loại: train 4.800, validation 1.600, test 1.600. Số thực tế phải tính từ manifest và báo cáo phần chênh lệch.

## 5. Kiểm tra chất lượng nguồn

- Xác thực file đọc được, không rỗng, không NaN/Inf và không hoàn toàn im lặng.
- Kiểm tra sample rate, số kênh và thời lượng.
- Tạo danh sách nghe kiểm tra nền: ưu tiên các clip có khả năng chứa giọng nói nền. Không mặc định nhãn ESC đảm bảo tuyệt đối không có “yes”.
- Nếu chưa có người nghe xác nhận toàn bộ nền, ghi rõ giới hạn nhãn tổng hợp trong báo cáo. Dùng công cụ sàng lọc tự động chỉ để gợi ý kiểm tra, không coi kết quả đó là ground truth.
- Đối với giọng đọc, cắt im lặng dư ở hai đầu bằng energy/VAD nhẹ, chừa khoảng 20–50 ms. Giữ toàn bộ từ; không cắt cứng còn 0,5 giây.
- Ghi file bị loại và lý do. Nếu nguồn của một phiên bản lỗi, xử lý lại hoặc loại cả nhóm biến thể của nền để tránh mất cân bằng âm/dương một cách ngầm định.

## 6. Sinh dữ liệu có/không có từ khóa

### 6.1. Thiết kế nhãn

Với mỗi nền tạo 4 biến thể:

| Biến thể | Âm thanh chèn | Nhãn |
| --- | --- | ---: |
| P1 | Một bản ghi `yes` | 1 |
| P2 | Một bản ghi `yes` khác | 1 |
| N1 | Một từ khác, ví dụ `no`, `up`, `stop` | 0 |
| N2 | Từ khác; riêng 20% nền dùng nền sạch | 0 |

Chọn 20% nền theo seed trong từng split; ghi rõ `variant_type`. Cách này giữ tổng số và tỷ lệ nhãn 1:1, đồng thời tạo phần lớn mẫu âm tính có tiếng người và một phần mẫu không có giọng nói.

- Lấy mẫu từ âm tính gần đều giữa các từ, tránh chỉ sử dụng `no`.
- Cố gắng chọn hai người nói khác nhau cho P1/P2 khi đủ dữ liệu.
- Phân phối người nói, vị trí và SNR không được cố ý gắn với nhãn. Báo cáo thống kê để phát hiện tương quan ngoài ý muốn.
- Không dùng duy nhất một bản ghi `yes` cho toàn bộ dataset.
- Nhãn ghi trong CSV là nhãn mới của nhiệm vụ; lớp ESC gốc chỉ là metadata.

### 6.2. Trộn âm thanh

1. Chuyển nền và giọng đọc sang float mono, 16 kHz.
2. Chuẩn hóa độ dài nền thành đúng 80.000 mẫu; ghi lại nếu nguồn cần pad/crop.
3. Chọn `start_sample` ngẫu nhiên trong khoảng từ 0 đến `80000 - speech_length`.
4. Chọn SNR theo phân phối uniform 0–15 dB cho cả dương và âm có giọng nói.
5. Đo RMS nền tại vùng chèn, RMS giọng đọc trên phần có tiếng; với RMS hữu ích, đặt:

   `alpha = background_rms * 10**(snr_db / 20) / speech_rms`

   Quy ước: SNR ở đây là tỷ lệ năng lượng giọng nói / nền, không phải ngược lại.

6. Nếu vùng nền gần im lặng, dùng mức RMS giọng nói mục tiêu cấu hình được, ví dụ -25 dBFS; ghi `mix_level_mode=fallback_dbfs`. Khi đó SNR yêu cầu không được coi là SNR thực đạt.
7. Trộn `mixed[start:end] += alpha * speech`. Chỉ fade cực ngắn ở biên khi cần giảm click, không làm mất âm đầu/cuối từ.
8. Nếu vượt biên, giảm gain cả file để tránh clipping; lưu gain và mức thực đạt. Áp dụng cùng quy tắc cho mọi mẫu có chèn.
9. Lưu WAV PCM_16; mẫu nền sạch cũng được đọc/ghi cùng pipeline, tránh định dạng file trở thành dấu hiệu nhãn.

Giữ tất cả mẫu dài 5 giây. Không nối thêm từ làm file dương dài hơn file âm.

### 6.3. Manifest bắt buộc

Các cột tối thiểu:

`sample_id, path, label, split, variant_type, background_id, background_source_id, background_category, background_fold, speech_id, speaker_id, inserted_word, start_sample, speech_samples, snr_db_requested, snr_db_measured, mix_level_mode, speech_gain, output_gain, seed`

Mẫu nền sạch có các trường giọng nói để trống. Lưu cấu hình sinh dữ liệu, phiên bản code và hash manifest. Dùng RNG theo `seed + sample_id` với hàm hash ổn định; không dùng `hash()` Python vì thay đổi giữa các process.

## 7. Đặc trưng và CNN baseline

Dependencies dự kiến: `torch`, `numpy`, `scipy`, `soundfile`, `librosa`, `pandas`, `scikit-learn`, `PyYAML`, `matplotlib`. Chọn phiên bản tương thích môi trường, ghi lại phiên bản thực dùng. Có CUDA thì sử dụng; CPU vẫn phải chạy được.

### Log-mel

- Sample rate: 16.000 Hz.
- 64 mel bins; `n_fft=512`, cửa sổ Hann 400 mẫu (25 ms), `hop_length=160` (10 ms).
- Dải tần 20–8.000 Hz; log năng lượng với epsilon rõ ràng.
- Chọn và lưu `center` nhất quán, ví dụ `False`.
- Chuẩn hóa theo thống kê tập train; lưu mean/std theo mel bin, không tính trên validation/test.
- Cache đặc trưng khi hữu ích; cache key bao gồm cấu hình preprocessing và nguồn audio.

### Kiến trúc

`LogMel → 3 × [Conv2D 3×3 + BatchNorm + ReLU + MaxPool] → GlobalMaxPool → Linear 64 + ReLU + Dropout 0.3 → Linear 1`

- Số kênh các khối: 32, 64, 128; convolution padding để giữ kích thước trước pooling.
- Model trả logits; train với `BCEWithLogitsLoss`. Chỉ sigmoid khi lấy xác suất, không sigmoid hai lần.
- GlobalMaxPool là lựa chọn baseline để bắt sự kiện ngắn; không khẳng định đây là kiến trúc tối ưu.
- Nếu dùng augmentation, chỉ bật trên train: thay đổi gain toàn file nhẹ hoặc masking spectrogram nhẹ. Không crop mất từ rồi giữ nguyên nhãn 1.

### Hyperparameters mặc định

| Tham số | Giá trị |
| --- | --- |
| Seed | 42 |
| Batch size | 32 |
| Epoch tối đa | 30 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Early stopping | Validation loss, patience 5 |
| Chọn checkpoint | Validation loss thấp nhất |

Lưu loss và F1 validation mỗi epoch. Không dùng test để chọn epoch, architecture hay tham số. Có thể giảm batch size khi OOM và ghi lại thay đổi. Không đưa ra cam kết về accuracy hay thời gian train trước khi đo.

## 8. Chọn ngưỡng và đánh giá

1. Sau khi chọn checkpoint, tính xác suất trên validation.
2. Quét các ngưỡng 0,05–0,95, bước 0,01; chọn ngưỡng F1 cao nhất. Khi hòa, ưu tiên FPR thấp hơn, rồi ngưỡng cao hơn.
3. Lưu ngưỡng trong artifact. Có thể báo thêm kết quả ngưỡng 0,5 nhưng không thay lựa chọn dựa trên test.
4. Chạy test sau khi khóa model, preprocessing và ngưỡng.

Chỉ số bắt buộc: accuracy, precision, recall, F1, AUROC, average precision, confusion matrix, FPR và FNR. Ghi rõ nhãn dương là 1; nếu nhóm thiếu một nhãn, chỉ số không xác định phải là null kèm lý do, không tự đặt bằng 0.

Báo cáo theo nhóm: từ âm tính, nền sạch, loại nền, SNR 0–5 / 5–10 / 10–15 dB; nhóm fallback dBFS tách riêng. Thêm kết quả test ở mức file và số nền nguồn độc lập; các biến thể cùng nền không phải mẫu độc lập hoàn toàn.

Xuất:

- `metrics.json`, `predictions.csv`, confusion matrix và đường cong loss.
- Danh sách false positives/false negatives để nghe lại, có xác suất và nguồn.
- Báo cáo số file, số người nói, phân bố nhãn, SNR và kết quả audit split.

Kiểm tra bổ sung dùng nguồn chỉ thuộc test: các nền sạch còn lại và các mức giọng nhỏ hơn để khảo sát. Đặt tên đây là tập kiểm tra bổ sung, không trộn với test chính hay dùng để điều chỉnh mô hình.

Nếu người dùng chưa cung cấp bản ghi thực tế, tạo thư mục/hướng dẫn thu âm để kiểm tra sau và ghi “chưa đánh giá trên bản ghi thực tế”. Không tự tuyên bố khả năng phát hiện trong hội thoại dài, giọng Việt hay môi trường triển khai.

## 9. CLI, checkpoint và dự đoán

Agent phải hiện thực các lệnh tương đương sau, có `--help` và báo lỗi rõ ràng:

```bash
python -m src.download --config configs/default.yaml
python -m src.prepare_sources --config configs/default.yaml
python -m src.build_dataset --config configs/default.yaml
python -m src.audit_dataset --config configs/default.yaml
python -m src.train --config configs/default.yaml
python -m src.evaluate --run-dir runs/baseline --split test
python -m src.predict --run-dir runs/baseline --audio path/to/sample.wav
```

Checkpoint phải chứa model weights, model config, feature config, thống kê chuẩn hóa, target word, mapping nhãn, seed và thông tin nguồn. Lưu ngưỡng đã chọn trong run directory. Inference phải dùng đúng các artifact này.

- Với file không quá 5 giây: pad về 5 giây, không kéo giãn thời gian.
- Với file dài hơn 5 giây: mặc định từ chối và giải thích giới hạn, không âm thầm lấy 5 giây đầu. Sliding window là phần mở rộng ngoài baseline.
- Output: `target_word`, `probability`, `threshold`, `prediction`, `duration_seconds`.
- Xác suất là điểm model, chưa được đảm bảo hiệu chuẩn thống kê.

README có các lệnh từ môi trường mới đến predict, dung lượng thực đo, CPU/GPU thực dùng, cách dùng dữ liệu đã tải và cách chạy trên Colab nếu máy thiếu tài nguyên.

## 10. Kiểm thử và thứ tự thực hiện

### Giai đoạn A — Chuẩn bị

- [ ] Đọc repository và hướng dẫn local; tạo môi trường, config và CLI.
- [ ] Tải hoặc nhận đường dẫn hai dataset, xác thực nguồn.
- [ ] Sinh source manifests, kiểm tra người nói và nguồn trùng.

### Giai đoạn B — Sinh dữ liệu

- [ ] Hiện thực mixer và nhãn, giữ cùng pipeline cho hai lớp.
- [ ] Test quan trọng: resampling, không cắt từ, độ dài 80.000 mẫu, vị trí chèn hợp lệ, RMS/SNR, không clipping, fallback nền im lặng và tái lập bằng seed.
- [ ] Sinh đầy đủ dataset, audit nhãn/split, xuất mẫu nghe kiểm tra.

### Giai đoạn C — Train

- [ ] Hiện thực log-mel, train-only normalization và CNN.
- [ ] Chạy smoke test ít batch để kiểm tra shapes, loss hữu hạn và lưu/nạp checkpoint.
- [ ] Chạy train đầy đủ; lưu best checkpoint và lịch sử.
- [ ] Chọn ngưỡng trên validation, lưu artifact.

### Giai đoạn D — Đánh giá và bàn giao

- [ ] Chạy test với model/ngưỡng đã khóa và xuất các báo cáo.
- [ ] Kiểm tra predict có preprocessing trùng với evaluate.
- [ ] Cung cấp ví dụ WAV dương/âm từ test và kết quả thực đo.
- [ ] Hoàn thiện README, report, môi trường và danh sách giới hạn.

Audit rò rỉ phải chạy trước full training. Smoke test dùng subset được đánh dấu, không báo kết quả subset như kết quả dataset đầy đủ. Không lặp lại training vô hạn để đạt một accuracy mong muốn.

## 11. Tiêu chí hoàn thành

- Pipeline chạy được từ dữ liệu nguồn đến dự đoán.
- Dataset có cả nhãn dương, âm có tiếng người và âm nền sạch.
- Split theo nền và người nói độc lập, có audit kiểm chứng.
- Có checkpoint đã train thật, feature stats và ngưỡng validation.
- Có metrics test thực đo, dự đoán từng file và phân tích lỗi.
- Có manifest đủ để tái tạo dữ liệu và README đủ để chạy lại.
- Nếu bước nào chưa chạy được, bàn giao trạng thái thực tế, lỗi và lệnh tiếp tục; không đánh dấu hoàn thành bước đó.

## 12. Tài liệu tham khảo

- ESC-50: https://github.com/karolpiczak/ESC-50
- Speech Commands: https://huggingface.co/datasets/google/speech_commands
- Paper Speech Commands: https://arxiv.org/abs/1804.03209
- Tutorial keyword recognition, tham khảo phương pháp; baseline kế hoạch này dùng PyTorch: https://www.tensorflow.org/tutorials/audio/simple_audio
- Google keyword spotting examples: https://github.com/google-research/google-research/tree/master/kws_streaming

Đây là kế hoạch thực nghiệm dữ liệu tổng hợp. Mục tiêu là kiểm chứng phát hiện từ khóa qua các giọng đọc và nền chưa thấy; kết quả đó không thay thế đánh giá trên bản ghi thực tế.
