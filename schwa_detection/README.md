# Nhận diện schwa /ə/ trên CPU

Trạng thái hiện tại: đã dừng theo yêu cầu người dùng sau lượt train thứ 30. Lượt cuối dùng đầy đủ phân bố âm vị suy ra tự động từ audio, train CPU 2 luồng trong 666,4 giây, kết thúc sau 12 epoch; checkpoint tốt nhất ở epoch 2 đạt precision 72,90%, recall 72,90%, accuracy 93,41% trên toàn bộ 4.884 vùng validation. Mục tiêu precision và recall trên 88% chưa đạt. Đã trích và kiểm tra đủ 9.702 vùng train / 4.884 vùng validation; bốn kiểm tra kỹ thuật và bước cập nhật gradient trên dữ liệu thật đều qua. Checkpoint head tại `runs/pr88_v5_full_prior_binary_gru/best.pt`, kết quả đầy đủ tại `runs/pr88_v5_full_prior_binary_gru/training_summary.json`. Checkpoint này cần encoder và runtime tương ứng, chưa phải gói inference độc lập. Gói đã kiểm chứng tại `test/` và `test.zip` vẫn dùng model trước đó. Toàn bộ 5.157 vùng test V5 chưa chấm; notebook giữ nguyên. Không còn tiến trình train của lượt này và không khởi động lượt mới. Các cập nhật bên dưới là lịch sử triển khai.

Tr?ng th?i m?i nh?t: word identity ?? ho?n t?t 14 epoch (precision 70,07% / recall 70,54%); BPE c?ng khai ?? ho?n t?t 12 epoch (68,13% / 68,01%), ch?a ??t 88%. ?? chu?n b?/audit 48 ??c tr?ng ph?/cao ??/n?ng l??ng t? WAV cho ?? 9.702 v?ng train v? 4.884 validation; hai ki?m tra t?n hi?u c? cao ??/bi?n ?? bi?t tr??c ?? qua. B?o c?o `reports/v5_raw_prosody_feature_audit.json`. Thi?t k? binary GRU k?t h?p XLSR, CTC lexical v? c?c ??c tr?ng th? ?? ch?t t?i `reports/precision_recall_88_raw_prosody_classifier_design.json`; classifier m?i ch?a tri?n khai/train, source ph?i ??ng k? tr??c d? ?o?n validation. ?? ho?n t?t 25 l??t train; m?c ti?u 88% v?n m?, test V5 ch?a c? ??c tr?ng/d? ?o?n m?i, g?i inference hi?n c? gi? nguy?n. C?c ?o?n ph?a d??i l? l?ch s? tri?n khai.

Tr?ng th?i m?i nh?t: head word identity ?? ho?n t?t 14 epoch, t?t nh?t precision 70,07% / recall 70,54%, ch?a ??t 88%. L??t m?i `runs/pr88_v5_whisper_text_binary_gru` ?? ??ng k? source tr??c d? ?o?n validation, d?ng token embedding 512 chi?u c?ng khai c?a Whisper ?? bi?u di?n c? t? ch?a th?y trong train; ASR ??u v?o v?n suy ra t? audio, kh?ng d?ng transcript/m?c ?m v? th? c?ng l?m ??c tr?ng. Ba ki?m tra subword/c?n th?i gian/t? ch?i test v? forward/backward tr?n hai b?n ghi TRAIN th?t ?? qua. GRU m?i ?ang chu?n b? ?? cache r?i train CPU 2 thread, gi? nguy?n 9.702 v?ng train / 4.884 validation, t?i ?a 40 epoch/patience 10. Ch?a c? b?ng ch?ng ch?t l??ng m?i; test V5 ch?a d?ng. Tr?ng th?i `reports/whisper_text_trial_state.json`.

Head h?c word identity ?? tri?n khai v? ??ng k? source tr??c d? ?o?n validation t?i `runs/pr88_v5_whisper_word_binary_gru/registered_protocol.json`. Vocabulary 777 t? ???c t?i d?ng ch? t? audio TRAIN; word ID l?u int32, word embedding 32 chi?u ???c h?c c?ng head GRU. B?n ki?m tra vocabulary/c?n th?i gian/gradient/resume optimizer-RNG ?? qua, forward/backward tr?n hai b?n ghi TRAIN th?t h?u h?n v? c?p nh?t embedding. L??t train CPU 2 thread ?ang ch?y, t?i ?a 40 epoch/patience 10, gi? nguy?n t?t c? 9.702 v?ng train / 4.884 validation. Tr?ng th?i `reports/whisper_word_trial_state.json`; m?c ti?u tr?n 88% v?n ch?a ???c x?c nh?n, test V5 ch?a d?ng.

H??ng ti?p theo sau khi Whisper acoustic kh?ng c?i thi?n: ki?m tra ch? tr?n TRAIN cho th?y prior t? ?i?n kh?ng bi?u di?n ?? bi?n th? AH0/AX trong c?c t? Whisper suy ra (v? d? OF). ??y l? th?ng k? c?n t? t? ??ng g?n ??ng, kh?ng ph?i metric ch?t l??ng. ?? t?o vocabulary 777 t? ch? t? audio TRAIN v? ch?t thi?t k? head h?c word embedding 32 chi?u t?i `reports/precision_recall_88_word_identity_design.json`. Implementation v? source freeze c?n ph?i ho?n t?t tr??c m?i d? ?o?n validation c?a ph??ng ?n m?i; nh?n v? to?n b? v?ng test gi? nguy?n.

L??t Whisper acoustic ?? ho?n t?t 15 epoch sau khi tr?ch/audit ?? 1.075 b?n ghi train v? 186 validation. Checkpoint t?t nh?t epoch 5 c? validation precision 67,79% / recall 68,01%, ch?a c?i thi?n c?c ph??ng ?n tr??c. Head m?i d?ng 1.936 chi?u (XLSR + t? Whisper + encoder ?m thanh Whisper), CPU 2 thread, to?n b? 9.702 v?ng train / 4.884 validation; source/data gi? ??ng protocol ?? ??ng k?. B?o c?o `runs/pr88_v5_whisper_acoustic_binary_gru/training_summary.json` v? `reports/whisper_acoustic_trial_state.json`. ?? ho?n t?t 23 l??t train; m?c ti?u c? precision/recall tr?n 88% v?n ch?a ??t. Test V5 ch?a ch?m, g?i inference hi?n c? gi? nguy?n. C?c ?o?n ph?a d??i l? l?ch s? tri?n khai.

Tr?ng th?i m?i nh?t: kh?i ph?c tu?n t? tr?n CPU ?? ho?n t?t. GRU to?n b?n ghi train 18 epoch, validation t?t nh?t precision 70,78% / recall 71,38%; Whisper GRU train 20 epoch, t?t nh?t 70,05% / 69,70%. ?? ki?m tra ?? cache audio-only 1.075 b?n ghi train v? 186 validation. Ph?p k?t h?p 4 head ?? ??ng k? tr??c khi t?nh score, t?i l?p ??ng s? li?u t?ng th?nh ph?n, ??t validation 73,22% / 72,73% tr?n ?? 4.884 v?ng; b?o c?o `runs/pr88_v5_validation_ensemble/validation_comparison.json`. ?? ho?n t?t 22 l??t train; m?c ti?u c? precision v? recall tr?n 88% v?n ch?a ??t. Test V5 ch?a ch?m, g?i inference ?? ki?m ch?ng t?i `test.zip` v?n gi? nguy?n. C?c `best.pt` trong run m?i l? checkpoint head, c?n pack v?i encoder c?ng khai tr??c khi inference ??c l?p. Tr?ng th?i t?i `reports/cpu_recovery_status.json`; c?c ?o?n c?p nh?t ph?a d??i l? l?ch s? tri?n khai.

Runtime Whisper đã được thêm tại `whisper_inference.py` (pack/predict/export đúng bốn file), tích hợp vào `lexical_inference.py freeze/evaluate` để chấm nguyên các vùng gốc từ WAV và khóa trước test. Ba kiểm tra kỹ thuật mới và mười kiểm tra hồi quy đã qua; mã standalone biên dịch/nạp được và khớp package Whisper đã pin. Bằng chứng tại `reports/whisper_inference_engineering_checks.json`, có ghi lại lỗi gọi nhầm tên module kiểm tra và lần chạy đúng sau đó. Các kiểm tra mới dùng fixture giả lập; chưa có checkpoint Whisper Schwa đầy đủ để xác minh parity hoặc chất lượng. Trích đặc trưng vẫn chạy, train Whisper đang chờ, test V5 chưa dùng và gói bàn giao chưa thay.

Đã đăng ký lượt train mới `runs/pr88_v5_whisper_binary_gru` bằng `whisper_train_cpu.py`: head GRU nhị phân khởi tạo mới, encoder XLSR/Whisper công khai cố định, giữ nguyên 9.702 vùng train và 4.884 vùng validation. Kiểm tra forward/backward trên hai bản ghi TRAIN thật và hai kiểm tra tiếp tục optimizer/RNG từ checkpoint đã qua (`reports/whisper_training_cpu_feasibility.json`). Tiến trình đang chờ trích đủ đặc trưng rồi tự train; chưa có precision/recall của lượt này. Cấu hình đã chốt ở `runs/pr88_v5_whisper_binary_gru/registered_protocol.json`; model cuối cần runtime validation, khóa trước test và kiểm tra gói inference độc lập.

Kết quả ASR trên đủ 186 bản ghi validation: Whisper Base English đạt WER 11,76%, CTC cũ 18,85%, Whisper tốt hơn ở cả sáu speaker với cùng cách chuẩn hóa chữ/số. Bảng đầy đủ tại `reports/whisper_validation_asr_comparison.json`. Đã thêm `whisper_lexical.py` để căn các từ/mốc thời gian suy ra từ audio vào tín hiệu âm vị công khai; bốn kiểm tra và thử nghiệm trên hai bản ghi TRAIN đã qua. Đang trích đặc trưng cho 1.075 bản ghi train và 186 validation trên CPU, không dùng transcript/nhãn/timing thủ công làm đặc trưng. Protocol tại `reports/precision_recall_88_whisper_lexical_protocol.json`. Đây chưa phải kết quả precision/recall Schwa; classifier dùng Whisper chưa train, test V5 chưa dùng, gói bàn giao chưa thay.

Runtime cho head toàn bản ghi: `recording_head_inference.py` hỗ trợ pack, predict và xuất gói độc lập đúng 4 file; `lexical_inference.py freeze/evaluate` hỗ trợ khóa model/ngưỡng/pooling/mã inference rồi chấm đủ vùng gốc. Hai kiểm tra runtime mới và tám kiểm tra hồi quy đã qua, gồm audio ngắn/stereo/dài và chạy Python độc lập từ ZIP. Báo cáo `reports/recording_head_inference_engineering_checks.json`; các kiểm tra dùng encoder nhỏ nên chưa thay thế kiểm tra checkpoint đầy đủ và chất lượng thực tế. Lượt train thật vẫn ở `runs/pr88_v5_recording_gru_suitcase`; chưa thay `test.zip`.

Cập nhật lượt cải thiện: WavLM MLP và GRU đã kết thúc, precision/recall tốt nhất lần lượt 66,72% / 67,17% và 67,56% / 68,01%, chưa đạt 88%. Đã bổ sung dữ liệu lời nói tự nhiên có nhãn thủ công của 10 speaker train tại `training_suitcase/`: 686,25 giây, 1.084 vùng cân bằng, không trùng dữ liệu held-out; các manifest V5 vẫn giữ nguyên. `recording_head_cpu.py` đang train head GRU dùng toàn bộ bản ghi trên encoder XLSR công khai, khởi tạo head mới, gồm 1.085 bản ghi train. Bốn kiểm tra timeline/context/pooling/resume đã qua và forward/backward trên hai bản ghi TRAIN thật hữu hạn. Protocol chốt trước validation tại `reports/precision_recall_88_recording_gru_protocol.json`; run `runs/pr88_v5_recording_gru_suitcase`. Chưa có metric chất lượng của lượt mới; test V5 và gói bàn giao chưa thay đổi.

Kiểm tra checkpoint WavLM đầy đủ epoch 3 đã qua: pipeline và gói ZIP chạy Python độc lập cho score giống nhau trên WAV TRAIN gốc, ngắn, stereo và dài 19,68 giây. Runtime dùng phép tính CPU với MKLDNN tắt để tránh sai khác giữa tiến trình; gói vẫn có đúng 4 file. Báo cáo `runs/pr88_v5_wavlm_phone_mlp/runtime_native_probe/parity_report.json`. Kiểm tra này không chứng minh precision/recall; checkpoint cuối được chọn vẫn cần kiểm tra lại và đánh giá validation bằng runtime thực tế. Lượt CTC đã kết thúc 12 epoch, tốt nhất precision 70,08% / recall 70,20%; WavLM đang train tiếp. Test V5 chưa được dùng.

Cập nhật kiểm tra WavLM: đặc trưng của 1.261 bản ghi train/validation đã trích xong; hai head MLP/GRU chờ lượt CTC kết thúc để train tuần tự. `wavlm_inference.py` hỗ trợ pack, predict và xuất gói đúng 4 file; `lexical_inference.py` hỗ trợ khóa model và đánh giá runtime cho WavLM. Hai kiểm tra inference đã qua, gồm chạy Python độc lập từ ZIP với audio ngắn/stereo/dài cho cả hai head và từ chối báo cáo có mã inference hoặc pooling khác. Sáu kiểm tra hồi quy CTC/lexical cũng qua. Đây là kiểm tra kỹ thuật với model nhỏ, chưa chứng minh chất lượng model WavLM sau train. Kiểm tra encoder thật trên ba bản ghi TRAIN có sai khác feature tối đa 0,00048828125 so với cache; cần đánh giá chất lượng bằng runtime thực tế trước khi khóa test. Báo cáo: `reports/wavlm_inference_engineering_checks.json` và `reports/wavlm_actual_feature_diagnostics.json`.

**Lượt cải thiện tiếp theo đang chạy:** yêu cầu mới là cả precision và recall trên khoảng 88%. Gói `test.zip` hiện tại vẫn là bản 71,80% precision / 72,62% recall dưới đây; chưa công bố đạt yêu cầu mới. Dữ liệu V5 có 19.743 vùng, dùng bản ghi test mới của cùng 6 speaker chưa có trong train/validation. Các speaker này đã được đánh giá ở lượt trước; bản ghi và prompt test mới được chốt trước khi train. Head mới phải khởi tạo từ đầu vì prompt test V5 từng thuộc train V4. Chi tiết ở `reports/precision_recall_88_protocol.json`.

Đã hoàn tất 17 lượt train V5; precision/recall validation tốt nhất đạt 72,99% / 73,23%, chưa đạt yêu cầu 88%. Lượt `runs/pr88_v5_tail2_ctc_joint` đang train trên CPU: cập nhật hai lớp cuối và bộ giải mã âm vị bằng CTC, đồng thời học nhãn âm vị theo frame và Schwa theo vùng 0,5 giây. Khởi tạo từ encoder công khai, không dùng weights đã train ở các lượt trước. Protocol và trạng thái ở `reports/precision_recall_88_ctc_adaptation_protocol.json` và `reports/improvement_progress.json`; test V5 chưa được chấm. Dataset nằm ở `v5/data/`, gồm train 9.702, validation 4.884, test 5.157 vùng. Các thư mục V2/V3/V4 không còn trong workspace; dữ liệu và cache V5 độc lập vẫn còn.

Lệnh tạo lại kiểm tra V5 hiện có và train một run fine-tune mới trên CPU (2 luồng, giữ 22 lớp đầu cố định):

```powershell
.\.venv\Scripts\python.exe schwa_detection\l2_dataset.py --version v5
.\.venv\Scripts\python.exe schwa_detection\finetune_cpu.py --backbone schwa_detection\data\pretrained\phoneme_xlsr53_fp16_linear.pt --threads 2 prepare
.\.venv\Scripts\python.exe schwa_detection\finetune_cpu.py --backbone schwa_detection\data\pretrained\phoneme_xlsr53_fp16_linear.pt --threads 2 train --run-dir schwa_detection\runs\my_finetune
```

Train tạo `best.pt` cho head, `adapted_encoder.pt` cho encoder và báo cáo trong run. Hai file weights này dùng cùng nhau trong pipeline; khi chọn được model đạt yêu cầu mới sẽ pack thành một `best.pt` chứa đủ weights cho gói inference. Notebook cũ được giữ nguyên.

Lượt CTC hiện tại dùng `--ctc-weight 1 --window-loss 1 --epochs 15 --warmup-epochs 1 --patience 5 --tail-lr 0.00003 --accumulate 4`. AH0/AX là Schwa; AH1/AH2 và ER/AXR giữ âm tính. Đoạn không rõ nhãn được tách khỏi target CTC; nhãn và danh sách vùng đánh giá giữ nguyên. Đã kiểm tra ánh xạ target trên 1.075 bản ghi train và forward/backward với encoder thật trên hai bản ghi; đây là kiểm tra kỹ thuật, chưa phải kết quả chất lượng.

`last.pt` của lượt CTC lưu hai lớp được cập nhật, decoder, head, optimizer và trạng thái ngẫu nhiên sau mỗi epoch. Nếu quá trình đã dừng, chạy lại đúng lệnh/cấu hình cũ với `--resume` để tiếp tục. Kiểm tra mô phỏng gián đoạn xác nhận weights tiếp tục giống chính xác lượt chạy liên tục. Bước đánh giá runtime hỗ trợ `lexical_inference.py freeze/evaluate` cho cả model lexical và model một encoder; chấm test cần báo cáo model/ngưỡng đã khóa trước test. Kiểm tra với encoder nhỏ đã xác nhận gói ZIP đúng 4 file và inference độc lập trên WAV gốc, ngắn, stereo 44,1 kHz, dài trên 10 giây. Cần kiểm tra lại với model đầy đủ sau train. Gói `test.zip` hiện tại vẫn dùng model bàn giao trước đó.

Đã kiểm tra thêm checkpoint đầy đủ của epoch 1: pipeline và Python độc lập từ gói 4 file cho cùng dự đoán trên bản ghi TRAIN 4,92 giây, clip 0,2 giây, stereo 44,1 kHz và audio TRAIN lặp dài 19,68 giây; sai khác score bằng 0. Báo cáo tại `runs/pr88_v5_tail2_ctc_joint/runtime_epoch1_probe/parity_report.json`. So với cache prefix dùng khi train, score có sai khác nhỏ tối đa 0,00000602 trên bản ghi kiểm tra; vì vậy chất lượng model cuối cần xác nhận bằng runtime triển khai trên validation trước khi khóa test. Kiểm tra epoch 1 không thay thế việc kiểm tra checkpoint cuối được chọn.

Đã chấm runtime checkpoint epoch 1 đủ 4.884 vùng validation, kết quả trùng khi train. Lượt CTC đang tiếp tục; hai lượt mới với WavLM Base+ đã đăng ký tại `reports/precision_recall_88_wavlm_protocol.json`, trích đặc trưng bằng `wavlm_train_cpu.py` rồi train tuần tự sau CTC. Encoder công khai có 94.381.168 tham số, lưu khoảng 249 MB; đặc trưng là trung bình lớp 6/9/12 và trừ trung bình từng kênh của chính bản ghi, không dùng câu chuẩn hoặc nhãn làm feature. Ba kiểm tra kỹ thuật đã qua, gồm tiếp tục train giống chính xác sau gián đoạn. Chưa có kết quả chất lượng WavLM; chưa thay gói `test.zip` và chưa chấm test V5.

**Đã tạo dataset, train thật trên CPU và kiểm thử model mới.** Weight hoàn chỉnh: [runs/schwa_context/best.pt](runs/schwa_context/best.pt). Gói gửi người chỉ inference: [test.zip](test.zip), gồm đúng 4 file `schwa.py`, `best.pt`, `requirements.txt`, `README.md`.

Test L2-ARCTIC V4 đạt **90,69% accuracy, precision 71,80%, recall 72,62%, F1 0,7221**. Đây là kết quả trên **4.735 vùng 0,5 giây của 170 bản ghi gốc**, thuộc 6 người nói và các câu đọc không có trong train/validation. Encoder dùng ngữ cảnh bản ghi gốc khi chấm từng vùng; **90,69% không phải accuracy trên clip 0,5 giây đã cắt rời hoặc accuracy quyết định có/không có schwa của cả file WAV**. Giọng và micro riêng chưa được kiểm chứng.

| Chỉ số test | Kết quả |
| --- | ---: |
| Accuracy | 90,69% |
| Precision | 71,80% |
| Recall | 72,62% |
| F1 | 0,7221 |
| AUROC | 0,9179 |
| Accuracy nếu luôn đoán âm tính | 83,34% |
| Phát hiện đúng / báo nhầm / bỏ sót | 573 / 225 / 216 |

Người nói test: NJS, TLV, TNI, TXHC, YKWK, ZHAA; accuracy theo người nói từ 84,60% đến 94,01%, giữ tất cả các vùng test đã chốt. Epoch 42, ngưỡng 0,44 và pooling 2 frame được chọn bằng validation F1 rồi khóa trước khi đánh giá test. Báo cáo đầy đủ, nhầm/bỏ sót của runtime bàn giao: [runs/schwa_context/evaluation/runtime_test/](runs/schwa_context/evaluation/runtime_test/). Không chỉnh model/ngưỡng theo kết quả test này.

Lần chấm đầu đạt 90,83%; kiểm tra đóng gói phát hiện sai khác CPU flash attention nhỏ bị khuếch đại qua các lớp int8. Đã cố định math attention cho encoder lượng tử hóa, rồi tính lại toàn bộ test với **cùng weights/ngưỡng/pooling**, được 90,69%. Đây là kiểm tra lại sau sửa tính toán, không phải test độc lập mới. Không dùng kết quả test để lựa chọn cách sửa. Bằng chứng và fingerprint ở [reports/runtime_parity_fix.json](reports/runtime_parity_fix.json); mọi cache mới có recipe riêng, không dùng lẫn đặc trưng trước sửa.

## Chạy ngay trên máy hiện tại

Mở PowerShell tại thư mục `Model_Am_Thanh`:

```powershell
.\.venv\Scripts\python.exe schwa_detection\test\schwa.py predict --audio "E:\audio\recording.wav"
.\.venv\Scripts\python.exe schwa_detection\test\schwa.py predict --audio "E:\audio\recording.wav" --output result.json
.\.venv\Scripts\python.exe schwa_detection\test\schwa.py predict --inspect
```

`test/schwa.py` tự nạp `test/best.pt`; không cần `src`, dataset hoặc notebook. Người nhận giải nén `test.zip`, cài thư viện theo README bên trong, rồi chạy `python schwa.py predict --audio recording.wav`. Giấy phép model nằm trong README, không có thư mục `licenses`.

WAV được đổi sang mono 16 kHz. Encoder phân tích bản ghi, chia đoạn tối đa 10 giây nếu cần; bộ phân loại chấm từng vùng 0,5 giây, quét cách nhau 50 ms. WAV ngắn được pad. Kết quả gồm `detected`, `score`, `threshold`, `windows`; `windows` là vùng dự đoán theo thời gian, chưa phải ranh giới âm vị chính xác. `score` là điểm cao nhất, chưa được hiệu chuẩn thành xác suất đúng; WAV dài có nhiều cơ hội báo nhầm hơn.

Model nhận diện âm /ə/ trong lời nói tiếng Anh. Backbone XLSR-53 đã được pretrain nhận diện âm vị, giữ cố định 24 lớp và lượng tử hóa linear int8; bộ phân loại 56.377 tham số được train mới. Checkpoint khoảng 338,7 MiB chứa đầy đủ encoder/head/ngưỡng; inference chạy offline trên CPU, 2 luồng. Kiểm tra gói độc lập: [reports/inference_package_context_check.json](reports/inference_package_context_check.json).

## Dataset và train lại trên CPU

Phần dưới mô tả lượt V4 đã bàn giao; thư mục V4 hiện không còn trong workspace. Dataset đang train để cải thiện precision/recall là V5 và các lệnh hiện tại nằm ở đầu README. Lượt V4 trước đó đã tạo/audit **19.023 vùng**: train 9.404 (4.702 dương / 4.702 âm), validation 4.884, test 4.735. Tách cả người nói và câu đọc giữa các tập.

Nhãn dùng âm thực sự được người chú giải đánh dấu trong `manual_events.perceived_raw`: AH0/AX/AX0 → schwa; AH1/AH2 và ER/AXR giữ riêng. Các vùng chồng lên đoạn chưa rõ nhãn bị loại trước khi chốt dataset. V4 sửa đoạn im lặng cuối thiếu tối đa một mẫu vì resampling; không sửa nhãn schwa theo dự đoán model.

Các lệnh sau tạo lại dữ liệu, trích đặc trưng và train một run mới; notebook cũ vẫn được giữ nguyên:

```powershell
.\.venv\Scripts\python.exe schwa_detection\l2_dataset.py --version v4
if (-not (Test-Path -LiteralPath "schwa_detection\data\pretrained\phoneme_xlsr53_int8.pt")) {
    .\.venv\Scripts\python.exe schwa_detection\frozen_phoneme.py --output schwa_detection\data\pretrained\phoneme_xlsr53_int8.pt --threads 2
}
.\.venv\Scripts\python.exe schwa_detection\improve.py extract --dataset v4 --splits train validation --context recording --backbone schwa_detection\data\pretrained\phoneme_xlsr53_int8.pt --threads 2
.\.venv\Scripts\python.exe schwa_detection\improve.py train --dataset v4 --context recording --backbone schwa_detection\data\pretrained\phoneme_xlsr53_int8.pt --run-dir schwa_detection\runs\my_context_run --task phones --hidden 128 --dropout 0.4 --window-loss 0.25 --lr 0.0005 --epochs 60 --patience 7 --seed 42 --threads 2
.\.venv\Scripts\python.exe schwa_detection\improve.py pack --model schwa_detection\runs\my_context_run\best.pt --backbone schwa_detection\data\pretrained\phoneme_xlsr53_int8.pt --output schwa_detection\runs\my_context_run\inference.pt
```

Run đã chọn train 49 epoch, early stopping, giữ epoch 42. Train head trên đặc trưng đã có mất khoảng 149 giây trên máy hiện tại; tải pretrained và trích đặc trưng tốn thêm thời gian. Đây là transfer learning với encoder giữ cố định. Model head được chọn/ngưỡng được chốt trên validation, không dùng test. Sau khi đã xem test V4, cần test độc lập mới nếu tiếp tục chỉnh model rồi muốn công bố một kết quả độc lập khác.

Nguồn dataset: [L2-ARCTIC v5.0](https://psi.engr.tamu.edu/l2-arctic-corpus/), bản 16 kHz [chikingsley/l2-arctic-manual-v5.0-16k](https://huggingface.co/datasets/chikingsley/l2-arctic-manual-v5.0-16k), revision `b7261545bc4e4a02910e9c85c659eb866e00db7a`, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Nguồn pretrained: [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft), revision `2c733782da5604684829819a5eb744c193fe9398`, Apache 2.0. Gói inference không kèm audio corpus.

## Lọc nhiễu và các bản cũ

```powershell
.\.venv\Scripts\python.exe schwa_detection\test\schwa.py denoise --audio input.wav --output cleaned.wav
```

Lọc nhiễu dùng soft spectral gating, giữ thời lượng và lưu WAV mới. Model mới mặc định dùng WAV gốc; lọc riêng thay đổi đầu vào, chưa được chứng minh tăng accuracy. Chưa kiểm thử model mới trên các bản ghi nhiễu thực tế của bạn.

`schwa_detection/schwa.py` vẫn là CLI CNN cũ với weight `runs/baseline/best.pt`. `test_improved.zip` là bản cũ đã test 87,48% trên KSP; đây là corpus/điều kiện khác, không phải phép so sánh cùng test với kết quả 90,69% mới. Tài liệu lịch sử ở `runs/baseline/README_history.md`, gói CNN cũ được giữ ở `runs/baseline/inference_backup/`. Model `yes` và các notebook trong `keyword_spotting/` được giữ nguyên.

Checksum và lựa chọn model trước test: [reports/model_selection_recording.json](reports/model_selection_recording.json). Protocol ngữ cảnh: [reports/recording_context_training_protocol.json](reports/recording_context_training_protocol.json). Audit hoàn tất và fingerprint bàn giao: [reports/completion_audit.json](reports/completion_audit.json).
