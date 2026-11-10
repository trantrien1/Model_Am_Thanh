# Plan mới: phát hiện âm schwa /ə/ và hỗ trợ lọc nhiễu trên CPU

Trạng thái hiện tại: đã dừng theo yêu cầu người dùng sau lượt train thứ 30. Lượt cuối train CPU 2 luồng trong 666,4 giây, kết thúc sau 12 epoch; checkpoint tốt nhất ở epoch 2 đạt precision 72,90%, recall 72,90%, accuracy 93,41% trên toàn bộ 4.884 vùng validation. Mục tiêu precision và recall trên 88% chưa đạt. Checkpoint head tại `schwa_detection/runs/pr88_v5_full_prior_binary_gru/best.pt`, báo cáo tại `schwa_detection/runs/pr88_v5_full_prior_binary_gru/training_summary.json`. Checkpoint này chưa có runtime inference độc lập. Gói đã kiểm chứng tại `schwa_detection/test/` và `schwa_detection/test.zip` vẫn dùng model trước đó. Toàn bộ 5.157 vùng test V5 chưa chấm; notebook giữ nguyên. Không còn tiến trình train của lượt này và không khởi động lượt mới. Các cập nhật bên dưới là lịch sử triển khai.

Tr?ng th?i m?i nh?t: word identity ?? ho?n t?t 14 epoch (precision 70,07% / recall 70,54%); BPE c?ng khai ?? ho?n t?t 12 epoch (68,13% / 68,01%), ch?a ??t 88%. ?? chu?n b?/audit 48 ??c tr?ng ph?/cao ??/n?ng l??ng t? WAV cho ?? 9.702 v?ng train v? 4.884 validation; hai ki?m tra t?n hi?u c? cao ??/bi?n ?? bi?t tr??c ?? qua. B?o c?o `schwa_detection/reports/v5_raw_prosody_feature_audit.json`. Thi?t k? binary GRU k?t h?p XLSR, CTC lexical v? c?c ??c tr?ng th? ?? ch?t t?i `schwa_detection/reports/precision_recall_88_raw_prosody_classifier_design.json`; classifier m?i ch?a tri?n khai/train, source ph?i ??ng k? tr??c d? ?o?n validation. ?? ho?n t?t 25 l??t train; m?c ti?u 88% v?n m?, test V5 ch?a c? ??c tr?ng/d? ?o?n m?i, g?i inference hi?n c? gi? nguy?n. C?c ?o?n ph?a d??i l? l?ch s? tri?n khai.

Tr?ng th?i m?i nh?t: head word identity ?? ho?n t?t 14 epoch, t?t nh?t precision 70,07% / recall 70,54%, ch?a ??t 88%. L??t m?i `schwa_detection/runs/pr88_v5_whisper_text_binary_gru` ?? ??ng k? source tr??c d? ?o?n validation, d?ng token embedding 512 chi?u c?ng khai c?a Whisper ?? bi?u di?n c? t? ch?a th?y trong train; ASR ??u v?o v?n suy ra t? audio, kh?ng d?ng transcript/m?c ?m v? th? c?ng l?m ??c tr?ng. Ba ki?m tra subword/c?n th?i gian/t? ch?i test v? forward/backward tr?n hai b?n ghi TRAIN th?t ?? qua. GRU m?i ?ang chu?n b? ?? cache r?i train CPU 2 thread, gi? nguy?n 9.702 v?ng train / 4.884 validation, t?i ?a 40 epoch/patience 10. Ch?a c? b?ng ch?ng ch?t l??ng m?i; test V5 ch?a d?ng. Tr?ng th?i `schwa_detection/reports/whisper_text_trial_state.json`.

Head h?c word identity ?? tri?n khai v? ??ng k? source tr??c d? ?o?n validation t?i `schwa_detection/runs/pr88_v5_whisper_word_binary_gru/registered_protocol.json`. Vocabulary 777 t? ???c t?i d?ng ch? t? audio TRAIN; word ID l?u int32, word embedding 32 chi?u ???c h?c c?ng head GRU. B?n ki?m tra vocabulary/c?n th?i gian/gradient/resume optimizer-RNG ?? qua, forward/backward tr?n hai b?n ghi TRAIN th?t h?u h?n v? c?p nh?t embedding. L??t train CPU 2 thread ?ang ch?y, t?i ?a 40 epoch/patience 10, gi? nguy?n t?t c? 9.702 v?ng train / 4.884 validation. Tr?ng th?i `schwa_detection/reports/whisper_word_trial_state.json`; m?c ti?u tr?n 88% v?n ch?a ???c x?c nh?n, test V5 ch?a d?ng.

H??ng ti?p theo sau khi Whisper acoustic kh?ng c?i thi?n: ki?m tra ch? tr?n TRAIN cho th?y prior t? ?i?n kh?ng bi?u di?n ?? bi?n th? AH0/AX trong c?c t? Whisper suy ra (v? d? OF). ??y l? th?ng k? c?n t? t? ??ng g?n ??ng, kh?ng ph?i metric ch?t l??ng. ?? t?o vocabulary 777 t? ch? t? audio TRAIN v? ch?t thi?t k? head h?c word embedding 32 chi?u t?i `schwa_detection/reports/precision_recall_88_word_identity_design.json`. Implementation v? source freeze c?n ph?i ho?n t?t tr??c m?i d? ?o?n validation c?a ph??ng ?n m?i; nh?n v? to?n b? v?ng test gi? nguy?n.

L??t Whisper acoustic ?? ho?n t?t 15 epoch sau khi tr?ch/audit ?? 1.075 b?n ghi train v? 186 validation. Checkpoint t?t nh?t epoch 5 c? validation precision 67,79% / recall 68,01%, ch?a c?i thi?n c?c ph??ng ?n tr??c. Head m?i d?ng 1.936 chi?u (XLSR + t? Whisper + encoder ?m thanh Whisper), CPU 2 thread, to?n b? 9.702 v?ng train / 4.884 validation; source/data gi? ??ng protocol ?? ??ng k?. B?o c?o `schwa_detection/runs/pr88_v5_whisper_acoustic_binary_gru/training_summary.json` v? `schwa_detection/reports/whisper_acoustic_trial_state.json`. ?? ho?n t?t 23 l??t train; m?c ti?u c? precision/recall tr?n 88% v?n ch?a ??t. Test V5 ch?a ch?m, g?i inference hi?n c? gi? nguy?n. C?c ?o?n ph?a d??i l? l?ch s? tri?n khai.

Tr?ng th?i m?i nh?t: kh?i ph?c tu?n t? tr?n CPU ?? ho?n t?t. GRU to?n b?n ghi train 18 epoch, validation t?t nh?t precision 70,78% / recall 71,38%; Whisper GRU train 20 epoch, t?t nh?t 70,05% / 69,70%. ?? ki?m tra ?? cache audio-only 1.075 b?n ghi train v? 186 validation. Ph?p k?t h?p 4 head ?? ??ng k? tr??c khi t?nh score, t?i l?p ??ng s? li?u t?ng th?nh ph?n, ??t validation 73,22% / 72,73% tr?n ?? 4.884 v?ng; b?o c?o `schwa_detection/runs/pr88_v5_validation_ensemble/validation_comparison.json`. ?? ho?n t?t 22 l??t train; m?c ti?u c? precision v? recall tr?n 88% v?n ch?a ??t. Test V5 ch?a ch?m, g?i inference ?? ki?m ch?ng t?i `schwa_detection/test.zip` v?n gi? nguy?n. C?c `best.pt` trong run m?i l? checkpoint head, c?n pack v?i encoder c?ng khai tr??c khi inference ??c l?p. Tr?ng th?i t?i `schwa_detection/reports/cpu_recovery_status.json`; c?c ?o?n c?p nh?t ph?a d??i l? l?ch s? tri?n khai.

Đã triển khai runtime Whisper Schwa pack/predict/export đúng bốn file và tích hợp khóa/đánh giá từ WAV theo toàn bộ vùng gốc. Ba kiểm tra kỹ thuật mới, mười kiểm tra hồi quy và nạp mã standalone đã qua; báo cáo `schwa_detection/reports/whisper_inference_engineering_checks.json`. Đây là kiểm tra fixture giả lập, không phải bằng chứng chất lượng hoặc parity của checkpoint cuối. Khi train Whisper xong, cần xác minh model đầy đủ giữa pipeline/gói độc lập, runtime validation rồi khóa model/ngưỡng/pooling/mã trước khi chấm toàn bộ test.

Lượt Schwa có ngữ cảnh Whisper đã được đăng ký tại `schwa_detection/runs/pr88_v5_whisper_binary_gru/registered_protocol.json`: binary GRU mới, CPU hai thread, 40 epoch tối đa / patience 10, encoder XLSR và Whisper công khai cố định. Forward/backward trên hai bản ghi TRAIN thật và hai kiểm tra resume optimizer/RNG đã qua; tiến trình chờ cache đầy đủ rồi tự train. Không thu hẹp validation/test và không dùng reference transcript/timing làm input. Chưa có kết quả chất lượng của lượt này; chỉ công nhận đạt mục tiêu sau actual runtime validation, khóa model/ngưỡng/pooling trước test, chấm toàn bộ test gốc và xác minh gói CPU đúng bốn file.

Đã so sánh ASR trên đủ 186 bản ghi validation: Whisper Base English WER 11,76% so với CTC 18,85%, tốt hơn ở cả sáu speaker với cùng chuẩn hóa chữ/số. Bảng kết quả tại `schwa_detection/reports/whisper_validation_asr_comparison.json`. Bước tiếp theo đang chạy: `whisper_lexical.py` trích ngữ cảnh từ audio cho toàn bộ 1.075 bản ghi train và 186 validation, căn từ bằng tín hiệu âm vị công khai, giữ riêng AH0/AH1/ER0; bốn kiểm tra và hai bản ghi TRAIN thật đã qua. Protocol `schwa_detection/reports/precision_recall_88_whisper_lexical_protocol.json` giữ nguyên mọi vùng validation/test, cấm dùng transcript/nhãn/timing thủ công làm đặc trưng. Classifier Schwa dùng Whisper chưa train; mục tiêu cả precision và recall trên 88% vẫn chưa đạt. GRU toàn bản ghi vẫn đang train, test V5 chưa dùng và gói bàn giao chưa thay.

Đã triển khai runtime cho GRU toàn bản ghi tại `schwa_detection/recording_head_inference.py`, gồm pack/predict/export đúng 4 file. Bộ khóa/đánh giá runtime chung hỗ trợ backend mới, kiểm tra mã inference, model/ngưỡng/pooling và danh sách mọi vùng test trước dự đoán. Hai kiểm tra mới và tám kiểm tra hồi quy đã qua với encoder nhỏ; báo cáo `schwa_detection/reports/recording_head_inference_engineering_checks.json`. Chất lượng và gói checkpoint đầy đủ vẫn cần xác nhận sau train; test V5 và gói bàn giao chưa thay đổi.

Lượt cải thiện tiếp theo đã chạy: `schwa_detection/recording_head_cpu.py` dùng GRU trên toàn bộ bản ghi, encoder XLSR công khai cố định và head khởi tạo mới. Dữ liệu train bổ sung gồm 10 bản ghi lời nói tự nhiên có nhãn thủ công của các speaker train, 686,25 giây / 1.084 vùng, tại `schwa_detection/training_suitcase/`; audit không có trùng với validation/test V5 và xác nhận manifest V5 giữ nguyên. Tổng train 1.085 bản ghi; validation/test vẫn là 4.884/5.157 vùng gốc. Bốn kiểm tra kỹ thuật đã qua, forward/backward trên hai bản ghi TRAIN thật hữu hạn. Protocol trước validation: `schwa_detection/reports/precision_recall_88_recording_gru_protocol.json`. Hai lượt WavLM đã hoàn thành nhưng chưa đạt 88%; test V5 chưa được chấm, gói bàn giao chưa thay đổi. Kết quả mới chỉ được công nhận sau runtime validation, khóa model/ngưỡng/pooling, chấm toàn bộ test và kiểm tra gói CPU đúng 4 file.

Kiểm tra runtime WavLM cập nhật: checkpoint đầy đủ epoch 3 đã khớp score giữa pipeline và ZIP độc lập trên bốn dạng audio TRAIN, gồm audio dài 19,68 giây, khi tắt MKLDNN tại inference. Báo cáo `schwa_detection/runs/pr88_v5_wavlm_phone_mlp/runtime_native_probe/parity_report.json`; checkpoint cuối vẫn phải kiểm tra lại. Lượt CTC hoàn thành 12 epoch, tốt nhất precision 70,08% / recall 70,20%, chưa đạt yêu cầu 88%. WavLM MLP đang train; GRU sẽ chạy sau. Test V5 chưa được chấm.

Cập nhật kiểm tra WavLM: đã trích xong đặc trưng 1.261 bản ghi train/validation; MLP và GRU sẽ train tuần tự khi CTC kết thúc. Đã triển khai gói inference đúng 4 file, kiểm tra ZIP chạy Python độc lập trên CPU và khóa model/mã inference/pooling trước khi chấm test. Hai kiểm tra WavLM và sáu kiểm tra hồi quy CTC/lexical đã qua; chưa có bằng chứng chất lượng WavLM sau train. Encoder thật có sai khác feature nhỏ với cache trên ba bản ghi TRAIN, nên bắt buộc xác nhận validation bằng runtime thực tế trước khi khóa và chấm test V5. Báo cáo trong `schwa_detection/reports/wavlm_inference_engineering_checks.json` và `schwa_detection/reports/wavlm_actual_feature_diagnostics.json`.

Cập nhật theo yêu cầu ngày 09/10/2026. Plan này thay thế nhiệm vụ phát hiện từ `yes`. Các notebook hiện có được giữ nguyên; pipeline mới dùng file Python và CPU.

Trạng thái ngày 10/10/2026: đã tạo/audit dataset V5 gồm 19.743 vùng và hoàn tất 17 lượt train; precision/recall validation tốt nhất 72,99% / 73,23%, chưa đạt mục tiêu mới trên 88%. Lượt `schwa_detection/runs/pr88_v5_tail2_ctc_joint` đang train trên CPU với nhãn âm vị CTC, loss theo frame và loss Schwa theo vùng; khởi tạo từ encoder công khai, cập nhật hai lớp cuối và decoder. Đã kiểm tra target trên 1.075 bản ghi train, forward/backward trên hai bản ghi thật, tiếp tục train chính xác sau gián đoạn và đóng gói 4 file với encoder nhỏ. Protocol chốt trước validation tại `schwa_detection/reports/precision_recall_88_ctc_adaptation_protocol.json`; kết quả/runtime của encoder đầy đủ cần kiểm tra sau train. Test V5 chưa được chấm. Gói `schwa_detection/test.zip` giữ model đã kiểm chứng trước đó: accuracy 90,69%, precision 71,80%, recall 72,62%; chưa coi là đạt yêu cầu 88%.

Checkpoint đầy đủ epoch 1 đã qua kiểm tra CPU pipeline/gói độc lập trên bốn dạng audio TRAIN, gồm WAV dài 19,68 giây, sai khác score bằng 0; báo cáo `schwa_detection/runs/pr88_v5_tail2_ctc_joint/runtime_epoch1_probe/parity_report.json`. Kiểm tra này chưa phải bằng chứng precision/recall 88% và cần lặp lại cho model cuối. Đang chấm validation bằng runtime triển khai; khóa model/ngưỡng/recipe trước khi chấm toàn bộ 5.157 vùng test V5, giữ nguyên nhãn, người nói và ngữ cảnh bản ghi.

Runtime validation của checkpoint CTC epoch 1 đã chấm đủ 4.884 vùng/186 bản ghi, trùng precision/recall khi train. Phân tích model trước đó cho thấy 91/161 báo nhầm cách AX hơn 0,5 giây, bỏ sót có ở nhiều độ dài âm; ASR hiện dùng có WER validation 18,83%. Nguồn manual đã tải không còn bản ghi train hợp lệ bổ sung khi giữ 12 speaker train và 120 prompt validation/test đã chốt. Vì vậy chuẩn bị thử encoder công khai WavLM Base+ của TorchAudio, không thay nhãn/split. Encoder chuẩn bị 249.285.100 byte, đã chạy CPU trên WAV TRAIN 4,92 giây; ba kiểm tra timeline, chuẩn hóa theo bản ghi và train/resume chính xác đã qua. Hai head phone MLP/GRU khởi tạo mới được chốt trước validation tại `schwa_detection/reports/precision_recall_88_wavlm_protocol.json`; đang trích đặc trưng train/validation, sẽ train lần lượt sau khi worker CTC hiện tại kết thúc để giới hạn RAM. Chưa có metric chất lượng WavLM và chưa dùng test V5.

## 1. Mục tiêu

- Nhận diện sự hiện diện của âm **/ə/** trong WAV lời nói tiếng Anh; target là âm vị, không phải một từ viết là “schwa”.
- Có tùy chọn lọc nhiễu tự động trước nhận diện và xuất WAV sau xử lý để nghe đối chiếu.
- Giả định làm việc hiện tại: nhận diện /ə/ là mục tiêu chính, lọc nhiễu là tiền xử lý. Câu hỏi phân biệt hướng này với bài toán chỉ khử nhiễu đã gửi cho người dùng; cập nhật phần liên quan khi có câu trả lời.
- Giữ riêng model, dataset và kết quả `yes` hiện có để tham khảo. Không đổi nhãn hoặc tên weights cũ thành model schwa.
- Không tạo notebook mới. Mã train/inference chạy bằng môi trường CPU `.venv` của workspace.

Video tham chiếu: [The most common sound /ə/: in English](https://www.youtube.com/watch?v=dd3xBSJ0vZE). Đã đọc mô tả video và [transcript chính thức của BBC được video dẫn tới](https://downloads.bbc.co.uk/learningenglish/features/the_pronunciation_lounge/pron_lounge_episode_1_transcript_.pdf); chưa phát/nghe trực tiếp video bằng công cụ trong phiên này. Nội dung cho thấy cần nhận diện âm giảm trong âm tiết không nhấn và lời nói nối tiếp, thay vì dựa vào một tên từ cố định.

## 2. Dữ liệu và nhãn

Nguồn khởi đầu: **CMU ARCTIC**, vì có WAV và các file nhãn âm vị theo thời gian. [Bài báo của nhóm tác giả](https://www.cs.cmu.edu/~awb/papers/ssw5/arctic.pdf) mô tả corpus và phone labels.

Đã kiểm tra trực tiếp:

- [File nhãn BDL arctic_a0001.lab](http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/lab/arctic_a0001.lab) có nhãn `ax` riêng với các nhãn như `ah`, `er`, `pau`.
- [README của BDL](http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/README) ghi thư mục `lab/` chứa nhãn tự động.

Yêu cầu khi triển khai:

1. Xác nhận bảng phone của từng nguồn; bước đầu dùng `ax` làm nhãn schwa. Không tự gộp `ah`, `er`, `axr` hay mọi nguyên âm không nhấn thành /ə/.
2. Đọc thời gian kết thúc các phone trong `.lab`, suy ra bắt đầu từ phone trước. Kiểm tra thứ tự, khoảng thời gian dương và không vượt thời lượng WAV.
3. Mỗi speaker chỉ ở một split. Chia danh sách speaker trước khi lấy đoạn âm thanh; lưu manifest split và seed.
4. Chia cả prompt/utterance ID để câu giống nhau đọc bởi speaker khác không xuất hiện xuyên train/validation/test. Mọi đoạn và biến thể của cùng recording ở cùng split.
5. Thu thập tối thiểu 5 speaker cho prototype: ít nhất 3 train, 1 validation, 1 test; xác minh nguồn thực tế có đủ nhãn và hai lớp trước train.
6. File âm tính cần có giọng nói và các âm dễ nhầm, đặc biệt nguyên âm khác; bổ sung pause/nền để khảo sát. Không chỉ học phân biệt tiếng người với im lặng.
7. Nhãn corpus là nhãn căn chỉnh tự động, có thể sai ranh giới và phát âm. Xuất danh sách nghe kiểm tra phân tầng theo speaker/phone/SNR; báo rõ số mẫu đã được người nghe xác nhận.
8. Dữ liệu Speech Commands hiện có chỉ có nhãn từ, nên không dùng trực tiếp thư mục từ làm ground truth schwa. Video hướng dẫn cũng không đủ làm dataset train nhiều speaker.

Tải WAV + `.lab` cần thiết, hạn chế tải các artifact tổng hợp giọng nói không sử dụng. Mỗi download có timeout/retry hữu hạn, file tạm và SHA-256. Tái sử dụng file đã xác thực; giữ attribution và giấy phép ở dữ liệu nguồn.

## 3. Tạo mẫu và audit

- WAV float mono 16 kHz; giữ thông tin sample rate/kênh/thời lượng gốc.
- Cửa sổ model khởi đầu **0,5 giây**, tương đương 8.000 mẫu. Pad khi thiếu; giữ timeline gốc. Không áp dụng giới hạn 5 giây của model `yes` cho toàn bộ WAV mới.
- Mẫu dương lấy quanh phone `ax`, bảo đảm toàn bộ đoạn target được giữ trong cửa sổ.
- Mẫu âm chỉ được chọn nếu cửa sổ không giao với bất kỳ đoạn `ax` nào trong recording. Phone nằm cạnh schwa không tự động là mẫu âm.
- Không kéo giãn hoặc cắt mất âm target rồi giữ nhãn dương. Loại và ghi lý do cho phone dài bất thường, nhãn lỗi hay audio không đọc được.
- Lấy mẫu âm theo phone/speaker để tránh một loại âm hoặc một người nói chi phối. Tỷ lệ 1:1 có thể dùng cho train; evaluation cần báo cả phân bố thật và cách lấy mẫu.
- Manifest gồm `sample_id, recording_id, speaker_id, prompt_id, split, audio_path, label, target_phone, phone_start, phone_end, window_start, window_end, source_sha256, seed`.
- Audit giao nhau speaker, prompt, recording và PCM hash; chuẩn hóa PCM trước khi so trùng. Audit phải qua trước full train.

Khởi đầu train trên dữ liệu sạch. Sau khi baseline chạy đúng, thêm dữ liệu trộn nền theo SNR và seed; nền có giọng nói cần kiểm tra/cách ly để không tạo target ngoài nhãn. Có thể tái sử dụng nguồn ESC-50 đã giải nén, với fold nền 1/2/3 train, 4 validation, 5 test.

## 4. Lọc nhiễu

- Triển khai tiền xử lý nhẹ trên CPU với hai chế độ `off` và `auto`; khởi đầu bằng soft spectral gating, không thêm mạng khử nhiễu lớn.
- `auto` ước lượng noise floor từ các frame năng lượng thấp, làm trơn mask và giữ minimum gain để hạn chế xóa giọng nhỏ. Nếu không có vùng nền đáng tin cậy thì giảm mức xử lý hoặc bỏ qua, và ghi trạng thái.
- Giữ nguyên số mẫu và vị trí thời gian; không xóa khoảng im lặng làm lệch annotation.
- Schwa thường yếu/ngắn: cần nghe đối chiếu raw/filtered, đo việc giữ target và kiểm tra lỗi tăng sau lọc nhiễu.
- Chọn chế độ/mức xử lý bằng validation, lưu trong checkpoint; dùng cùng công thức và tham số khi train/evaluate/inference. Không chọn chế độ dựa trên test.
- Báo riêng kết quả sạch, nhiễu chưa lọc và nhiễu đã lọc. Dữ liệu noisy/clean ghép cặp có thể dùng để đo thay đổi SNR/MSE, nhưng không thay thế đánh giá khả năng giữ schwa.

Nếu người dùng chọn mục tiêu **chỉ khử nhiễu**, thay nhánh phân loại bằng pipeline phục hồi WAV; khi đó dữ liệu noisy/clean ghép cặp và metric phục hồi audio là trọng tâm. Không train classifier schwa để giả định nó là model khử nhiễu.

## 5. Model nhỏ và cấu hình CPU

Baseline đã triển khai và train thử:

`0,5 s audio → LogMel 32 bins → CNN 16/32/64 channels → global max pooling → Linear 32 → logit`

Sau run đầu, thử thêm pooling chỉ theo thời gian và giữ 4 dải tần đầu ra trước lớp Linear. Phiên bản này có **31.777 tham số**, nhằm giữ thông tin vị trí tần số để phân biệt nguyên âm. So sánh global pooling, giữ dải tần với dữ liệu gốc, và giữ dải tần với lọc nhiễu `auto` trên validation; chọn run theo F1 tại ngưỡng validation, hòa thì validation loss thấp hơn. Chỉ run đã chọn được dùng đánh giá test; không chỉnh model theo kết quả test.

- STFT: `n_fft=512`, Hann 400, hop 160, `center=False`, Slaney mel 20–8.000 Hz, log power epsilon `1e-10`. Cửa sổ 0,5 s có 47 frame với cấu hình này.
- Mean/std từng mel bin tính từ train và lưu trong checkpoint.
- Dropout 0,2; BCEWithLogitsLoss; seed 42; Adam lr 0,001.
- CPU threads khởi đầu 2, batch 16, DataLoader workers 0 trên Windows. Đo lại trên dataset thật trước điều chỉnh.
- Tối đa 20 epoch, early stopping theo validation loss với patience 5; chọn best bằng validation loss.
- Cache đặc trưng để tránh decode/STFT lại mỗi epoch. Cache key có audio hash, cửa sổ, feature config và denoise config.

Máy kiểm tra: Intel i5-12500H, 12 lõi/16 luồng, RAM 16.869.548.032 bytes. Benchmark CNN **25.633 tham số**, batch 16, đã đo 10 batch forward/backward/Adam sau warmup:

| Threads | Thời gian 10 batch |
| --- | ---: |
| 2 | 0,176 giây |
| 4 | 0,210 giây |
| 8 | 0,188 giây |

Chi tiết ở [cpu_benchmark.json](schwa_detection/reports/cpu_benchmark.json). Đây là đo tốc độ trên tensor ngẫu nhiên, không phải train/evaluate schwa; chưa tính decode audio, features, validation và I/O. Kết quả hỗ trợ lựa chọn CNN nhỏ chạy CPU, chưa đủ để cam kết tổng thời gian train hoặc accuracy.

## 6. Đánh giá và inference

1. Smoke test riêng: xác minh shape, loss hữu hạn, normalization train-only, checkpoint nạp được; không báo metric smoke như kết quả đầy đủ.
2. Train thật, lưu best/last, lịch sử, môi trường, danh sách speaker/prompt và fingerprint dataset.
3. Chọn ngưỡng trên validation, quét 0,05–0,95 bước 0,01 theo F1; hòa thì FPR thấp hơn rồi ngưỡng cao hơn.
4. Chạy test sau khi khóa weights, feature/denoise config và threshold.
5. Báo precision/recall/F1, accuracy, AUROC, AP, confusion matrix, FPR/FNR; nhóm thiếu điều kiện tính metric trả null kèm lý do. Báo theo speaker, phone âm tính, SNR và chế độ nhiễu.
6. Với WAV mới, quét cửa sổ 0,5 s theo bước 50 ms, xử lý từng batch trên CPU. Output có `target_symbol, score, threshold, detected, windows`.
7. `windows` là các vùng cửa sổ dự đoán, không tự coi là ranh giới âm vị chính xác. Muốn định vị từng phone cần thêm nhãn/evaluation theo thời gian.
8. Đánh giá bổ sung trên WAV người dùng thu riêng, có người nghe kiểm tra schwa. Không suy ra chất lượng thực tế chỉ từ corpus đọc câu hoặc hai WAV ví dụ.

Checkpoint mới phải chứa weights, model/feature/denoise config, normalization, target phone mapping, threshold và metadata nguồn. Score chưa được hiệu chuẩn thống kê.

## 7. Mã nguồn và cách bàn giao

Ưu tiên ít file, tổ chức dự kiến:

```text
schwa_detection/
  schwa.py             # CLI download/prepare/train/evaluate/predict/denoise
  README.md
  requirements.txt
  data/                # nguồn và manifest khi tạo dữ liệu
  runs/                # checkpoint và báo cáo khi train
  reports/cpu_benchmark.json
```

Các CLI cần triển khai, có `--help` và lỗi rõ ràng:

```powershell
..\.venv\Scripts\python.exe schwa.py download
..\.venv\Scripts\python.exe schwa.py prepare
..\.venv\Scripts\python.exe schwa.py train --device cpu --threads 2
..\.venv\Scripts\python.exe schwa.py evaluate --split test
..\.venv\Scripts\python.exe schwa.py predict --audio path/to/sample.wav
..\.venv\Scripts\python.exe schwa.py denoise --audio input.wav --output cleaned.wav
```

Các lệnh trên đã có mã nguồn. Gói gửi người chỉ inference có 4 file: `schwa.py`, `best.pt`, `requirements.txt`, `README.md`; ngưỡng nhúng trong checkpoint. Mã trong gói chỉ có CLI predict/denoise. Hướng dẫn dùng WAV riêng, không kèm dataset/WAV nguồn để giảm số file.

## 8. Trạng thái và tiêu chí hoàn thành

- [x] Đọc nội dung tham chiếu qua mô tả video và transcript BBC; xác định target /ə/.
- [x] Kiểm tra phần cứng và benchmark CNN nhỏ trên CPU.
- [x] Kiểm tra nguồn mẫu có nhãn `ax` theo thời gian và ghi giới hạn auto-label.
- [x] Triển khai download/prepare/audit: 760 bản ghi, 9.150 WAV; train 5.152, validation 2.064, test 1.934. Không giao nhau speaker/prompt/recording/PCM; không recording nào bị loại vì nhãn lỗi.
- [x] Triển khai lọc nhiễu và đo năng lượng trên 400 cửa sổ dương train/validation; giữ timeline, RMS target thấp nhất bằng 0,737 lần bản gốc. Chưa nghe/xác nhận phát âm bằng người.
- [x] Triển khai CLI train/evaluate/predict trên CPU; 7 kiểm tra kỹ thuật đã qua.
- [x] Train thật 3 run CPU, chọn `frequency_raw` bằng validation; epoch 10/15, ngưỡng 0,33, 31.777 tham số. Checkpoint `schwa_detection/runs/baseline/best.pt`, SHA-256 `741bde6e9eee29d927210933a377d3cf6c286dc01d63ba3b87ef17910db0f2db`.
- [x] Xuất metric test và danh sách lỗi: F1 0,4447, precision 31,63%, recall 74,83%, AUROC 0,8022; 482 báo nhầm và 75 bỏ sót. Chưa có độ tin cậy đủ cho ứng dụng thực tế; không điều chỉnh model theo test đã xem.
- [x] Gói inference `schwa_detection/test.zip` có 4 file; kiểm tra độc lập cho kết quả giống pipeline, hỗ trợ WAV thật dài 5,825 giây.
- [ ] Người nghe xác nhận nhãn/phát âm trên corpus và các bản ghi riêng; đánh giá giọng/micro/nhiễu môi trường thực tế. Chưa có dữ liệu người dùng để thực hiện bước này.
- [ ] Train bổ sung với noise augmentation/nguồn đa giọng và chọn lại bằng validation; nhiễu trắng đã khảo sát nhưng chưa cải thiện bằng lọc. Cần một test mới khi tiếp tục phát triển sau khi đã xem kết quả test hiện tại.

Chỉ đánh dấu hoàn thành train khi có checkpoint học từ dữ liệu schwa thật. Không dùng weights `yes`, tensor benchmark hay âm thanh tổng hợp thiếu ground truth để báo rằng đã train model mới. Các notebook `keyword_spotting/notebooks/` được giữ nguyên.

## 9. Mục tiêu cải thiện khoảng 90% (10/10/2026)

Yêu cầu mới: nâng độ chính xác lên khoảng 90%. Mặc định đo accuracy trên WAV test độc lập, đồng thời báo precision/recall/F1 và accuracy của cách đoán tất cả âm tính. Không coi việc loại hết mẫu dương là đạt yêu cầu. Câu hỏi phân biệt accuracy và precision đã gửi; cập nhật tiêu chí nếu người dùng chọn ưu tiên precision.

- Dùng Wav2Vec2 BASE được pretrain trên LibriSpeech để trích đặc trưng, giữ 6 lớp transformer. Backbone được giữ cố định; chỉ train bộ phân loại âm vị nhỏ trên CPU, sau đó lấy xác suất `ax` trong cửa sổ 0,5 giây.
- Nhãn train theo mốc thời gian của từng phone; bỏ supervision sát biên 5 ms để hạn chế lỗi căn chỉnh. Nhãn WAV và cách đánh giá sự hiện diện schwa giữ nguyên.
- Lượt thử trên train/validation cũ đạt accuracy validation 91,96%, precision 66,67%, recall 77,94%, F1 0,7186. Đây là kết quả validation, chưa phải chứng minh đạt 90% trên test.
- V2: 1.740 bản ghi, 300 câu × 5 speaker train (BDL/CLB/RMS/JMK/SLT), 120 câu AWB validation, 120 câu KSP test. Người nói và câu đọc không giao nhau giữa các tập trong V2. JMK test cũ được chuyển thành dữ liệu phát triển; kết quả test cũ không còn là đánh giá độc lập cho lượt cải thiện.
- Chỉ chọn kiến trúc/ngưỡng/epoch bằng validation. Test KSP mới chỉ được chạy sau khi khóa checkpoint. Xuất đầy đủ metrics, nhầm/bỏ sót và fingerprint; không sửa ngưỡng dựa trên test để báo đạt 90%.
- Giữ notebook cũ và khả năng train CPU. Gói inference cuối vẫn chỉ 4 file; model pretrained và head sẽ được nhúng chung trong checkpoint để chạy offline.

Tiến độ và mã thử nghiệm ở `schwa_detection/improve.py`, `schwa_detection/runs/transfer_v1/`, `schwa_detection/v2/`. Chưa đánh dấu mục tiêu 90% hoàn thành khi chỉ có kết quả validation.

Nguồn kỹ thuật: [TorchAudio 2.5 Wav2Vec2 feature extraction](https://docs.pytorch.org/audio/2.5.0/tutorials/speech_recognition_pipeline_tutorial.html), [mã bundle và giấy phép](https://github.com/pytorch/audio/blob/v2.5.1/src/torchaudio/pipelines/_wav2vec2/impl.py).

Kết quả V2 sau khi chốt model: validation AWB 92,25% accuracy/F1 0,7279; test KSP **87,48% accuracy, precision 52,73%, recall 40,80%, F1 0,4600**. Cách đoán toàn bộ âm tính đạt 86,92%, nên chưa đáp ứng mục tiêu nhận diện tốt. Checkpoint `runs/improved/best.pt` chỉ là ứng viên hiện tại, chưa thay gói CNN mặc định. Test KSP đã được xem và không còn là test độc lập để chọn model tiếp theo.

V3 chuyển sang phần có nhãn thủ công của L2-ARCTIC v5.0, gồm 24 người nói thuộc 6 ngôn ngữ mẹ đẻ, có tiếng Việt. Dùng `manual_events.perceived_raw`, không dùng cột 39-phone đã gộp AX/AH. Quy ước AH0/AX/AX0 → ax, AH1/AH2 → ah, ER/AXR giữ riêng; xét nhãn thay thế/thêm/xóa theo âm thực tế. Các đoạn `err` hoặc thiếu nhãn là `unk` và mọi cửa sổ chồng lên chúng bị loại trước khi chốt dataset.

Nguồn pinned: `chikingsley/l2-arctic-manual-v5.0-16k`, revision `b7261545bc4e4a02910e9c85c659eb866e00db7a`, CC BY-NC 4.0. Tách 12/6/6 speaker theo nguồn, đồng thời chia prompt toàn cục 60/20/20 với seed 20261011. Đã chuyển đổi 1.444 bản ghi (train 1.088, validation 186, test 170), không loại bản ghi vì lỗi timeline. V3 test chỉ dùng sau khi khóa kiến trúc, epoch và ngưỡng bằng validation. Không công bố đạt 90% từ validation hoặc từ một tập chủ yếu âm tính.

Nguồn nhãn: [L2-ARCTIC documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/), [bảng AH0=/ə/, AH=/ʌ/ ở phụ lục nghiên cứu](https://psi.engr.tamu.edu/wp-content/uploads/2020/06/1362168820931888.pdf).

V4 sửa riêng lỗi biên do resampling: 955 đoạn cuối đã được chú giải im lặng thiếu tối đa một mẫu 16 kHz được kéo dài đến cuối WAV. Không thay đổi nhãn schwa hoặc các đoạn thực sự thiếu nhãn. Dataset được tạo lại và audit trước train, gồm 19.023 WAV (9.404/4.884/4.735). V3 vẫn giữ nguyên; V4 test 6 speaker và 60 prompt chưa dùng để chọn model/ngưỡng.

Đã train thêm các head trên CPU, thử phổ mel, ngữ cảnh theo thời gian, encoder ASR 9 lớp và thêm 16.006 WAV native CMU vào train sau khi loại toàn bộ prompt validation/test L2. Validation V4 của BASE 6 lớp + mel + native đạt accuracy 89,62%, precision 57,87%, recall 53,87%, F1 0,5580. Đây chưa phải test 90%. Đang thử XLSR-53 6 lớp và đặc trưng cao độ/năng lượng; chọn bằng validation F1 như trước.

Ứng viên đã test 87,48% có gói `schwa_detection/test_improved.zip` đúng 4 file. Đã kiểm tra độc lập tính tự chứa và đối chiếu inference với pipeline; không dùng kiểm tra đóng gói làm bằng chứng tăng accuracy. Gói CNN cũ, weights `yes` và các notebook giữ nguyên.

Encoder XLSR-53 giữ 6 lớp và prosody chưa cải thiện F1 rõ. Lượt tiếp theo dùng [checkpoint âm vị Common Voice của Meta](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft), revision `2c733782da5604684829819a5eb744c193fe9398`, checksum nguồn `04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551`. Giữ đầy đủ 24 lớp/đầu CTC, lượng tử hóa linear int8 trên CPU; encoder 354.917.004 byte. Trích logits âm vị làm đặc trưng cho bộ phân loại schwa mới, kích hoạt lượng tử hóa từng waveform độc lập.

Để giới hạn thời gian train CPU, chọn trước 1.200 WAV train cân bằng speaker/nhãn (600/600, 12 speaker), seed 42. Validation/test V4 giữ nguyên toàn bộ, không thay đổi target AH0/AX. Ba head presence dự kiến: linear ngẫu nhiên, MLP 64 và linear khởi tạo từ logit ký hiệu /ə/ của encoder; sau đó train lại với nhãn WAV. Chọn/ngưỡng/pooling bằng validation F1. Protocol lưu trước khi chấm validation tại `reports/phoneme_ctc_training_protocol.json`; chưa đánh giá test mới. Đã có 29 kiểm tra kỹ thuật qua, gồm khởi tạo đúng ký hiệu bare schwa, lưu/nạp quantized model và độc lập giữa các waveform trong batch; đây chưa phải bằng chứng tăng accuracy.

Ba head trên clip riêng đã train xong và đạt F1 validation 0,496/0,507/0,494, chưa cải thiện nên không được chọn. Lượt tiếp theo dùng nguyên bản ghi làm ngữ cảnh: giữ nguyên 19.023 nhãn vùng 0,5 giây V4 và mọi speaker/prompt/recording thuộc từng split. Encoder chạy một lần trên bản ghi (đoạn tối đa 10 giây, phần lõi 6 giây và ngữ cảnh 2 giây mỗi phía cho WAV dài), sau đó nội suy về 24 tâm frame của mỗi vùng chấm. Không lấy schwa ở nơi khác trong câu làm mẫu dương của vùng đó.

Benchmark mới phải được ghi rõ **có ngữ cảnh bản ghi gốc**, không được trình bày như accuracy trên clip 0,5 giây cắt rời. Đã đăng ký ba head dùng đủ 9.404 mẫu train: linear khởi tạo từ /ə/, MLP presence 64 và MLP phone 128. Validation 4.884 vùng và test 4.735 vùng giữ nguyên toàn bộ. Protocol `reports/recording_context_training_protocol.json` lưu trước khi chấm validation; test chưa dùng để chọn model/ngưỡng. Có 34 kiểm tra kỹ thuật qua, gồm timeline/chia đoạn/phạm vi vùng chấm và ngăn xuất validation thành metric test. Mục tiêu 90% vẫn chưa đạt.

### Kết quả bàn giao ngày 10/10/2026

- [x] Đã tạo/audit V4 19.023 vùng và train trên CPU. Chọn head phone 128 từ run `transfer_v4_recording_phones_extended`: train 49 epoch, giữ epoch 42 theo validation F1; ngưỡng 0,44, pooling 2 frame. Encoder âm vị 24 lớp giữ cố định, linear int8; head học mới 56.377 tham số.
- [x] Chốt checkpoint trước test: `schwa_detection/runs/schwa_context/best.pt`, SHA-256 `20be5dccceee27e932f7dce37a5712db70bb7042238f268b00164fbac0806c63`. Test giữ đủ 4.735 vùng của 170 bản ghi, 6 speaker và các prompt riêng; không chọn lại model/ngưỡng bằng test.
- [x] Bản runtime bàn giao đạt **90,69% accuracy, precision 71,80%, recall 72,62%, F1 0,7221, AUROC 0,9179**. Cách luôn đoán âm tính đạt 83,34%. Có 573 phát hiện đúng, 225 báo nhầm, 216 bỏ sót. Accuracy từng speaker từ 84,60% đến 94,01%; không loại speaker khó.
- [x] Sửa ổn định tính toán sau kiểm tra đóng gói: CPU flash attention có sai khác nhỏ bị khuếch đại qua các lớp int8. Chọn math attention bằng kiểm tra độ khớp trên dữ liệu phát triển cũ, rồi chấm lại test với **cùng weights/ngưỡng/pooling**. Test đầu 90,83% được giữ nguyên ở `evaluation/test/`; kết quả runtime thực sự bàn giao nằm tại `evaluation/runtime_test/`. Đây là kiểm tra lại sau sửa runtime, không phải một test độc lập mới. Recipe phiên bản 2 giữ cache riêng; không dùng test accuracy để chọn cách sửa.
- [x] 36 kiểm tra kỹ thuật qua. Gói `schwa_detection/test.zip` đúng 4 file, đã giải nén vào thư mục độc lập và kiểm tra CPU: hai clip 0,5 giây, bản ghi gốc 3,46 giây, stereo 44,1 kHz, WAV 0,2 giây và WAV dài hơn 10 giây. Điểm/quyết định/vùng dự đoán khớp pipeline với sai số tối đa 1e-6. Không dùng kiểm tra đóng gói làm metric accuracy.
- [x] Cập nhật README/lệnh inference và model mặc định của gói gửi bạn. CNN cũ giữ trong `runs/baseline/inference_backup/`; notebook và weight `yes` không thay đổi. Fingerprint bàn giao tại `schwa_detection/reports/completion_audit.json`.

**Mục tiêu khoảng 90% đã đạt trên benchmark vùng 0,5 giây có ngữ cảnh bản ghi gốc.** Không công bố đây là accuracy của clip cắt rời, quyết định cho cả WAV, hay bản ghi micro/nhiễu của người dùng. Giữ đúng target /ə/ (AH0/AX); chưa xác nhận nhãn bằng việc nghe lại từng vùng trong dự án. Nếu tiếp tục chọn model sau kết quả này, cần một test độc lập mới.

## 10. Yêu cầu cả precision và recall trên khoảng 88%

Tiến độ lượt cải thiện: bảy head giữ encoder cố định đã train xong trên V5 (năm INT8, hai FP16). Mức thấp hơn giữa precision/recall validation cao nhất hiện khoảng 69,8%; chưa đạt yêu cầu. Kiểm tra scale train cho thấy embedding có độ lệch chuẩn khoảng 0,28, logits khoảng 3,02; đã thử LayerNorm riêng hai nhóm, không thay đổi nhãn/split/test. Không dùng accuracy khoảng 92% trên validation để kết luận đạt precision/recall.

Encoder công khai giữ nguyên nguồn và revision, phiên bản mới lưu linear float16 và tính toán float32 từng lớp trên CPU. Đã trích đủ đặc trưng train/validation và cache prefix của 1.261 bản ghi; fine-tune hai lớp cuối đang chạy, 22 lớp trước giữ cố định. Điểm làm tròn float16 của cache prefix cũng áp dụng trong inference, đã kiểm tra khớp cả WAV ngắn và dài hơn 10 giây. Benchmark thật một bản ghi train dài 4,92 giây: prefix khoảng 2,00 giây, forward/backward/Adam hai lớp cuối khoảng 0,61 giây; đây là benchmark kỹ thuật, chưa phải chất lượng model. Protocol ở `reports/precision_recall_88_fp16_protocol.json`, `precision_recall_88_group_protocol.json`, `precision_recall_88_tail_protocol.json`. Tập test V5 vẫn chưa dùng để chọn model/ngưỡng.

Lượt binary adaptation tiếp theo được đăng ký tại `reports/precision_recall_88_tail_binary_protocol.json`: AX dương, các âm đã biết khác âm tính; giữ nguyên vùng/nhãn test và các frame bỏ qua. Head 128, hai lớp cuối train với LR 0,00003, window loss 1,0, tối đa 15 epoch. Chạy sau lượt phone hiện tại để giữ giới hạn RAM. Bốn kiểm tra kỹ thuật qua, gồm cả train/lưu/pack/nạp lại inference cho task phones và binary; chưa phải bằng chứng đạt chất lượng 88%.

Thử encoder công khai LV60 khác nguồn, revision `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`, giữ cùng geometry/nhãn/split V5. Đã chuẩn bị weights và đo CPU trên một WAV train, không dùng phép đo kỹ thuật này làm metric chất lượng. Đang trích train/validation với một luồng CPU, sau đó train hai head group-normalized MLP/GRU mới; protocol lưu trước dự đoán tại `reports/precision_recall_88_lv60_protocol.json`. Báo cáo `reports/v5_validation_error_analysis.json` chỉ phân tích train/validation, không đọc test: Schwa một frame khó hơn các đoạn dài và lỗi phân bố ở nhiều nhóm âm/người nói.

Đã giải phóng khoảng 2,53 GB bằng cách xóa đúng hai bản tải checkpoint công khai dạng FP32 sau khi kiểm tra SHA của cả bản gốc lẫn encoder chuẩn bị dùng CPU. Giữ weights runtime, provenance/checksum nguồn, mọi dataset, notebook và gói inference. Báo cáo: `reports/redundant_public_checkpoint_cleanup.json`.

Lượt fine-tune phone đã hoàn thành 10 epoch, chọn epoch 8: precision/recall validation đều 69,87%, chưa đạt mục tiêu. Lượt binary adaptation tiếp tục chạy. Hướng tiếp theo bổ sung ngữ cảnh từ lời nói tự nhận dạng: ASR BASE 960h giữ đủ 12 lớp và CTC ký tự, từ điển CMUdict revision `74790861f652b15e4ac49015a90074ad62a27690`, căn chỉnh âm vị bằng CTC. Câu chuẩn và nhãn manual chỉ dùng cho kiểm tra/target train, không làm đầu vào feature. Giữ đúng 0,5 giây/vùng và các split V5.

Prototype audio-only `lexical_context.py` đã chạy trên một WAV train 4,92 giây: ASR khoảng 1,84 giây, căn chỉnh/tra phát âm khoảng 0,61 giây. Sáu kiểm tra kỹ thuật qua, gồm đối chiếu xác suất CTC với liệt kê mọi path nhỏ, giữ riêng AH0/AH1/AH2/ER0, và train/lưu/nạp lại head phone/binary. Đây chưa phải bằng chứng chất lượng 88%. Hai head mới được đăng ký trước khi chấm validation tại `reports/precision_recall_88_lexical_protocol.json`, chạy sau LV60 để giới hạn RAM. Chưa triển khai model thử này vào gói inference công khai.

Mục tiêu mới: precision > 88%, recall > 88% trên test, giữ accuracy khoảng 90% trở lên và khả năng train CPU/gói inference 4 file. Bản 71,80% / 72,62% chưa đáp ứng yêu cầu này.

V5 giữ cùng 24 speaker và quy ước nhãn/geometry của V4. Dùng seed 20261012 chốt 60 prompt test mới từ nhóm prompt train V4; giữ 60 prompt validation cũ, 180 prompt còn lại làm train. Speaker test vẫn tách khỏi train/validation. Các speaker này đã được đánh giá trước, nhưng các bản ghi test mới chưa dùng trong pipeline V4. Không đổi danh sách test theo kết quả model hoặc loại speaker khó. Vì prompt test mới từng thuộc train cũ, không khởi tạo từ head/encoder đã train trên V4; chỉ dùng encoder pretrained công khai và head mới từ đầu.

Thử đặc trưng kết hợp embedding 1.024 chiều và logits âm vị 392 chiều của encoder 24 lớp, có cùng ngữ cảnh bản ghi/24 tâm frame. Cache có tên riêng theo representation để không dùng lẫn với logits cũ. Chọn ngưỡng, pooling, epoch bằng `min(precision, recall)` trên validation; hòa thì F1, accuracy, FPR. Test chỉ chạy sau khi khóa một model bằng validation. Protocol ở `schwa_detection/reports/precision_recall_88_protocol.json`.

Trước train đã xóa 7 cặp cache BASE/ASR V2/V3 cũ có thể tái tạo, khoảng 1,69 GB, sau xác minh checksum. Giữ mọi WAV/nhãn/dataset/checkpoint/notebook và báo cáo đánh giá; metadata cache bị xóa lưu tại `reports/retired_cache_cleanup_pr88.json`. Gói `test.zip` hiện tại vẫn được giữ trong lúc thử model mới.

?? tri?n khai `schwa_detection/lexical_inference.py` ?? pack/predict/export model lexical tr?n CPU, nh?ng hai encoder v? t? ?i?n trong `best.pt`; g?i v?n ??ng 4 file. Ki?m tra synthetic ?? qua tr?n hai head MLP/GRU v? ba d?ng WAV (ng?n, stereo 44,1 kHz, d?i h?n 10 gi?y): cache/runtime v? b?n xu?t Python ??c l?p kh?p t?i ?a 1e-6. B?o c?o `reports/lexical_inference_engineering_checks.json`; hai ki?m tra k? thu?t ?? qua, ??y kh?ng ph?i metric ch?t l??ng. C? CLI `freeze/evaluate` kh?a SHA weights/dataset/m? inference, ng??ng/pooling v? to?n b? v?ng test; ch?m tr?n b?n ghi g?c v? ti?p t?c cache ??ng fingerprint khi gi?n ?o?n. Ch?a ch?y c?c l?nh n?y tr?n test V5, ch?a thay g?i inference c?ng khai.

Hai head lexical INT8 ?? ho?n t?t: MLP epoch 21/31 (precision validation 72,03%, recall 71,55%); binary GRU epoch 8/18 (72,99% / 73,23%). Ch?a ??t m?c ti?u. Ki?m tra model ??y ?? tr?n TRAIN ph?t hi?n cache/inference INT8 ch?a kh?p; weights ??ng g?i gi?ng ngu?n, m? helper c?/m?i t?nh gi?ng nhau, nh?ng s? lu?ng thay ??i ??c tr?ng. Kh?ng coi ki?m tra synthetic tr??c ?? l? ch?ng minh parity cho model ??y ??. Encoder FP16 t?i t?o ch?nh x?c cache TRAIN v?i hai lu?ng CPU trong ch?n ?o?n; c?n c? ??nh c?u h?nh v? ki?m tra ??y ?? tr??c recipe lexical ti?p theo. B?o c?o `reports/lexical_fullsize_cpu_parity.json` v? `reports/lexical_fp16_encoder_cache_diagnosis.json`. G?i c?ng khai ch?a thay, test V5 ch?a ch?m.

?? c? ??nh CPU ri?ng t?ng pha trong recipe lexical FP16: encoder ?m v? 2 lu?ng, ASR 1 lu?ng, c?n ch?nh 1 lu?ng, head 2 lu?ng. Full-size feature parity ?? qua tr?n ba b?n ghi TRAIN (ABA, ASI, EBVS) v? audio TRAIN l?p d?i 19,68 gi?y; b?n ghi th?c trong train ??u d??i 10 gi?y. ??c tr?ng ?m h?c/lexical/c?a s? ??u kh?p ch?nh x?c v?i cache ngu?n; ki?m tra d?i l? consistency c?a chia ?o?n, kh?ng ph?i metric ch?t l??ng. Hai l??t m?i kh?i t?o head t? ??u, ??ng k? tr??c validation t?i `reports/precision_recall_88_fp16_lexical_protocol.json`, session 65365 ?ang tr?ch validation/train. C?c source/caches/metrics c? gi? nguy?n, test V5 ch?a ch?m, m?c ti?u 88% ch?a ??t.
