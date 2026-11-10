# Kiểm tra bản ghi thực tế sau khi train

Hiện **chưa đánh giá trên bản ghi thực tế**. Đặt WAV tự thu trong thư mục này và dùng `src.predict`.

- Thu từng đoạn tối đa 5 giây, để nguyên toàn bộ từ. Có thể dùng mono/stereo và sample rate khác; pipeline resample về 16 kHz và pad, không kéo giãn thời gian.
- Có đoạn chứa `yes`, đoạn chứa từ khác (`no`, `up`, `stop`, ...), và đoạn chỉ có nền. Ghi nhãn bằng cách nghe xác nhận.
- Thu ở nhiều khoảng cách, mức nền và người nói; tách người nói/phiên thu dùng đánh giá khỏi nguồn train. Không dùng những đoạn đánh giá để chọn ngưỡng.
- File dài hơn 5 giây bị từ chối. Kết quả dataset trộn chưa chứng minh hiệu quả trên hội thoại dài hay giọng Việt.

```powershell
python -m src.predict --run-dir runs/baseline --audio data/real_recordings/example.wav --device cpu
```
