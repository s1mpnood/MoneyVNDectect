# Vietnamese Banknote Detection

Pipeline nhận dạng mệnh giá tiền Việt Nam bằng cách kết hợp phát hiện theo đặc trưng ORB và phân loại lai giữa đặc trưng thủ công (màu sắc + texture) và CNN nhẹ.

## Kiến trúc

- **Template matching (ORB)**: mỗi mệnh giá có ảnh mẫu. Hệ thống trích xuất keypoint ORB và so khớp với ảnh đầu vào, tính homography để suy ra vị trí tờ tiền.
- **Đặc trưng màu & texture**: tính histogram HSV và Local Binary Pattern để mô tả màu sắc + hoa văn.
- **CNN nhẹ**: mô hình tích chập đơn giản nhận ảnh crop của tờ tiền, có thể fine-tune với tập dữ liệu ảnh chụp.
- **Kết hợp**: điểm tin cậy cuối cùng là trung bình trọng số giữa phân loại thủ công và CNN.

## Chuẩn bị

1. Thu thập ảnh mẫu cho từng mệnh giá (đặt tên `10k.jpg`, `20k.jpg`, ...).
2. Đặt các ảnh mẫu vào một thư mục, ví dụ `templates/`.
3. (Tuỳ chọn) Huấn luyện CNN và lưu trọng số vào file `.pt`.

## Sử dụng

```bash
python -m pip install -r requirements.txt
python -m src.main path/to/image.jpg --templates templates/ --labels 10k 20k 50k --weights weights/banknote.pt --output output.jpg
```

Nếu không cung cấp `--weights`, mô hình CNN sẽ dùng trọng số khởi tạo ngẫu nhiên nên chỉ nên sử dụng sau khi đã fine-tune.

## Ứng dụng web đơn giản

Có thể chạy ứng dụng Flask để tải ảnh trực tiếp và nhận kết quả phân loại ngay trên trình duyệt.

1. Cài đặt phụ thuộc: `python -m pip install -r requirements.txt`.
2. Thiết lập biến môi trường đường dẫn template (ví dụ `export BANKNOTE_TEMPLATE_DIR=templates`).
   - (Tuỳ chọn) Thiết lập `BANKNOTE_LABELS=10k,20k,50k` nếu muốn chỉ định danh sách nhãn thay vì tự suy ra từ tên file mẫu.
   - (Tuỳ chọn) Thiết lập `BANKNOTE_WEIGHTS=weights/banknote.pt` nếu có trọng số CNN đã huấn luyện.
3. Chạy lệnh `python -m src.app` và mở trình duyệt tại `http://localhost:5000`.

Endpoint `/api/predict` nhận POST form-data (trường `image`) và trả về JSON kết quả.

## Huấn luyện CNN (gợi ý)

- Chuẩn hoá các ảnh tờ tiền, resize về 128x64.
- Chia train/val, sử dụng `torch.utils.data.Dataset` để load ảnh và nhãn.
- Dùng optimizers như Adam với learning rate nhỏ (1e-3), fine-tune vài epoch.
- Lưu trọng số tốt nhất bằng `torch.save(model.state_dict(), path)`.

## Giấy phép

MIT
