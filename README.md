
# Sơ qua về project

Ứng dụng mô hình CNN để train dữ liệu nhận diện các gương mặt. Các gương mặt được cắt trực tiếp nhờ model YOLO đã được pre-train.

![ảnh](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/10/59954intro-to-CNN.webp)

# Cách chạy:

## 1. Cài đặt các thư viện cần thiết (python 3.10)

```
pip install -r requires.txt
```

## 2. Chuẩn bị data
### 2.1 Chạy theo data đã có sẵn từ trước (theo code mẫu):

- Chỉ cần `run` file `main.py` trong thư mục chính chứa code: `main code\main.py`

### 2.2 Chạy theo data tự chuẩn bị:

- **Bước 1:** Xóa các thư mục trong foder `data_image_raw`

- **Bước 2:** Nếu đã có data về các gương mặt, chỉ cần thêm vào thư mục `data_image_raw` theo cấu trúc sau:


```
Mỗi thư mục của mỗi người là ảnh các gương mặt của người đó:

data_image_raw   
        ├── tên người thứ 1    
        ├── tên người thứ 2
        ├──  -------------    
        └── tên người thứ n
```

> Số lượng các ảnh mỗi người nên đồng đều nhau và các tính chất như màu sắc và độ sáng nên tương đồng nhau


- Nếu chưa có bất kỳ data nào (cần ít nhất 2 người để lấy data):

  - Vào file `main code\make_data_face` và chỉnh tên người cần lấy dữ liệu, rồi bấm `run` để code tự động lấy đủ 500 gương mặt của người đó. Nên quay nhiều hướng khác nhau để data đa dạng, tránh overfitting.

    > Truyền tên người cần lấy data vào dòng này trong file: 
    ``MakeDataFace('viet anh')``
  - Làm tương tự cho những người còn lại đến khi hết


## 3. Xử lý data

- **Bước 1:** Vào file `main code\make_data_face.py` và 
chỉnh tên người.
- **Bước 2:** Chạy file `main code\processed_raw_data.py` để xử lý data thô.
- **Bước 3:** Chạy tất cả các file trong `training_data.ipynb` để tiến hành training.

## 4. Chạy code

Chỉ cần chạy file `main code\main.py` để bắt đầu sử dụng.

# Nhận xét:

Mô hình huấn luyện tương đối hiệu quả trong phạm vi tập data lớn gồm nhiều gương mặt được train, nhưng lại dễ bị overfiting hoặc kém hiệu quả hơn với tập data ít, số người ít vì mô hình học được rất dễ bị một đặc điểm trội nào đó (màu sắc, góc độ) từ 1 gương mặt làm sai lệch đi kết quả dự đoán.