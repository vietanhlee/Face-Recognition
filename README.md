# Sơ qua về project

## Lý thuyết áp dụng  

Ứng dụng mô hình CNN để train dữ liệu nhận diện các gương mặt. Các gương mặt trên được cắt trực tiếp nhờ model phát hiện gương mặt người được train trên cấu trúc YOLO dựa theo dataset về gương mặt con người đã được gán nhãn.

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/10/59954intro-to-CNN.webp)
<p align = 'center'> Minh họa cấu trúc CNN </p>

![](https://oditeksolutions.com/wp-content/uploads/2025/01/Fashionable-Blog-Banner.webp)
<p align = 'center'> Phát hiện gương mặt với YOLO </p>

# Cách chạy

## 1. Cài đặt các thư viện cần thiết (python 3.10)

- Clone dự án về và chạy dòng lệnh sau trên command Prompt để cài thư viện cần thiết:

``` bash
    pip install -r requires.txt
```

## 2. Chuẩn bị data



- **Bước 1:** Xóa sạch các thư mục trong foder `data_image_raw` nếu nó có tồn tại

- **Bước 2:**

  - Nếu đã có data về các gương mặt, chỉ cần thêm vào thư mục `data_image_raw` theo cấu trúc sau:

    Mỗi thư mục của mỗi người là ảnh các gương mặt của người đó:

    ```
    data_image_raw   
            ├── tên người thứ 1    
            ├── tên người thứ 2
            ├──  -------------    
            └── tên người thứ n
    ```

    > Số lượng các ảnh mỗi người nên đồng đều nhau và các tính chất như màu sắc và độ sáng nên tương đồng nhau


  - Nếu chưa có bất kỳ data nào (cần ít nhất 2 người để lấy data):

    - Vào file `main code\make_data_face` và chỉnh tên người cần lấy dữ liệu:
      
      > Truyền tên người cần lấy data vào dòng này trong file: 
        ``` python
            MakeDataFace('Viet Anh')
         ```

      nhìn thẳng vào camerea rồi bấm `run` để chương trình tự động lấy đủ 500 gương mặt. Nên quay nhiều hướng khác nhau để data đa dạng, tránh overfitting. Code tự điều chỉnh ánh sáng và tương phản nên không nhất thiết cần thu thập gương mặt mọi người ở cùng vị trí (nhưng có vẫn là hơn)

    - Làm tương tự cho những người còn lại đến khi hết.

    ![](https://raw.githubusercontent.com/vietanhlee/Face-Recognizer/refs/heads/main/display_github/thu%20thap.png)
    <p align = 'center'> Thu thập hình ảnh gương mặt </p>
    
    > Ảnh sau khi được thu thập sẽ được lưu vào thư mục `data_image_raw/ten_người_đó`


## 3. Xử lý data

- Chạy file `training_data.ipynb` để tiến hành xử lý dữ liệu thô sau đó training và xuất ra model cuối ở: `model/model_cnn.h5`.

## 4. Chạy code

Chỉ cần chạy file `main code\main.py` để bắt đầu sử dụng.
![](https://raw.githubusercontent.com/vietanhlee/Face-Recognizer/refs/heads/main/display_github/chay.png)
    <p align = 'center'> Chạy thử </p>
# Nhận xét:

Mô hình huấn luyện tương đối hiệu quả trong phạm vi tập data lớn gồm nhiều gương mặt được train, nhưng lại dễ bị overfiting hoặc kém hiệu quả hơn với tập data ít, số người ít vì mô hình học được rất dễ bị một đặc điểm trội nào đó (màu sắc, góc độ) từ 1 gương mặt làm sai lệch đi kết quả dự đoán mặc dù đã tăng cường làm giàu dữ liệu như tăng giảm độ sáng và độ tương phản.

> Với một số ít dataset có thể dùng so sánh khoảng cách norm từ ảnh được cắt đến tập dataset và chọn ra ảnh có khoảng cách nhỏ nhất làm nhãn.

# Dự án đang tích hợp thêm trên Qt5

