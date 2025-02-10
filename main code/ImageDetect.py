import cv2
import os
from ultralytics import YOLO
import cv2
from keras.api.layers import RandomBrightness, RandomContrast
from keras.api.models import Sequential
import numpy as np
import cvzone

# Tạo lớp tăng cường dữ liệu
data_augmentation = Sequential([
    RandomBrightness(0.25),           # Thay đổi độ sáng ngẫu nhiên +- 25%
    RandomContrast(0.25),             # Thay đổi độ tương phản +- 25%
])

# Load model pre train
facemodel = YOLO('model/yolov11n-face.pt')

class ImageDetect():
    '''Đầu vào là mảng np.darray được đọc từ cv2, tên người lấy data và index để đặt tên thư mục chứa ảnh
class này chủ yếu sẽ lưu tạo thư mục chứa tên là name_label và thực hiện nhận diện gương mặt sau đó cắt và lưu ảnh vào thư mục ấy
đồng thời hiển thị một số chú thích về quá trình lên ảnh đầu ra là image_output'''
    def __init__(self, image_input, name_label, index):
        self.image_output = image_input.copy() # Ảnh số hóa được đưa vào cv2.read(img_path)
        self.name_label = name_label # Nhãn được đánh 
        self.index = index # Dùng đặt tên tệp ảnh gương mặt sau crop
        self.check = 1 # kiểm tra sự tồn tại của gương mặt trong khung hình
        self.img_face = None
        # Thông số ảnh crop khuôn mặt, phục vụ cho trích xuất data bên ngoài
        self.x = 0
        self.y = 0
        self.w = 0 # width: chiều rộng
        self.h = 0 # height: chiều cao

        # Chạy hàm process ngay
        self.process()

    def process(self):
        '''Khi chạy hàm này sẽ lưu tạo thưu mục chứa tên là name_label và thực hiện nhận diện gương mặt sau đó cắt và lưu ảnh vào thư mục ấy
đồng thời hiển thị một số chú thích về quá trình lên ảnh đầu ra là image_output'''
        # Tạo thư mục chứa ảnh: data_image_raw\name_label\out{index}.jpg
        os.makedirs("data_image_raw" + "\\" + self.name_label, exist_ok= True)
        # Data dự đoán
        face_result = facemodel.predict(self.image_output, conf = 0.6, verbose = False)

        # Thông số các box
        boxes_xyxy = face_result[0].boxes.xyxy.tolist()
        
        # Kiểm tra và cắt khuôn mặt
        if len(boxes_xyxy) == 0:
            self.check = 0 # Update biến check
            print("Không phát hiện khuôn mặt nào trong ảnh.")
        else:
            for box in boxes_xyxy:
                # Chuyển về kiểu nguyên vì openCV yêu cầu nguyên
                x1, y1, x2, y2 = map(int, box)
                h, w = y2 - y1, x2 - x1
                
                # Update thông số box
                self.x = x1
                self.y = y1
                self.w = w
                self.h = h

                # Ảnh mặt được cắt ra và resize theo tiêu chuẩn
                img_cut = self.image_output[y1: y1 + h, x1: x1 + w]
                img_cut = cv2.resize(img_cut, (128, 128))
                self.img_face = img_cut.copy()
                
                # Thực hiện tăng cường dữ liệu, thêm batch dimension do yêu cầu input_shape có bath_size thêm ở đầu
                augmented_image = data_augmentation(np.expand_dims(img_cut, axis= 0), training= True) 

                # Chuyển tensor thành định dạng NumPy
                image_to_save = augmented_image[0].numpy().astype("float32")  # Loại bỏ batch dimension và chuyển kiểu dữ liệu
                
                # Lưu ảnh vào thư mục vừa tạo thông qua đường dẫn đã tạo
                cv2.imwrite(f'data_image_raw\\{self.name_label}\\out{self.index}.jpg', img= image_to_save)

                # Thông báo ra màn hình
                print(f'Đã lưu ảnh thứ {self.index} của {self.name_label}')

                # Vẽ bounding box
                cvzone.cornerRect(self.image_output, [x1, y1, w, h], rt = 0)
                
                # Vẽ text chỉ dẫn nằm ngay trên bouding box
                cv2.putText(img = self.image_output, text = f'Da luu anh thu {self.index} cua {self.name_label}',
                        org = (int(x1 - 70), int(y1 - 20)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1, color = (0, 128, 255), thickness = 2)  