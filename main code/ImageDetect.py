import cv2
import os
from ultralytics import YOLO
import cv2

# Load model pre train
facemodel = YOLO('yolov11n-face.pt')

class ImageDetect():
    '''Input is image read by cv2.read, name_lable is name of persion is taken'''
    def __init__(self, image, name_lable, index):
        self.image = image # Ảnh số hóa được đưa vào cv2.read(img_path)
        self.name_lable = name_lable # Nhãn được đánh 
        self.index = index # Dùng đặt tên tệp ảnh gương mặt sau crop
        self.check = 1 # kiểm tra sự tồn tại của gương mặt trong khung hình
        # Thông số ảnh crop khuôn mặt, phục vụ cho trích xuất data bên ngoài
        self.x = 0
        self.y = 0
        self.w = 0 # width: chiều rộng
        self.h = 0 # height: chiều cao

        # Tạo thư mục chứa ảnh: data_image_raw\name_lable\out{index}.jpg
        os.makedirs("data_image_raw" + "\\" + self.name_lable, exist_ok= True)
        
        # Data về gương mặt
        face_result = facemodel.predict(self.image,conf = 0.6)

        # Kiểm tra và cắt khuôn mặt
        if len(face_result[0].boxes) == 0:
            self.check = 0 # Update biến check
            print("Không phát hiện khuôn mặt nào trong ảnh.")
        else:
            for info in face_result:
                parameters = info.boxes # Lấy các box
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0] # Do mảng 2D
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Chuyển về kiểu nguyên vì openCV yêu cầu nguyên
                    h, w = y2 - y1, x2 - x1
                    
                    # Update thông số
                    self.x = x1
                    self.y = y1
                    self.w = w
                    self.h = h

                    # Ảnh mặt được cắt ra và resize theo tiêu chuẩn
                    img_cut = self.image[y1: y1 + h, x1: x1 + w]
                    img_cut = cv2.resize(img_cut, (128, 128))
                    # Lưu ảnh vào thư mục vừa tạo thông qua đường dẫn đã tạo
                    cv2.imwrite(f'data_image_raw\\{self.name_lable}\\out{self.index}.jpg', img= img_cut)
                    # Thông báo ra màn hìn
                    print(f'Đã lưu ảnh thứ {self.index} của {self.name_lable}')