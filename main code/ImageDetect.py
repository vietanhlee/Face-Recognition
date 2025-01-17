import cv2
import os
from ultralytics import YOLO
import cv2

facemodel = YOLO('yolov11n-face.pt')

class ImageDetect():
    '''Input is image read by cv2.read, name_lable is name of persion is taken'''
    def __init__(self, image, name_lable, index):
        self.image = image
        self.name_lable = name_lable
        self.index = index
        self.check = 1
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

        # Tạo thư mục chứa ảnh
        os.makedirs("data_image_raw" + "\\" + self.name_lable, exist_ok= True)
        # Bộ lọc ảnh
        face_result = facemodel.predict(self.image,conf = 0.6)

        # Kiểm tra và cắt khuôn mặt
        if len(face_result[0].boxes) == 0:
            self.check = 0
            print("Không phát hiện khuôn mặt nào trong ảnh.")
        else:
            for info in face_result:
                parameters = info.boxes
                for box in parameters:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h,w = y2 - y1,x2 - x1
                    
                    # Set data
                    self.x = x1
                    self.y = y1
                    self.w = w
                    self.h = h

                    # Ảnh mặt được cắt ra
                    img_cut = self.image[y1: y1 + h, x1: x1 + w]
                    # Resize 64x64
                    img_cut = cv2.resize(img_cut, (128, 128))
                    
                    # Lưu ảnh vào thư mục vừa tạo thông qua đường dẫn
                    cv2.imwrite(f'data_image_raw\\{self.name_lable}\\out{self.index}.jpg', img= img_cut)
                    print(f'Đã lưu ảnh thứ {self.index} của {self.name_lable}')