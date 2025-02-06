import cv2
from ImageDetect import ImageDetect
import time

class MakeDataFace():
    def __init__(self, name_lable):
        self.name_lable = name_lable # Tên gương mặt đánh nhãn
        self.count = 0 # Biến đếm số ảnh đã được chụp

        # Khởi tạo camera
        cap = cv2.VideoCapture(0)  # 0 là chỉ số của camera mặc định, có thể thay bằng đường dẫn đến tệp test
        if not cap.isOpened():
            print("Không mở được camera")
            exit()

        while self.count < 500:
            check_done, frame = cap.read() 
            if not check_done:
                print("Không đọc được ảnh")
                break
            
            # Đảo ngược ảnh
            frame = cv2.flip(frame, 1) 

            # Gọi đối tượng
            ID = ImageDetect(image_input= frame, name_lable= self.name_lable, index= self.count)

            # Nếu tồn tại gương mặt thì mới tăng count lên 1
            if(ID.check == 1):
                self.count += 1
                            
            cv2.imshow('Make data face', ID.image_output)

            if (cv2.waitKey(1) & 0xFF == ord('q')) or self.count == 500:
                break
            time.sleep(0.06)
        
        # Giải phóng tài nguyên camera
        cap.release()
        cv2.destroyAllWindows()

# Tạo lệnh 
MakeDataFace('Viet Anh')

