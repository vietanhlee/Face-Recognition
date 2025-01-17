import cv2
from ImageDetect import ImageDetect
import time
import cvzone

# Mặc dù biến count của bạn được định nghĩa trong hàm __init__,
# nhưng khi bạn viết vòng lặp while True trực tiếp trong định nghĩa lớp (thay vì trong một phương thức),
# bạn đang tạo ra một phạm vi khác ngoài phạm vi của __init__.
# Điều này làm cho Python không hiểu biến count vì nó là một thuộc tính của instance (được truy cập bằng self.count),
# nhưng bạn lại sử dụng nó như một biến cục bộ trong vòng lặp.

class MakeDataFace():
    def __init__(self, name_lable):
        self.name_lable = name_lable
        self.count = 0 # Biến đếm số ảnh đã được chụp

        # Khởi tạo camera
        cap = cv2.VideoCapture(0)  # 0 là chỉ số của camera mặc định
        
        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            exit()
        while True:
            check_done, frame = cap.read()  # Đọc một khung hình từ camera
            if not check_done:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            # frame = cv2.resize(frame, (700, 500))
            # Gọi module ImageDetect để lưu ảnh vào thư mục chứa tên là name_lable
            try:
                ID = ImageDetect(image=frame, name_lable= self.name_lable, index= self.count)
                if(ID.check == 1):
                    self.count += 1
                    cvzone.cornerRect(frame, [ID.x, ID.y, ID.w, ID.h],l = 3,rt = 3)
                    cv2.putText(
                            img = frame,
                            text = f'Da luu anh thu {self.count} cua {self.name_lable}',
                            org = (int(ID.x - 35), int(ID.y - 5)),
                            fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 0.5,  
                            color = (0,255,255),
                            thickness = 2
                        )
            except Exception as e:
                print(f"Error in ImageDetect: {e}")
            # Hiển thị khung hình trong cửa sổ
            cv2.imshow('Camera', frame)

            # Thoát khỏi vòng lặp khi nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.count == 500:
                break
            # Giới hạn tốc độ xử lý (nếu cần)
            time.sleep(0.1)

        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()

MakeDataFace('viet anh')