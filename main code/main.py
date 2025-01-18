import numpy as np
from tensorflow.keras.models import load_model
import cvzone
from ultralytics import YOLO
import cv2
import pickle

path_model = 'model_cnn.h5'

# Lấy lable của data là file chứa categories của OneHotEnCoder()
with open('categories.pkl', 'rb') as f:
    cat = pickle.load(f)
lb = np.array(cat[0]) # cat là mảng 2 chiều vd: [['lable']], chuyển về numpy để thao tác tiện luôn

# Load model đã train nhận diện gương mặt
model = load_model(path_model)

# Load model detect face pre-train trên YOLO
facemodel = YOLO('yolov11n-face.pt')
# cam = cv2.VideoCapture('obama.mp4')
cam = cv2.VideoCapture(0) # set camera thay 0 bởi path video demo để test

if cam.isOpened() == 0:
    print('cam is not open')
while True:
    check_done, frame = cam.read()
    frame = cv2.flip(frame, 1) # Đảo ngược ảnh

    # Dùng model pre train nhận diện gương mặt
    # Trả về data về các gương mặt 
    face_result = facemodel.predict(frame, conf = 0.6)
    
    if(check_done == 0):
        print('cant open read frame')
    else:
        for info in face_result:
            parameters = info.boxes # Lấy ra mảng chứa thông tin về các box là bounding các face được detect
            for box in parameters:
                # x, y, w, h = box.xywh[0] # Đây là mảng 2D: [[data]]
                # x, y, w, h = int(x), int(y), int(w), int(h) # Ép về kiểu nguyên mới vẽ box được

                x, y, x2, y2 = box.xyxy[0] # Đây là mảng 2D: [[data]]
                x, y, x2, y2 = int(x), int(y), int(x2), int(y2)  # Ép về kiểu nguyên mới vẽ box được
                w = x2 - x # width: chiều rộng
                h = y2 - y # height: chiều cao

                # Vẽ bounding box
                cvzone.cornerRect(frame, [x, y, w, h],l= 3,rt= 3)

                # Crop and preprocess face
                img_cut = frame[y:y + h, x:x + w] # Ảnh gương mặt
                img_cut = cv2.resize(img_cut, (128, 128)) # Chuyển size về đúng định dạng theo model
                img_cut = img_cut.astype('float32') / 255 # Chuẩn hóa dữ liệu
                img_cut_gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY) # Chuyển về ảnh đen trắng 
                # Mở rộng chiều dữ liệu ảnh vi model cần ảnh đầu vào dạng 4D: (1, 128, 128, 1) : (bacth_size = 1, 128, 128, số chiều: 1)
                img_cut_expanded_0 = np.expand_dims(img_cut_gray, axis=0)
                img_cut_expanded_end = np.expand_dims(img_cut_expanded_0, axis=-1)

                # Đoán nhãn
                arr_predict = model.predict(img_cut_expanded_end) 
                predicted_label_index = np.argmax(arr_predict, axis=1) # Lấy index có tỉ lệ cao nhất, axis = 1 vì đây là mảng 2 chiều [[data]]
                accuracy = arr_predict[0][predicted_label_index[0]] # Lấy tỉ lệ phần trăm, các mảng ở đây đều là 2D nên cần trỏ truy cập vào phần tử đầu

                # Set tiêu đề hiển thị unknow nếu tỉ lệ quá nhỏ 
                txt = 'unknow' if accuracy < 0.99 else lb[predicted_label_index[0]] + ' ' + str(round(accuracy * 100, 2)) + ' %'
                cv2.putText(img= frame,
                            org= (x + w // 2 - 45, y - 25), 
                            text= f'{txt}',
                            fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale= 0.5,
                            color= (0, 255, 255),
                            thickness= 2
                            )
                print(arr_predict, lb[predicted_label_index[0]]) # Hiển thị data lên màn console, mảng 2D
    
    # Hiển thị frame ra bên ngoài cửa sổ
    cv2.imshow('Face Recognizer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giải phóng camera (quyền truy cập) và hủy các các cửa sổ window
cam.release()
cv2.destroyAllWindows()