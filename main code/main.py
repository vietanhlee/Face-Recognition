import numpy as np
from tensorflow.keras.models import load_model
import cvzone
from ultralytics import YOLO
import cv2
import pickle

path_model = 'model/model_cnn.h5'

# Lấy lable của data là file chứa categories của OneHotEnCoder()
with open('model/categories.pkl', 'rb') as f:
    cat = pickle.load(f)
lb = np.array(cat[0]) # cat là mảng 2 chiều vd: [['lable']], chuyển về numpy để thao tác tiện luôn

# Load model đã train nhận diện gương mặt
model = load_model(path_model)

# Load model detect face pre-train trên YOLO
facemodel = YOLO('model/yolov11n-face.pt')
cam = cv2.VideoCapture(0) # set camera thay 0 bởi path video demo để test

if cam.isOpened() == 0:
    print('cam is not open')
while True:
    check_done, frame = cam.read()
    frame = cv2.flip(frame, 1) # Đảo ngược ảnh

    face_result = facemodel.predict(frame, conf = 0.6, verbose = False)
    
    if(check_done == 0):
        print('cant open read frame')
    else:
        boxes_xyxy = face_result[0].boxes.xyxy.tolist()
        for box in boxes_xyxy:
            x, y, x2, y2 = map(int, box)
            w = x2 - x # width: chiều rộng
            h = y2 - y # height: chiều cao

            # Vẽ bounding box
            cvzone.cornerRect(frame, [x, y, w, h], rt = 0)

            # Cắt và chuẩn hóa dữ liệu
            img_cut = frame[y:y + h, x:x + w] 
            img_cut = cv2.resize(img_cut, (128, 128)) 
            img_cut = img_cut.astype('float32') / 255 
            
            # yêu cầu về đầu vào của input_shape của model (thêm batch size = 1)
            img_cut_expanded_0 = np.expand_dims(img_cut, axis = 0) 

            # Gán nhãn
            arr_predict = model.predict(img_cut_expanded_0, verbose = 0)
            # Lấy index có tỉ lệ cao nhất, axis = 1 vì đây là mảng 2 chiều [[data]] 
            predicted_label_index = np.argmax(arr_predict, axis = 1)
            # Lấy tỉ lệ phần trăm tương ứng, các mảng ở đây đều là 2D nên cần trỏ truy cập vào phần tử đầu 
            accuracy = arr_predict[0][predicted_label_index[0]] 

            # Set tiêu đề hiển thị unknow nếu tỉ lệ quá nhỏ 
            txt = 'unknow' if accuracy < 0.97 else lb[predicted_label_index[0]] + ' ' + str(round(accuracy * 100, 2)) + ' %'
            cv2.putText(img = frame, org= (x + w // 2 - 90, y - 25), text = f'{txt}', 
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0,0), thickness= 3)
            print(arr_predict[0], lb[predicted_label_index[0]], str(round(accuracy * 100, 2)) + ' %') # Hiển thị data lên màn console
    
    cv2.imshow('Face Recognizer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()