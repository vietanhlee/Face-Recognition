import pickle
import numpy as np
import cv2
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Process_and_train_data():
    def __init__(self):
        # Lấy thông tin các tệp con (trùng tên với lable) trong tệp data_raw_image
        list_lable = os.listdir('data_image_raw')

        # List lưu ảnh đã mã hóa và lable tương ứng
        data_img = []
        lable = []

        for item in list_lable:
            # Tạo dường dẫn đến từng tệp con (có thể dùng cộng xâu bth) cách bên dưới an toàn hơn
            path_lable = os.path.join('data_image_raw', item)
            list_image = os.listdir(path_lable) # Trả về tên các tệp ảnh
            
            for image in list_image:
                # Tạo đường dẫn đến thư mục ảnh
                path_image = os.path.join(path_lable, image)
                
                # Đọc ảnh
                matrix = cv2.imread(path_image)
                
                # Thêm ảnh và lable tương ứng vào các list
                data_img.append(matrix)
                lable.append(item)

            # In ra màn thông báo
            print(f'Đã xử lý xong ảnh của: {item} với số ảnh: {len(list_image)}')

        # Chuyển data và lable về np.array vì tensorflow yêu cầu đầu vào là np.array, lable cần đưa về dạng 2D
        data_img = np.array(data_img) 
        cat_lable = set(lable.copy())
        lable = np.array(lable).reshape(-1, 1) # Có thể dùng expand_dim cũng được

        # Hiển thị ra màn console
        print(f'shape của data: {data_img.shape} với các lable {cat_lable}')

        # Encoder lable
        encoder = OneHotEncoder(sparse_output= False)
        lable_processed = encoder.fit_transform(lable)
        # Chuẩn hóa data đầu vào
        data_processed = data_img.astype('float32') / 255

        with open('model/categories.pkl', 'wb') as f:
            pickle.dump(encoder.categories_, f)
        with open('model/categories.pkl', 'rb') as f:
            cat = pickle.load(f)
        lb = np.array(cat[0])
        num_class = lb.size # Số class dùng để phân biệt


        xtrain, xtest, ytrain, ytest = train_test_split(data_processed, lable_processed, test_size= 0.2)

        model_cnn = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), # Yêu cầu input_shape phải là 3D
            MaxPool2D((2, 2), padding = 'same'),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D((2, 2), padding = 'same'),
            
            Flatten(),
            
            Dense(128, activation='relu'),
            Dropout(0.5),
            
            Dense(num_class, activation= 'softmax'),
        ])
        model_cnn.compile(optimizer= Adam(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['acc', 'precision', 'recall'])

        model_cnn.summary() 

        model_cnn.fit(xtrain, ytrain, epochs= 10, validation_data = (xtest, ytest) , batch_size= 32)