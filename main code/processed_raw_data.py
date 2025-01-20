import cv2
import numpy as np
import os

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
        
        # Mã hóa ảnh và chuyển về dạng gray
        matrix = cv2.imread(path_image)
        matrix_gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        # mtr = np.concatenate([mtr, matrix])
        
        # Thêm ảnh mã hóa và lable tương ứng vào các list
        data_img.append(matrix_gray)
        lable.append(item)

    # In ra màn thông báo
    print(f'Đã xử lý xong ảnh của: {item} với số ảnh: {len(list_image)}')

# Chuyển data và lable về np.array vì tensorflow yêu cầu đầu vào là np.array, lable cần đưa về dạng 2D
data_img = np.array(data_img) 
cat_lable = set(lable.copy())
lable = np.array(lable).reshape(-1, 1) # Có thể dùng expand_dim cũng được

# Hiển thị ra màn console
print(f'shape của data: {data_img.shape} với các lable {cat_lable}')

# Lưu data và lable vào tệp để xong train
import pickle
os.makedirs('data_processed', exist_ok= True)
with open('data_processed\\data.pkl', 'wb') as f:
    pickle.dump(data_img, f)
with open('data_processed\\lable.pkl', 'wb') as f:
    pickle.dump(lable, f)