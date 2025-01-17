import cv2
import numpy as np
import os

list_item = os.listdir('data_image_raw')

data_img = []
lable = []

for item in list_item:
    path_lable = os.path.join('data_image_raw', item)
    list_image = os.listdir(path_lable)
    
    for image in list_image:
        path_image = os.path.join(path_lable, image)
        matrix = cv2.imread(path_image)
        matrix_gray = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
        # mtr = np.concatenate([mtr, matrix])
        
        data_img.append(matrix_gray)
        lable.append(item)

    print(item, len(list_image))

data_img = np.array(data_img)
fix = lable.copy()
fix = set(fix)
lable = np.array(lable).reshape(-1, 1)

print(data_img.shape)
print(fix)

import pickle
os.makedirs('processed', exist_ok= True)
with open('processed\\data.pkl', 'wb') as f:
    pickle.dump(data_img, f)
with open('processed\\lable.pkl', 'wb') as f:
    pickle.dump(lable, f)