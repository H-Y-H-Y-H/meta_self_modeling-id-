import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
img_list = os.listdir('img')
all_img = []

row_num = 14
col_num = 8
for i in range(col_num):
    img_row = []
    for j in range(row_num):
        img = plt.imread('img/'+img_list[j+i*row_num])
        paddings = np.zeros((img.shape[0],30,4),dtype=img.dtype)

        img_row.append(img)
        img_row.append(paddings)
    img_row = np.hstack(img_row)
    all_img.append(img_row)
all_img = np.vstack(all_img)
row_shape, col_shape = all_img.shape[:2]
row_shape,col_shape = row_shape//10,col_shape//10

all_img = cv2.resize(all_img,(col_shape, row_shape))
# plt.imshow(all_img)
plt.imsave('comb.png',all_img)

