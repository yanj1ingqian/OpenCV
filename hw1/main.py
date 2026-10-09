import cv2 
import numpy as np
from hw1.sobel import *

if __name__ == "__main__":
    img = cv2.imread("input/03.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if img is None:
        print("读图失败")
        exit()
    print(img.shape,img.dtype,img.size)
    gray = gauss(img)
    grad_x = sobel_x(gray)
    grad_y = sobel_y(gray)
    grad_mag = sobel_mag(grad_x,grad_y)
    grad_angle = sobel_angle(grad_x,grad_y,grad_mag)
    gx_show = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gy_show = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gm_show = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("output/sobel_x.jpg",gx_show)
    cv2.imwrite("output/sobel_y.jpg",gy_show)
    cv2.imwrite("output/sobel_mag.jpg",gm_show)
    cv2.imwrite("output/sobel_angle.jpg",grad_angle)
    cv2.imwrite("output/gray.jpg",gray.astype(np.uint8))