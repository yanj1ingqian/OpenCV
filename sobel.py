import cv2 
import numpy as np

img = cv2.imread("input/03.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
if img is None:
    print("读图失败")
    exit()
print(img.shape,img.dtype,img.size)

d = [[x,y] for x in range(-1,2) for y in range(-1,2)]
#x方向梯度
def gauss(img):
    H, W = img.shape
    pad = np.pad(img, 1, mode='edge')
    out = np.zeros_like(img, dtype=np.float32) 
    k = [1,2,1,
        2,4,2,
        1,2,1]
    for y in range(H):
            for x in range(W):
                s = 0
                for index,p in enumerate(d):
                    s+=pad[y+p[0]][x+p[1]]*k[index]
                out[y][x] = s/16
    return out
def sobel_x(img):
    H, W = img.shape
    pad = np.pad(img, 1, mode='edge')
    out = np.zeros_like(img, dtype=np.float32)
    k = [-1,0,1,
         -2,0,2,
         -1,0,1]
    
    for y in range(H):
        for x in range(W):
            s = 0
            for index,p in enumerate(d):
                s+=pad[y+p[0]][x+p[1]]*k[index]
            out[y][x] = s/8
    return out
#y方向梯度
def sobel_y(img):
    H, W = img.shape
    pad = np.pad(img, 1, mode='edge')
    out = np.zeros_like(img, dtype=np.float32)
    k = [-1,-2,-1,
         0,0,0,
         1,2,1]
    
    for y in range(H):
        for x in range(W):
            s = 0
            for index,p in enumerate(d):
                s+=pad[y+p[0]][x+p[1]]*k[index]
            out[y][x] = s/8
    return out

def sobel_mag(grad_x,grad_y):
    grad_mag = np.sqrt(grad_x**2+grad_y**2)
    return grad_mag

def sobel_angle(grad_x,grad_y):
    out = np.zeros_like(grad_x, dtype=np.float32)
    theta = np.arctan2(grad_y,grad_x) 
    angle = np.degrees(theta)%360
    angle_u8 = (angle / 360*255.0).astype(np.uint8)
    out = cv2.applyColorMap(angle_u8,cv2.COLORMAP_JET)
    return out

gray = gauss(img)
grad_x = sobel_x(gray)
grad_y = sobel_y(gray)
grad_mag = sobel_mag(grad_x,grad_y)
grad_angle = sobel_angle(grad_x,grad_y)
gx_show = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gy_show = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
gm_show = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("output/sobel_x.jpg",gx_show)
cv2.imwrite("output/sobel_y.jpg",gy_show)
cv2.imwrite("output/sobel_mag.jpg",gm_show)
cv2.imwrite("output/sobel_angle.jpg",grad_angle)
cv2.imwrite("output/gray.jpg",gray.astype(np.uint8))