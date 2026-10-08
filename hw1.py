import cv2 
import numpy as np

img = cv2.imread("input/01.jpg")
if img is None:
    print("读图失败")
print(img.shape,img.dtype,img.size)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

grad_x = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.float32)
grad_y = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.float32)
grad_mag = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.float32)
grad_angle = np.zeros((*gray.shape, 3), np.uint8)

H, W = gray.shape
#处理边缘防止越界
padded = np.pad(gray,1,mode = 'edge')

d = [[x,y] for x in range(-1,2) for y in range(-1,2)]

def sobel_x(gray):
    k = [-1,0,1,
         -2,0,2,
         -1,0,1]
    
    for y in range(H):
        for x in range(W):
            s = 0
            for index,p in enumerate(d):
                s+=padded[y+p[0]][x+p[1]]*k[index]
            grad_x[y][x] = s
    return grad_x

def sobel_y(gray):
    k = [-1,-2,-1,
         0,0,0,
         1,2,1]
    
    for y in range(H):
        for x in range(W):
            s = 0
            for index,p in enumerate(d):
                s+=padded[y+p[0]][x+p[1]]*k[index]
            grad_y[y][x] = s
    return grad_y

def sobel_mag(grad_x,grad_y):
    grad_mag = np.sqrt(grad_x**2+grad_y**2)
    return grad_mag

def sobel_angle(grad_x,grad_y):
    grad_mag = np.sqrt(grad_x**2+grad_y**2)
    nz = np.maximum(grad_mag,1e-6)
    

    ux = grad_x/nz
    uy = grad_y/nz
    scale = max(float(grad_mag.max()), 1e-6)

    R = ((ux+1.0)*127.5).astype(np.uint8)
    G = ((uy+1.0)*127.5).astype(np.uint8)
    B = np.clip(grad_mag / scale * 255.0, 0, 255).astype(np.uint8)

    return np.dstack([R,G,B])
    

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
cv2.imwrite("output/sobel_angle.jpg",cv2.cvtColor(grad_angle, cv2.COLOR_RGB2BGR))
cv2.imwrite("output/gray.jpg",gray.astype(np.uint8))