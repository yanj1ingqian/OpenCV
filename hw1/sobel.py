import cv2 
import numpy as np

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

def sobel_angle(grad_x,grad_y,grad_mag):
    grad_x = grad_x
    grad_y = grad_y
    out = np.zeros_like(grad_x, dtype=np.float32)
    theta = np.arctan2(grad_y,grad_x) 
    angle = np.degrees(theta)%360
    angle_u8 = (angle / 360*255.0).astype(np.uint8)
    mag = cv2.normalize(grad_mag,None,0,1,cv2.NORM_MINMAX)
    out = cv2.applyColorMap(angle_u8,cv2.COLORMAP_JET)
    
    out = out * mag[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
