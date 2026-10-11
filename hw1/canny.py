import cv2 
import numpy as np
from collections import deque
from sobel import *

T1 = 0.15
T2 = 0.05

img = cv2.imread("hw1/input/02.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("读图失败")
    exit()
img = img.astype(np.float32)

def canny(img):
    H,W = img.shape
    img = gauss(img)
    gx = sobel_x(img)
    gy = sobel_y(img)
    gm = sobel_mag(gx,gy)
    pad = np.pad(gm, 1, mode='edge')
    theta = np.arctan2(gy,gx)
    angle = np.degrees(theta)%180
    
    for y in range(H):
        for x in range(W):
            a = angle[y][x]
            if a < 22.5 or a >= 157.5:
                p1, p2 = (0, -1), (0, 1)
            elif a < 67.5:
                p1, p2 = (-1, 1), (1, -1)
            elif a < 112.5:
                p1, p2 = (-1, 0), (1, 0)
            else:
                p1, p2 = (-1, -1), (1, 1)
            if pad[y+1][x+1] <pad[y+1+p1[0]][x+1+p1[1]] or pad[y+1][x+1] <pad[y+1+p2[0]][x+1+p2[1]]:
                gm[y,x] = 0.0
    t1 = T1 * np.max(gm)
    t2 = T2 * np.max(gm)

    STRONG, WEAK = 255, 128
    label = np.zeros((H, W), dtype=np.uint8)
    ys, xs = np.where(gm >= t1)
    label[ys, xs] = STRONG
    ys, xs = np.where((gm >= t2) & (gm < t1))
    label[ys, xs] = WEAK

    q = deque(zip(*np.where(label == STRONG)))

    while q:
        y,x = q.popleft()
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and label[ny,nx] == WEAK:
                    label[ny,nx] = STRONG
                    q.append((ny,nx))
    out = np.zeros((H, W), dtype=np.uint8)
    out[label == STRONG] = 255   
    return out

img_canny= canny(img)
img_canny = cv2.normalize(img_canny, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("hw1/output/canny.png",img_canny)