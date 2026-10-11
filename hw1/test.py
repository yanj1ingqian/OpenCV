import os
import cv2
import numpy as np
from sobel import *
from canny import canny
from matplotlib import pyplot as plt

if __name__ == "__main__":

    BASE = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)

    #读图
    img = cv2.imread(os.path.join(BASE, "input/02.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("读图失败")
        exit()
    img = img.astype(np.float32)
    print(img.shape, img.dtype, img.size)

    #Sobel
    gray = gauss(img)
    grad_x = sobel_x(gray)
    grad_y = sobel_y(gray)
    grad_mag = sobel_mag(grad_x, grad_y)
    grad_angle = sobel_angle(grad_x, grad_y, grad_mag)

    gx_show = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gy_show = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gm_show = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    canny_show = canny(img)

    #保存 
    cv2.imwrite(os.path.join(BASE, "output/sobel_x.jpg"), gx_show)
    cv2.imwrite(os.path.join(BASE, "output/sobel_y.jpg"), gy_show)
    cv2.imwrite(os.path.join(BASE, "output/sobel_mag.jpg"), gm_show)
    cv2.imwrite(os.path.join(BASE, "output/sobel_angle.jpg"), grad_angle)
    cv2.imwrite(os.path.join(BASE, "output/canny.jpg"), canny_show)
    cv2.imwrite(os.path.join(BASE, "output/gray.jpg"), gray.astype(np.uint8))

    #显示
    grad_angle_rgb = cv2.cvtColor(grad_angle, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(gx_show, cmap='gray')
    axes[0, 1].set_title("Sobel X")

    axes[0, 2].imshow(gy_show, cmap='gray')
    axes[0, 2].set_title("Sobel Y")

    axes[1, 0].imshow(gm_show, cmap='gray')
    axes[1, 0].set_title("Sobel Magnitude")

    axes[1, 1].imshow(grad_angle_rgb)
    axes[1, 1].set_title("Sobel Angle")

    axes[1, 2].imshow(canny_show, cmap='gray')
    axes[1, 2].set_title("Canny")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "output/result.png"), dpi=150)
    plt.show()