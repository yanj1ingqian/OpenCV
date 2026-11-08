import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
img1 = cv2.imread(os.path.join(BASE, 'input', 'left.jpg'))
img2 = cv2.imread(os.path.join(BASE, 'input', 'right.jpg'))
assert img1 is not None, "Image 1 not found"
assert img2 is not None, "Image 2 not found"
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(g1, None)
kp2, des2 = sift.detectAndCompute(g2, None)
print(f"Number of keypoints: {len(kp1)}, {len(kp2)}")

knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1,des2,k=2)
good = [m for m,n in knn if m.distance < 0.75*n.distance]
print(f"Number of good matches: {len(good)}")

pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

def normalize_points(pts):
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1.0]])
    h = np.hstack([pts, np.ones((len(pts), 1))])
    return ((T @ h.T).T)[:, :2], T


def dlt(pts1, pts2):
    p1, T1 = normalize_points(pts1)
    p2, T2 = normalize_points(pts2)
    A = []
    for (x, y), (u, v) in zip(p1, p2):
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
    return H / H[2, 2]


def apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    q = (H @ h.T).T
    w = q[:, 2:3].copy()
    w[np.abs(w) < 1e-12] = 1e-12
    return q[:, :2] / w


def ransac(pts1, pts2, thresh=4.0, max_iter=3000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pts1)
    best_H, best_mask, best_cnt = None, np.zeros(n, bool), 0

    for _ in range(max_iter):
        idx = rng.choice(n, 4, replace=False)

        # 任意三点共线 → A 秩亏，重抽
        ok = True
        for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            v1 = pts1[idx[b]] - pts1[idx[a]]
            v2 = pts1[idx[c]] - pts1[idx[a]]
            if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1.0:
                ok = False
                break
        if not ok:
            continue

        H = dlt(pts1[idx], pts2[idx])
        err = np.linalg.norm(apply_h(H, pts1) - pts2, axis=1)
        mask = err < thresh
        if mask.sum() > best_cnt:
            best_cnt, best_mask, best_H = int(mask.sum()), mask, H

    if best_cnt < 10:
        return None, best_mask

    # 用全部内点重算（对应 4.4 第 4 步）
    H = dlt(pts1[best_mask], pts2[best_mask])
    mask = np.linalg.norm(apply_h(H, pts1) - pts2, axis=1) < thresh
    return H, mask


H, mask = ransac(pts1, pts2)
assert H is not None, "RANSAC 失败，内点不足"
print("内点率: %d/%d = %.1f%%" % (mask.sum(), len(mask), 100 * mask.mean()))
print("H =")
print(np.round(H, 6))

def stitch(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 包围盒：图1 四角经 H 变换后，与图2 四角求并集
    c1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(c1, H).reshape(-1, 2)
    c2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
    allc = np.vstack([warped, c2])
    xmin, ymin = np.floor(allc.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=0)).astype(int)
    ow, oh = int(xmax - xmin), int(ymax - ymin)

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    M = T @ H

    # 图1：变换到画布，同时得到它的有效区域掩码
    canvas1 = cv2.warpPerspective(img1, M, (ow, oh))
    mask1 = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8), M, (ow, oh),
                                flags=cv2.INTER_NEAREST)

    # 图2：平移到画布（其余区域为 0），同样生成掩码
    canvas2 = np.zeros_like(canvas1)
    mask2 = np.zeros((oh, ow), np.uint8)
    ox, oy = -xmin, -ymin
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + w2, ow), min(oy + h2, oh)
    if x1 > x0 and y1 > y0:
        canvas2[y0:y1, x0:x1] = img2[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        mask2[y0:y1, x0:x1] = 255

    # 距离变换 → 羽化权重：离接缝越远，这一侧越可信
    d1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3)
    d2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)
    total = d1 + d2 + 1e-6
    a1 = (d1 / total)[..., None]
    a2 = (d2 / total)[..., None]

    out = canvas1.astype(np.float32) * a1 + canvas2.astype(np.float32) * a2
    return np.clip(out, 0, 255).astype(np.uint8)


OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
inliers = [good[i] for i in np.where(mask)[0]]
cv2.imwrite(os.path.join(OUT, "inliers.png"),
            cv2.drawMatches(img1, kp1, img2, kp2, inliers, None,
                            matchColor=(0, 200, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
cv2.imwrite(os.path.join(OUT, "stitched.jpg"), stitch(img1, img2, H))
print("完成 -> output/inliers.png, output/stitched.jpg")
