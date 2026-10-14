"""SIFT 教程配套脚本（配合《SIFT教程.md》使用）

用法：
    python tutorial_sift.py          # 跑全部 4 个 Demo
    python tutorial_sift.py 1        # 只跑 Demo 1
    python tutorial_sift.py 1 3      # 跑 Demo 1 和 3

Demo 1  关键点检测、属性统计、尺度可视化        —— 教程第 2、3 章
Demo 2  参数对比实验（表格输出）                —— 教程第 5 章
Demo 3  全流程匹配：比率检验→交叉验证→RANSAC     —— 教程第 4 章
Demo 4  物体定位 + 全景拼接                     —— 教程第 6、7 章

调用方式（可作为模块复用）：
    from tutorial_sift import match_sift, detect_sift, draw_rich_keypoints
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
IMG = os.path.join(BASE, "..", "hw1", "input", "02.jpg")   # 1024x683，教程实测用的就是它
SCENE = os.path.join(BASE, "..", "hw1", "input", "01.jpg")  # 2499x1562，Demo 4 当场景用

RATIO = 0.75          # Lowe 比率检验阈值
RANSAC_THRESH = 5.0   # RANSAC 重投影误差阈值（像素）


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def imread_gray(path: str) -> np.ndarray:
    """读图并转灰度，失败时给出明确错误。"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"读图失败: {os.path.abspath(path)}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def imread_color(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"读图失败: {os.path.abspath(path)}")
    return img


def detect_sift(gray: np.ndarray, **kwargs):
    """一步到位：返回 (keypoints, descriptors)。参考教程 2.1 / 2.2 节。"""
    sift = cv2.SIFT_create(**kwargs)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def draw_rich_keypoints(img: np.ndarray, kp) -> np.ndarray:
    """带尺度(圆半径)和主方向(圆内短线)的关键点可视化。参考教程第 3 章。"""
    return cv2.drawKeypoints(
        img, kp, None, color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def match_sift(img1, img2, ratio: float = RATIO, ransac_thresh: float = RANSAC_THRESH):
    """SIFT 标准匹配配方：比率检验 → 交叉验证 → RANSAC。

    参数 img1/img2 可以是灰度图或 BGR 图（函数内部统一转灰度）。

    返回 dict:
        kp1, des1, kp2, des2 : 检测结果
        good     : 比率检验后的匹配
        mutual   : 交叉验证后的匹配
        inliers  : RANSAC 内点（几何一致，最终结果）
        H, mask  : 单应矩阵（把 img1 坐标映射到 img2 坐标）与内点掩码
    """
    g1 = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = img2 if img2.ndim == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    out = dict(kp1=kp1, des1=des1, kp2=kp2, des2=des2,
               good=[], mutual=[], inliers=[], H=None, mask=None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        print("  [警告] 描述子为空，检查图像是否过小/过平/过糊")
        return out

    bf = cv2.BFMatcher(cv2.NORM_L2)   # SIFT 描述子是 float32，固定用 L2

    # ① 比率检验：去掉"没有区分度"的匹配
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]

    # ② 交叉验证：互相是对方的最佳匹配
    knn_back = bf.knnMatch(des2, des1, k=2)
    back = {m.queryIdx: m.trainIdx for m, n in knn_back if m.distance < ratio * n.distance}
    mutual = [m for m in good if back.get(m.trainIdx) == m.queryIdx]

    out.update(good=good, mutual=mutual)
    print(f"  关键点: {len(kp1)} vs {len(kp2)}")
    print(f"  ① 比率检验({ratio})      -> {len(good)} 条")
    print(f"  ② 交叉验证              -> {len(mutual)} 条")

    # ③ RANSAC 几何验证
    if len(mutual) < 4:
        print("  [警告] 匹配少于 4 对，无法求单应矩阵（两张图可能没有共同内容）")
        return out

    src = np.float32([kp1[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        print("  [警告] findHomography 返回 None")
        return out

    inliers = [m for m, ok in zip(mutual, mask.ravel()) if ok]
    out.update(H=H, mask=mask, inliers=inliers)
    print(f"  ③ RANSAC 内点           -> {len(inliers)}/{len(mutual)} "
          f"({100.0 * len(inliers) / max(len(mutual), 1):.0f}%)")
    return out


def save(name: str, img: np.ndarray) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f"  已保存 {os.path.relpath(path, BASE)}")
    return path


# --------------------------------------------------------------------------- #
# Demo 1：关键点检测与可视化（教程第 2、3 章）
# --------------------------------------------------------------------------- #
def demo1() -> None:
    print("\n=== Demo 1: 关键点检测、属性与可视化（第 2、3 章）===")
    img = imread_color(IMG)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  图像尺寸: {gray.shape}")

    kp, des = detect_sift(gray)
    print(f"  关键点数: {len(kp)}   描述子: {des.shape} {des.dtype}")

    # --- 单个关键点的字段含义（教程 2.2 节表格） ---
    k = kp[0]
    print("  第一个关键点:")
    print(f"    pt=({k.pt[0]:.2f}, {k.pt[1]:.2f})  size={k.size:.2f} (σ≈{k.size / 2:.2f})")
    print(f"    angle={k.angle:.2f}°  response={k.response:.5f}  octave={k.octave}(编码值,勿直接解读)")

    # --- 描述子的存储细节（教程 1.6 节） ---
    norms = np.linalg.norm(des.astype(np.float32), axis=1)
    print(f"  描述子 L2 范数: min={norms.min():.2f} max={norms.max():.2f} "
          f"mean={norms.mean():.2f}（≈512，OpenCV 的存储约定）")
    print(f"  描述子取值范围: {des.min():.1f} ~ {des.max():.1f}")

    # --- 尺度分布 ---
    sizes = np.array([x.size for x in kp])
    print(f"  尺度 size: 中位数={np.median(sizes):.2f} 最大={sizes.max():.2f}")
    print("  尺度分布:")
    for lo, hi in [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]:
        n = int(((sizes >= lo) & (sizes < hi)).sum())
        print(f"    size {lo:>2}-{hi if hi < 1e9 else '∞':<3}: {n:5d} 个 {'#' * (n * 40 // max(len(kp), 1))}")

    # --- 方向分布 ---
    angles = np.array([x.angle for x in kp])
    print(f"  主方向 angle: {angles.min():.1f}° ~ {angles.max():.1f}°（覆盖全区间说明各朝向都有点）")

    # --- 可视化 1：带尺度+方向 ---
    save("demo1_keypoints_rich.png", draw_rich_keypoints(img, kp))

    # --- 可视化 2：按尺度着色 ---
    canvas = img.copy()
    for x in sorted(kp, key=lambda q: -q.size):     # 大尺度先画，避免被小圆盖住
        col = (0, 255, 0) if x.size > 20 else (0, 165, 255) if x.size > 10 else (0, 0, 255)
        cv2.circle(canvas, (int(x.pt[0]), int(x.pt[1])), int(x.size / 2), col, 1)
    save("demo1_keypoints_scales.png", canvas)
    print("  颜色: 绿=大尺度(结构)  橙=中尺度  红=小尺度(细节)")

    # --- mask 做 ROI 筛选 ---
    h, w = gray.shape
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 250, 255, -1)
    kp_m, _ = cv2.SIFT_create().detectAndCompute(gray, mask)
    print(f"  用圆形 mask 后关键点: {len(kp)} -> {len(kp_m)}")

    # --- 手写一个极简描述子，理解 128 维的来历（教程 3 章） ---
    def simple_descriptor(g: np.ndarray, x: float, y: float, r: int = 16, bins: int = 8):
        x, y = int(x), int(y)
        if y - r < 1 or y + r + 1 > g.shape[0] or x - r < 1 or x + r + 1 > g.shape[1]:
            return None
        patch = g[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.rad2deg(np.arctan2(gy, gx)) + 360) % 360
        hist, _ = np.histogram(ang, bins=bins, range=(0, 360), weights=mag)
        return hist / (np.linalg.norm(hist) + 1e-7)

    hd = simple_descriptor(gray, kp[0].pt[0], kp[0].pt[1])
    if hd is not None:
        print(f"  手写 8 维类描述子: {np.round(hd, 3).tolist()}")
        print("  (真实 SIFT 只是把它升级为 4x4 子区域 x 8 方向 = 128 维，并按主方向旋转)")


# --------------------------------------------------------------------------- #
# Demo 2：参数对比实验（教程第 5 章）
# --------------------------------------------------------------------------- #
def demo2() -> None:
    print("\n=== Demo 2: 参数对比实验（第 5 章）===")
    gray = imread_gray(IMG)
    print(f"  图像尺寸: {gray.shape}")
    print(f"  {'配置':<20}{'关键点数':>8}{'size中位数':>12}{'size最大':>10}{'最小response':>14}")

    configs = [
        ("默认",               {}),
        ("nfeatures=50",       dict(nfeatures=50)),
        ("nfeatures=500",      dict(nfeatures=500)),
        ("contrast=0.02",      dict(contrastThreshold=0.02)),
        ("contrast=0.06",      dict(contrastThreshold=0.06)),
        ("contrast=0.09",      dict(contrastThreshold=0.09)),
        ("edge=5",             dict(edgeThreshold=5)),
        ("edge=20",            dict(edgeThreshold=20)),
        ("sigma=3.0",          dict(sigma=3.0)),
        ("nOctaveLayers=5",    dict(nOctaveLayers=5)),
    ]
    rows = []
    for name, kw in configs:
        kp, des = detect_sift(gray, **kw)
        if not kp:
            print(f"  {name:<20}{'0':>8}")
            continue
        sizes = np.array([k.size for k in kp])
        resp = np.array([k.response for k in kp])
        rows.append((name, len(kp), np.median(sizes), sizes.max(), resp.min()))
        print(f"  {name:<20}{len(kp):>8}{np.median(sizes):>12.2f}"
              f"{sizes.max():>10.2f}{resp.min():>14.4f}")

    print("\n  要点:")
    print("    * contrastThreshold 会被内部 /nOctaveLayers(默认3)，所以 0.09 才对应 Lowe 论文的 0.03")
    print("    * edgeThreshold 越大 -> 保留越多（名字有歧义）")
    print("    * nfeatures 按 response 截断，会同时改变尺度分布（最大 size 变小）")

    # 尺度对比：不同分辨率下的关键点数量（教程第 5 章结论 2）
    print("\n  图像分辨率的影响（为什么提速要先缩图）:")
    for s in (0.5, 1.0, 2.0):
        g = cv2.resize(gray, None, fx=s, fy=s,
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        kp, _ = detect_sift(g)
        print(f"    缩放 x{s:<4} 尺寸={str(g.shape):<14} 关键点={len(kp):6d} "
              f"(面积比 {s * s:.2f}x)")


# --------------------------------------------------------------------------- #
# Demo 3：全流程匹配（教程第 4 章）
# --------------------------------------------------------------------------- #
def demo3() -> None:
    print("\n=== Demo 3: 匹配全流程 比率检验→交叉验证→RANSAC（第 4 章）===")
    img1 = imread_color(IMG)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 造一个"另一个视角"：旋转 25° + 缩放 0.8 + 裁剪
    h, w = gray1.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 0.8)
    warp = cv2.warpAffine(img1, M, (w, h), borderValue=(255, 255, 255))
    img2 = warp[100:600, 200:900].copy()
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    save("demo3_view2.png", img2)

    # --- 反面教材：不做过滤的 bf.match() ---
    kp1, des1 = detect_sift(gray1)
    kp2, des2 = detect_sift(gray2)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.match(des1, des2)
    print(f"  [反面教材] bf.match() 直接给 {len(raw)} 条匹配（几乎每个点都有'匹配'）")
    save("demo3_match_raw.png", cv2.drawMatches(
        img1, kp1, img2, kp2, raw[:50], None,
        matchColor=(0, 128, 255), singlePointColor=(200, 200, 200), flags=2))
    print("    -> 连线杂乱，说明 match() 不能作为结果使用")

    # --- 标准流程 ---
    res = match_sift(gray1, gray2)
    good, inliers, H = res["good"], res["inliers"], res["H"]
    kp1, kp2 = res["kp1"], res["kp2"]

    save("demo3_match_good.png", cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
    print(f"  ① 比率检验结果已保存（{len(good)} 条）")

    if inliers:
        save("demo3_match_inliers.png", cv2.drawMatches(
            img1, kp1, img2, kp2, inliers, None,
            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
        print(f"  ③ 最终内点已保存（{len(inliers)} 条）")

        # 验证 H 的物理意义：用真实变换做对照
        # 图2 的生成过程是两步：① 以图1中心旋转25°+缩放0.8(M) ② 在 warp 结果上裁剪 [100:600, 200:900]
        # 注意矩阵要从右往左作用：先旋转缩放，再裁剪平移
        #   -> H_true = T_crop @ M
        # （写成 M @ T_crop 是常见错误：那表示"先裁剪再旋转"，测出来四角会差约 97 像素）
        crop_T = np.array([[1, 0, -200], [0, 1, -100], [0, 0, 1]], float)   # warp图 -> 图2
        M3 = np.vstack([M, [0, 0, 1]])
        M_true = crop_T @ M3                                               # 图1 -> 图2
        print("\n  单应矩阵校验（H 把图1坐标映射到图2坐标）:")
        print(f"    实测 H 平移量 = ({H[0, 2]:8.1f}, {H[1, 2]:8.1f})")
        print(f"    真实变换平移量 = ({M_true[0, 2]:8.1f}, {M_true[1, 2]:8.1f})")
        print(f"    实测 H 线性部分 = [[{H[0, 0]:7.3f}, {H[0, 1]:7.3f}],"
              f" [{H[1, 0]:7.3f}, {H[1, 1]:7.3f}]]")
        print(f"    真实 线性部分   = [[{M_true[0, 0]:7.3f}, {M_true[0, 1]:7.3f}],"
              f" [{M_true[1, 0]:7.3f}, {M_true[1, 1]:7.3f}]]")
        # 由内点拟合一个 4 自由度相似变换，和真值（缩放0.8 + 旋转-25°）对比
        src = np.float32([kp1[m.queryIdx].pt for m in inliers]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in inliers]).reshape(-1, 1, 2)
        A_meas, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        print(f"    由匹配点拟合的相似变换: 缩放={np.hypot(A_meas[0, 0], A_meas[1, 0]):.4f} "
              f"旋转={np.degrees(np.arctan2(A_meas[1, 0], A_meas[0, 0])):.2f}° "
              f"（真值: 缩放 0.8, 旋转 -25°）")
        corners_t = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        d_err = np.linalg.norm(
            cv2.perspectiveTransform(corners_t, H).reshape(-1, 2) -
            cv2.perspectiveTransform(corners_t, M_true).reshape(-1, 2), axis=1)
        print(f"    用 H 与真值投影图1四角的平均误差 = {d_err.mean():.2f} 像素"
              f"（亚像素级，说明 H 估计正确）")

        # 比率阈值对比
        print("\n  ratio 阈值的影响:")
        knn = bf.knnMatch(res["des1"], res["des2"], k=2)
        for r in (0.6, 0.7, 0.75, 0.8, 0.9):
            n = sum(1 for m, q in knn if m.distance < r * q.distance)
            print(f"    ratio={r:<5} -> {n:5d} 条")

    # --- FLANN 对比 ---
    print("\n  FLANN 加速对比:")
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn_f = flann.knnMatch(np.float32(res["des1"]), np.float32(res["des2"]), k=2)
    good_f = [m for m, n in knn_f if m.distance < RATIO * n.distance]
    print(f"    BFMatcher  -> {len(good)} 条")
    print(f"    FLANN      -> {len(good_f)} 条（结果几乎一致，大图时快得多）")


# --------------------------------------------------------------------------- #
# Demo 4：物体定位 + 全景拼接（教程第 6、7 章）
# --------------------------------------------------------------------------- #
def demo4() -> None:
    print("\n=== Demo 4: 物体定位 + 全景拼接（第 6、7 章）===")

    # ---------------- 6.1 物体定位 ----------------
    print("\n  [6.1] 在场景里找物体")
    obj_src = imread_color(IMG)
    scene_big = imread_color(SCENE)

    # 从 02.jpg 裁一块，旋转 30° + 放大 1.3x 贴到场景里
    patch = obj_src[150:400, 300:620].copy()
    ph, pw = patch.shape[:2]
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), 30, 1.3)
    nw, nh = int(pw * 1.6), int(ph * 1.6)
    M[0, 2] += (nw - pw) / 2
    M[1, 2] += (nh - ph) / 2
    warped_patch = cv2.warpAffine(patch, M, (nw, nh), borderValue=(255, 255, 255))

    # 场景缩小到长边 1600 再处理（教程第 5 章：尺寸是最大的成本来源）
    scale = 1600.0 / max(scene_big.shape[:2])
    scene = cv2.resize(scene_big, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    y0, x0 = int(400 * scale), int(600 * scale)
    scene[y0:y0 + nh, x0:x0 + nw] = warped_patch
    save("demo4_obj_patch.png", patch)
    save("demo4_scene.png", scene)

    res = match_sift(patch, scene)
    if res["H"] is None:
        print("  定位失败")
    else:
        H = res["H"]
        # 物体四角 -> 场景坐标（H 的方向：src=物体, dst=场景）
        corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)

        vis = scene.copy()
        cv2.polylines(vis, [np.int32(proj)], True, (0, 0, 255), 4)
        save("demo4_located.png", vis)

        # 真值对照
        true_corners = cv2.perspectiveTransform(
            np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2),
            np.vstack([M, [0, 0, 1]])).reshape(-1, 2) + np.array([x0, y0])
        err = np.linalg.norm(proj.reshape(-1, 2) - true_corners, axis=1)
        print(f"  定位误差（4 个角 vs 真值）: {np.round(err, 1).tolist()} 像素")
        print("  -> 红框应该正好套住贴上去的图案（打开 output/demo4_located.png 确认）")

        save("demo4_matches.png", cv2.resize(
            cv2.drawMatches(patch, res["kp1"], scene, res["kp2"], res["good"], None,
                            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS),
            None, fx=0.5, fy=0.5))

    # ---------------- 7.1 全景拼接 ----------------
    print("\n  [7.1] 全景拼接（手写版 + OpenCV Stitcher）")
    full = imread_color(IMG)
    a = full[:, 0:700].copy()      # 左图
    b = full[:, 500:].copy()       # 右图，重叠 200 像素
    save("demo4_pano_left.png", a)
    save("demo4_pano_right.png", b)

    res = match_sift(a, b)
    if res["H"] is not None:
        kp1, kp2 = res["kp1"], res["kp2"]      # kp1 来自 a（左图），kp2 来自 b（右图）
        inl = res["inliers"]
        # 要把 b 变换到 a 的坐标系，需要"正向"的单应：src 取 b 的点，dst 取 a 的点
        #   src = kp2[m.trainIdx]（b 的点）   dst = kp1[m.queryIdx]（a 的点）
        # 注意别写反：写反了得到的是"a -> b"的矩阵，喂给 warpPerspective 会得到一张错位图
        src = np.float32([kp2[m.trainIdx].pt for m in inl]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.queryIdx].pt for m in inl]).reshape(-1, 1, 2)
        H = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)[0]
        print(f"    b -> a 的单应: 平移量 = ({H[0, 2]:.1f}, {H[1, 2]:.1f})，"
              f"真实值应为 (500, 0)")

        W = a.shape[1] + b.shape[1]
        canvas = cv2.warpPerspective(b, H, (W, a.shape[0]))
        canvas[:, :a.shape[1]] = a
        save("demo4_pano_manual.png", canvas)

        # 羽化融合：消除接缝
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        valid = cv2.warpPerspective(np.ones(gb.shape, np.uint8) * 255, H, (W, a.shape[0]))
        alpha = np.clip(cv2.GaussianBlur(valid.astype(np.float32) / 255.0, (0, 0), 40), 0, 1)
        left = np.zeros_like(canvas)
        left[:, :a.shape[1]] = a
        alpha = np.where(left.any(axis=2), 1.0 - alpha, alpha)
        blend = (canvas.astype(np.float32) * (1 - alpha[..., None]) +
                 left.astype(np.float32) * alpha[..., None]).astype(np.uint8)
        save("demo4_pano_blend.png", blend)

        # 检验对齐质量：重叠区与原图逐像素比较（对齐好则应接近 0）
        ref = full[:, :W]
        for x0, x1, who in [(0, 200, "仅左图"), (300, 500, "仅左图"),
                            (700, 1000, "重叠/右图")]:
            d = np.abs(canvas[:, x0:x1].astype(np.float32) - ref[:, x0:x1].astype(np.float32)).mean()
            tag = "对齐良好" if d < 8 else "有明显错位（检查 H 的方向！）"
            print(f"    x {x0:>4}-{x1:<4} ({who}): 与原图平均差 {d:6.2f}  -> {tag}")

    # OpenCV 自带的 Stitcher：省事，但失败时会抛 C++ 异常，必须包 try/except
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch([a, b])
        if status == cv2.Stitcher_OK:
            save("demo4_pano_stitcher.png", pano)
        else:
            print(f"    Stitcher 失败，status={status}（常见原因：重叠不足 / 场景太单调）")
    except cv2.error as e:
        print(f"    Stitcher 抛出异常（重叠不足时会这样）: {str(e).splitlines()[0][:80]}")

    # 反面例：两张毫无重叠的图
    print("\n    试试两张无关的图（Stitcher 的表现）:")
    try:
        other = cv2.resize(imread_color(IMG)[:, ::-1], (a.shape[1], a.shape[0]))
        status, pano = cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch([a, other])
        print(f"      status={status}（非 0 表示失败，属预期）")
    except cv2.error as e:
        print(f"      抛出异常: {str(e).splitlines()[0][:70]}")
        print("      -> 所以生产代码里 Stitcher 必须包 try/except")


# --------------------------------------------------------------------------- #
DEMOS = {1: demo1, 2: demo2, 3: demo3, 4: demo4}


def main(argv: list[str]) -> int:
    os.makedirs(OUT, exist_ok=True)
    wanted = [int(a) for a in argv[1:] if a.isdigit()] or sorted(DEMOS)
    print(f"工作目录: {BASE}")
    print(f"输出目录: {OUT}")
    print(f"运行 Demo: {wanted}")
    for i in wanted:
        if i in DEMOS:
            DEMOS[i]()
        else:
            print(f"\n[跳过] 没有 Demo {i}")
    print("\n全部完成，请查看 output/ 目录下的图片。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
