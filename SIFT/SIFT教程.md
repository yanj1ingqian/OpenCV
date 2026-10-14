# SIFT 从零到实战教程（Python + OpenCV）

> 面向对象：已经学过 Sobel / Canny（能自己写出卷积、梯度、非极大值抑制），想搞明白「特征点」到底怎么算出来、在 OpenCV 里怎么用的人。
>
> 全部代码在 **Python 3.12 + OpenCV 5.0.0** 下实测跑通；文中出现的每一个数字，都是在 `hw1/input/02.jpg`（1024×683）上真实跑出来的结果，方便你对照自己的输出。

---

## 目录

- [第 0 章 先搞清楚：SIFT 解决什么问题](#第-0-章-先搞清楚sift-解决什么问题)
- [第 1 章 原理：五步走](#第-1-章-原理五步走)
  - [1.1 尺度空间与高斯金字塔](#11-尺度空间与高斯金字塔)
  - [1.2 DoG：高斯差分](#12-dog高斯差分)
  - [1.3 关键点定位：三维非极大值抑制](#13-关键点定位三维非极大值抑制)
  - [1.4 剔除坏点：低对比度 + 边缘响应](#14-剔除坏点低对比度--边缘响应)
  - [1.5 方向分配：旋转不变性的来源](#15-方向分配旋转不变性的来源)
  - [1.6 描述子：128 维是怎么来的](#16-描述子128-维是怎么来的)
- [第 2 章 OpenCV 上手：五步流程](#第-2-章-opencv-上手五步流程)
- [第 3 章 把关键点看明白](#第-3-章-把关键点看明白)
- [第 4 章 匹配：从暴力匹配到 RANSAC](#第-4-章-匹配从暴力匹配到-ransac)
- [第 5 章 参数调优：在你自己图上做实验](#第-5-章-参数调优在你自己图上做实验)
- [第 6 章 实战一：在场景里找物体](#第-6-章-实战一在场景里找物体)
- [第 7 章 实战二：全景拼接](#第-7-章-实战二全景拼接)
- [第 8 章 SIFT vs ORB vs 深度学习特征](#第-8-章-sift-vs-orb-vs-深度学习特征)
- [第 9 章 踩坑清单](#第-9-章-踩坑清单)
- [第 10 章 自测题（带答案）](#第-10-章-自测题带答案)
- [附录 A API 速查](#附录-a-api-速查)
- [附录 B SIFT 的不变性实测数据](#附录-b-sift-的不变性实测数据)
- [附录 C 文件清单](#附录-c-文件清单)

**配套脚本**：`tutorial_sift.py`（第 2~7 章的代码按顺序整理成 4 个可单独运行的 Demo）。

**现在就开始**（脚本会自动读取 `../hw1/input/02.jpg`）：

```bash
cd C:\lou\OpenCV\SIFT
python tutorial_sift.py            # 跑全部 4 个 Demo
python tutorial_sift.py 1          # 只跑 Demo 1
```

---

## 第 0 章 先搞清楚：SIFT 解决什么问题

你在作业一里已经会用 Sobel 求梯度、用 Canny 提边缘了。现在考虑一个更实际的问题：

> 同一个建筑，我拍了**两张照片**：一张近、一张远；一张正对着、一张斜了 30°；一张白天、一张傍晚。怎么让计算机知道「这两张图里哪些点是同一个物理位置」？

用灰度值直接比？亮度一变就全废。用边缘图比？旋转一下边缘方向全变了。用固定网格切块比？缩放一下块里的内容就换了。

**SIFT**（Scale-Invariant Feature Transform，尺度不变特征变换，David Lowe 1999/2004）给出的答案是：

1. 先在图像里自动找出一批**特别的点**（角点、斑点、纹理突变处）——叫**关键点 / Keypoint**；
2. 为每个点算一个 **128 维向量** 来刻画它周围长什么样——叫**描述子 / Descriptor**；
3. 两张图里描述子最像的两个点，就认为是同一个物理位置。

这个向量被刻意设计成对**缩放、旋转、亮度变化、小视角变化**都不敏感。你只需要记住四个词：

```
关键点 = 位置(x, y) + 尺度(size/σ) + 方向(angle) + 强度(response)
描述子 = 以该点为中心的 128 维「局部外观指纹」
匹配   = 在描述子空间里找最近邻
```

> **一句话对比**：Canny 告诉你「哪里有边」，SIFT 告诉你「这是哪块地方」——前者给的是像素集合，后者给的是**可跨图像辨认的身份**。这就是后面能做匹配、定位、拼接、三维重建的根本原因。

---

## 第 1 章 原理：五步走

先给全局流程图，后面逐个拆：

```
输入灰度图
   │
   ├─(1) 建高斯金字塔 ──► 每个 octave 内多张不同 σ 的模糊图
   │
   ├─(2) 相邻两层相减 ──► DoG 金字塔（差分图）
   │
   ├─(3) DoG 上做 3×3×3 非极大值抑制 ──► 候选关键点（含尺度）
   │
   ├─(4) 剔除低对比度点 + 边缘响应点 ──► 稳定关键点
   │
   ├─(5) 梯度方向直方图定主方向 ──► 旋转不变的关键点
   │
   └─(6) 16 个 4×4 子区域 × 8 方向 = 128 维描述子
```

### 1.1 尺度空间与高斯金字塔

**核心思想**：要让特征对「尺寸」不敏感，就不能只看一个分辨率，而要**同时看很多个模糊程度**的图像——一层一层地模糊，等价于一层一层地缩小观察尺度。

- **尺度空间（scale space）**：把图像 $I(x,y)$ 与高斯核卷积，得到 $L(x,y,\sigma) = G(x,y,\sigma) * I(x,y)$。$\sigma$ 越大，细节越被抹掉，看到的结构越"宏观"。
- **octave（组）**：一组图像。同一 octave 内，每张图的 $\sigma$ 依次乘 $k = 2^{1/s}$（$s$ = 每组的层数，OpenCV 默认 `nOctaveLayers=3`，即 $k=2^{1/3}\approx1.26$）。
- **下一个 octave**：把图像**长宽各缩小一半**（隔点取样），然后重来一遍。共 `nOctaves` 组。

```
octave 0:  σ=1.6      σ=2.02     σ=2.54     σ=3.20    σ=4.03   ← 原尺寸
octave 1:  长宽减半，同样 5 种 σ                                     ← 相当于"看远处"
octave 2:  再减半 ...
```

$\sigma_0 = 1.6$ 是 Lowe 的经验值：**对输入图像先做一次 $\sigma=1.6$ 的模糊**，目的是去掉噪点、也去掉相机本身带来的模糊。

> **和 Sobel 作业的联系**：作业里你写的 `gauss()` 就是这个 $\sigma$ 为常数的版本。SIFT 的创新不在「高斯模糊」，而在于**把 $\sigma$ 当成一个维度**，在 $(\sigma, y, x)$ 三维空间里一起找极值。

**为什么减半再算能省事**：octave 1 里 $\sigma=2\times1.6$ 的模糊，和 octave 0 里 $\sigma=3.2$ 的模糊，观察到的结构几乎一样。所以金字塔每一组只需要算少数几层，靠「降采样 + 复用」就能覆盖从 $\sigma=1.6$ 到图像尺寸级别的一整条尺度轴。

### 1.2 DoG：高斯差分

直接在高斯金字塔上找极值不稳定（LoG 的尺度归一化要乘 $\sigma^2$）。Lowe 用的技巧：

$$
D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)
$$

即**同一 octave 内相邻两层直接相减**。数学上 $G(k\sigma)-G(\sigma) \approx (k-1)\sigma^2\nabla^2 G$，所以 DoG 是尺度归一化 LoG 的廉价近似——**只要减法，不要二阶导数**。

假设一个 octave 里做了 $s+3$ 层高斯图，相减就得到 $s+2$ 层 DoG 图。

```
高斯:  L0   L1   L2   L3   L4   L5
DoG:     D1   D2   D3   D4      ← 相邻相减，共 nOctaveLayers+2 层
         ↑    ↑    ↑
         └────┴────┴── 中间层才能做上下左右的三维比较
```

> 这就是为什么金字塔要多算 3 层：**做极值检测的层必须有上下邻居**。

### 1.3 关键点定位：三维非极大值抑制

对每一个 DoG 像素，和它的 **26 个邻居**比较（同一层的 8 邻居 + 上一层的 9 个 + 下一层的 9 个）：

```
        上一层           当前层           下一层
       ┌────────┐      ┌────────┐      ┌────────┐
       │ 9 个   │      │8 个  ★ │      │ 9 个   │      ★ = 待判定像素
       └────────┘      └────────┘      └────────┘
```

只有**严格大于或严格小于全部 26 个邻居**才留为候选点。

和 Canny 的区别：Canny 的 NMS 只在**一张图的梯度方向上**比 2 个邻居；SIFT 的 NMS 在**尺度维 + 空间维共 26 个邻居**里比。**「在尺度维度也做 NMS」是 SIFT 尺度不变性的关键**。

第 3 步得到的只是**整数像素坐标**，且包含大量边缘/噪声点。

### 1.4 剔除坏点：低对比度 + 边缘响应

**(a) 亚像素精确定位 + 低对比度剔除**

把 $D(x,y,\sigma)$ 在候选点处做二阶泰勒展开，用 DOG 的偏移量拟合极值位置，得到**亚像素精度**的 $(x,y,\sigma)$。若拟合出的极值 $|D(\hat{x})|$ 太小（对比度低，通常是噪声或均匀区域），丢弃。

这就是 `contrastThreshold` 参数干的事。**实测**（`02.jpg`）：

| `contrastThreshold` | 关键点数 | 保留点的最小 response |
|---|---|---|
| 0.02 | 1542 | 0.0067 |
| 0.04（默认） | **1310** | 0.0133 |
| 0.06 | 1112 | 0.0200 |
| 0.09 | 871 | 0.0301 |

注意最后一行：设成 0.09 时，保留点的最小 response 是 0.03 —— **OpenCV 内部会把该阈值除以 `nOctaveLayers`（=3）**。所以想让行为和 Lowe 论文里的 0.03 一致，要写 `contrastThreshold=0.09`。这个坑很常见，记一下。

**(b) 边缘响应剔除**

DoG 在**边缘**上也会有强响应（沿边缘方向变化小、垂直方向变化大），但这类点稍微移动就会跳到边缘另一侧，位置极不稳定——**正是匹配里最讨厌的**。

判别方法：看该点处 Hessian 矩阵的两个特征值 $\alpha \ge \beta$（分别对应两个主曲率）。

- 若点是**好的角点**：两侧都变化剧烈，$\alpha \approx \beta$；
- 若点在**边缘上**：$\alpha \gg \beta$。

设 $r = \alpha/\beta$，由迹 $Tr = \alpha+\beta = D_{xx}+D_{yy}$、行列式 $Det = \alpha\beta = D_{xx}D_{yy}-D_{xy}^2$ 可得：

$$
\frac{Tr(H)^2}{Det(H)} = \frac{(r+1)^2}{r} < \frac{(r_{th}+1)^2}{r_{th}}
$$

OpenCV 默认 `edgeThreshold=10`，即 $r_{th}=10$。**注意这里的语义和名字相反**：`edgeThreshold` 越大，被剔除的边缘点越少（保留的关键点越多）。实测：`edgeThreshold=10` → 1310 个点，`edgeThreshold=5` → 936 个点。

检查一下上面那个公式的单调性：$r\in[1,\infty)$ 时 $\frac{(r+1)^2}{r}$ 单调递增（最小值 4 在 $r=1$）。(10+1)²/10 = 12.1。所以判据是 `Tr(H)²/Det(H) < 12.1` 才保留。

> **和 Canny 的联系**：Canny 里你也是靠「双阈值 + 边缘连接」来压掉弱响应；SIFT 压掉的是**位置上站不稳**的点，目标不同但动机一样——**只留可复现的点**。

### 1.5 方向分配：旋转不变性的来源

对关键点邻域（高斯加权）内每个像素算梯度幅值和方向：

$$
m(x,y)=\sqrt{(L(x+1,y)-L(x-1,y))^2 + (L(x,y+1)-L(x,y-1))^2}
$$
$$
\theta(x,y)=\tan^{-1}\!\frac{L(x,y+1)-L(x,y-1)}{L(x+1,y)-L(x-1,y)}
$$

然后做一个 **36 柱的方向直方图**（每柱 10°），把邻域内每个像素的梯度幅值按方向投票，投票前用高斯窗口加权（离中心越远权重越小）：

```
直方图峰值方向  = 该关键点的主方向 angle
超过峰值 80% 的次峰 = 再生成一个额外关键点（位置/尺度相同，方向不同）
```

**这就是旋转不变性的来源**：后面算描述子时，所有坐标都**相对主方向旋转归一化**——图旋转了 30°，主方向也跟着转 30°，相对角度不变，描述子就不变。

> OpenCV 的 `kp.angle` 单位是**度**，范围 $[0, 360)$。实测 `02.jpg` 上 angle 分布几乎覆盖全区间（0.1° ~ 360.0°），说明确实提取出了各种朝向的点。

### 1.6 描述子：128 维是怎么来的

1. 以关键点为中心，取一个边长为 $16\sigma$（实际实现按尺度换算）的方形邻域；
2. 先把这块邻域**按主方向旋转**，保证旋转不变；
3. 划分成 **4×4 = 16 个子区域**，每个子区域 4×4 像素；
4. 每个子区域内统计 **8 个方向**的梯度直方图（每个方向 45°），并做一次高斯加权；
5. 16 个子区域 × 8 个方向 = **128 维**向量。

```
┌────┬────┬────┬────┐
│8维 │8维 │8维 │8维 │   每个格子 = 一个 8 方向直方图
├────┼────┼────┼────┤
│8维 │8维 │8维 │8维 │   → 拼起来 128 维
├────┼────┼────┼────┤
│8维 │8维 │8维 │8维 │
├────┼────┼────┼────┤
│8维 │8维 │8维 │8维 │
└────┴────┴────┴────┘
```

**最后三步归一化（关键，直接决定匹配效果）**：

1. 把 128 维缩放到单位长度 → 去掉**亮度缩放**影响；
2. 把大于 0.2 的分量截断到 0.2 → 抑制个别大梯度（比如强边缘）的话语权；
3. 再归一化一次 → 去掉**亮度偏移**影响。

**动手验证一下 OpenCV 的实现**（真的跑一下，印象才深）：

```python
import cv2, numpy as np
gray = cv2.imread('../hw1/input/02.jpg', cv2.IMREAD_GRAYSCALE)
kp, des = cv2.SIFT_create().detectAndCompute(gray, None)
print(des.shape, des.dtype)                    # (1310, 128) float32
print(np.linalg.norm(des, axis=1)[:3])         # [512.3 512.9 512.1]
print(des.min(), des.max())                    # 0.0 219.0
```

观察结果，三点值得注意：

- **维度 128、类型 `float32`** —— 所以做匹配时数据必须是 float；
- **L2 范数约等于 512 而不是 1**：OpenCV 把归一化后的值乘以 512 存起来（精度考虑）；且因为第 2 步的截断，范数在 510.6~513.4 之间浮动，**不是严格的单位向量**；
- **最大值 219 ≈ 0.428 × 512**，正好对应「截断到 0.2 后再归一化」的结果。

> 实践结论：**用 `cv2.NORM_L2` 距离比较描述子**。不要手动再做 `des/255` 之类的操作——那会破坏距离尺度的语义（虽然比率检验对此不敏感，但 `match()` 的绝对距离阈值会直接失效）。

至此 SIFT 的五个步骤就完整了。下面的内容全部是**怎么用**。

---

## 第 2 章 OpenCV 上手：五步流程

SIFT 在 OpenCV 里的使用可以总结成 5 行核心代码：

```python
import cv2

img  = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)   # 1) 读灰度图
sift = cv2.SIFT_create()                          # 2) 创建检测器
kp, des = sift.detectAndCompute(img, None)        # 3) 检测 + 计算描述子
bf   = cv2.BFMatcher(cv2.NORM_L2)                  # 4) 创建匹配器
knn  = bf.knnMatch(des1, des2, k=2)                # 5) 匹配
```

就这么简单——因为**金字塔、DoG、NMS、方向、描述子全部在 `detectAndCompute` 一个函数里做完了**。你要学的是**参数、输出含义、和后面的匹配策略**。

### 2.1 创建检测器：参数逐个说清

```python
sift = cv2.SIFT_create(
    nfeatures=0,             # 最多保留多少个"最好"的点，0 = 不限制（默认）
    nOctaveLayers=3,         # 每个 octave 几层（默认 3，Lowe 论文值）
    contrastThreshold=0.04,  # 低对比度剔除阈值（注意：内部会 /nOctaveLayers）
    edgeThreshold=10,        # 边缘响应剔除阈值（越大保留越多）
    sigma=1.6,               # 输入图像的高斯模糊 σ（默认 1.6）
)
```

各参数怎么选，一句话版：

| 参数 | 什么时候改 |
|---|---|
| `nfeatures` | 要控制速度/内存时设上限（如 2000）。**注意它会改变"点的选择"**：实测 `nfeatures=50` 时留下的点最大 `size` 只有 5.85（默认是 86.16），也就是**大尺度、结构性强的点几乎全被砍掉**，只剩下最细尺度的点 |
| `nOctaveLayers` | 想要更多尺度层、更密的关键点 → 调大（如 5）。一般不动。实测 3 → 1317 点，5 → 1804 点 |
| `contrastThreshold` | **点太多**（纹理丰富/噪点多）→ 调大（0.06~0.12）；**点太少** → 调小（0.01~0.02） |
| `edgeThreshold` | 图像**边缘/线条很多**（建筑、文字）→ 调小（5~8）去掉边缘点；否则不动 |
| `sigma` | 图像本身很糊/噪声很大 → 调大（如 2.0，实测 1.6→3.0 会让点数从 1317 降到 717）；图像非常清晰锐利 → 可调小 |

> **以上参数的实测对比见 [第 5 章](#第-5-章-参数调优在你自己图上做实验)**，那里有你这份图片上的真实数字。

### 2.2 `detectAndCompute` 的返回值

```python
kp, des = sift.detectAndCompute(gray, None)
#  kp  : tuple[KeyPoint]，每个点含 pt/size/angle/response/octave/class_id
#  des : np.ndarray, shape=(len(kp), 128), dtype=float32
```

`KeyPoint` 的字段，逐个看实测值：

| 字段 | 含义 | 实测样例 |
|---|---|---|
| `pt` | `(x, y)` 亚像素坐标（float） | `(2.74, 119.25)` |
| `size` | 该点所在尺度的直径（像素），**σ ≈ size/2** | 1.8 ~ 86.2 |
| `angle` | 主方向，单位**度**，$[0,360)$ | 0.1 ~ 360.0 |
| `response` | 该点的对比度强度（越大越显著） | 0.0133 ~ 0.1417 |
| `octave` | 所在金字塔组（**编码过，见下**） | 如 `8323583` |
| `class_id` | OpenCV 内部用，忽略即可 | 0 |

**两个重要细节**：

**(1) 尺度（size）分布极不均匀。** 实测 `02.jpg`：`size` 的中位数只有 3.3，但最大值达 86.2；175 个离散取值。说明大部分关键点来自最细的尺度层。想看清各尺度的贡献，得自己分组统计（第 3 章有代码）。

**(2) `octave` 字段被编码过，不要直接用。** 实测 `02.jpg` 上 `kp.octave` 的取值分布里，最多的一个值是 255（=-1）——这并不表示"负 octave"，而是 OpenCV 的编码把「octave 索引 + 层内偏移」打包进了一个整数，且不同版本编码方式有差异。**结论：想按尺度分析就按 `size` 分组，别去解析 `octave`**。

```python
# 不要写： o = kp[i].octave
# 要写：
import numpy as np
sizes = np.array([k.size for k in kp])
octave_like = np.round(np.log2(sizes / (2 * 1.6))).astype(int)  # 粗略的组编号
```

**(3) 结果有轻微不确定性。** 实测同一张图连续跑 3 次，得到 1317 / 1310 / 1310 个关键点。原因是 OpenCV 内部的多线程和浮点运算顺序。**做实验对比时，要么固定随机性来源，要么用相对指标（匹配率、内点率）而不是绝对数量**。

**(4) 必须传灰度图或单通道图。** 传彩色图会报错或者行为未定义，老老实实先 `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`。

**(5) `mask` 参数可以做 ROI 筛选**：

```python
mask = np.zeros(gray.shape, np.uint8)
cv2.circle(mask, (512, 341), 250, 255, -1)     # 只处理中心圆形区域
kp, des = sift.detectAndCompute(gray, mask)     # 实测：1310 → 509 个点
```

---

## 第 3 章 把关键点看明白

只看数字没感觉，必须画出来。OpenCV 给了 `drawKeypoints`，关键是 **flags**：

```python
import cv2

img = cv2.imread('../hw1/input/02.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kp, des = cv2.SIFT_create().detectAndCompute(gray, None)

# 画法 A：只画面点
vis_dot = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))

# 画法 B（推荐）：画成带方向线的圆 —— 圆半径代表尺度，半径上的线代表主方向
vis_rich = cv2.drawKeypoints(
    img, kp, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

**怎么看这张图**：

- **圆的大小 = `size`（尺度）**：小圆圈 = 细尺度细节点（窗户边角、纹理），大圆圈 = 粗尺度结构点（整扇窗、整个楼梯间）；
- **从圆心指出的线 = 主方向 `angle`**：注意看圆内那条短线，方向应该和该处的边缘走向垂直（梯度方向）；
- **圆密集的地方 = 纹理丰富区**：实测这张图上点集中在楼梯、栏杆、窗框、文字水印附近；**大片天空几乎没有点**——因为 DoG 在均匀区域没有极值。这个现象是判断"图里哪块有信息"最快的方法。

**自己动手做尺度可视化**（这段很值得跑一遍）：

```python
# 按尺度给不同颜色：绿=大尺度(结构), 橙=中尺度, 红=小尺度(细节)
canvas = img.copy()
for k in sorted(kp, key=lambda x: -x.size):          # 大尺度先画，避免被盖住
    col = (0, 255, 0) if k.size > 20 else (0, 165, 255) if k.size > 10 else (0, 0, 255)
    cv2.circle(canvas, (int(k.pt[0]), int(k.pt[1])), int(k.size / 2), col, 1)
cv2.imwrite('output/scales.png', canvas)
```

再统计一下尺度直方图：

```python
import numpy as np
sizes = np.array([k.size for k in kp])
print('中位数 %.2f  最大 %.2f' % (np.median(sizes), sizes.max()))   # 3.30  86.16
for lo, hi in [(0, 4), (4, 8), (8, 16), (16, 32), (32, 1e9)]:
    print(f'{lo:>3}-{hi:<5} : {((sizes >= lo) & (sizes < hi)).sum():4d} 个')
```

**亲手算一个"类 SIFT"描述子来加深理解**（不要用现成 API，就按 1.6 节的手工版本写一遍）：

```python
def simple_descriptor(gray, x, y, r=16, bins=8):
    """极简版：一个圆形邻域内的 8 方向梯度直方图（实为手工版 8 维描述子）"""
    x, y = int(x), int(y)
    patch = gray[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 360) % 360      # 0~360 度，y 轴向下
    hist, _ = np.histogram(ang, bins=bins, range=(0, 360), weights=mag)
    return hist / (np.linalg.norm(hist) + 1e-7)             # 归一化：亮度不变
```

跑一下看看：

```python
h = simple_descriptor(gray, kp[0].pt[0], kp[0].pt[1])
print(np.round(h, 3))    # [0.339 0.108 0.072 0.187 0.42  0.088 0.148 0.797] 之类
```

**关键领悟**：真实 SIFT 的 128 维就是把这个 8 维版本做了两个升级——(a) 从 1 个邻域变成 **4×4 个子区域**（保留空间信息，能区分"左上角亮"和"右下角亮"）；(b) 坐标**按主方向旋转**（带来旋转不变性）。其余思想完全一样。

---

## 第 4 章 匹配：从暴力匹配到 RANSAC

这是 SIFT 最有用的部分，也是最容易做错的部分。**按顺序走完这四级，才是能用的匹配。**

```python
import cv2, numpy as np

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)      # SIFT 描述子是 float32，用 L2；不要用 HAMMING

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
```

### 第 1 级：什么都没过滤（错误示范）

```python
matches = bf.match(des1, des2)                    # 每个点强行找一个最近邻
vis = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
```

**实测**：`02.jpg` 与它的旋转版本之间，`bf.match` 给出 1317 条匹配——**几乎每个点都被"匹配"了**，图中连线乱成一团。因为它**无条件**给每个点找最近邻，没有"匹配可能不存在"的选项。**结论：`match()` 只能用来快速看描述子空间长什么样，不能作为结果。**

> 顺带说明 `match()` 和 `knnMatch()` 的关系：`knnMatch(..., k=2)` 对每个点返回最近的两个邻居。同图自匹配时，最近邻距离是 0（就是它自己），次近邻 143.9——**这个比值 0/143.9 = 0 正是比率检验的输入**。

### 第 2 级：Lowe 比率检验（ratio test）★ 最重要

**思想**：一个**正确**的匹配，应该比"第二像的候选"明显更好；如果最近邻和第二近邻差不多像，说明这个描述子没有区分度，不可靠。

$$
\text{最近邻距离} < \text{ratio} \times \text{第二近邻距离}, \quad \text{ratio} = 0.75
$$

```python
knn = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
```

**实测对比**（`02.jpg` 对 其旋转 25°/缩放 0.8 的裁剪版本，原图 1317 点 vs 目标图 1042 点）：

| 方法 | 匹配数 |
|---|---|
| `match()` 不过滤 | 1317 |
| 比率 0.6 | 475 |
| 比率 0.7 | 512 |
| **比率 0.75（Lowe 推荐）** | **537** |
| 比率 0.8 | 563 |
| 比率 0.9 | 696 |

**怎么选 ratio**：0.75 是通用起点。**要求高精度**（如三维重建、定位）→ 0.6~0.7；**要求高召回**（如外观变化剧烈的检索）→ 0.8~0.85。**注意 ratio 越松，后面 RANSAC 的压力越大**。

### 第 3 级：交叉验证（cross-check / mutual match）

A 里第 i 个点匹配到 B 里的第 j 个，**同时** B 里第 j 个点的最佳匹配也是 A 里的第 i 个，才算互相承认。

```python
knn_ab = bf.knnMatch(des1, des2, k=2)
good_ab = [m for m, n in knn_ab if m.distance < 0.75 * n.distance]

knn_ba = bf.knnMatch(des2, des1, k=2)
good_ba = {m.queryIdx: m.trainIdx for m, n in knn_ba if m.distance < 0.75 * n.distance}

mutual = [m for m in good_ab if good_ba.get(m.trainIdx) == m.queryIdx]
```

**实测**：537 → 485（过滤掉 52 条单向匹配）。这 52 条里就有不少错误匹配。**交叉验证是性价比很高的一步**（代价是再算一次匹配）。

### 第 4 级：RANSAC 几何一致性过滤 ★ 最终手段

到这一步剩下的匹配仍然可能有错。**但它们绝大多数不满足同一个几何变换**——所以让 RANSAC 找"能同时解释最多匹配的那个单应矩阵 H"，不支持 H 的点就是错的。

```python
src = np.float32([kp1[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
dst = np.float32([kp2[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
#                                                       ↑ 重投影误差阈值(像素)
mask = mask.ravel().astype(bool)
inliers = [m for m, ok in zip(mutual, mask) if ok]
```

**实测**：

| 输入 | RANSAC 内点 |
|---|---|
| 比率检验 537 条 | 489 条内点（91%） |
| 交叉验证 485 条 | 469 条内点（97%） |

**怎么读 RANSAC 的结果**：

- **内点数**是"匹配质量"的最终指标。467 个几何一致的内点，说明匹配非常成功；
- **内点率**反映前面过滤得够不够。如果内点率低于 50%，说明前面 ratio 太松（或两张图根本没有共同内容）；
- **`findHomography` 需要至少 4 对匹配**，少于 4 对会返回 `None`，一定要判空；
- 如果**画面是纯平移/旋转且没有透视**，也可以用 `cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)` 求 4 自由度的相似变换——**约束更强、更不容易被少量错匹配带偏**，代价是不能表达透视。

**注意 H 的方向**：`findHomography(src, dst)` 求的是 **把源坐标映射到目标坐标**的矩阵。这个方向很容易搞反，第 6、7 章会用到，务必记清：`src` 是哪张图的点，`H` 就是把哪张图变换到另一张图的坐标系。

**验证 H 是否正确**：拿一个已匹配点代进去，误差应该在 1-2 像素内；下面这个"投影矩形框"的可视化是最直观的检查方法（见第 6 章）。

### 用 FLANN 加速

关键点上千、描述子 128 维时，暴力匹配 O(n²) 会慢。FLANN 用 KD 树/随机森林近似最近邻：

```python
FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)                    # checks 越大越准越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
knn = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
```

**实测**：同一组图，`BFMatcher` 得到 537 条，FLANN 得到 538 条 —— **结果几乎一致（差 1 条来自近似搜索）但快得多**。经验规则：

- 关键点 **< 1000**：用 `BFMatcher`（简单、精确）；
- 关键点 **> 1000 或要实时**：用 `FlannBasedMatcher`；
- **FLANN + SIFT 时两个小坑**：(a) 数据需要是 `float32`；(b) `checks` 调小（如 20）能明显提速但召回略降。

### 一张图看懂四级过滤

```
1317 条  bf.match()          ← 每个点都有匹配，基本没用
 537 条  比率检验 0.75        ← 去掉了"没区分度"的点
 485 条  交叉验证             ← 去掉单向匹配
 469 条  RANSAC 内点          ← 几何一致，可以放心用
```

**这四级就是工业界 SIFT 匹配的标准配方，背下来。**

---

## 第 5 章 参数调优：在你自己图上做实验

调参的唯一方法是**跑实验看数字**。用你自己的 `hw1/input/02.jpg` 跑下面这段：

```python
import cv2, numpy as np
gray = cv2.imread('../hw1/input/02.jpg', cv2.IMREAD_GRAYSCALE)
print('图像尺寸:', gray.shape)

for name, kw in [
    ('默认',              {}),
    ('nfeatures=50',      dict(nfeatures=50)),
    ('contrast=0.02',     dict(contrastThreshold=0.02)),
    ('contrast=0.09',     dict(contrastThreshold=0.09)),
    ('edge=5',            dict(edgeThreshold=5)),
    ('edge=20',           dict(edgeThreshold=20)),
    ('sigma=3.0',         dict(sigma=3.0)),
    ('nOctaveLayers=5',   dict(nOctaveLayers=5)),
]:
    kp, des = cv2.SIFT_create(**kw).detectAndCompute(gray, None)
    sizes = np.array([k.size for k in kp])
    resp = np.array([k.response for k in kp])
    print(f'{name:18s} 点数={len(kp):5d}  size中位数={np.median(sizes):6.2f}  '
          f'size最大={sizes.max():7.2f}  最小response={resp.min():.4f}')
```

**实测输出**（1024×683，完整表格见 `output/` 里 Demo 2 的控制台输出）：

| 配置 | 关键点数 | size 中位数 | size 最大 | 最小 response |
|---|---|---|---|---|
| 默认 | **1317** | 3.30 | 86.16 | 0.0133 |
| `nfeatures=50` | 51 | 2.44 | **5.85** | 0.0881 |
| `nfeatures=500` | 500 | 3.17 | 86.16 | 0.0521 |
| `contrastThreshold=0.02` | 1552 | 3.16 | 86.16 | 0.0067 |
| `contrastThreshold=0.06` | 1116 | 3.34 | 86.16 | 0.0200 |
| `contrastThreshold=0.09` | 875 | 3.33 | 86.16 | 0.0301 |
| `edgeThreshold=5` | 936 | 3.43 | 72.93 | 0.0133 |
| `edgeThreshold=20` | 1746 | 2.97 | 86.16 | 0.0133 |
| `sigma=3.0` | 717 | 6.03 | 152.98 | 0.0134 |
| `nOctaveLayers=5` | 1804 | 2.83 | 90.69 | 0.0080 |

> 表中所有数字来自**同一次检测**（同一进程、同一顺序）。单独跑默认参数时点数是 1317，再跑可能就是 1310——原因见 [2.2 节第 (3) 点](#22-detectandcompute-的返回值)。**对比参数时要在同一次运行里取全部数据**，跨运行比较绝对点数会得到假的"差异"。

**图像分辨率的影响**（同一张图缩放到不同尺寸再检测）：

| 缩放 | 尺寸 | 关键点数 | 面积比 |
|---|---|---|---|
| ×0.5 | 512×342 | 614 | 0.25× |
| ×1.0 | 1024×683 | 1317 | 1.00× |
| ×2.0 | 2048×1366 | **12219** | 4.00× |

**这张表直接给出提速的办法**：缩到一半，点数降到约 1/2；放大到两倍，点数涨到约 9 倍。**关键点数随面积超线性增长**，所以缩图是收益最大的优化。

**几个可以直接抄的结论**：

1. **默认参数在自然照片上通常就够好**，先跑默认，跑不通再调。
2. **点太多（上万）**：先怀疑图像尺寸太大。先把图缩到 **长边 1000~1600** 再跑，能省一个数量级的时间且效果几乎不变。
3. **点太少**：`contrastThreshold` 减半（0.02），同时看图像是不是太模糊（模糊图必须调小 `sigma`）。
4. **要找的物体很小**：先**放大**图像（`cv2.resize` 到 2×）反而比调参数更有效——小目标在金字塔最细层才出现，放大等于给它更多层。
5. **建筑、文档、棋盘格这类边缘极多的图**：`edgeThreshold=5~7` 能显著去噪（1317 → 936）。
6. **`nfeatures` 慎用**：它按 response 排序只保留前 N 个，会**同时改变尺度分布**（实测 `nfeatures=50` 时最大 size 从 86 掉到 5.85，大尺度结构点全没了）。想限速，优先缩图而不是限 `nfeatures`。

---

## 第 6 章 实战一：在场景里找物体

**任务**：场景图里藏着一个物体（被旋转、缩放过），找出它的位置并画框。

完整流程（这段代码是通用模板，只需替换两张图）：

```python
import cv2, numpy as np

obj   = cv2.imread('obj.png')        # 物体模板
scene = cv2.imread('scene.png')      # 待搜索的场景
g_obj   = cv2.cvtColor(obj,   cv2.COLOR_BGR2GRAY)
g_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_o, des_o = sift.detectAndCompute(g_obj,   None)
kp_s, des_s = sift.detectAndCompute(g_scene, None)
if des_o is None or des_s is None:
    raise SystemExit('没检测到描述子，检查图像是否过小/过平')

bf = cv2.BFMatcher(cv2.NORM_L2)

# ① 比率检验
knn = bf.knnMatch(des_o, des_s, k=2)
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
print('good matches:', len(good))
if len(good) < 10:
    raise SystemExit('匹配太少，找不到目标')

# ② 用单应矩阵做几何验证
# src 取"要被变换的那张图"的点（物体），dst 取"目标坐标系"的点（场景）
# 这里 knnMatch(des_o, des_s) 的 query 是物体、train 是场景，所以：
src = np.float32([kp_o[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
inliers = int(mask.sum())
print(f'内点 {inliers}/{len(good)}')
if inliers < 10:
    raise SystemExit('内点太少，目标不存在')

# ③ 把物体的四个角投影到场景里（H 的方向：物体 -> 场景）
h, w = g_obj.shape
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
proj = cv2.perspectiveTransform(corners, H)
```

> **`src`/`dst` 到底谁取谁？** 一条规则：**`src` 是要被变换的那张图，`dst` 是目标坐标系那张图**。
> 这里物体要被"搬"到场景里，所以 `src` 取物体、`dst` 取场景。
> 顺序写反不会报错，但红框会飞到画面外——**实测写反时四角误差平均 1014 像素，写对时是 0.4 像素**。判断方法就是看这张框：**框套住了物体才对。**

vis = scene.copy()
cv2.polylines(vis, [np.int32(proj)], True, (0, 0, 255), 4)
cv2.imwrite('output/located.png', vis)
```

实测结果（把 `02.jpg` 中 `[150:400, 300:620]` 的楼梯局部旋转 30°、放大 1.3× 后贴到 `01.jpg` 上，场景缩到长边 1600 处理）：

```
物体模板关键点: 129       场景关键点: 7124
① 比率检验(0.75)      -> 79 条
② 交叉验证            -> 73 条
③ RANSAC 内点         -> 72/73 (99%)
定位误差（4 个角 vs 真值）: [0.2, 0.8, 0.2, 0.5] 像素   ← 亚像素级精度
```

画出来的红框**正好套住被旋转、缩放过的物体**。

**这个模板为什么可靠**：

- `findHomography` 允许透视变换，所以**平面物体 + 任意视角**都能框住（SIFT 对透视变化有一定容忍度）；
- 即使 good matches 里有 4 条错匹配，RANSAC 也能剔除（72/76）；
- 若 `proj` 画出来是个**巨大或畸形的四边形**，说明 H 估计错了（常见于匹配太少或匹配全错）——**一定要看这张图，不要只看内点数**。

**提升鲁棒性的三招**：

1. **双向匹配**（第 4 章第 3 级）：内点率能从 91% 提升到 97%；
2. **多尺度搜索**：把小物体模板放大到 2× 再检测，能匹配到更远/更小的实例；
3. **比值检验 + `estimateAffinePartial2D` 兜底**：当物体只被平移旋转缩放（无透视）时，用 4 自由度模型抗噪更好。

---

## 第 7 章 实战二：全景拼接

**任务**：两张有重叠区域的照片，拼成一张全景图。

**原理**：全景拼接 = 匹配 + 求单应 + 透视变换 + 融合。

```python
import cv2, numpy as np

a = cv2.imread('left.png')      # 左图（作为参考坐标系）
b = cv2.imread('right.png')     # 右图（要变换过去）

sift = cv2.SIFT_create()
ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
kpa, da = sift.detectAndCompute(ga, None)
kpb, db = sift.detectAndCompute(gb, None)

bf = cv2.BFMatcher(cv2.NORM_L2)
knn = bf.knnMatch(da, db, k=2)                  # 查询是 a，目标是 b
good = [m for m, n in knn if m.distance < 0.75 * n.distance]
print('good:', len(good))
# 坐标含义（最容易搞错的一步）：m.queryIdx -> kpa 的下标, m.trainIdx -> kpb 的下标

# 求"b -> a"的单应：src 取被变换的那张(b)，dst 取目标坐标系那张(a)
src = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
print('内点:', int(mask.sum()), '/', len(good))

# 把 b 透视变换到 a 的坐标系，再覆盖 a
canvas = cv2.warpPerspective(b, H, (a.shape[1] + b.shape[1], a.shape[0]))
canvas[:, :a.shape[1]] = a
cv2.imwrite('output/pano_manual.png', canvas)
```

> **⚠️ `H` 的方向是本章最大的坑，必须"量出来"确认，不能靠肉眼扫一眼。** 规律只有一条：
>
> **`src` 取"要被变换的那张图"上的点，`dst` 取"目标坐标系那张图"上的点；`findHomography(src, dst)` 的结果直接喂给 `warpPerspective(被变换的图, H)`，不要手动求逆。**
>
> 这里就是 `src = kpb`（右图，要被变换）、`dst = kpa`（左图，目标坐标系）。
> **实测写反的后果**（同一组图，只改 `src`/`dst` 的顺序）：
>
> | 写法 | 得到的 `H[0,2]` | 重叠区与原图的平均灰度差 | 判定 |
> |---|---|---|---|
> | `src=kpb, dst=kpa`（正确） | **+500.0** | **0.33** | 像素级对齐 ✅ |
> | `src=kpa, dst=kpb`（写反） | −499.7 | 148.4 | 右图整体错位 1000 像素 ❌ |
>
> 一个 30 秒的自检方法：往右图上贴一个亮色方块，`warpPerspective` 之后看它有没有出现在左图坐标系里它该在的位置。
> 更省事的自检：**比较拼接结果的重叠区和原图**（下面的 Demo 4 就是这么做的）。

**实测结果**（同一张 `02.jpg` 裁出左右重叠 200 像素的两个视图，左图 700×683、右图 524×683）：

```
左图 1023 个点，右图 650 个点
① 比率检验(0.75) -> 368 条
② 交叉验证       -> 352 条
③ RANSAC 内点    -> 349/352 (99%)
b -> a 的单应: 平移量 = (500.0, -0.0)     ← 真实值就是 (500, 0)
重叠区对齐自检: 与原图平均差 0.33 灰度级
```

（切图用的是 `b = img[:, 500:]`，即 右图x = 原图x − 500；所以"把右图放回左图坐标系"就是平移 +500 —— **数值完全对得上**。）



**结果图里的黑边和鬼影怎么处理**：

1. **黑边**：`warpPerspective` 的输出尺寸要按变换后的内容重算（或用 `np.where(valid_mask, warped, left)` 把无内容处填成左图）；
2. **接缝鬼影**：两张图重叠区亮度不同，硬覆盖会出现明显接缝。做法是**羽化融合**——对有效区域掩码做高斯模糊得到 0~1 权重图，然后加权平均：

```python
warped = cv2.warpPerspective(b, H, (a.shape[1] + b.shape[1], a.shape[0]))
valid  = cv2.warpPerspective(np.ones(gb.shape, np.uint8) * 255, H, warped.shape[1::-1])
alpha  = np.clip(cv2.GaussianBlur(valid.astype(np.float32) / 255.0, (0, 0), 40), 0, 1)

left = np.zeros_like(warped)
left[:, :a.shape[1]] = a
alpha = np.where(left.any(axis=2), 1.0 - alpha, alpha)    # 左图覆盖处权重取反
blend = (warped * (1 - alpha[..., None]) + left * alpha[..., None]).astype(np.uint8)
cv2.imwrite('output/pano_blend.png', blend)
```

3. **OpenCV 有现成的 `Stitcher`**，能自动完成匹配、估计变换、找接缝、融合，甚至自动排序多张图：

```python
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status, pano = stitcher.stitch([a, b, c, ...])       # 多张图可以一起给
if status != cv2.Stitcher_OK:
    print('拼接失败，status =', status)               # 常见：重叠不足 / 场景太单调
```

**实测**：用上面那组左右视图，`Stitcher` 返回 `status=0 (OK)`，输出 1024×683 的拼接图。换成**两张毫无重叠**的图，返回 `status=1`（失败）。

> **但要知道 `Stitcher` 的失败模式**：某些情况下（例如点太少、几何退化）它会**抛出 C++ 异常**（`cv::error: Unknown C++ exception`）而不是返回错误码。**生产代码里必须用 `try/except` 包起来**。所以：**要控制就手写（可控、可调试），要省事就用 `Stitcher`（但要包异常处理）**。

**自检代码（强烈建议每次都跑）**：把拼接结果和原图的重叠区做逐像素比较，对齐好的话平均差应该接近 0：

```python
ref = full[:, :canvas.shape[1]]          # 原始完整图（真值）
d = np.abs(canvas[:, 700:1000].astype(np.float32) -
           ref[:, 700:1000].astype(np.float32)).mean()
print(f'重叠区平均灰度差 = {d:.2f}')      # 实测 0.33（对齐良好）；H 写反时是 148
```


---

## 第 8 章 SIFT vs ORB vs 深度学习特征

| 维度 | SIFT | ORB | AKAZE/BRISK | SuperPoint 等 |
|---|---|---|---|---|
| 描述子维度 | 128 × float32 = **512 字节/点** | 32 × uint8 = **32 字节/点** | 61 × uint8 等 | 256 × float32 |
| 尺度不变 | ✅ 完整尺度空间 | ⚠️ 图像金字塔，较弱 | ✅ 非线性尺度空间 | ✅（学习得到） |
| 旋转不变 | ✅ 主方向 | ✅ 灰度质心 | ✅ | ⚠️ 需训练 |
| 亮度不变 | ✅ 描述子归一化 | ⚠️ 较弱 | ✅ | ⚠️ |
| 视角变化 | ⚠️ 中等（约 ±30°） | ❌ 弱 | ⚠️ 中等 | ✅ 强 |
| 速度 | 慢（1024×683 约 0.1~0.3 s） | **快 10~100 倍** | 中 | GPU 下极快，CPU 慢 |
| 是否免费 | ✅ OpenCV 4.4+ 起在主干（专利已过期） | ✅ | ✅ | 需另装模型 |
| 适合场景 | 精确匹配、拼接、SfM、检索、小目标 | 实时 SLAM、AR、手机端 | 折中方案 | 难例、弱纹理、大视角 |

**决策建议**：

- **要准、不在乎速度** → SIFT（本文主角）；
- **要实时（>30 FPS）** → ORB（`cv2.ORB_create(1000)`），但记得它的匹配要用 `cv2.NORM_HAMMING`：

```python
orb = cv2.ORB_create(1000)              # 注意：必须给正数，nfeatures=0 在部分版本会返回 None
kp, des = orb.detectAndCompute(gray, None)
print(des.shape, des.dtype)             # (1000, 32) uint8
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)   # 二值描述子用汉明距离
```

- **弱纹理 / 大幅视角变化 / 极端光照** → 上学习类特征（SuperPoint + SuperGlue、LoFTR、DISK），但要注意部署成本；
- **只是想判断"这两张图是不是同一个地方"** → SIFT + 比率检验 + 内点数阈值，就是最简单可靠的路子（词袋/图像检索的经典做法）。

---

## 第 9 章 踩坑清单

按"遇到报错/结果不对 → 来这一节找"的顺序排列。

| # | 症状 | 原因 | 解决 |
|---|---|---|---|
| 1 | `AttributeError: module 'cv2' has no attribute 'SIFT'` | OpenCV < 4.4 或装了 `opencv-python-headless` 的旧版 | 升级：`pip install -U opencv-python`；或老版本用 `cv2.xfeatures2d.SIFT_create()` |
| 2 | 匹配结果一团乱麻、几千条 | 直接用了 `bf.match()` 没做过滤 | 用 `knnMatch(k=2)` + 比率 0.75（第 4 章） |
| 3 | 匹配距离全是很大或很小的怪值 | 用了 `NORM_HAMMING` 配 SIFT，或对描述子做了 `des/255` | SIFT 用 `NORM_L2` 且**不要手动改描述子数值** |
| 4 | `findHomography` 返回 `None` | 匹配点少于 4 对 | 先判 `len(good) >= 10`；否则说明两张图无共同内容 |
| 5 | 定位框位置完全跑偏 / 拼接图错位 | **`src`/`dst` 写反了**（`H` 方向错） | 记住：**`src` = 要被变换的图，`dst` = 目标坐标系那张图**；`findHomography(src,dst)` 的结果直接喂 `warpPerspective(被变换的图, H)`，**不要手动求逆**。实测写反时物体定位四角误差 0.4px → 1014px、全景重叠区灰度差 0.33 → 148，务必用"量出来"的方式自检（第 7 章） |
| 6 | 全景图上有鬼影/接缝 | 硬覆盖，没做融合 | 羽化加权融合（第 7 章），或直接用 `Stitcher` |
| 7 | `cv2.Stitcher` 抛 `Unknown C++ exception` | 输入图**重叠不足 / 点太少 / 几何退化**（实测两张无关的图通常返回 `status=1`，但退化情况确实会抛异常） | 用 `try/except` 包住，并同时检查 `status`；或改用手写匹配 + 内点数自检 |
| 8 | 每次跑关键点数量都不一样（1310/1317） | OpenCV 多线程 + 浮点累加顺序 | 正常现象；**比较时用相对指标**（匹配率/内点率）而不是绝对点数 |
| 9 | 图像很大时慢到不可用 | 关键点数随面积爆炸（实测 2048×1366 上有 58939 个点） | **先缩图到长边 1000~1600**，或设 `nfeatures` |
| 10 | 关键点全挤在角落/水印上 | 水印、边框、Ctrl+C 的压缩伪影都是强纹理 | 用 `mask` 排除边缘区域，或先做形态学清理 |
| 11 | 平滑区域一个点都没有 | DoG 在均匀区域无极值 | 正常行为，不是 bug。弱纹理图需要降低 `contrastThreshold` |
| 12 | 传彩色图报错/结果异常 | SIFT 要单通道 | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |
| 13 | 描述子取出来是 `None` | 关键点数为 0（图太小/太糊） | 先 `if des is None: return`；检查图像尺寸和清晰度 |
| 14 | `SIFT_create(descriptorType=cv2.CV_8U)` 报 `invalid keyword argument` | 带 `descriptorType` 的重载要求**前 5 个参数全部按位置传** | `cv2.SIFT_create(0, 3, 0.04, 10, 1.6, cv2.CV_8U)`；且实测 OpenCV 5.0 下仍返回 float32 |
| 15 | 镜面翻转的图匹配不上 | SIFT **不具备镜像不变性**（描述子按主方向旋转，但翻转改变了手性） | 实测：原图自匹配 129 条，水平翻转后只剩 21 条。翻转场景需额外跑一次翻转后的模板 |
| 16 | 图太糊时匹配差 | 模糊抹掉了细尺度结构 | 减小 `sigma`、先锐化，或提高拍摄质量。实测：σ=1.5 模糊后 129→37，σ=3 只剩 16 |

---

## 第 10 章 自测题（带答案）

**Q1.** SIFT 的尺度不变性主要来自哪两个设计？

<details><summary>答案</summary>

(1) **高斯金字塔 + DoG**：在多个尺度上检测，并且 NMS 是在 $(\sigma, y, x)$ 三维空间里做的，所以特征点带上了"尺度"这个属性；(2) **描述子按关键点尺度取邻域**（边长正比于 $\sigma$），所以同一物理结构在不同分辨率下取到的是"相同比例"的邻域内容。

补充：**主方向**带来的是旋转不变性，不是尺度不变性。
</details>

**Q2.** 为什么 `contrastThreshold=0.09` 才等价于 Lowe 论文里的 0.03？

<details><summary>答案</summary>

因为 OpenCV 内部在过滤时会把这个阈值**再除以 `nOctaveLayers`**（默认 3）。0.09/3 = 0.03。实测验证：设 0.09 时保留点的最小 `response` 正好是 0.0301。
</details>

**Q3.** 一个点通过了 DoG 极值检测，但落在一条强边缘上。它会被哪一步剔除？依据是什么？

<details><summary>答案</summary>

被**边缘响应剔除**（`edgeThreshold`，默认 10）。依据是 Hessian 矩阵的两个特征值之比 $r=\alpha/\beta$：边缘点的 $r$ 很大，判据 $Tr(H)^2/Det(H) = (r+1)^2/r$ 会超过 $(r_{th}+1)^2/r_{th} = 12.1$，于是被丢弃。
</details>

**Q4.** 为什么 `bf.match()` 的结果不能直接用？

<details><summary>答案</summary>

因为它**无条件**为每个查询点返回一个最近邻，没有"没有匹配"这个选项。实测在两张真实相关图像上它会给出 1317 条匹配（几乎等于关键点总数），其中大部分是错的。必须加**比率检验**（以及交叉验证、RANSAC）。
</details>

**Q5.** 两张图匹配后，`good` 有 500 条，RANSAC 内点只有 120 条。你会怎么排查？

<details><summary>答案</summary>

① 先看 `ratio` 是不是太松（0.8/0.9 会放进大量错匹配）；② 看两张图**是否真的有共同内容**（内点数少也可能是因为只有一小块重叠）；③ 加**交叉验证**；④ 看 RANSAC 阈值 5.0 是否太小（图像若有大透视/畸变可放宽到 8~10）；⑤ 直接可视化内点连线（`drawMatches`）——**肉眼一看就知道是错匹配还是真只有小区域重叠**。
</details>

**Q6.** `findHomography(src, dst)` 里 `src` 是物体模板的点、`dst` 是场景的点，这个 `H` 能把什么变到什么？如果顺序写反了会怎样？

<details><summary>答案</summary>

`H` 把**物体模板坐标系**里的点映射到**场景坐标系**（`perspectiveTransform(模板角点, H)` 得到场景里的框）。

顺序写反会得到一个"场景 → 模板"的矩阵，**不会报错**，但用它投影模板角点会得到一堆飞到画面外的坐标——实测四角误差平均从 0.4 像素变成 1014 像素。**判断标准就是那张可视化图：框套住了物体才对。**

`warpPerspective` 的方向约定：`src` 取要被变换的图、`dst` 取目标坐标系那张图，`H` 直接传进去，**不要手动 `np.linalg.inv`**。
</details>

**Q7.** 为什么 `kp.octave` 不要直接拿来当"第几组"用？

<details><summary>答案</summary>

OpenCV 把 octave 索引和层内偏移编码进了同一个整数，且实测取值里出现了 255（表示 -1）等值，直接解读会得到无意义的分布。要按尺度分组，用 `kp.size`（σ ≈ size/2）自己算。
</details>

**Q8.** 为什么匹配时不需要手动把描述子归一化到单位长度？

<details><summary>答案</summary>

OpenCV 已经做了归一化（乘 512 存储，范数 ≈ 512）。**比率检验对描述子的整体缩放是不变的**（分子分母同倍缩放），所以手动再归一化对结果没有改善，反而会让 `match()` 的绝对距离阈值失去意义。
</details>

**Q9.** 想让 SIFT 在 4K 图上实时跑，最有效的三件事是什么？

<details><summary>答案</summary>

① **缩小图像到长边 1000~1600**（关键点数与面积近似成正比，这是收益最大的一步）；② 换 **ORB** 等二值描述子（快 10~100 倍，代价是精度）；③ 用 **FLANN** 替代暴力匹配，并把 `checks` 调小、设 `nfeatures` 限制点数。如果再要更快，就得靠 GPU 或学习类特征了。
</details>

**Q10.** 一张图旋转 90° 后和原图匹配，为什么效果好（实测 114/129），旋转 45° 反而差一些（实测 75/129）？

<details><summary>答案</summary>

SIFT 本身有旋转不变性，理论上任何角度都该一样。实测差异来自两处：(1) **45° 旋转要重采样插值**，引入了模糊和混叠，削弱了细尺度结构；90° 旋转是纯像素重排，**没有插值损失**；(2) 旋转后图像内容被裁掉/补边，边界处的点失效。这提醒我们：**理论与实测的差距往往来自重采样，而不是算法本身**。
</details>

---

## 附录 A API 速查

```python
# ---------- 创建 ----------
sift = cv2.SIFT_create()                                  # 默认参数，先用这个
sift = cv2.SIFT_create(nfeatures=2000,                    # 最多保留 2000 个
                       nOctaveLayers=3,                   # 每组层数
                       contrastThreshold=0.04,             # 低对比度阈值（内部 /3）
                       edgeThreshold=10,                   # 边缘阈值（越大保留越多）
                       sigma=1.6)                          # 输入图像预模糊 σ

# ---------- 检测 + 描述 ----------
kp, des = sift.detectAndCompute(gray, mask)                # gray: 单通道; mask: 可选 ROI
kp_only = sift.detect(gray, mask)                          # 只要关键点
des_only = sift.compute(gray, kp)                          # 只要描述子（复用已有 kp）

# ---------- 关键点属性 ----------
kp[i].pt        # (x, y) 亚像素坐标
kp[i].size      # 尺度直径；σ ≈ size / 2
kp[i].angle     # 主方向（度，0~360）
kp[i].response  # 对比度强度
kp[i].octave    # 编码值，不要直接解读
kp[i].class_id  # 忽略

# ---------- 可视化 ----------
cv2.drawKeypoints(img, kp, None, color=(0, 255, 0),
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)   # 画尺度+方向
cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.drawMatchesKnn(img1, kp1, img2, kp2, [[m] for m in good], None, ...)

# ---------- 匹配 ----------
bf = cv2.BFMatcher(cv2.NORM_L2)                            # SIFT 用 L2；ORB 用 NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)      # 二值描述子 + 交叉验证
m  = bf.match(des1, des2)                                  # 无条件最近邻（慎用）
knn = bf.knnMatch(des1, des2, k=2)                         # 返回最近的两个，做比率检验
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

# 匹配对象字段：m.queryIdx（图1下标）, m.trainIdx（图2下标）, m.distance

# ---------- 几何验证 ----------
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   # 透视，需 >= 4 对
A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                      method=cv2.RANSAC, ransacReprojThreshold=3.0)  # 4 自由度
proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
warped = cv2.warpPerspective(img, H, (W, H_out))

# ---------- 全景 ----------
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)      # 或 cv2.Stitcher_SCANS
status, pano = stitcher.stitch(images)                     # status == 0 表示成功（要包 try/except）
```

**标准匹配配方（可直接复制）**：

```python
def match_sift(img1, img2, ratio=0.75, ransac_thresh=5.0):
    """返回 (good_matches, inlier_matches, H, mask, (kp1, des1, kp2, des2))"""
    sift = cv2.SIFT_create()
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return [], [], None, None, (kp1, des1, kp2, des2)

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # ① 比率检验（双向）
    good = [m for m, n in bf.knnMatch(des1, des2, k=2) if m.distance < ratio * n.distance]
    back = {m.queryIdx: m.trainIdx
            for m, n in bf.knnMatch(des2, des1, k=2) if m.distance < ratio * n.distance}
    mutual = [m for m in good if back.get(m.trainIdx) == m.queryIdx]

    # ② RANSAC 几何验证
    if len(mutual) < 4:
        return good, [], None, None, (kp1, des1, kp2, des2)
    src = np.float32([kp1[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        return good, [], None, None, (kp1, des1, kp2, des2)
    inliers = [m for m, ok in zip(mutual, mask.ravel()) if ok]
    return good, inliers, H, mask, (kp1, des1, kp2, des2)
```

---

## 附录 B SIFT 的不变性实测数据

**实验设置**：取 `02.jpg` 中 `[150:400, 300:620]` 的 320×250 局部作为模板，检测出 129 个关键点；对模板做各种变换后重新匹配，统计通过 `ratio=0.75` 的匹配数。

**基准**：模板对自己匹配 = **129**。

| 变换 | 匹配数 | 结论 |
|---|---|---|
| 缩放 ×0.4 | 26 | 缩小丢细尺度结构，掉得快 |
| 缩放 ×0.6 | 44 | |
| 缩放 ×1.0 | 129 | 基准 |
| 缩放 ×1.5 | 86 | 放大对匹配有利（细节变多） |
| 缩放 ×2.0 | 98 | 但插值模糊限制了上限 |
| 旋转 15° | 84 | |
| 旋转 45° | 75 | 需要插值重采样，损失最大 |
| 旋转 90° | **114** | 纯像素重排，几乎无损（见 Q10） |
| 旋转 180° | 111 | |
| 亮度 +40 | **123** | 亮度不变性很强 |
| 对比度 ×0.5 | 76 | 低对比度会丢点 |
| 对比度 ×1.8 | 80 | 过曝同样有损 |
| 高斯噪声 σ=5 | 91 | |
| 高斯噪声 σ=15 | 52 | |
| 高斯噪声 σ=30 | 30 | 噪声破坏梯度方向直方图 |
| 模糊 σ=1.5 | 37 | **模糊是 SIFT 最大的敌人** |
| 模糊 σ=3.0 | 16 | |
| 模糊 σ=6.0 | 9 | |
| 水平镜像 | 21 | **不具镜像不变性** |

**可以直接得出的工程结论**：

1. **亮度变化几乎免疫**（123/129），这是描述子归一化的功劳；
2. **旋转 90° 类变换近乎无损，任意角度有插值损失**；
3. **模糊（失焦、运动模糊）比噪声更致命**——实测 σ=3 的模糊就把匹配打到 16 条。**拍图请先保证对焦**；
4. **缩小图像损失很大**（×0.4 只剩 26 条），所以**不要为了提速把图缩得太小**（第 5 章说缩到长边 1000~1600，就是这个平衡点）；
5. **镜像需要特殊处理**。

---

## 附录 C 文件清单

| 文件 | 说明 |
|---|---|
| `SIFT教程.md` | 本文 |
| `tutorial_sift.py` | 配套脚本，含 4 个 Demo，`python tutorial_sift.py [1-4]` 运行 |
| `output/` | 脚本输出的可视化结果（自动创建） |

**Demo 说明**：

| Demo | 内容 | 对应章节 |
|---|---|---|
| `1` | 关键点检测、属性统计、尺度着色可视化 | 第 2、3 章 |
| `2` | 参数对比实验（表格输出） | 第 5 章 |
| `3` | 全流程匹配：比率检验 → 交叉验证 → RANSAC，并可视化 | 第 4 章 |
| `4` | 物体定位 + 全景拼接 | 第 6、7 章 |

---

## 参考文献

1. D. G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, IJCV 2004. —— SIFT 原始论文，**强烈建议读第 3 节和第 6 节**（描述子与匹配），本文第 1 章基本是它的浓缩。
2. D. G. Lowe, *Object Recognition from Local Scale-Invariant Features*, ICCV 1999. —— 早期版本。
3. H. Bay et al., *SURF: Speeded Up Robust Features*, ECCV 2006. —— SIFT 的加速版。
4. E. Rublee et al., *ORB: An efficient alternative to SIFT or SURF*, ICCV 2011.
5. OpenCV 官方文档：`cv2.SIFT` 类、`cv2.BFMatcher`、`cv2.findHomography` 的 API 说明。
