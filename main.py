import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2 import ximgproc

# ====== 入力画像 ======
imgL_full = cv2.imread("left.JPG", cv2.IMREAD_GRAYSCALE)
imgR_full = cv2.imread("right.JPG", cv2.IMREAD_GRAYSCALE)
if imgL_full is None or imgR_full is None:
    raise ValueError("画像が読み込めていません")

# ====== ダウンサンプリング ======
SCALE = 0.25
imgL = cv2.resize(imgL_full, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
imgR = cv2.resize(imgR_full, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
h, w = imgL.shape

# ====== 推定内部パラメータ K ======
# EXIF: iPhone SE (3rd gen), 35mm換算 28mm
# 35mmフルサイズ幅 36mm → FOV_h = 2*atan(36/(2*28)) ≈ 65.5°
# f_pixel_full = W_full / (2*tan(FOV_h/2))
FOV_H_DEG = 2.0 * np.degrees(np.arctan(36.0 / (2.0 * 28.0)))
f_full = imgL_full.shape[1] / (2.0 * np.tan(np.radians(FOV_H_DEG / 2.0)))
f = f_full * SCALE
cx, cy = w / 2.0, h / 2.0
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
print(f"推定 f = {f:.1f}px (フル {f_full:.1f}px), 画像 {w}x{h}")

# ====== 特徴点マッチング ======
sift = cv2.SIFT_create(nfeatures=4000)
kpL, desL = sift.detectAndCompute(imgL, None)
kpR, desR = sift.detectAndCompute(imgR, None)

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
raw = flann.knnMatch(desL, desR, k=2)
good = [m for m, n in raw if m.distance < 0.7 * n.distance]
print(f"特徴点: L={len(kpL)}, R={len(kpR)}, 良マッチ={len(good)}")
if len(good) < 30:
    raise RuntimeError("マッチが少なすぎます")

ptsL = np.float32([kpL[m.queryIdx].pt for m in good])
ptsR = np.float32([kpR[m.trainIdx].pt for m in good])

# ====== Essential Matrix + 相対姿勢復元 ======
E, mask_e = cv2.findEssentialMat(ptsL, ptsR, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
mask_e = mask_e.ravel().astype(bool)
print(f"Essential RANSAC インライア: {mask_e.sum()}/{len(mask_e)}")

ptsL_in = ptsL[mask_e]
ptsR_in = ptsR[mask_e]

n_in, R, t, mask_pose = cv2.recoverPose(E, ptsL_in, ptsR_in, K)
print(f"recoverPose チェイラリティ内点: {n_in}")
print(f"R=\n{R}")
print(f"t (unit, scale unknown) = {t.ravel()}")

# 並進方向の判定：正面方向(t_z)が支配的だとステレオとして苦しい
t_norm = t.ravel() / np.linalg.norm(t)
print(f"t 成分比 |tx|={abs(t_norm[0]):.2f} |ty|={abs(t_norm[1]):.2f} |tz|={abs(t_norm[2]):.2f}")
if abs(t_norm[2]) > 0.8:
    print("⚠️ 並進が光軸方向に偏っています。横移動ステレオではなく、視差がうまく出ない可能性があります。")

# ====== stereoRectify で正しい平行化 ======
dist = np.zeros(5)
R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
    K, dist, K, dist, (w, h), R, t, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY
)
mapL1, mapL2 = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_16SC2)
rectL = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

# ====== SGBM + WLS ======
block_size = 7
num_disp = 16 * 12
stereo_left = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * block_size**2,
    P2=32 * block_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
stereo_right = ximgproc.createRightMatcher(stereo_left)

disp_left = stereo_left.compute(rectL, rectR)
disp_right = stereo_right.compute(rectR, rectL)

wls = ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
wls.setLambda(8000.0)
wls.setSigmaColor(1.5)
disp_filtered = wls.filter(disp_left, rectL, disparity_map_right=disp_right)
disparity = disp_filtered.astype(np.float32) / 16.0

# 無効領域をマスク
valid = disparity > 0
print(f"有効視差ピクセル: {valid.sum()}/{valid.size} ({100*valid.mean():.1f}%)")

# ====== 可視化 ======
disp_vis_src = disparity.copy()
disp_vis_src[~valid] = np.nan
vmin = np.nanpercentile(disp_vis_src, 5) if valid.any() else 0
vmax = np.nanpercentile(disp_vis_src, 95) if valid.any() else 1

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0, 0].imshow(rectL, cmap="gray")
axes[0, 0].set_title("Rectified Left")
axes[0, 1].imshow(rectR, cmap="gray")
axes[0, 1].set_title("Rectified Right")

im = axes[1, 0].imshow(disp_vis_src, cmap="inferno", vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Disparity (WLS filtered)")
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# 平行化検証用：横並びに緑の水平線
stacked = np.hstack([rectL, rectR])
stacked_rgb = cv2.cvtColor(stacked, cv2.COLOR_GRAY2RGB)
for y in range(0, h, max(1, h // 20)):
    cv2.line(stacked_rgb, (0, y), (stacked_rgb.shape[1], y), (0, 255, 0), 1)
axes[1, 1].imshow(stacked_rgb)
axes[1, 1].set_title("Rectification check")

for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig("output.png", dpi=120)
print("保存: output.png")
