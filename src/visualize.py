import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------- 随机模拟3D CT和分割标签 ----------------------
img_arr = np.random.rand(64,64,32) * 200 - 100
lab_arr = np.zeros_like(img_arr)
lab_arr[20:40,20:40,10:25] = 1

# 模拟affine、spacing
affine = np.diag([1.0,1.0,2.0,1])
spacing = np.diag(affine)[:3]

print("图像 shape：", img_arr.shape)
print("label shape：", lab_arr.shape)
print("spacing：", spacing)
print("标签唯一值：", np.unique(lab_arr))

# ========== 题目要求的两句assert断言 ==========
assert img_arr.shape == lab_arr.shape, "图像和标签尺寸不一致！"
assert set(np.unique(lab_arr)).issubset({0, 1}), "标签出现非0/1异常值！"
print("✅ assert断言校验通过")

# 挑选切片
z_slice = 15
ct_slice = img_arr[:, :, z_slice]
label_slice = lab_arr[:, :, z_slice]

# 保存输出到 outputs/figures
save_dir = "../outputs/figures/"
os.makedirs(save_dir, exist_ok=True)

plt.imsave(os.path.join(save_dir, "ct_slice.png"), ct_slice, cmap="gray")
plt.imsave(os.path.join(save_dir, "label_slice.png"), label_slice, cmap="gray")

plt.figure(figsize=(6,6))
plt.imshow(ct_slice, cmap="gray")
plt.contour(label_slice, colors="red")
plt.axis("off")
plt.savefig(os.path.join(save_dir, "overlay.png"), bbox_inches="tight")
plt.close()

print("✅ 三张图片已经保存到 outputs/figures 文件夹！")
