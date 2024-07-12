"""计算结果和官方代码稍有区别"""
import cv2
import numpy as np

# 读取需要比较的图像
img1 = cv2.imread('./images/bird.png')
img2 = cv2.imread('./images/bird_SwinIR.png')

img1 = img1.astype(np.float32) / 255.
img2 = img2.astype(np.float32) / 255.

img1_y = np.dot(img1, [24.996, 128.553, 65.481]) + 16.0
img2_y = np.dot(img2, [24.996, 128.553, 65.481]) + 16.0

img1_y = img1_y[..., None]
img2_y = img2_y[..., None]

# 计算Y通道的MSE，然后计算psnr
mse = np.mean((img1_y - img2_y) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * np.log10(255. / np.sqrt(mse))
 
print(f'PSNR value: {psnr:.6f}dB.')