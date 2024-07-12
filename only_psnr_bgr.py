import cv2
import numpy as np

# 读取需要比较的图像
img1 = cv2.imread('./images/bird.png')
img2 = cv2.imread('./images/bird_SwinIR.png')

# 数据类型转换，结果更精确
img1 = img1.astype(np.float64)
img2 = img2.astype(np.float64)

# 计算3个通道的平均MSE，然后计算psnr
mse = np.mean((img1 - img2) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * np.log10(255. / np.sqrt(mse))
 
print(f'PSNR value: {psnr:.6f}dB.')