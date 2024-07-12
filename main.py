import cv2
import numpy as np

import _utils_calculate_psnr_ssim as util

# 加载需要计算的两张图片, opencv读取方式是<numpy.ndarray>, HWC, BGR, <numpy.uint8>[0, 255]
img1 = cv2.imread('./images/bird.png')
img2 = cv2.imread('./images/bird_SwinIR.png')

# 设定图像边缘裁剪像素
crop_border = 0

# 计算psnr与ssim (输入图片的格式需要是ndarray)
psnr = util.calculate_psnr(img1, img2, crop_border, test_y_channel=True)
ssim = util.calculate_ssim(img1, img2, crop_border, test_y_channel=True)

print(f'PSNR: {psnr:.6f}dB\nSSIM: {ssim:.6f}')
