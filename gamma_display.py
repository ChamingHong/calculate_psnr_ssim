import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('./images/bird.png')

# 伽马校正函数
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 模拟编码伽马（通常在保存图像时进行）
encoded_image = gamma_correction(image, 2.2)

# 模拟显示伽马（通常在显示图像时进行）
displayed_image = gamma_correction(encoded_image, 1/2.2)

# 显示原始图像和经过伽马校正后的图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
encoded_image_rgb = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
displayed_image_rgb = cv2.cvtColor(displayed_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(encoded_image_rgb)
plt.title('Encoded Image (Gamma 2.2)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(displayed_image_rgb)
plt.title('Displayed Image (Gamma 1/2.2)')
plt.axis('off')

plt.show()
