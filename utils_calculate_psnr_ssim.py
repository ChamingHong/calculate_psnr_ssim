import cv2
import numpy as np
import torch

def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """计算峰值信噪比(PSNR)
    Args:
        img1 (ndarray): ndarray类型的图片, [0-255]
        img2 (ndarray): ndarray类型的图片, [0-255]
        crop_border (int): 计算时图片边缘裁剪的像素数量
        input_order (str): 输入图片的通道顺序
            Defalut: 'HWC'
        test_y_channel (bool): 是否在YCbCr的Y通道上计算
            Defalut: False
    Returns:
        (float): 计算的浮点型psnr结果 
    """

    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_order are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def reorder_image(img, input_order='HWC'):
    """调整图像通道顺序为HWC
    Args:
        img (ndarray): 输入图像
        input_order (str): 输入图像的通道顺序
            Default: 'HWC'
    Returns:
        (ndarray): HWC图像
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_order are ' '"HWC" and "CHW"')
    if(len(img.shape) == 2):
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """转换到Y通道
    Args:
        img (ndarray): 输入图像, [0-255]
    Returns:
        (ndarray): 输出图像Y通道, [0-255](float type)
    """

    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """转换输入图像img的类型和范围
    Args:
        img (ndarray): 输入图像, 需满足:
            1. np.uint8,   [0, 255]
            2. np.float32, [0, 1]
    Returns:
        (ndarray): 转换完成的图像, np.float32, [0, 1]
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    """根据dst_type的类型和范围转换图像img
    Args:
        img (ndarray): 输入图像, np.float32, [0, 255]
        dst_type (np.uint8 | np.float32):
            1. np.uint8:   img ---> np.uint8,   [0, 255]
            2. np.float32: img ---> np.float32, [0, 1]
    Returns:
        (ndarray): 转换完成的图像 
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """将BGR转换为YCbCr
    Args:
        img (ndarray): 输入图像,需满足:
            1. np.uint8    [0, 255];
            2. np.float32  [0, 1]
        y_only (bool): 是否只保留Y通道
            Default: False
    Returns:
        (ndarray): 转换好的YCbCr图像,与输入的数据类型范围相同
    """

    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

