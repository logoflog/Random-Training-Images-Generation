import numpy as np
import cv2
import os
from scipy import ndimage


def generate_random_noise_image(height, width, channels=3):
    """生成随机噪声图像"""
    return np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)


def create_irregular_path_mask(height, width, complexity=5, smoothness=2):
    """
    创建不规则路径分割掩码
    :param height: 图像高度
    :param width: 图像宽度
    :param complexity: 路径复杂度 (1-10)
    :param smoothness: 平滑度 (1-5)
    :return: 二值掩码
    """
    # 生成随机噪声作为基础
    noise = np.random.rand(height, width)

    # 应用高斯滤波使边界平滑
    sigma = smoothness * 2
    smoothed_noise = ndimage.gaussian_filter(noise, sigma=sigma)

    # 添加一些结构化元素增加复杂度
    for _ in range(complexity):
        # 随机选择一个点作为"种子"
        seed_y = np.random.randint(0, height)
        seed_x = np.random.randint(0, width)

        # 创建从该点向外扩散的梯度
        y_grid, x_grid = np.ogrid[:height, :width]
        distance = np.sqrt((y_grid - seed_y) ** 2 + (x_grid - seed_x) ** 2)
        influence = np.exp(-distance / (min(height, width) * 0.3))

        # 随机决定是增加还是减少影响
        if np.random.random() > 0.5:
            smoothed_noise += influence * 0.5
        else:
            smoothed_noise -= influence * 0.5

    # 归一化到0-1范围
    smoothed_noise = (smoothed_noise - smoothed_noise.min()) / (smoothed_noise.max() - smoothed_noise.min())

    # 使用自适应阈值创建二值掩码
    threshold = np.percentile(smoothed_noise, 50)
    mask = (smoothed_noise > threshold).astype(np.uint8) * 255

    return mask


def split_image_with_mask_keep_values(image, mask, keep_range=(0.2, 0.5)):
    """
    使用掩码分割图像，但保留被"去掉"部分的部分值
    :param image: 输入图像
    :param mask: 二值掩码 (0或255)
    :param keep_range: 保留范围 (min_ratio, max_ratio)，例如(0.2, 0.5)表示保留20%-50%
    :return: 分割后的两部分图像
    """
    # 将掩码归一化到0-1
    mask_normalized = mask.astype(np.float32) / 255.0

    # 随机选择保留比例（在指定范围内）
    keep_ratio = np.random.uniform(keep_range[0], keep_range[1])

    # 第一部分：掩码区域保留100%，非掩码区域保留keep_ratio
    part1 = image.astype(np.float32)
    part1 = part1 * mask_normalized[:, :, np.newaxis] + \
            part1 * (1 - mask_normalized[:, :, np.newaxis]) * keep_ratio

    # 第二部分：掩码区域保留keep_ratio，非掩码区域保留100%
    part2 = image.astype(np.float32)
    part2 = part2 * (1 - mask_normalized[:, :, np.newaxis]) + \
            part2 * mask_normalized[:, :, np.newaxis] * keep_ratio

    # 转换回uint8
    part1 = np.clip(part1, 0, 255).astype(np.uint8)
    part2 = np.clip(part2, 0, 255).astype(np.uint8)

    return part1, part2


def split_image_with_mask_varying_keep(image, mask, min_keep=0.2, max_keep=0.5):
    """
    使用掩码分割图像，每个像素的保留比例在范围内随机变化
    :param image: 输入图像
    :param mask: 二值掩码 (0或255)
    :param min_keep: 最小保留比例
    :param max_keep: 最大保留比例
    :return: 分割后的两部分图像
    """
    # 将掩码归一化到0-1
    mask_normalized = mask.astype(np.float32) / 255.0

    # 为每个像素生成随机保留比例（在范围内）
    random_keep = np.random.uniform(min_keep, max_keep, (image.shape[0], image.shape[1], 1))

    # 第一部分：掩码区域保留100%，非掩码区域保留随机比例
    part1 = image.astype(np.float32)
    part1 = part1 * mask_normalized[:, :, np.newaxis] + \
            part1 * (1 - mask_normalized[:, :, np.newaxis]) * random_keep

    # 第二部分：掩码区域保留随机比例，非掩码区域保留100%
    part2 = image.astype(np.float32)
    part2 = part2 * (1 - mask_normalized[:, :, np.newaxis]) + \
            part2 * mask_normalized[:, :, np.newaxis] * random_keep

    # 转换回uint8
    part1 = np.clip(part1, 0, 255).astype(np.uint8)
    part2 = np.clip(part2, 0, 255).astype(np.uint8)

    return part1, part2


def create_directories():
    """创建必要的目录结构"""
    source_a_dir = r'F:\dataset\Random\SourceA'
    source_b_dir = r'F:\dataset\Random\SourceB'
    os.makedirs(source_a_dir, exist_ok=True)
    os.makedirs(source_b_dir, exist_ok=True)
    return source_a_dir, source_b_dir


def generate_and_split_batch_irregular(num_images=500, height=500, width=500,
                                       complexity=5, smoothness=2,
                                       keep_range=(0.2, 0.5),
                                       varying_keep=True):
    """
    批量生成图像并用不规则路径分割，保留部分值
    :param num_images: 生成的图像数量
    :param height: 图像高度
    :param width: 图像宽度
    :param complexity: 路径复杂度
    :param smoothness: 平滑度
    :param keep_range: 保留范围 (min_ratio, max_ratio)
    :param varying_keep: 是否使用变化的保留比例（True=每个像素不同，False=统一比例）
    """
    source_a_dir, source_b_dir = create_directories()

    print(f"开始生成并分割 {num_images} 张 {width}x{height} 的随机噪声图像...")
    print(f"分割方式: 不规则小路径分割")
    print(f"保留范围: {keep_range[0] * 100:.0f}% - {keep_range[1] * 100:.0f}%")
    print(f"保留模式: {'变化保留比例' if varying_keep else '统一保留比例'}")

    for i in range(num_images):
        # 生成一张随机噪声图像
        original_image = generate_random_noise_image(height, width)

        # 创建不规则掩码
        mask = create_irregular_path_mask(height, width, complexity, smoothness)

        # 分割图像（保留部分值）
        if varying_keep:
            part1, part2 = split_image_with_mask_varying_keep(
                original_image, mask, keep_range[0], keep_range[1]
            )
        else:
            part1, part2 = split_image_with_mask_keep_values(
                original_image, mask, keep_range
            )

        # 保存分割后的图像
        filename = f"{i:03d}.png"
        path_a = os.path.join(source_a_dir, filename)
        path_b = os.path.join(source_b_dir, filename)

        cv2.imwrite(path_a, part1)
        cv2.imwrite(path_b, part2)

        if (i + 1) % 50 == 0:
            print(f"已完成 {i + 1}/{num_images} 张图像")

    print(f"所有 {num_images} 张图像分割完成！")
    print(f"SourceA (掩码区域100% + 非掩码区域{keep_range[0] * 100:.0f}-{keep_range[1] * 100:.0f}%): {source_a_dir}")
    print(f"SourceB (非掩码区域100% + 掩码区域{keep_range[0] * 100:.0f}-{keep_range[1] * 100:.0f}%): {source_b_dir}")


if __name__ == "__main__":
    NUM_IMAGES = 500
    IMAGE_HEIGHT = 500
    IMAGE_WIDTH = 500

    # 路径复杂度参数
    COMPLEXITY = 6  # 路径复杂度 (1-10)
    SMOOTHNESS = 3  # 边界平滑度 (1-5)

    # 保留比例参数
    KEEP_RANGE = (0.2, 0.5)  # 保留20%到50%的值

    # 保留模式
    # True: 每个像素的保留比例在20-50%之间随机变化（更自然）
    # False: 所有像素使用统一的随机保留比例
    VARYING_KEEP = True

    try:
        generate_and_split_batch_irregular(
            NUM_IMAGES,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            COMPLEXITY,
            SMOOTHNESS,
            KEEP_RANGE,
            VARYING_KEEP
        )
    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        print("请确保 F:\\ 盘存在并且有写入权限")