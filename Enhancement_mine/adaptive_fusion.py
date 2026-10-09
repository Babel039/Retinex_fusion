"""空间自适应残差融合（《项目流程.md》3.1 节）的推理实现。

算法（对应 3.1.4 的伪代码）：
    Y     = rgb_to_luminance(X)                            # 取亮度通道
    L     = gaussian_blur(Y, sigma1)                       # 局部亮度（模糊抗噪）
    alpha = clip((L_high - L) / (L_high - L_low), 0, 1)    # 逐像素增强强度
    E     = retinexformer(X)                               # 增强图（整图前向一次）
    y     = X + alpha * (E - X)                           # 残差融合

与 my_prediction.py 的关系：前向推理部分（读图 → 补边 → 分块推理 → 裁边 → 存图）
完全沿用 my_prediction.py 的写法；本脚本在"拿到增强图 E"之后额外加入 α(p) 的
估计与残差融合，并把 E 与最终结果 y 分别输出到不同目录，便于对照。

运行方式（Anaconda Prompt，项目根目录）：
    conda activate course_design
    python Enhancement_mine/adaptive_fusion.py

只需修改下面的"手动配置区"，不需要命令行参数。
"""

import os
import re
import sys
from pathlib import Path


# ============================== 手动配置区 ==============================
# 相对路径均以 Retinexformer 项目根目录为基准，也可以填写绝对路径。

OPT_PATH = r'Options/RetinexFormer_NTIRE.yml'
WEIGHTS_PATH = r'pretrained_weights/NTIRE.pth'
INPUT_DIR = r'data/small_test'

# ---- 输出路径（四条路径相互独立，按需修改）----
# 1) Retinexformer 直出结果 E（未融合）——用于对照，看"整图增强"的过曝情况
RETINEX_OUTPUT_DIR = r'results/mine/retinex_out'
# 2) 空间自适应残差融合后的最终结果 y ——本算法的正式输出
OUTPUT_DIR = r'results/mine/final_out'
# 3) 增强强度图 alpha(p)（0~255 灰度 PNG）——用于验证 alpha 是否按设计分布
ALPHA_OUTPUT_DIR = r'results/mine/alpha'
# 4) 四联对比图（原图 | E | alpha 伪彩 | y），用于报告插图
COMPARISON_OUTPUT_DIR = r'results/mine/compare'

# 指定可见 GPU。单卡填 '0'；多卡可填 '0,1'。
GPU_IDS = '0'

# 与原测试脚本一致：0 表示整图推理；4GB 显存建议保持 512 或更低。
TILE_SIZE = 512
TILE_OVERLAP = 32

# True 会对 8 种翻转/旋转结果取平均，效果可能更好，但推理时间约为 8 倍。
SELF_ENSEMBLE = False

# ---- 3.1 算法参数（三个待标定参数，含义见 3.1.4 参数表）----
# 亮度阈值按 8-bit 填写（与文档一致），脚本内部会除以 255 归一化。
L_LOW = 70          # 低于此亮度视为欠曝，全力增强
L_HIGH = 150        # 高于此亮度不增强

# 【sigma1】计算局部亮度 L。它是"判决的空间分辨率"：L 在 ±2*sigma1 范围内做平均，一个尺度
# 小于 4*sigma1 的目标填不满这个邻域，中心处的 L 必然被周围拉走。所以上界由人脸尺度决定
# （实测人脸中心 alpha 的保留率）：
#     sigma1 = 人脸尺度 x 1/8  -> 保留 100%
#     sigma1 = 人脸尺度 x 1/4  -> 保留 100%
#     sigma1 = 人脸尺度 x 1/2  -> 保留  31%
#
# 本项目标定：路面远景人脸约 30px。依据是 results/my_prediction/6789.jpg_wh860.png 中的
# 骑手——头盔宽约 80px、人脸约 60px，而路面上的人脸约为其一半。于是硬约束
# sigma1 <= 人脸/4 = 7.5，取 5.0（= 人脸/6），理由：
#   - 人脸中心 alpha = 1.000，全力增强；
#   - 即使实际人脸小到 20px 仍能全力增强，为"人脸尺寸估计不准"留了余量；
#   - 比取 7.5 时人脸内部更均匀（脸内 alpha 均值 0.83 vs 0.71）；
#   - 代价是 alpha 的颗粒度升到值域的约 1%（sigma1=7.5 时为 0.6%），绝对值很小。
#
# 注：alpha 只做这一次高斯平滑。曾评估过"对 alpha 再平滑一次以防边界光晕"，经验证无必要
# 且会稀释人脸处的增强，已去除。
SIGMA1 = 5.0

# ---- 输出开关 ----
SAVE_RETINEX = True         # 保存 Retinexformer 直出结果 E
SAVE_ALPHA = True           # 保存 alpha(p) 灰度图
SAVE_COMPARISON = True      # 保存四联对比图
COMPARISON_MAX_WIDTH = 4096  # 对比图超过此宽度时等比缩小，避免生成超大文件

# 3.1.3 工程细节 ②：若整张图 L 均 >= L_HIGH，则 alpha ≡ 0、输出恒等于输入，
# 此时跳过 Retinexformer 前向是数学上严格等价的（不是"猜测该不该处理"）。
ENABLE_SAFE_SKIP = True
# ======================================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENHANCEMENT_DIR = PROJECT_ROOT / 'Enhancement'
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SIZE_FACTOR = 4

# BT.601 亮度权重。注意输入张量是 RGB 顺序（utils.load_img 返回 RGB），
# 因此这里必须按 R/G/B 排列，不能直接套用 cv2 对 BGR 的默认灰度公式。
LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114

# CUDA_VISIBLE_DEVICES 必须在 CUDA 运行时初始化之前设置，故放在 import torch 之前。
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS
# 复用 Enhancement/ 下已有的 utils 与 inference_utils。
for _path in (str(PROJECT_ROOT), str(ENHANCEMENT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_project_path(path_value):
    """把手动配置中的相对路径转换为相对于项目根目录的绝对路径。"""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_image_paths(input_dir):
    """返回输入目录第一层中所有受支持图片，并按文件名自然排序。"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f'输入目录不存在：{input_dir}')
    if not input_dir.is_dir():
        raise NotADirectoryError(f'输入路径不是目录：{input_dir}')

    image_paths = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(
        image_paths,
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)
        ],
    )


# --------------------------------------------------------------------------
# 3.1 算法本体：局部亮度 -> alpha(p) -> 残差融合
# --------------------------------------------------------------------------
def gaussian_blur(x, sigma):
    """可分离高斯模糊，等价于 sigma 标准差的高斯核卷积。

    x: [B, C, H, W]；sigma <= 0 时原样返回。
    边界填充优先用 reflect（镜像），小图上 reflect 的填充量不能超过边长，
    此时退化为 replicate，避免直接报错。
    """
    if sigma <= 0:
        return x

    # 半径取 4*sigma（与 torchvision.transforms.gaussian_blur 的取法一致）。
    radius = max(1, int(round(4.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()

    # repeat 不支持 -1（它不是"保持原尺寸"的意思），维度一律写死。
    channels = x.shape[1]
    kernel_x = kernel.view(1, 1, 1, 2 * radius + 1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, 2 * radius + 1, 1).repeat(channels, 1, 1, 1)

    mode_x = 'reflect' if radius < x.shape[3] else 'replicate'
    mode_y = 'reflect' if radius < x.shape[2] else 'replicate'

    x = F.pad(x, (radius, radius, 0, 0), mode=mode_x)
    x = F.conv2d(x, kernel_x, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode=mode_y)
    x = F.conv2d(x, kernel_y, groups=channels)
    return x


def estimate_alpha(rgb, l_low, l_high, sigma1):
    """按 3.1.3 计算逐像素增强强度 alpha(p)。

    rgb:    [1, 3, H, W]，范围 [0, 1]。
    l_low / l_high: 已归一化到 [0, 1] 的亮度阈值。
    返回:   [1, 1, H, W]，取值 [0, 1]，在通道维上广播到 3 通道使用。

    注意 clip 写法与 3.1.3 的分段函数完全等价：
      L <= l_low  时分子 >= 分母 -> 截断为 1；
      L >= l_high 时分子 <= 0    -> 截断为 0。
    """
    # 逐通道加权求和，避免生成 rgb 的整份中间副本（省显存）。
    y = (rgb[:, 0:1] * LUMA_R + rgb[:, 1:2] * LUMA_G + rgb[:, 2:3] * LUMA_B)

    local = gaussian_blur(y, sigma1)              # 局部亮度 L = G_sigma1 * Y
    return torch.clamp((l_high - local) / (l_high - l_low), 0.0, 1.0)


def fuse(original, enhanced, alpha):
    """残差融合 y = X + alpha * (E - X)，并裁到合法像素范围。"""
    return torch.clamp(original + alpha * (enhanced - original), 0.0, 1.0)


# --------------------------------------------------------------------------
# 输出辅助
# --------------------------------------------------------------------------
def to_uint8_image(tensor):
    """[1, C, H, W] (0~1) -> [H, W, C] uint8（RGB 顺序，可直接交给 utils.save_img）。"""
    array = tensor.clamp(0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def alpha_to_uint8(alpha):
    """[1, 1, H, W] (0~1) -> [H, W] uint8。alpha=0 对应 0，alpha=1 对应 255。"""
    array = alpha.squeeze().cpu().numpy()
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def overexposure_ratio(uint8_image, threshold=250):
    """过曝率：像素值 >= threshold 的占比（3.1.5 指标 2）。"""
    return float((uint8_image >= threshold).mean())


def build_comparison(panels, labels, max_width):
    """把若干张同高 RGB uint8 图横向拼接，并在左上角写标签。返回 BGR 图。

    标签用 ASCII，因为 cv2.putText 不支持中文。
    """
    montage = np.concatenate(panels, axis=1)
    if montage.shape[1] > max_width:
        scale = max_width / montage.shape[1]
        montage = cv2.resize(montage, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)

    montage = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    panel_width = montage.shape[1] // len(panels)
    font_scale = max(0.8, montage.shape[0] / 1200.0)
    for index, label in enumerate(labels):
        origin = (index * panel_width + 12, int(font_scale * 40))
        # 先描黑边再写白字，保证在亮暗背景上都看得清。
        cv2.putText(montage, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(montage, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return montage


def main():
    """加载模型，依次增强输入目录中的图片并保存为 PNG。"""
    opt_path = resolve_project_path(OPT_PATH)
    weights_path = resolve_project_path(WEIGHTS_PATH)
    input_dir = resolve_project_path(INPUT_DIR)
    retinex_dir = resolve_project_path(RETINEX_OUTPUT_DIR)
    output_dir = resolve_project_path(OUTPUT_DIR)
    alpha_dir = resolve_project_path(ALPHA_OUTPUT_DIR)
    comparison_dir = resolve_project_path(COMPARISON_OUTPUT_DIR)

    if not opt_path.is_file():
        raise FileNotFoundError(f'YAML 配置文件不存在：{opt_path}')
    if not weights_path.is_file():
        raise FileNotFoundError(f'模型权重不存在：{weights_path}')
    if not L_LOW < L_HIGH:
        raise ValueError('必须满足 L_LOW < L_HIGH。')
    if SIGMA1 <= 0:
        raise ValueError('SIGMA1 必须大于 0，否则局部亮度退化为单像素值。')

    input_paths = find_image_paths(input_dir)
    if not input_paths:
        raise RuntimeError(f'输入目录中没有找到受支持的图片：{input_dir}')

    for directory in (output_dir, retinex_dir, alpha_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    import utils
    from tqdm import tqdm

    from basicsr.models import create_model
    from basicsr.utils.options import parse
    from inference_utils import self_ensemble, tiled_forward

    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用的 CUDA GPU，请在 conda course_design 环境中运行。')
    if TILE_SIZE < 0:
        raise ValueError('TILE_SIZE 不能小于 0。')
    if TILE_SIZE > 0 and not 0 <= TILE_OVERLAP < TILE_SIZE:
        raise ValueError('TILE_OVERLAP 必须大于等于 0 且小于 TILE_SIZE。')

    l_low = L_LOW / 255.0
    l_high = L_HIGH / 255.0

    print(f'使用 GPU：{GPU_IDS}')
    print(f'输入目录：{input_dir}')
    print(f'Retinex 直出：{retinex_dir}')
    print(f'融合结果：{output_dir}')
    print(f'alpha 图：{alpha_dir}')
    print(f'对比图：{comparison_dir}')
    print(f'参数：L_low={L_LOW}, L_high={L_HIGH}, sigma1={SIGMA1}')
    print(f'共找到 {len(input_paths)} 张图片')

    # YAML 只提供网络结构等配置，不使用 datasets.val 中的数据路径。
    opt = parse(str(opt_path), is_train=False)
    opt['dist'] = False
    model_restoration = create_model(opt).net_g

    checkpoint = torch.load(str(weights_path), map_location='cpu')
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except RuntimeError:
        # 兼容参数名称带有 module. 前缀的 DataParallel 权重。
        new_checkpoint = {
            'module.' + key: value
            for key, value in checkpoint['params'].items()
        }
        model_restoration.load_state_dict(new_checkpoint)

    model_restoration.cuda()
    if not isinstance(
            model_restoration,
            (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    def forward_model(input_tensor, model):
        """根据手动配置选择整图推理或重叠分块推理。"""
        return tiled_forward(input_tensor, model, TILE_SIZE, TILE_OVERLAP)

    skipped_count = 0

    with torch.inference_mode():
        for input_path in tqdm(input_paths, desc='正在增强'):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # [H, W, C] RGB uint8 -> [1, C, H, W] float32，范围 [0, 1]。
            image = np.float32(utils.load_img(str(input_path))) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            original = image.unsqueeze(0).cuda()

            _, _, height, width = original.shape

            # 第一步：在原始分辨率上估计 alpha(p)。
            # alpha 由图像自身亮度决定，不需要目标检测，也不需要 ROI。
            alpha = estimate_alpha(original, l_low, l_high, SIGMA1)
            alpha_max = float(alpha.max())

            # 工程细节 ②：alpha 处处为 0 时输出恒等于输入，前向可安全跳过。
            skipped = ENABLE_SAFE_SKIP and alpha_max == 0.0
            if skipped:
                skipped_count += 1
                enhanced = original
                fused = original
                retinex_uint8 = to_uint8_image(original)
            else:
                # 第二步：补边到 4 的倍数后跑一次 Retinexformer 得到 E。
                padded_height = ((height + SIZE_FACTOR) // SIZE_FACTOR) * SIZE_FACTOR
                padded_width = ((width + SIZE_FACTOR) // SIZE_FACTOR) * SIZE_FACTOR
                pad_height = padded_height - height if height % SIZE_FACTOR != 0 else 0
                pad_width = padded_width - width if width % SIZE_FACTOR != 0 else 0
                input_tensor = F.pad(
                    original, (0, pad_width, 0, pad_height), 'reflect')

                if TILE_SIZE > 0:
                    if SELF_ENSEMBLE:
                        enhanced = self_ensemble(
                            input_tensor, model_restoration,
                            forward_fn=forward_model)
                    else:
                        enhanced = forward_model(input_tensor, model_restoration)
                else:
                    if SELF_ENSEMBLE:
                        enhanced = self_ensemble(input_tensor, model_restoration)
                    else:
                        enhanced = model_restoration(input_tensor)

                # 裁掉补边，恢复原始尺寸后再融合，保证 X 与 E 逐像素对齐。
                enhanced = torch.clamp(enhanced[:, :, :height, :width], 0.0, 1.0)

                # 第三步：残差融合 y = X + alpha * (E - X)。
                fused = fuse(original, enhanced, alpha)
                retinex_uint8 = to_uint8_image(enhanced)

            original_uint8 = to_uint8_image(original)
            fused_uint8 = to_uint8_image(fused)
            alpha_uint8 = alpha_to_uint8(alpha)

            stem = input_path.stem
            # 跳过的图片没有真正的 Retinex 输出，E 只是输入图的副本，故不写入。
            if SAVE_RETINEX and not skipped:
                utils.save_img(str(retinex_dir / f'{stem}.png'), retinex_uint8)
            if SAVE_ALPHA:
                cv2.imwrite(str(alpha_dir / f'{stem}.png'), alpha_uint8)
            utils.save_img(str(output_dir / f'{stem}.png'), fused_uint8)

            if SAVE_COMPARISON:
                # alpha 用 JET 伪彩显示：蓝=0（不增强），红=1（全力增强）。
                alpha_color = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
                alpha_color = cv2.cvtColor(alpha_color, cv2.COLOR_BGR2RGB)
                comparison = build_comparison(
                    [original_uint8, retinex_uint8, alpha_color, fused_uint8],
                    ['input', 'retinex E', 'alpha', 'fused y'],
                    COMPARISON_MAX_WIDTH,
                )
                cv2.imwrite(str(comparison_dir / f'{stem}.png'), comparison)

            # 逐图打印：alpha 分布 + 过曝率（3.1.5 指标 2），用于快速核对算法是否按设计工作。
            alpha_mean = float(alpha.mean())
            enhanced_ratio = float((alpha > 0.05).float().mean())
            if skipped:
                alpha_desc = '跳过前向(alpha≡0)'
            else:
                alpha_desc = (f'alpha均值={alpha_mean:.3f}, 峰值={alpha_max:.2f}, '
                              f'alpha>0.05 占比={enhanced_ratio:.1%}')
            print(
                f'{input_path.name}: {alpha_desc}, 过曝率 input/E/y = '
                f'{overexposure_ratio(original_uint8):.2%}/'
                f'{overexposure_ratio(retinex_uint8):.2%}/'
                f'{overexposure_ratio(fused_uint8):.2%}'
            )

    print(f'推理完成。跳过前向 {skipped_count}/{len(input_paths)} 张（alpha 处处为 0）。')
    print(f'最终增强结果：{output_dir}')
    print(f'Retinex 直出结果：{retinex_dir}')


if __name__ == '__main__':
    main()
