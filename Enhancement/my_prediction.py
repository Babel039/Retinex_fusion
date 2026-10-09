"""使用 Retinexformer 批量增强指定目录中的低光图片。

运行前只需修改“手动配置区”，然后在项目根目录执行：
    python Enhancement/my_prediction.py

YAML 文件仅用于创建网络，不会读取其中配置的验证集路径。
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
OUTPUT_DIR = r'results/my_prediction'

# 指定可见 GPU。单卡填 '0'；多卡可填 '0,1'。
GPU_IDS = '0'

# 与原测试脚本一致：0 表示整图推理；显存不足时可设置为 512 或 256。
TILE_SIZE = 512
TILE_OVERLAP = 32

# True 会对 8 种翻转/旋转结果取平均，效果可能更好，但推理时间约为 8 倍。
SELF_ENSEMBLE = False
# ======================================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SIZE_FACTOR = 4


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


def main():
    """加载模型，依次增强输入目录中的图片并保存为 PNG。"""
    opt_path = resolve_project_path(OPT_PATH)
    weights_path = resolve_project_path(WEIGHTS_PATH)
    input_dir = resolve_project_path(INPUT_DIR)
    output_dir = resolve_project_path(OUTPUT_DIR)

    if not opt_path.is_file():
        raise FileNotFoundError(f'YAML 配置文件不存在：{opt_path}')
    if not weights_path.is_file():
        raise FileNotFoundError(f'模型权重不存在：{weights_path}')

    input_paths = find_image_paths(input_dir)
    if not input_paths:
        raise RuntimeError(f'输入目录中没有找到受支持的图片：{input_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)

    # 必须在导入 torch 前指定可见 GPU。
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from skimage import img_as_ubyte
    from tqdm import tqdm

    import utils
    from basicsr.models import create_model
    from basicsr.utils.options import parse
    from inference_utils import self_ensemble, tiled_forward

    if not torch.cuda.is_available():
        raise RuntimeError('未检测到可用的 CUDA GPU，当前脚本与原测试脚本一样使用 GPU 推理。')
    if TILE_SIZE < 0:
        raise ValueError('TILE_SIZE 不能小于 0。')
    if TILE_SIZE > 0 and not 0 <= TILE_OVERLAP < TILE_SIZE:
        raise ValueError('TILE_OVERLAP 必须大于等于 0 且小于 TILE_SIZE。')

    print(f'使用 GPU：{GPU_IDS}')
    print(f'输入目录：{input_dir}')
    print(f'输出目录：{output_dir}')
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
        return tiled_forward(
            input_tensor, model, TILE_SIZE, TILE_OVERLAP)

    with torch.inference_mode():
        for input_path in tqdm(input_paths, desc='正在增强'):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # [H, W, C] RGB uint8 -> [1, C, H, W] float32，范围 [0, 1]。
            image = np.float32(utils.load_img(str(input_path))) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            input_tensor = image.unsqueeze(0).cuda()

            _, _, height, width = input_tensor.shape
            padded_height = ((height + SIZE_FACTOR) // SIZE_FACTOR) * SIZE_FACTOR
            padded_width = ((width + SIZE_FACTOR) // SIZE_FACTOR) * SIZE_FACTOR
            pad_height = padded_height - height if height % SIZE_FACTOR != 0 else 0
            pad_width = padded_width - width if width % SIZE_FACTOR != 0 else 0
            input_tensor = F.pad(
                input_tensor, (0, pad_width, 0, pad_height), 'reflect')

            if TILE_SIZE > 0:
                if SELF_ENSEMBLE:
                    restored = self_ensemble(
                        input_tensor,
                        model_restoration,
                        forward_fn=forward_model,
                    )
                else:
                    restored = forward_model(input_tensor, model_restoration)
            elif height < 3000 and width < 3000:
                if SELF_ENSEMBLE:
                    restored = self_ensemble(
                        input_tensor, model_restoration)
                else:
                    restored = model_restoration(input_tensor)
            else:
                # 保留原脚本的大图策略：奇数列、偶数列分别推理后交错合并。
                input_odd = input_tensor[:, :, :, 1::2]
                input_even = input_tensor[:, :, :, 0::2]
                if SELF_ENSEMBLE:
                    restored_odd = self_ensemble(
                        input_odd, model_restoration)
                    restored_even = self_ensemble(
                        input_even, model_restoration)
                else:
                    restored_odd = model_restoration(input_odd)
                    restored_even = model_restoration(input_even)

                restored = torch.zeros_like(input_tensor)
                restored[:, :, :, 1::2] = restored_odd
                restored[:, :, :, 0::2] = restored_even

            # 裁掉补边，并转回 [H, W, C] RGB uint8 图片。
            restored = restored[:, :, :height, :width]
            restored = (
                torch.clamp(restored, 0, 1)
                .cpu()
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .numpy()
            )

            output_path = output_dir / f'{input_path.stem}.png'
            utils.save_img(str(output_path), img_as_ubyte(restored))

    print(f'推理完成，结果已保存到：{output_dir}')


if __name__ == '__main__':
    main()
