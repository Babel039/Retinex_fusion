# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from inference_utils import self_ensemble, tiled_forward

# 命令行参数控制配置文件、模型权重、输出目录及推理方式。
parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')
parser.add_argument('--tile_size', default=0, type=int,
                    help='Tile size for memory-efficient inference; 0 disables tiling')
parser.add_argument('--tile_overlap', default=32, type=int,
                    help='Overlap between adjacent inference tiles')

args = parser.parse_args()

# 只让当前进程看到用户指定的 GPU，例如 --gpus 0 或 --gpus 0,1。
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

# 读取 YAML 测试配置。网络结构和验证集路径都由该文件提供。
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
# 测试脚本不启用分布式训练环境。
opt['dist'] = False

# 读取原始 YAML，并移除网络类型字段。create_model 会根据 opt 创建实际网络。
x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')

# 创建图像增强网络 net_g。
model_restoration = create_model(opt).net_g

# 加载训练好的模型参数。权重文件中的 params 保存网络状态字典。
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    # 兼容使用 DataParallel 保存、参数名称需要带 module. 前缀的权重。
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
# 多 GPU 环境下由 DataParallel 分发输入；单 GPU 时也可以使用该包装。
if not isinstance(model_restoration,
                  (nn.DataParallel, nn.parallel.DistributedDataParallel)):
    model_restoration = nn.DataParallel(model_restoration)
# 切换到评估模式，关闭 Dropout 并固定 BatchNorm 等层的行为。
model_restoration.eval()


def forward_model(input_tensor, model):
    """统一的推理入口：按参数选择整图推理或重叠分块推理。

    input_tensor 的形状为 [B, C, H, W]，像素值范围通常为 [0, 1]。
    tile_size 为 0 时 tiled_forward 会直接调用 model(input_tensor)。
    """
    return tiled_forward(input_tensor, model, args.tile_size,
                         args.tile_overlap)

# Retinexformer 要求输入宽高能被 4 整除，推理前不足的部分会补齐。
factor = 4
dataset = args.dataset
# 默认结果目录由“数据集/配置文件名/权重文件名”组成，便于区分实验。
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []

# SID、SMID 和 SDSD 有专用 Dataset 类，需要保留序列或场景目录结构。
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    if '~' in opt['dataroot_gt']:
        opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    # 每次处理一张图片，测试时不打乱样本顺序。
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    # inference_mode 关闭梯度记录，降低推理时的显存和计算开销。
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            # 尝试释放上一轮遗留的 CUDA 缓存，降低连续测试时的显存压力。
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # lq 是低光输入；gt 是正常曝光参考图，用于计算客观指标。
            input_ = data_batch['lq']
            # 保存用数组从 [1, C, H, W] 转为 [H, W, C]。
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # 右侧和底部采用镜像填充，使宽高成为 4 的倍数。
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                # 对翻转/旋转后的 8 个输入分别推理，再将结果还原并取平均。
                restored = self_ensemble(
                    input_, model_restoration, forward_fn=forward_model)
            else:
                restored = forward_model(input_, model_restoration)

            # 裁掉补边区域，恢复输入图片的原始尺寸。
            restored = restored[:, :, :h, :w]

            # 限制像素范围，并从 [1, C, H, W] 转为 NumPy 的 [H, W, C]。
            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # 使用 GT 的平均灰度校正输出亮度，与 KinD、LLFlow 等测试设置一致。
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            # PSNR 使用 [0, 1] 浮点图；SSIM 使用转换后的 [0, 255] uint8 图。
            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            # 按输入图片所属的场景子目录分别保存增强图、输入图和 GT。
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:
    # 其他数据集（如 NTIRE）直接从 YAML 的 val 路径读取成对图片。
    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    # 自然排序可避免 10.png 排在 2.png 前面；输入与 GT 按排序后的位置配对。
    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

    target_paths = natsorted(glob(os.path.join(
        target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

    # 测试只执行前向传播，不构建梯度计算图。
    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):

            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # OpenCV 读取结果由 utils 转为 RGB，再归一化到 [0, 1]。
            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            # [H, W, C] -> [C, H, W] -> [1, C, H, W]，然后送入 GPU。
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # 保存原始尺寸，并在右侧、底部镜像补齐到 4 的倍数。
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.tile_size > 0:
                # 分块推理可以显著降低高分辨率图片的峰值显存占用。
                if args.self_ensemble:
                    restored = self_ensemble(
                        input_, model_restoration, forward_fn=forward_model)
                else:
                    restored = forward_model(input_, model_restoration)
            elif h < 3000 and w < 3000:
                # 未指定分块且图片较小时，直接进行整图推理。
                if args.self_ensemble:
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = model_restoration(input_)
            else:
                # 超大图片按奇数列和偶数列拆成两幅图推理，再交错合并。
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2

            # 移除补边，恢复原始宽高。
            restored = restored[:, :, :h, :w]

            # 将网络输出转成保存和指标计算所需的 [H, W, C] NumPy 数组。
            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # 可选：按照 GT 与输出的平均灰度比修正增强结果的整体亮度。
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            # 逐张记录指标，循环结束后计算整个测试集的平均值。
            psnr.append(utils.PSNR(target, restored))
            ssim.append(utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored)))
            # output_dir 优先级高于自动生成的 result_dir。
            if output_dir != '':
                utils.save_img((os.path.join(output_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            else:
                utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

# 汇总并打印整个测试集的平均 PSNR 和平均 SSIM。
psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
