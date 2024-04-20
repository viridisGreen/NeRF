import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
import json
import ipdb

class Evaluator:

    def __init__(self,):
        self.psnrs = [] #* 用于存储每个图像的峰值信噪比（PSNR）评估结果
        #* 创建文件夹用于存储输出数据
        os.system('mkdir -p ' + cfg.result_dir)
        os.system('mkdir -p ' + cfg.result_dir + '/vis') #* 存储可视化输出

    def evaluate(self, output, batch):
        # assert image number = 1
        H, W = batch['meta']['H'].item(), batch['meta']['W'].item() #* 从batch['meta']中提取原始图像的高度和宽度（H和W）
        #* 将预测rgb和真实rgb从张量转为np数组，并调整形状
        # ipdb.set_trace()
        pred_rgb = output[0][0].detach().cpu().numpy()
        
        batch = batch['rays_rgb']
        batch = batch.reshape(-1, 3, 3)
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        
        gt_rgb = target_s[0].detach().cpu().numpy()
        #* 用skimage.metrics中的psnr函数计算预测图像相对于真实图像的PSNR值，并append到self.psnrs中
        psnr_item = psnr(gt_rgb, pred_rgb, data_range=1.)
        self.psnrs.append(psnr_item)
        save_path = os.path.join(cfg.result_dir, 'vis/res.jpg')
        # imageio.imwrite(save_path, img_utils.horizon_concate(gt_rgb, pred_rgb))
        # image_float64 = img_utils.horizon_concate(gt_rgb, pred_rgb) * 255.0 #* 真实|预测，拼接
        # image_int8 = image_float64.astype(np.uint8) #* 类型转换，float[0, 1] -> int[0, 255]
        # imageio.imwrite(save_path, image_int8) #* 存储到指定path

    def summarize(self):
        '''汇总评估过程中收集的所有PSNR值, 并打印和保存评估结果'''
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        print(ret)
        self.psnrs = [] #* 清空，为下一次评估准备
        print('Save visualization results to {}'.format(cfg.result_dir))
        #* 将评估结果（存储在字典ret中）以JSON格式写入到一个文件中
        json.dump(ret, open(os.path.join(cfg.result_dir, 'metrics.json'), 'w')) 
        return ret
