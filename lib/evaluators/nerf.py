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
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        hwf = [batch['meta']['H'].item(), batch['meta']['W'].item(), batch['meta']['focal']]
        K = batch['meta']['K']
        chunk = cfg.task_arg.chunk_size
        save_path = os.path.join(cfg.result_dir, 'vis/res.jpg')
        
        self.render_path(render_poses, hwf, K, chunk, savedir=save_path)

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

    def render_path(self, render_poses, hwf, K, chunk, render_kwargs=None, gt_imgs=None, savedir=None, render_factor=0):

        H, W, focal = hwf

        if render_factor!=0:
            # Render downsampled for speed
            H = H//render_factor
            W = W//render_factor
            focal = focal/render_factor

        rgbs = []
        disps = []

        t = time.time()
        for i, c2w in enumerate(tqdm(render_poses)):
            print(i, time.time() - t)
            t = time.time()
            # rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
            rgb, disp, acc = output[0], output[1], output[2]
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())
            if i==0:
                print(rgb.shape, disp.shape)

            """
            if gt_imgs is not None and render_factor==0:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                print(p)
            """
            
            to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                # filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(savedir, rgb8)


        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

        return rgbs, disps
    
#=================================================================================================#

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w