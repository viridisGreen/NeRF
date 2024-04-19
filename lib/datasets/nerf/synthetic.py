import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2
import ipdb

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene #* 这里的scene是lego
        cams = kwargs['cams'] #? 不太清楚是干什么的，初步判断为选图片的
        self.input_ratio = kwargs['input_ratio'] #* 预先定义的比例值，用于指定图像的缩放比例
        self.basedir = os.path.join(data_root, scene)
        self.split = split #* 'train/test'
        self.white_bkgd = cfg.task_arg.white_bkgd
        
        with open(os.path.join(self.basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
            
        all_imgs = []
        all_poses = []
        
        imgs = []
        poses = []
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        imgs = imgs[cams] #todo 比源码多出来的
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        if self.input_ratio != 1.: #* 对应源码里的half_res
            H = H // self.input_ratio #? 可能会有错
            W = W // self.input_ratio #? 可能会有错
            focal = focal / self.input_ratio #? 可能会有错
            
            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
        
        # set variables
        self.imgs = imgs
        self.poses = poses
        self.hwf = [H, W, focal]
        
        print('Loaded blender', images.shape, hwf)
        
    
    def __getitem__(self, index):
        if self.white_bkgd:
            images = self.imgs[...,:3] * self.imgs[...,-1:] + (1. - self.imgs[...,-1:])
        else:
            images = self.imgs[...,:3]