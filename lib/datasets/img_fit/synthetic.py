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
        #* 这里的scene是'train/val/test'
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene 
        view = kwargs['view'] #* 不太清楚是干什么的
        self.input_ratio = kwargs['input_ratio'] #* 预先定义的比例值，用于指定图像的缩放比例
        self.data_root = os.path.join(data_root, scene)
        self.split = split #* 'train/test'
        self.batch_size = cfg.task_arg.N_pixels

        # read image
        image_paths = []
        #* 返回值为py字典
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train')))) 
        for frame in json_info['frames']: #* frames也是一个字典
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))

        #* 将像素值全部变换为0~1的浮点数，同时取出了第view张图片
        img = imageio.imread(image_paths[view])/255. 
        
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:]) #* 根据透明度处理图像，将rgba转换为rgb图像
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        # set image
        self.img = np.array(img).astype(np.float32)
        # set uv
        H, W = img.shape[:2]
        #* X是每个像素的水平坐标值，Y是垂直坐标值
        X, Y = np.meshgrid(np.arange(W), np.arange(H)) #* 分别代表图像在像素网格上的坐标值
        #! uv坐标是把像素点的位置映射到了在uv坐标系下的坐标
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1) #* 映射到一个正方形
        #* 重塑为(N=H*W, 2)的形状，每一行包含uv坐标
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32) 


    def __getitem__(self, index):
        if self.split == 'train':
            #* 从uv中随机选取batch size的索引，无放回抽样；这意味着每个索引只能被选择一次，用于从数据集中随机抽取一个批次的数据。
            ids = np.random.choice(len(self.uv), self.batch_size, replace=False) 
            uv = self.uv[ids]
            rgb = self.img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.img.reshape(-1, 3)
        ret = {'uv': uv, 'rgb': rgb} # input and output. they will be sent to cuda
        ret.update({'meta': {'H': self.img.shape[0], 'W': self.img.shape[1]}}) # meta means no need to send to cuda
        return ret

    def __len__(self):
        # we only fit 1 images, so we return 1
        return 1
