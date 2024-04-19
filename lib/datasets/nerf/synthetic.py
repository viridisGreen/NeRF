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
        self.N_rays = cfg.task_arg.N_rays
        
        with open(os.path.join(self.basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            meta = json.load(fp)
            
        all_imgs = []
        all_poses = []
        
        imgs = []
        poses = []
        for frame in meta['frames']:
            fname = os.path.join(self.basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        all_imgs.append(imgs)
        all_poses.append(poses)
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        imgs = imgs[cams] #! 比源码多出来的, 初步判断取代了源码中skip的作用
        poses = poses[cams]
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        if self.input_ratio != 1.: #* 对应源码里的half_res
            H = int(H * self.input_ratio) #? 可能会有错, 后续修改：// -> *
            W = int(W * self.input_ratio) #? 可能会有错, 后续修改：// -> *
            focal = focal / self.input_ratio #? 可能会有错
                        
            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            
        hwf = [H, W, focal]    
        
        if self.white_bkgd:
            images = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
        else:
            images = imgs[...,:3]
        
        print('Loaded blender', images.shape, hwf)
        
        # # set variables
        # self.imgs = images
        # self.poses = poses
        self.hwf = [H, W, focal]
        
        K = None
        
        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])  
        # ipdb.set_trace()
        # Prepare raybatch tensor if batching random rays
        N_rand = cfg.task_arg.N_rays #* 一批采样几条光线
        #todo 默认use_batching
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [N*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        # print('shuffle rays')
        # np.random.shuffle(rays_rgb) #* 考虑到后面随机抽取，这里不用也不能shuffle了
        # print('done') 
        # ipdb.set_trace()
        self.rays_rgb = rays_rgb   
        
    
    def __getitem__(self, index):   
        N = len(self.rays_rgb) #* rays总数
        H, W, _ = self.hwf
        N_item = int(N / H / W) #* 一张图片能产生的rays数量
        # ipdb.set_trace()
        if self.split == "train":
            ids = np.random.choice(N_item, self.N_rays, replace=True) 
            ids = ids + index * N_item
            rays_rgb = self.rays_rgb[ids]
        else:
            ids = np.random.choice(N_item, self.N_rays, replace=True) 
            rays_rgb = self.rays_rgb[ids]
        # ipdb.set_trace()
        return rays_rgb
    

    def __len__(self):
        # N = self.rays_rgb.shape[0]
        # # ipdb.set_trace()
        # if self.split == "train":
        #     return int(N / self.N_rays / cfg.train.batch_size)
        # else:
        #     return int(N / self.N_rays / cfg.test.batch_size)
        return 3






#=====================================================================================================#

def get_rays_np(H, W, K, c2w): #* K是相机内参
    '''用于计算从相机坐标系到世界坐标系的射线方向和起点'''
    #todo 生成像素索引网格
    #* np.arrange: 生成等间距的值构成的数组
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    #todo 计算射线方向
    #? 分别计算 x 和 y 方向的归一化坐标，考虑到了焦距和主点偏移
    #? -np.ones_like(i) 添加了 z 方向的坐标，这里固定为 -1 表示所有射线方向都指向 z 负方向（相机前方）
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    #todo 将射线方向旋转到世界坐标系
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    #todo 计算射线的起点
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    #* rays_o: 每个射线的起点都相同，即相机的世界坐标位置
    #* rays_d: 这些方向已从相机坐标系转换到世界坐标系
    return rays_o, rays_d #* 返回射线的起点和方向