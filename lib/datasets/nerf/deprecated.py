import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2

class Dataset(data.Dataset):
    def __init__(self, **kwargs): 
        data_root = kwargs['data_root']
        scene = cfg.scene
        basedir = os.path.join(data_root, scene)

        split = kwargs['split']
        input_ratio = kwargs['input_ratip']
        cams = kwargs['cams']

################# LLFF data type ####################
        # poses, bds, imgs = self._load_data(basedir)
        # print('Loaded', basedir, bds.min(), bds.max())

        # # Correct rotation matrix ordering and move variable dim to axis 0
        # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        # poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        # imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        # images = imgs
        # bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # poses = self.recenter_poses(poses) # recenter
        # poses, render_poses, bds = self.pherify_poses(poses, bds) #spherify

        # render_poses = np.array(render_poses).astype(np.float32)

        # c2w = self.poses_avg(poses)
        # print('Data:')
        # print(poses.shape, images.shape, bds.shape)
        
        # dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
        # i_test = np.argmin(dists)
        # print('HOLDOUT view is', i_test)
        
        # images = images.astype(np.float32)
        # poses = poses.astype(np.float32)

        # self.images = images
        # self.poses = poses
        # self.bds = bds
        # self.render_poses = render_poses
        # self.i_test = i_test
################# LLFF data type ####################

################# Blender data type ####################
        metas = {}
        with open(os.path.join(basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
            metas[split] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        
        meta = metas[split]
        imgs = []
        poses = []
        skip = 1
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(1)]
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([self.pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        self.imgs = imgs
        self.poses = poses
        self.render_poses = render_poses
        self.hwf = [H, W, focal]
        self.i_split = i_split
################# Blender data type ####################



################# LLFF data type ####################
    def _load_data(self, basedir, load_imgs=True):
    
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        bds = poses_arr[:, -2:].transpose([1,0])
        
        img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape
        
        imgdir = basedir
        if not os.path.exists(imgdir):
            print( imgdir, 'does not exist, returning' )
            return
        
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        if poses.shape[-1] != len(imgfiles):
            print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
            return
        
        sh = imageio.imread(imgfiles[0]).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1./factor
        
        if not load_imgs:
            return poses, bds
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
            
        imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
        imgs = np.stack(imgs, -1)  
        
        print('Loaded image data', imgs.shape, poses[:,-1,0])
        return poses, bds, imgs

    def recenter_poses(self, poses):

        poses_ = poses+0
        bottom = np.reshape([0,0,0,1.], [1,4])
        c2w = poses_avg(poses)
        c2w = np.concatenate([c2w[:3,:4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
        poses = np.concatenate([poses[:,:3,:4], bottom], -2)

        poses = np.linalg.inv(c2w) @ poses
        poses_[:,:3,:4] = poses[:,:3,:4]
        poses = poses_
        return poses

    def spherify_poses(self, poses, bds):
    
        p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
        
        rays_d = poses[:,:3,2:3]
        rays_o = poses[:,:3,3:4]

        def min_line_dist(rays_o, rays_d):
            A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
            b_i = -A_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        
        center = pt_mindist
        up = (poses[:,:3,3] - center).mean(0)

        vec0 = self.normalize(up)
        vec1 = self.normalize(np.cross([.1,.2,.3], vec0))
        vec2 = self.normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)

        poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
        
        sc = 1./rad
        poses_reset[:,:3,3] *= sc
        bds *= sc
        rad *= sc
        
        centroid = np.mean(poses_reset[:,:3,3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad**2-zh**2)
        new_poses = []
        
        for th in np.linspace(0.,2.*np.pi, 120):

            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0,0,-1.])

            vec2 = self.normalize(camorigin)
            vec0 = self.normalize(np.cross(vec2, up))
            vec1 = self.normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)

            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        
        new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
        poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
        
        return poses_reset, new_poses, bds

    def poses_avg(self, poses):

        hwf = poses[0, :3, -1:]

        center = poses[:, :3, 3].mean(0)
        vec2 = self.normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self.viewmatrix(vec2, up, center), hwf], 1)
        
        return c2w

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def viewmatrix(self, z, up, pos):
        vec2 = self.normalize(z)
        vec1_avg = up
        vec0 = self.normalize(np.cross(vec1_avg, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m
################# LLFF data type ####################

################# Blender data type ####################
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

    def pose_spherical(self, theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
################# Blender data type ####################


    def __getitem__(self, idx):
        images = self.imgs
        poses = self.poses
        render_poses = self.recenter_poses
        hwf = self.hwf
        i_train, i_val, i_test = self.i_split

        near = 2.
        far = 6.

        if cfg.task_arg.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        K = None
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)






    def __len__(self):
        pass
       

