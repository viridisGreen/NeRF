import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg
import ipdb

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        net_cfg = cfg.network
        
        #todo 获取网络参数
        W, D, V_D = net_cfg.nerf.W, net_cfg.nerf.D, net_cfg.nerf.V_D
        self.xyz_encoder, self.xyz_input_ch = get_encoder(net_cfg.xyz_encoder) 
        self.dir_encoder, self.dir_input_ch = get_encoder(net_cfg.dir_encoder) 
        self.skips = [4]
        
        #todo 网络的主干部分，rgb和α都要经过这一部分
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.xyz_input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.xyz_input_ch, W) for i in range(D-1)])
        
        #todo 网络的α输出部分
        self.alpha_linear = nn.Linear(W, 1) #* 从主干部分的输出 到 α
        
        #todo 网络的rgb输出部分
        self.feature_linear = nn.Linear(W, W) #* 主干部分 -> 方向输入
        #* 相机方向作为输入的网络部分，到输出颜色的前一层，共一层
        self.views_linears = nn.ModuleList([nn.Linear(self.dir_input_ch + W, W//2)]) 
        self.rgb_linear = nn.Linear(W//2, 3) #* 方向输入部分的输出 -> rgb值
        

    def forward(self, batch): #* [ro+rd, B, 3*?]
############### desperate #############   
        # input_pts, input_views = torch.split(batch, [self.xyz_input_ch, self.dir_input_ch], dim=-1)
        # h = input_pts
        # #todo 主干部分的前向传播
        # for i, l in enumerate(self.pts_linears):
        #     h = self.pts_linears[i](h)
        #     h = F.relu(h)
        #     if i in [self.skips]:
        #         h = torch.cat([input_pts, h], -1)

        # #todo 使用观察方向作为输入，就分别输出rgb和α
        # alpha = self.alpha_linear(h)
        
        # feature = self.feature_linear(h)
        # h = torch.cat([feature, input_views], -1)
    
        # for i, l in enumerate(self.views_linears):
        #     h = self.views_linears[i](h)
        #     h = F.relu(h)
        # rgb = self.rgb_linear(h)
        
        # outputs = torch.cat([rgb, alpha], -1) #* 将分别输出的rgb和α连接后再返回

        # return outputs           
############### desperate #############    

        batch = batch['rays_rgb']
        batch = batch.reshape(-1, 3, 3)
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        # use provided ray batch
        rays_o, rays_d = batch_rays #* [1, batchsize, 3]
        
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) #* 标准化
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() #* [1, batchsize, 3] -> [batchsize, 3]

        sh = rays_d.shape # [..., 3]
        # if ndc:
        #     # for forward facing scenes
        #     rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1,3]).float() #* [1, batchsize, 3] -> [batchsize, 3]
        rays_d = torch.reshape(rays_d, [-1,3]).float() #* [1, batchsize, 3] -> [batchsize, 3]

        #* 取出第一维度主要是为了获取形状, [batchsize, 1]
        near, far = 0., 1.
        near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) 
        rays = torch.cat([rays_o, rays_d, near, far], -1) #* [batchsize, 3+3+1+1]
        rays = torch.cat([rays, viewdirs], -1) #* [batchsize, 3+3+1+1 + 3]
        
        #todo 构造render arguments
        network_query_fn = lambda inputs, viewdirs, network_fn : self.run_network(inputs, viewdirs, network_fn,
                                                            embed_fn=self.xyz_encoder,
                                                            embeddirs_fn=self.dir_encoder,
                                                            netchunk=cfg.task_arg.chunk_size)
        
        render_kwargs_train = {
            'network_query_fn' : network_query_fn, #* 执行力网络的前向传播
            'perturb' : 1., #* 用于控制渲染过程中的随机扰动，应该是布尔值
            'N_importance' : cfg.task_arg.cascade_samples[1], #* 细网络采样点数
            'network_fine' : self.network_fn, #! 细网络
            'N_samples' : cfg.task_arg.cascade_samples[0], #* 粗网络采样点数
            'network_fn' : self.network_fn, #! 粗网络
            # 'use_viewdirs' : True, #* 方向是否使用embedding
            'white_bkgd' : cfg.task_arg.white_bkgd, #* bool值，是否使用white bkgd
            'raw_noise_std' : 0., #* 应该是和perturb对应的值
        }
        
        # Render and reshape
        # ipdb.set_trace()
        all_ret = self.batchify_rays(rays, cfg.task_arg.chunk_size, **render_kwargs_train) #* batchify_rays的返回值是字典
    
        #todo 根据原始射线数据的形状对返回值进行重塑
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:]) #* [1, batchsize] + ...
            all_ret[k] = torch.reshape(all_ret[k], k_sh) #* 形状全部重塑为[1, batchsize, ...]

        k_extract = ['rgb_map', 'disp_map', 'acc_map'] #* 想单独返回的键名
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract} #* 除了单独返回的，其他的返回值
        
        return ret_list + [ret_dict]
    
    
    
#=====================================================================================================#    
    
    #todo 把rays分chunk去render，连接并返回
    def batchify_rays(self, rays_flat, chunk=1024*32, **kwargs):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self.render_rays(rays_flat[i:i+chunk], **kwargs) #* render_rays的返回值是一个字典
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} #* 将每个结果列表沿着第一个维度（即批次维度）合并
        return all_ret #* 返回值是字典
    

    #todo 只在batchify_rays里被调用, nerf模型的核心渲染函数
    def render_rays(self, 
                    ray_batch, 
                    network_fn, #! 直接使用init里定义好的模块
                    network_query_fn, #* 直接使用init里定义好的模块
                    N_samples=64, #* 增加default值为64
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=128, #* 修改default值为128
                    network_fine=None,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    verbose=False,
                    pytest=False):
        """Volumetric rendering.
        Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        network_fn: function. Model for predicting RGB and density at each point
            in space.
        network_query_fn: function used for passing queries to network_fn.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        #? 简单来说就是, 是均匀采样还是xx
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
        network_fine: "fine" network with same spec as network_fn.
        white_bkgd: bool. If True, assume a white background.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0] #* rays的数量
        #* ray_batch-[N_rays, 8 or 11], rays_o(3), rays_d(3), near(1), far(1), (viewdir(3))
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) #* [N_rays, 1, 2]
        near, far = bounds[...,0], bounds[...,1] # [-1,1] #? [N_rays, 1, 1]

        t_vals = torch.linspace(0., 1., steps=N_samples) #* 生成一个从 0 到 1 均匀分布的数值数组，包含 N_samples 个元素
        if not lindisp: #* 采样点会均匀分布
            z_vals = near * (1.-t_vals) + far * (t_vals) #* 计算就先不管了  [N_samples, ]
        else: #* 会使采样点更集中在near端
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) #* 计算就先不管了  [N_samples, ]

        z_vals = z_vals.expand([N_rays, N_samples]) #* [N_rays, N_samples]

        #todo 在z_vals中引入随机性
        if perturb > 0.:
            #* 建议去看nerf速通(下)，讲的很好
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) #* 计算每对相邻采样点之间的中点
            upper = torch.cat([mids, z_vals[...,-1:]], -1) #* 中点s + 最后一个采样点
            lower = torch.cat([z_vals[...,:1], mids], -1) #* 第一个采样点 + 中点s
            # stratified samples in those intervals
            #* 生成与z_vals形状相同的随机数张量t_rand，每个元素的值在0~1之间。这些随机数用于在每个定义的间隔内随机选择采样深度
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest: #* 测试模式下 (pytest=True)，使用固定的随机种子生成随机数，确保结果的可重复性
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand #* 根据t_rand更新采样点，使其随机化

        #* 采样点 = 起点 + 方向 * 采样模板（类似梳函数
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


        # raw = run_network(pts)
        # ipdb.set_trace()
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        #todo 计算细网络的输出
        if N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map #* 存储粗网络的一些输出结果

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) #* 计算每对相邻采样点之间的中点，就是前面的mids
            #todo 和粗网络构造采样点的方式不同，根据密度进行采样
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach() #* 从计算图中分离张量，不会进行梯度追踪，减少内存使用

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1) #* 将粗采样和细采样合并，并进行升序排序，64+128?
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            run_fn = network_fn if network_fine is None else network_fine
            # raw = run_network(pts, fn=run_fn)
            raw = network_query_fn(pts, viewdirs, run_fn)

            rgb_map, disp_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        #todo 如果启用了调试模式 (DEBUG)，则检查返回的任何数据是否包含非法值（如 NaN 或无穷大）
        # for k in ret:
        #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
        #         print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret #* 返回一个字典
    
    
    #todo 只在render_rays里被调用, 辅助函数
    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        #* 论文里的渲染公式（不然就是渲染公式旁边的那个公式
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        #todo 计算每个采样间隔的距离
        dists = z_vals[...,1:] - z_vals[...,:-1] #* 首先计算相邻采样点之间的距离
        #* 在最后一个采样点，添加一个非常大的值（1e10），这样处理是为了确保最后一个采样区间封闭
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        
        #* norm(rays_d[...,None,:], dim=-1) 的形状是 [num_rays, 1]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1) #* 距离与ray_d的长度进行缩放，确保距离在三维空间中是正确的

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        #todo 增加噪声并计算alpha的值
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        #* 权重，不透明度的另一种体现
        #* torch.cumprod: 计算累积乘积，即每个点的权重等于该点透明度的补集与之前所有点的透明度的补集的乘积，
        #*                结果表示到达当前点之前，光线没有被阻挡的概率
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3], 使用权重加权把num_samples along ray 积分积掉

        depth_map = torch.sum(weights * z_vals, -1) #* 使用权重计算每条射线的加权平均深度
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) #* 视差图，深度的倒数
        acc_map = torch.sum(weights, -1) #* 所有采样点的权重和，表示整条射线的总不透明度

        if white_bkgd: #* 1-acc_map计算射线未被吸收的光量，并将其作为背景光加到 rgb_map 上，实现射线穿透效果的模拟
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map
    
    
    #todo 做一遍网络的前馈
    def network_fn(self, batch):
        input_pts, input_views = torch.split(batch, [self.xyz_input_ch, self.dir_input_ch], dim=-1)
        h = input_pts
        #todo 主干部分的前向传播
        # ipdb.set_trace()
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        #todo 使用观察方向作为输入，就分别输出rgb和α
        alpha = self.alpha_linear(h)
        
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
    
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
        rgb = self.rgb_linear(h)
        
        outputs = torch.cat([rgb, alpha], -1) #* 将分别输出的rgb和α连接后再返回

        return outputs   
        
        
    #todo 真正进行网络推理的地方
    def run_network(self, inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
        """Prepares inputs and applies network 'fn'. 
        #? fn就是network funtion, 暂时不知道具体是什么
        #* workflow: 展平 -> 计算 -> 重塑(逆展平)
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) #* 仅保留最后一维，其余全部展平
        embedded = embed_fn(inputs_flat) #* 从使用的embed函数来看，这个inputs应该是rgb

        if viewdirs is not None:
            input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) #* 仅保留最后一维，其余全部展平
            embedded_dirs = embeddirs_fn(input_dirs_flat) #* 计算方向的embedding
            embedded = torch.cat([embedded, embedded_dirs], -1) #* 按照rgb dir的顺序，在最后一维进行连接：N * (rgb，dir)

        #* bachify看看是否分chunk运行：如果分就返回分批运行的函数，不分就直接返回传入的fn
        outputs_flat = self.batchify(fn, netchunk)(embedded) #* bachify的返回值是一个函数，调用了返回的函数
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]) #* 将形状重塑为进来的形状
        return outputs
        
    
    #todo 只在run_network里被调用，将输入分batch进入网络
    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        if chunk is None:
            return fn
        def ret(inputs):
            #* 在第0个维度上，分chunk把inputs输入网络fn()，然后再把返回值连接
            return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        return ret
        
#=====================================================================================================#

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples