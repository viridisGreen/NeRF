import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

#? NetworkWrapper要的net(work)是什么？net就是最底层的net
#* 封装了网络部分和统计数据的更新
class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader): #* train_loade 并没有用上，net就是最底层的net
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, batch):
        batch = batch['rays_rgb']
        batch = batch.reshape(-1, 3, 3)
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        
        rgb, disp, acc, extras = self.net(batch)
        output = {'rgb': rgb, 'disp': disp, 'acc': acc, 'extras': extras}

        scalar_stats = {} #* 统计数据：包含color_mse，psnr，loss
        loss = 0
        img_loss = self.img2mse(rgb, target_s)
        scalar_stats.update({'img_mse': img_loss})
        loss += img_loss
        
        #todo 如果有细网络的话，loss再加上粗网络的值
        if 'rgb0' in extras: 
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        psnr = -10. * torch.log(img_loss.detach()) / \
                torch.log(torch.Tensor([10.]).to(img_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
