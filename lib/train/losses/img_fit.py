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
        self.color_crit = nn.MSELoss(reduction='mean') #* 用于计算颜色均方误差的函数
        #* 将mse转换为psnr的函数，
        #? 但是不知道为什么forward里没有用上
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, batch):
        output = self.net(batch)

        scalar_stats = {} #* 统计数据：包含color_mse，psnr，loss
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss

        psnr = -10. * torch.log(color_loss.detach()) / \
                torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
