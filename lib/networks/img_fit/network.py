import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg
import ipdb

'''输入是一个batch的uv坐标，输出是一个batch的rgb值'''
class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        #* get_encoder会返回encoder和encoder_outdim
        self.uv_encoder, input_ch = get_encoder(net_cfg.uv_encoder) 
        D, W  = net_cfg.D, net_cfg.W

        self.backbone_layer = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D-1)])
        self.output_layer = nn.Sequential(
                nn.Linear(W, 3),
                nn.Sigmoid()
                )

    def render(self, uv, batch): #* 似乎并没有真的用到batch
        uv_encoding = self.uv_encoder(uv)
        x = uv_encoding
        for i, l in enumerate(self.backbone_layer):
            x = self.backbone_layer[i](x)
            x = F.relu(x)
        # ipdb.set_trace()
        rgb = self.output_layer(x)
        return {'rgb': rgb}

    def batchify(self, uv, batch): #* 似乎也没有真的用到batch
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, uv.shape[0], chunk):
            ret = self.render(uv[i:i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret #* 有一个key，对应一个list，里面存储着所有的rgb值

    def forward(self, batch): #* batch是一个字典，里面有uv，rgb，H，W
        # ipdb.set_trace()
        B, N_pixels, C = batch['uv'].shape #* B代表batch，N代表H*W，C代表channel，通常是2
        ret = self.batchify(batch['uv'].reshape(-1, C), batch)
        return {k:ret[k].reshape(B, N_pixels, -1) for k in ret}
