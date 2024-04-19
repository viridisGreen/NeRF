import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam, #* 自定义的优化器
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net):
    '''
    #? net是哪个类的实例? 
    net: 待训练的神经网络模型，预期实现了.named_parameters()方法，以遍历模型的参数
    '''
    
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    for key, value in net.named_parameters(): #* 注意遍历的是网络的参数
        if not value.requires_grad:
            continue #* 如果是不需要更新的参数，直接跳过就行了
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "eps": eps}]

    if 'adam' in cfg.train.optim: #* adam再config里被作为默认优化器
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
