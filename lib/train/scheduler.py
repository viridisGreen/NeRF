from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR


def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        #* 当训练到指定的epoch（milestones）时，学习率会乘上一个因子（gamma），从而实现分段常数减少的学习率
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        #* 每过decay_epochs个epoch，学习率乘上一个因子（gamma），实现指数级减少的学习率
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    '''根据配置cfg来调整已存在的学习率调度器scheduler的参数'''
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        #* Counter：字典的子类，元素被存储为字典的键，而元素的计数则存储为字典的值
        scheduler.milestones = Counter(cfg_scheduler.milestones) #* 如果是multistep就更新milesstones
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs #* 如果算exponential就更新decay epochs
    scheduler.gamma = cfg_scheduler.gamma #* gamma都有，所以一定更新
