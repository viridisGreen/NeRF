from collections import deque, defaultdict
import torch
from tensorboardX import SummaryWriter
import os
from lib.config.config import cfg

from termcolor import colored


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    设计用来跟踪一系列值，并提供对这些值的平滑处理能力
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size) #* 双端队列
        self.total = 0.0
        self.count = 0

    def update(self, value):
        #* 向双端队列中添加一个新值，并更新total和count。如果队列已满(即达到window_size)，则最旧的值会被自动移除
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        #* 计算并返回队列中所有值的中位数
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        #* 计算队列中所有值的平均值
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        #* 计算并返回从开始到现在所有值的全局平均值
        return self.total / self.count


def process_volsdf(image_stats):
    for k, v in image_stats.items():
        image_stats[k] = torch.clamp(v[0].permute(2, 0, 1), min=0., max=1.)
    return image_stats

process_neus = process_volsdf #* 只在这个地方出现过，暂时不知道干什么的，好像没什么用，先不管了

class Recorder(object):
    def __init__(self, cfg):
        if cfg.local_rank > 0: #* 分布式训练，该recorder不在主进程上，不执行任何操作
            return

        log_dir = cfg.record_dir
        if not cfg.resume: #* 如果不是恢复训练（cfg.resume为False），说明是一次全新的训练，则会清空日志目录
            print(colored('remove contents of directory %s' % log_dir, 'red'))
            os.system('rm -r %s' % log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

        # scalars
        self.epoch = 0
        self.step = 0
        #* defaultdict的主要特点是在访问不存在的键时，
        #* 会自动创建这个键并将其值设为由提供的默认工厂函数返回的值（这里是SmoothValue），
        #* 而不是像普通字典那样抛出一个KeyError
        self.loss_stats = defaultdict(SmoothedValue) #* 用于存储不同种类的损失值
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()

        # images
        self.image_stats = defaultdict(object)
        #* globals()会返回全局符号表字典，包含了当前模块所有可以访问的变量、函数、类等
        if 'process_' + cfg.task in globals(): 
            self.processor = globals()['process_' + cfg.task]
        else:
            self.processor = None

    def update_loss_stats(self, loss_dict):
        if cfg.local_rank > 0:
            return
        for k, v in loss_dict.items():
            #* detach：返回新的tensor，requires_grad标志被设置为False，用于提高内存效率
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        if cfg.local_rank > 0:
            return
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        if cfg.local_rank > 0:
            return

        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        if cfg.local_rank > 0:
            return
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        if cfg.local_rank > 0:
            return
        self.step = scalar_dict['step']

    def __str__(self):
        if cfg.local_rank > 0:
            return
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(cfg):
    return Recorder(cfg)
