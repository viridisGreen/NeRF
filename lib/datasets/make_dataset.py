from . import samplers
from .dataset_catalog import DatasetCatalog
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
from torch.utils.data import DataLoader, ConcatDataset
import ipdb

torch.multiprocessing.set_sharing_strategy('file_system')

def _dataset_factory(is_train, is_val): # 全程没有用到
    if is_val:
        module = cfg.val_dataset_module
        path = cfg.val_dataset_path
    elif is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    dataset = imp.load_source(module, path).Dataset
    return dataset


def make_dataset(cfg, is_train=True):
    if is_train:
        args = cfg.train_dataset
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
    else:
        args = cfg.test_dataset
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
    # ipdb.set_trace()
    #* 将某个模块的代码载入到当前的运行环境中，使得该模块中定义的函数、类和变量可以在当前环境下使用
    dataset = imp.load_source(module, path).Dataset #* 返回的是一个Dataset类，讲Dataset类导入到当前模块
    dataset = dataset(**args) #* 用类创建了实例，并返回
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset) #* 随机采样，在每个 epoch 随机打乱数据的顺序
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset) #* 顺序采样，按照数据集中的原始顺序加载数据
    return sampler #* 返回的是采样器对象，可用于创建DataLoader，在这里用于创建batch sampler（sampler的一种


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, #* 是否丢弃最后一个不完整的batch
                            max_iter, #* 最大迭代次数，用于限制数据加载的批次数量
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler #* 在configs里面，默认是'default'
        sampler_meta = cfg.train.sampler_meta #? 是一个CN({})，暂时不知道是干什么的
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        #* 如果是default，采用pytorch的标准BatchSampler
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        #* 自定义的BatchSampler
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1: #* 在这里specify了最大迭代次数
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter) #* 这里的返回值回由Sampler类变为BatchSampler类
    return batch_sampler


def worker_init_fn(worker_id):
    """
    * 作用是为每个数据加载器(DataLoader)的工作进程设置一个独特的随机种子。
    * 这在并行加载数据时非常有用，确保每个工作进程生成的随机数序列不同，避免数据加载过程中的潜在随机性重复
    """
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:
        batch_size = cfg.train.batch_size
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset = make_dataset(cfg, is_train) #* 获取当前任务的数据集的实例
    #* sampler只决定是随机采样还是顺序采样，作为batch sampler的参数输入
    sampler = make_data_sampler(dataset, shuffle, is_distributed) #* class：Sampler
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train) #* class：BatchSampler
    num_workers = cfg.train.num_workers #* 把他当作一个超参数就好
    collator = make_collator(cfg, is_train) #* 用于把数据处理成批次的，也当成超参数吧~
    data_loader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            num_workers=num_workers,
                            collate_fn=collator,
                            worker_init_fn=worker_init_fn,
                            pin_memory=True) #* 设为true可以加速cpu到gpu之间的数据传输
    # ipdb.set_trace()
    return data_loader
