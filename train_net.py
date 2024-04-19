from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, save_trained_config, load_pretrain
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
import ipdb
torch.autograd.set_detect_anomaly(True)

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    #* make_data_loader函数specify了：
    #*     哪个数据集，shuffle的时候随机采样or顺序采样，
    #*     batch size，drop last，max iter 
    #?     以及worker_init_fn（姑且看作没什么用的超参数）
    #?     和collate_fn（用于处理成批次的，姑且也当作没什么用的超参数）
    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)
    val_loader = make_data_loader(cfg, is_train=False)
    #* Trainer > NetworkWrapper > ?network?
    #* Tranier封装了前向传播和反向传播，train和validation，NW封装了底层网络本身和forward
    trainer = make_trainer(cfg, network, train_loader) #* 返回Trainer类的实例

    optimizer = make_optimizer(cfg, network) #* 返回一个优化器的实例，默认是adam
    scheduler = make_lr_scheduler(cfg, optimizer) #* 根据训练的进度（比如当前是第几个epoch）动态调整学习率
    recorder = make_recorder(cfg) #* 返回一个Recorder类，里面成员有很多记录的数据，方法有很多记录的方法
    evaluator = make_evaluator(cfg)

    #? 主要是看有没有续训，如果是的话会加载续训模型
    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume) #todo 用于训练中断恢复
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain) #todo 用于加载预训练模型

    set_lr_scheduler(cfg, scheduler) #* 就算加载了续训模型，也用我们最新定义的scheduler

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_loader.dataset.epoch = epoch #? 不太懂，在数据集里没看到epoch

        #* 每次train就是一个epoch，把loader里的data全部过一遍
        trainer.train(epoch, train_loader, optimizer, recorder) #* train内部没用上epoch，不用疑惑，就是没用
        scheduler.step()

        #* 每cfg.save_ep个epoch，保存一次模型
        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)

        #* cfg.save_latest_ep默认为1，也就是每个epoch都会存
        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        #* cfg.eval_ep默认也是1，也就是默认每个epoch都要evaluate以下
        if (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            trainer.val(epoch, val_loader, evaluator, recorder) #* 传进去的epoch会在recorder里用

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)
    if cfg.local_rank == 0:
        print('Success!')
        print('='*80)
    os.system('kill -9 {}'.format(os.getpid()))


if __name__ == "__main__":
    # ipdb.set_trace()
    main()
