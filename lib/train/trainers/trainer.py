import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from lib.config import cfg
from lib.utils.data_utils import to_cuda


class Trainer(object):
    def __init__(self, network): #* Trainer要的network是一个NetworkWrapper类的实例
        device = torch.device('cuda:{}'.format(cfg.local_rank)) 
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
            network = DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank,
                find_unused_parameters=True
           )
        self.network = network
        self.local_rank = cfg.local_rank #* 进程索引，用于分布式训练，如果有四块gpu，那么他就是0123
        self.device = device
        self.global_step = 0 #* 用于记录迭代的总次数

    def reduce_loss_stats(self, loss_stats): #? 传进来的loss stats是什么类型？存储了什么数据？
        #? loss stats应该是字典，key是loss的名称，value是loss的值（多个
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()} #* 计算每种loss的mean，并重新存储
        return reduced_losses

    def to_cuda(self, batch): #? batch是什么数据类型？数据以怎样的形式存储？
        #* batch的类型是字典，里面的value可能是各种类型
        for k in batch: #* 遍历字典的key
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                #batch[k] = [b.cuda() for b in batch[k]]
                batch[k] = [b.to(self.device) for b in batch[k]]
            elif isinstance(batch[k], dict):
                batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
            else:
                # batch[k] = batch[k].cuda()
                batch[k] = batch[k].to(self.device)
        return batch

    #! 每次train就是一个epoch
    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader) #* 计算批次总数，即这个epoch中将执行的迭代次数
        self.network.train()
        end = time.time() #* 记录当前时间
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end #* 计算从上一个批次处理完到当前批次数据加载完成的时间
            iteration = iteration + 1 #? 可能是为了方便后面取整数？

            batch = to_cuda(batch, self.device)
            batch['step'] = self.global_step #? 在批次数据中添加当前的全局步数信息，可能用于后续操作
            #* network是NetworkWrapper类的实例，封装了底层网络和统计数据
            output, loss, loss_stats, image_stats = self.network(batch) #? network是什么还需要以后再看

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            #* 对梯度值进行裁剪，防止梯度爆炸，这里将梯度限制在-40 ~ 40以内
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40) 
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats) #* loss stats是network的返回值
            recorder.update_loss_stats(loss_stats) #? 应该是recorder所属类的方法

            batch_time = time.time() - end #* 训练一个batch所用的时间
            end = time.time()
            recorder.batch_time.update(batch_time) #? update应该也是recorder所属类的方法
            recorder.data_time.update(data_time)

            self.global_step += 1
            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                #* 计算预计剩余时间Estimated Time of Arrival，迄今为止处理批次的平均时间 * 剩余迭代次数来得到的。
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration) 
                #* 将预计剩余时间转换为 hh:mm:ss格式的字符串
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds))) 
                #* optimizer有多个参数组，0是默认参数组
                lr = optimizer.param_groups[0]['lr'] #* training state的数据
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 #* 转化为MB

                #* 准备一个格式化字符串
                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats) #* image stats是network返回值之一
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {} #* 存储各项损失的求和，默认为0
        image_stats = {}
        data_size = len(data_loader) #* 用于计算损失的平均值
        for batch in tqdm.tqdm(data_loader):
            batch = to_cuda(batch, self.device)
            batch['step'] = recorder.step
            with torch.no_grad():
                output, loss, loss_stats, _ = self.network(batch)
                if evaluator is not None:
                    image_stats_ = evaluator.evaluate(output, batch) #? evaluator还需要再看
                    if image_stats_ is not None:
                        image_stats.update(image_stats_) #* update：更新旧值，增加新值

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size #* 计算某项损失的平均值
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize() #? evaluator的成员函数，再看
            val_loss_stats.update(result) #* update：更新旧值，增加新值

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
