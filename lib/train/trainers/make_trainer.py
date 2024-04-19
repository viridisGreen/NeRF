from .trainer import Trainer
import imp


#? NetworkWrapper要的network是什么？
def _wrapper_factory(cfg, network, train_loader=None): #* train_loader是训练数据的DataLoader
    module = cfg.loss_module
    path = cfg.loss_path
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network, train_loader)
    return network_wrapper #* 返回的是NetworkWrapper的类的实例


def make_trainer(cfg, network, train_loader=None):
    network = _wrapper_factory(cfg, network, train_loader) #* train_loader是训练数据的DataLoader
    return Trainer(network)
