import os
import imp


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    network = imp.load_source(module, path).Network() #* 这个就是我们自己specify的network
    return network
