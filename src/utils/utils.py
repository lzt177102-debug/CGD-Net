# -*- coding: utf-8 -*-
import os
import json
import yaml
import time
import torch
import random
import numpy as np

# =====================================================
# Config loader (FULL, UNCHANGED LOGIC)
# =====================================================
def load_yaml_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ========= bool =========
    bool_keys = [
        'order_by_degree', 'use_lap_pe', 'use_rw_pe', 'use_grid_pe',
        'if_pool', 'load_pretrained', 'multi_gpu'
    ]
    for k in bool_keys:
        if k in config:
            config[k] = bool(config[k])

    # ========= float =========
    float_keys = [
        'learning_rate', 'weight_decay', 'val_size',
        'node_loss_weight', 'global_loss_weight',
        'consistency_weight',
        'lr_scheduler_factor', 'lr_scheduler_min_lr', 'lr_scheduler_gamma'
    ]
    for k in float_keys:
        if k in config:
            config[k] = float(config[k])

    # ========= int =========
    int_keys = [
        'patch_size', 'lap_pe_k', 'stride', 'min_nodes_threshold',
        'channels', 'pe_dim', 'num_layers', 'shuffle_ind',
        'd_state', 'd_conv', 'node_dim', 'edge_dim',
        'node_label_dim', 'global_label_dim',
        'batch_size', 'num_workers',
        'epochs', 'patience', 'random_state',
        'lr_scheduler_patience', 'lr_scheduler_step_size'
    ]
    for k in int_keys:
        if k in config:
            config[k] = int(config[k])

    # ========= defaults =========
    defaults = {
        'val_size': 0.2,
        'pool_type': 'add',
        'lr_scheduler_type': 'ReduceLROnPlateau',
        'lr_scheduler_mode': 'max',
        'lr_scheduler_factor': 0.7,
        'lr_scheduler_patience': 10,
        'lr_scheduler_min_lr': 1e-6,
        'lr_scheduler_step_size': 30,
        'lr_scheduler_gamma': 0.5,
        'load_pretrained': False,
        'multi_gpu': False,
        'consistency_weight': 1.0,
    }
    for k, v in defaults.items():
        config.setdefault(k, v)

    return config


# =====================================================
# Random seed
# =====================================================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================
# Average meter
# =====================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


# =====================================================
# Directory & IO
# =====================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# GPU & time utils
# =====================================================
def get_max_gpu_memory_usage(device):
    if device.type != 'cuda':
        return 0.0
    gpu_id = device.index if device.index is not None else 0
    mem = torch.cuda.max_memory_allocated(gpu_id) / 1024 ** 2
    torch.cuda.reset_max_memory_allocated(gpu_id)
    return round(mem, 2)


def convert_time_units(seconds):
    minutes = seconds / 60
    hours = minutes / 60
    return {
        'seconds': round(seconds, 2),
        'minutes': round(minutes, 2),
        'hours': round(hours, 4),
        'gpu_days': round(hours / 24, 6),
    }


def load_model(model, checkpoint_path, multi_gpu=False):
    """
    通用加载模型函数（轻量简洁，兼容多卡/单卡权重）。

    :param model: 要加载状态字典的PyTorch模型。
    :param checkpoint_path: 模型权重文件的路径。
    :param multi_gpu: 布尔值，指示是否使用多GPU加载模型。
    :return: 加载了权重的模型。
    """
    # 加载状态字典
    pretrain = torch.load(checkpoint_path)
    if 'model_state_dict' in pretrain.keys():
        state_dict = pretrain['model_state_dict']
    else:
        # 兼容直接保存state_dict的情况，避免key不存在报错
        state_dict = pretrain.get('state_dict', pretrain)
    
    # 检查是否为多卡模型保存的状态字典，移除'module.'前缀（多卡到单卡）
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    
    # 逐参数匹配加载，不匹配则跳过
    for name, param in model.named_parameters():
        if name in state_dict and param.size() == state_dict[name].size():
            param.data.copy_(state_dict[name])
            # 如需打印加载成功的层，取消注释下方一行
            # print(f"Loaded layer: {name}")
        else:
            print(f"Skipped layer: {name}")
    
    # 如果需要在多GPU上运行模型
    if multi_gpu:
        model = nn.DataParallel(model)

    return model

def makedirs(path):
    """创建目录（与分类代码风格一致）"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

