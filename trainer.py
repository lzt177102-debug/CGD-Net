import os
import time
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

# 自定义工具类/函数（从 src.utils.utils 导入）
from src.utils.utils import (
    makedirs, 
    load_model, 
    convert_time_units, 
    get_max_gpu_memory_usage,
    AverageMeter,
    load_yaml_config
)

# 数据相关
from RSGCN_DataLoader import GraphDataBuilder, GraphDataLoaderManager

# 模型定义
from src.networks.GraphMamba import GraphGDP
from src.networks.GCN import GraphGDP_GCN
from src.networks.GraphGPS import GraphGDP_GraphGPS
from src.networks.Graphormer import GraphGDP_Graphormer

# 损失函数
from src.losses.dual_regression_loss import DualRegressionLoss

# 评估指标
from src.metrics.metrics import (
    r2_score,
    mae_score
)



# ==================== 训练器类（核心封装，支持从YAML加载学习率策略与参数） ====================
class GraphGDPTrainer:
    """GraphGDP 模型训练器（提取提前生成的对数标签 + MAE/R² 指标 + 配置化学习率策略 + 50轮后启用一致性损失）"""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.epoch = 0
        self.best_val_r2 = -float('inf')  # 最优指标改为R²（越大越好）
        self.patience_counter = 0
        # 学习率策略标识（用于后续调用判断）
        self.lr_scheduler_type = self.config.get('lr_scheduler_type', 'ReduceLROnPlateau')
        # 新增：最优模型记录起始轮数阈值（固定为50轮，可配置化扩展）
        self.best_model_start_epoch = 50
        
        # 参考代码风格：生成唯一时间戳（仅用于实验目录命名，文件无时间戳）
        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        
        # 【核心】初始化最新YAML要求的目录结构
        self._init_exp_dir()
        
        # 构建数据相关（修改：仅加载训练集和验证集）
        self.train_dataset, self.val_dataset = self._build_dataset()
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 构建模型相关（含从YAML加载学习率策略与参数）
        self.model, self.criterion, self.optimizer, self.scheduler = self._build_model_components()
        
        # 新增：调用简洁版load_model加载预训练模型（从YAML配置读取参数）
        if self.config['load_pretrained']:
            print(f"\n===== 加载预训练模型（路径：{self.config['pretrained_model_path']}） =====")
            self.model = load_model(
                model=self.model,
                checkpoint_path=self.config['pretrained_model_path'],
                multi_gpu=self.config['multi_gpu']
            )
            # 确保模型移至指定设备（兼容多GPU）
            self.model = self.model.to(self.device)
            print(f"🎉 预训练模型加载完成！")
        
        # 训练日志
        self.train_log = self._load_train_log()
        
        # 新增：显存和时间统计（核心：记录峰值显存、每轮时间、总时间）
        self.total_training_start = 0.0  # 总训练开始时间戳
        self.total_training_time = 0.0  # 总训练时间（秒）
        self.epoch_train_times = []  # 每轮训练时间（秒）
        self.max_gpu_memory_used = 0.0  # 整个训练过程的最大显存占用（MB）
        self.epoch_gpu_memories = []  # 每轮训练的峰值显存占用（MB）

    def _init_exp_dir(self):
        """【精准匹配】初始化目录结构：./result/exp_时间戳/ 内含config/logs/models"""
        # 1. 根目录：./result（最新YAML指定，确保存在）
        self.result_root = makedirs(self.config['output_dir'])
        
        # 2. 核心：创建带时间戳的实验目录 ./result/exp_20251228_123456/
        self.exp_dir = makedirs(os.path.join(self.result_root, f'exp_{self.timestamp}'))
        
        # 3. 实验目录下创建子目录（config/logs/models，无额外时间戳）
        self.config_dir = makedirs(os.path.join(self.exp_dir, 'config'))
        self.log_dir = makedirs(os.path.join(self.exp_dir, 'logs'))
        self.model_dir = makedirs(os.path.join(self.exp_dir, 'models'))
        
        # 4. 定义无时间戳的文件路径（完全匹配要求，与参考代码一致）
        self.train_config_path = os.path.join(self.config_dir, 'train_config.json')
        self.train_log_path = os.path.join(self.log_dir, 'train_log.json')
        self.best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        self.checkpoint_prefix = os.path.join(self.model_dir, 'checkpoint_epoch')
        # 新增：显存/时间统计结果保存路径
        self.resource_log_path = os.path.join(self.log_dir, 'resource_stats.json')
        
        # 5. 归档当前生效配置（无时间戳，符合目录内文件命名要求）
        with open(self.train_config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 实验目录创建完成：{self.exp_dir}")
        print(f"✅ 配置文件归档至：{self.train_config_path}")
        print(f"✅ 模型将保存至：{self.model_dir}")
        print(f"✅ 日志将保存至：{self.log_dir}")
        print(f"✅ 全局池化类型配置：{self.config.get('pool_type', 'add')}")
        print(f"✅ 学习率策略：{self.lr_scheduler_type}（参数从YAML配置加载）")
        print(f"✅ 最优模型记录起始轮数：{self.best_model_start_epoch} 轮（前{self.best_model_start_epoch-1}轮不更新最佳模型）")
        print(f"✅ 一致性损失配置：50轮后启用，权重 {self.config.get('consistency_weight', 1.0)}")

    def _build_dataset(self):
        """构建并划分数据集（修改：仅划分为训练集和验证集，取消测试集）"""
        # 数据集保存路径（YAML未指定，使用默认值，确保目录存在）
        self.dataset_save_path = "./dataset/graph_data_with_lappe_and_node_labels"
        makedirs(self.dataset_save_path)
        
        print(f"\n===== 构建/加载含对数标签的数据集（保存至：{self.dataset_save_path}） =====")
        builder = GraphDataBuilder(
            gdp_file_path=self.config['gdp_file_path'],
            patch_size=self.config['patch_size'],
            lap_pe_k=self.config['lap_pe_k']
        )
        
        dataset = builder.build_graph_dataset(
            features_dir=self.config.get('features_dir', './features'),
            output_dir=self.dataset_save_path,
            stride=self.config['stride'],
            max_counties=self.config['max_counties'],
            random_patches=False,
            min_nodes_threshold=self.config['min_nodes_threshold']
        )
        
        if dataset is None:
            raise ValueError("❌ 数据集构建失败或为空，请检查数据路径与配置")
        
        # 【核心修改1】仅划分训练集和验证集（取消测试集划分）
        unique_counties = list(set(builder.patch_county_mapping))
        unique_counties.sort()  # 默认按字符串字典序排序（县名通常是字符串，效果稳定）
        from sklearn.model_selection import train_test_split
        train_counties, val_counties = train_test_split(
            unique_counties,
            test_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        # 根据县划分数据集
        train_indices = [i for i, county in enumerate(builder.patch_county_mapping) 
                        if county in train_counties]
        val_indices = [i for i, county in enumerate(builder.patch_county_mapping) 
                      if county in val_counties]
        
        # 创建子数据集
        train_dataset = dataset.subset(train_indices)
        val_dataset = dataset.subset(val_indices)
        
        # 打印划分结果
        print(f"📊 按县划分数据集（仅训练/验证）:")
        print(f"  训练县: {len(train_counties)} 个, 图块: {len(train_indices)} 个")
        print(f"  验证县: {len(val_counties)} 个, 图块: {len(val_indices)} 个")
        
        return train_dataset, val_dataset

    def _build_dataloaders(self):
        """构建数据加载器（仅创建训练集和验证集加载器，取消测试集）"""
        print(f"\n===== 创建数据加载器（支持变长图 + 对数双标签） =====")
        loader_manager = GraphDataLoaderManager(
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        # 仅传入训练集和验证集，取消测试集
        data_loaders = loader_manager.create_data_loaders(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=None,
            shuffle_train=True
        )
        
        # 仅返回训练加载器和验证加载器
        return data_loaders['train'], data_loaders['val']

    def _build_model_components(self):
        """构建模型、损失函数、优化器、调度器（从YAML加载学习率策略与参数，新增一致性损失权重）"""
        print(f"\n===== 初始化模型、优化器与配置化学习率调度器 =====")
        
        # -------------------------- 核心修改：动态选择模型 --------------------------
        # 1. 定义模型映射（新增GraphGPS和Graphormer）
        model_mapping = {
            "GraphGDP": GraphGDP,
            "GraphGDP_GCN": GraphGDP_GCN,
            "GraphGDP_GraphGPS": GraphGDP_GraphGPS,  # 新增GraphGPS模型
            "GraphGDP_Graphormer": GraphGDP_Graphormer  # 新增Graphormer模型
        }

        # 2. 获取并校验配置中的模型类型
        model_cls_name = self.config.get("model_cls", "GraphGDP")  # 默认使用GraphGDP
        supported_models = list(model_mapping.keys())
        if model_cls_name not in model_mapping:
            raise ValueError(f"❌ 仅支持 model_cls={'/'.join(supported_models)}，当前配置为：{model_cls_name}")
        model_cls = model_mapping[model_cls_name]
        print(f"📌 初始化模型类型：{model_cls_name}")

        # 3. 准备模型通用参数
        base_model_kwargs = {
            "channels": self.config['channels'],
            "pe_dim": self.config['pe_dim'],
            "num_layers": self.config['num_layers'],
            "use_rw_pe": self.config['use_rw_pe'],
            "use_lap_pe": self.config['use_lap_pe'],
            "use_grid_pe": self.config['use_grid_pe'],
            "node_dim": self.config['node_dim'],
            "edge_dim": self.config['edge_dim'],
            "if_pool": self.config['if_pool'],
            "pool_type": self.config.get('pool_type', 'add'),
            "drop": self.config['drop'],
            "node_label_dim": self.config['node_label_dim'],
            "global_label_dim": self.config['global_label_dim']
        }

        # 4. 为不同模型补充专属参数
        model_kwargs = base_model_kwargs.copy()

        if model_cls_name == "GraphGDP":
            # GraphGDP（Mamba版）专属参数
            model_kwargs.update({
                "model_type": self.config['model_type'],
                "shuffle_ind": self.config['shuffle_ind'],
                "d_state": self.config['d_state'],
                "d_conv": self.config['d_conv'],
                "order_by_degree": self.config['order_by_degree']
            })
        elif model_cls_name == "GraphGDP_GraphGPS":
            # GraphGPS（GINE+Performer）专属参数
            model_kwargs.update({
                "performer_heads": self.config.get('performer_heads', 4),  # 默认4头
                "performer_dim_head": self.config.get('performer_dim_head', 32),  # 默认dim_head=32
                "performer_depth": self.config.get('performer_depth', 1)  # Performer必填的depth参数
            })
        elif model_cls_name == "GraphGDP_Graphormer":
            # Graphormer专属参数
            model_kwargs.update({
                "graphormer_heads": self.config.get('graphormer_heads', 8)  # 默认8头注意力
            })
        # GraphGDP_GCN无专属参数，无需额外补充

        # 5. 初始化模型
        model = model_cls(**model_kwargs).to(self.device)
        
        # 初始化损失函数（新增：传入一致性损失权重）
        criterion = DualRegressionLoss(
            node_weight=self.config['node_loss_weight'],
            global_weight=self.config['global_loss_weight'],
            consistency_weight=self.config.get('consistency_weight', 1.0)  # 新增：一致性损失权重，默认1.0
        )
        
        # 初始化优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 【核心修改】从YAML加载学习率策略与参数，支持动态切换
        scheduler = None
        if self.lr_scheduler_type == 'ReduceLROnPlateau':
            # 加载ReduceLROnPlateau专属参数（从YAML读取，带默认值）
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.config.get('lr_scheduler_mode', 'max'),
                factor=self.config.get('lr_scheduler_factor', 0.7),
                patience=self.config.get('lr_scheduler_patience', 10),
                min_lr=self.config.get('lr_scheduler_min_lr', 1e-6)
            )
            print(f"📌 已初始化 ReduceLROnPlateau 调度器，参数：")
            print(f"   - mode: {self.config.get('lr_scheduler_mode', 'max')}")
            print(f"   - factor: {self.config.get('lr_scheduler_factor', 0.7)}")
            print(f"   - patience: {self.config.get('lr_scheduler_patience', 10)}")
            print(f"   - min_lr: {self.config.get('lr_scheduler_min_lr', 1e-6)}")
        elif self.lr_scheduler_type == 'StepLR':
            # 加载StepLR专属参数（从YAML读取，带默认值）
            scheduler = StepLR(
                optimizer,
                step_size=self.config.get('lr_scheduler_step_size', 30),
                gamma=self.config.get('lr_scheduler_gamma', 0.5),
                verbose=self.config.get('lr_scheduler_verbose', True)
            )
            print(f"📌 已初始化 StepLR 调度器，参数：")
            print(f"   - step_size: {self.config.get('lr_scheduler_step_size', 30)}")
            print(f"   - gamma: {self.config.get('lr_scheduler_gamma', 0.5)}")
        else:
            raise ValueError(f"❌ 不支持的学习率策略：{self.lr_scheduler_type}，可选：['ReduceLROnPlateau', 'StepLR']")
        
        # 打印预训练配置摘要
        print(f"\n📌 预训练模型配置摘要：")
        print(f"   - 是否加载预训练：{self.config['load_pretrained']}")
        if self.config['load_pretrained']:
            print(f"   - 预训练模型路径：{self.config['pretrained_model_path']}")
            print(f"   - 多GPU加载：{self.config['multi_gpu']}")
        
        # 打印一致性损失配置
        print(f"\n📌 一致性损失配置摘要：")
        print(f"   - 节点损失权重：{self.config['node_loss_weight']}")
        print(f"   - 全局损失权重：{self.config['global_loss_weight']}")
        print(f"   - 一致性损失权重：{self.config.get('consistency_weight', 1.0)}")
        print(f"   - 启用时机：第50轮及以后（前49轮不计算一致性损失）")
        
        return model, criterion, optimizer, scheduler

    def _load_train_log(self):
        """加载已有训练日志（若存在）"""
        if os.path.exists(self.train_log_path):
            with open(self.train_log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_train_log(self):
        """保存训练日志（无时间戳，符合目录内文件命名要求）"""
        with open(self.train_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_log, f, indent=4, ensure_ascii=False)

    def _save_resource_stats(self):
        """保存显存/时间统计结果（训练结束后调用）"""
        # 转换总时间单位
        total_time_units = convert_time_units(self.total_training_time)
        # 转换平均每轮时间单位
        avg_epoch_seconds = np.mean(self.epoch_train_times) if self.epoch_train_times else 0.0
        avg_epoch_time_units = convert_time_units(avg_epoch_seconds)
        
        resource_stats = {
            'device': str(self.device),
            'total_training_time': total_time_units,
            'avg_epoch_training_time': avg_epoch_time_units,
            'epoch_train_times_seconds': self.epoch_train_times,
            'max_gpu_memory_used_mb': self.max_gpu_memory_used,
            'epoch_gpu_memories_mb': self.epoch_gpu_memories,
            'total_epochs_completed': len(self.epoch_train_times),
            'best_val_r2': self.best_val_r2,
            'best_model_start_epoch': self.best_model_start_epoch,
            'timestamp': self.timestamp,
            'consistency_weight': self.config.get('consistency_weight', 1.0),
            'consistency_enable_epoch': 50  # 新增：记录一致性损失启用轮数
        }
        
        with open(self.resource_log_path, 'w', encoding='utf-8') as f:
            json.dump(resource_stats, f, indent=4, ensure_ascii=False)

    def _save_checkpoint(self, is_best=False):
        """保存模型检查点（无时间戳，匹配参考代码风格）"""
        checkpoint_dict = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_r2': self.best_val_r2,
            'best_model_start_epoch': self.best_model_start_epoch,
            'timestamp': self.timestamp,
            'config': self.config,
            'max_gpu_memory_used_mb': self.max_gpu_memory_used,
            'total_training_time_seconds': self.total_training_time,
            'consistency_enable_epoch': 50  # 新增：记录一致性损失启用轮数
        }
        
        if is_best:
            torch.save(checkpoint_dict, self.best_model_path)
            print(f"✅ 最佳模型已保存至：{self.best_model_path}（基于验证集R²最优，≥{self.best_model_start_epoch}轮）")
        else:
            checkpoint_path = f'{self.checkpoint_prefix}_{self.epoch}.pth'
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"✅ 普通检查点已保存至：{checkpoint_path}")

    def _process_batch(self, batch):
        """处理批次数据（核心：正确提取数据代码中已存储的对数标签，不做多余计算，返回batch_idx）"""
        # 把batch对象移至设备
        batch = batch.to(self.device)
        
        # 提取基础数据
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch  # 保留batch_idx，用于按图块聚合节点
        
        # 处理edge_attr占位符（兼容有/无边属性）
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None and batch.edge_attr.nelement() > 0:
            edge_attr = batch.edge_attr
        else:
            edge_dim = self.config.get('edge_dim', 1)
            edge_attr = torch.zeros((edge_index.shape[1], edge_dim), device=self.device)
        
        # 处理LapPE编码（兼容无LapPE的情况）
        if hasattr(batch, 'lap_pe') and batch.lap_pe is not None:
            lap_pe = batch.lap_pe
        else:
            lap_pe = torch.zeros((batch.num_nodes, self.config['pe_dim']), device=self.device)
        
        # 提取节点级对数标签
        if hasattr(batch, 'y_node') and batch.y_node is not None and batch.y_node.dim() == 2:
            node_target = batch.y_node[:, 1:2]
        else:
            raise ValueError("❌ 未找到节点级标签（y_node），或标签维度不正确，请检查数据构建代码")
        
        # 提取图块级对数标签
        if hasattr(batch, 'y') and batch.y is not None and batch.y.dim() == 2:
            global_target = batch.y[:, 1:2]
        else:
            raise ValueError("❌ 未找到图块级标签（y），或标签维度不正确，请检查数据构建代码")
        
        # 新增：返回batch_idx
        return x, edge_index, edge_attr, lap_pe, batch_idx, node_target, global_target

    def train_one_epoch(self):
        """训练单个Epoch（50轮后启用一致性损失，根据学习率策略判断是否调用scheduler.step()）"""
        self.model.train()
        meters = {
            'total_loss': AverageMeter(),
            'node_loss': AverageMeter(),
            'global_loss': AverageMeter(),
            'consistency_loss': AverageMeter(),  # 新增：一致性损失统计
            'node_r2': AverageMeter(),
            'global_r2': AverageMeter(),
            'node_mae': AverageMeter(),
            'global_mae': AverageMeter()
        }
        
        # 核心逻辑：50轮后启用一致性损失
        enable_consistency = self.epoch >= 50
        consistency_status = "Enabled" if enable_consistency else "Disabled"
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.epoch}/{self.config["epochs"]} (Consist: {consistency_status})')
        for batch in pbar:
            # 处理批次数据（获取batch_idx）
            x, edge_index, edge_attr, lap_pe, batch_idx, node_target, global_target = self._process_batch(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            node_pred, global_pred = self.model(
                x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx, lap_pe=lap_pe
            )
            # 新增：传入enable_consistency，控制是否计算一致性损失
            total_loss, node_loss, global_loss, consistency_loss = self.criterion(
                node_pred, node_target, global_pred, global_target, batch_idx, enable_consistency
            )
            
            # 计算评价指标
            node_r2 = r2_score(node_pred, node_target).item()
            global_r2 = r2_score(global_pred, global_target).item()
            node_mae = mae_score(node_pred, node_target).item()
            global_mae = mae_score(global_pred, global_target).item()
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            
            # 更新指标（新增：一致性损失）
            batch_size = batch.num_graphs
            meters['total_loss'].update(total_loss.item(), batch_size)
            meters['node_loss'].update(node_loss.item(), batch_size)
            meters['global_loss'].update(global_loss.item(), batch_size)
            meters['consistency_loss'].update(consistency_loss.item(), batch_size)  # 新增
            meters['node_r2'].update(node_r2, batch_size)
            meters['global_r2'].update(global_r2, batch_size)
            meters['node_mae'].update(node_mae, batch_size)
            meters['global_mae'].update(global_mae, batch_size)
            
            # 更新进度条（新增：一致性损失展示）
            pbar.set_postfix({
                'Total Loss': f'{meters["total_loss"].val:.6f}',
                'Node Loss': f'{meters["node_loss"].val:.6f}',
                'Global Loss': f'{meters["global_loss"].val:.6f}',
                'Consist Loss': f'{meters["consistency_loss"].val:.6f}',  # 新增
                'Node R²': f'{meters["node_r2"].val:.6f}',
                'Global R²': f'{meters["global_r2"].val:.6f}',
                'Node MAE': f'{meters["node_mae"].val:.6f}',
                'Global MAE': f'{meters["global_mae"].val:.6f}'
            })
        
        # 【核心判断】仅StepLR在训练后调用scheduler.step()，ReduceLROnPlateau在验证后调用
        if self.lr_scheduler_type == 'StepLR' and self.scheduler:
            self.scheduler.step()
        
        # 打印训练结果（新增：一致性损失 + 启用状态）
        print(f"\nTrain Epoch [{self.epoch}/{self.config['epochs']}] (Consist: {consistency_status})")
        print(f"  总损失: {meters['total_loss'].avg:.6f} | 节点损失: {meters['node_loss'].avg:.6f} | 全局损失: {meters['global_loss'].avg:.6f} | 一致性损失: {meters['consistency_loss'].avg:.6f}")
        print(f"  节点 R²: {meters['node_r2'].avg:.6f} | 全局 R²: {meters['global_r2'].avg:.6f}")
        print(f"  节点 MAE: {meters['node_mae'].avg:.6f} | 全局 MAE: {meters['global_mae'].avg:.6f}")
        
        # 计算平均R²
        avg_r2 = (meters['node_r2'].avg + meters['global_r2'].avg) / 2
        
        return {
            'loss': meters['total_loss'].avg,
            'node_loss': meters['node_loss'].avg,
            'global_loss': meters['global_loss'].avg,
            'consistency_loss': meters['consistency_loss'].avg,  # 新增
            'node_r2': meters['node_r2'].avg,
            'global_r2': meters['global_r2'].avg,
            'avg_r2': avg_r2,
            'node_mae': meters['node_mae'].avg,
            'global_mae': meters['global_mae'].avg,
            'consistency_enabled': enable_consistency  # 新增：记录启用状态
        }

    def evaluate(self, mode='val'):
        """评估模型（仅支持验证集评估，50轮后启用一致性损失，返回平均R²用于学习率调度）"""
        self.model.eval()
        meters = {
            'total_loss': AverageMeter(),
            'node_loss': AverageMeter(),
            'global_loss': AverageMeter(),
            'consistency_loss': AverageMeter(),  # 新增：一致性损失统计
            'node_r2': AverageMeter(),
            'global_r2': AverageMeter(),
            'node_mae': AverageMeter(),
            'global_mae': AverageMeter()
        }
        
        # 核心逻辑：50轮后启用一致性损失（与训练保持一致）
        enable_consistency = self.epoch >= 50
        consistency_status = "Enabled" if enable_consistency else "Disabled"
        
        # 仅支持验证集
        data_loader = self.val_loader
        pbar = tqdm(data_loader, desc=f'{mode.capitalize()} Epoch {self.epoch} (Consist: {consistency_status})')
        
        with torch.no_grad():
            for batch in pbar:
                # 处理批次数据（获取batch_idx）
                x, edge_index, edge_attr, lap_pe, batch_idx, node_target, global_target = self._process_batch(batch)
                
                # 前向传播
                node_pred, global_pred = self.model(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx, lap_pe=lap_pe
                )
                # 新增：传入enable_consistency，控制是否计算一致性损失
                total_loss, node_loss, global_loss, consistency_loss = self.criterion(
                    node_pred, node_target, global_pred, global_target, batch_idx, enable_consistency
                )
                
                # 计算评价指标
                node_r2 = r2_score(node_pred, node_target).item()
                global_r2 = r2_score(global_pred, global_target).item()
                node_mae = mae_score(node_pred, node_target).item()
                global_mae = mae_score(global_pred, global_target).item()
                
                # 更新指标（新增：一致性损失）
                batch_size = batch.num_graphs
                meters['total_loss'].update(total_loss.item(), batch_size)
                meters['node_loss'].update(node_loss.item(), batch_size)
                meters['global_loss'].update(global_loss.item(), batch_size)
                meters['consistency_loss'].update(consistency_loss.item(), batch_size)  # 新增
                meters['node_r2'].update(node_r2, batch_size)
                meters['global_r2'].update(global_r2, batch_size)
                meters['node_mae'].update(node_mae, batch_size)
                meters['global_mae'].update(global_mae, batch_size)
                
                # 更新进度条（新增：一致性损失展示）
                pbar.set_postfix({
                    'Total Loss': f'{meters["total_loss"].val:.6f}',
                    'Node Loss': f'{meters["node_loss"].val:.6f}',
                    'Global Loss': f'{meters["global_loss"].val:.6f}',
                    'Consist Loss': f'{meters["consistency_loss"].val:.6f}',  # 新增
                    'Node R²': f'{meters["node_r2"].val:.6f}',
                    'Global R²': f'{meters["global_r2"].val:.6f}',
                    'Node MAE': f'{meters["node_mae"].val:.6f}',
                    'Global MAE': f'{meters["global_mae"].val:.6f}'
                })
        
        # 打印评估结果（新增：一致性损失 + 启用状态）
        print(f"\n{mode.capitalize()} Epoch [{self.epoch}/{self.config['epochs']}] (Consist: {consistency_status})")
        print(f"  总损失: {meters['total_loss'].avg:.6f} | 节点损失: {meters['node_loss'].avg:.6f} | 全局损失: {meters['global_loss'].avg:.6f} | 一致性损失: {meters['consistency_loss'].avg:.6f}")
        print(f"  节点 R²: {meters['node_r2'].avg:.6f} | 全局 R²: {meters['global_r2'].avg:.6f}")
        print(f"  节点 MAE: {meters['node_mae'].avg:.6f} | 全局 MAE: {meters['global_mae'].avg:.6f}")
        
        # 计算平均R²
        avg_r2 = (meters['node_r2'].avg + meters['global_r2'].avg) / 2
        return {
            'loss': meters['total_loss'].avg,
            'node_loss': meters['node_loss'].avg,
            'global_loss': meters['global_loss'].avg,
            'consistency_loss': meters['consistency_loss'].avg,  # 新增
            'node_r2': meters['node_r2'].avg,
            'global_r2': meters['global_r2'].avg,
            'avg_r2': avg_r2,
            'node_mae': meters['node_mae'].avg,
            'global_mae': meters['global_mae'].avg,
            'consistency_enabled': enable_consistency  # 新增：记录启用状态
        }

    def _print_resource_summary(self):
        """训练结束后，打印显存/时间汇总（补充完整最优指标，含一致性损失）"""
        print("\n" + "="*80)
        print("📊 训练资源消耗汇总（仅最后显示）")
        print("="*80)
        
        # 设备信息
        print(f"\n1. 训练设备: {self.device}")
        
        # 显存信息
        if self.device.type == 'cuda':
            print(f"\n2. 显存占用统计（单位：MB）")
            print(f"   - 整个训练过程峰值显存: {self.max_gpu_memory_used} MB")
            print(f"   - 每轮峰值显存范围: {np.min(self.epoch_gpu_memories):.2f} ~ {np.max(self.epoch_gpu_memories):.2f} MB")
            print(f"   - 平均每轮峰值显存: {np.mean(self.epoch_gpu_memories):.2f} MB")
        else:
            print(f"\n2. 显存占用统计: 未使用GPU，无显存数据")
        
        # 时间信息
        total_time_units = convert_time_units(self.total_training_time)
        avg_epoch_seconds = np.mean(self.epoch_train_times) if self.epoch_train_times else 0.0
        avg_epoch_time_units = convert_time_units(avg_epoch_seconds)
        
        print(f"\n3. 训练时间统计")
        print(f"   - 总训练时间: {total_time_units['seconds']} 秒 = {total_time_units['minutes']} 分钟 = {total_time_units['hours']} 小时 = {total_time_units['gpu_days']} GPU天")
        print(f"   - 完成轮数: {len(self.epoch_train_times)} / {self.config['epochs']}")
        print(f"   - 平均每轮时间: {avg_epoch_seconds:.2f} 秒 = {avg_epoch_time_units['minutes']:.2f} 分钟")
        
        # 4. 训练效果最优指标（核心：补充完整节点+全局 MAE/MSE/R² + 一致性损失）
        print(f"\n4. 训练效果最优指标（对应最佳模型，验证集，≥{self.best_model_start_epoch}轮）")
        # 反向查找最优模型对应的完整日志条目（解决浮点精度问题）
        best_log_entry = None
        if self.train_log:
            for log_entry in self.train_log:
                current_val_avg_r2 = log_entry['val']['avg_r2']
                if abs(current_val_avg_r2 - self.best_val_r2) < 1e-8:  # 浮点误差兼容
                    best_log_entry = log_entry
                    break
        
        if best_log_entry:
            # 提取完整最优指标（节点级+全局级，MAE/MSE/R² + 一致性损失）
            val_metrics = best_log_entry['val']
            node_mae = val_metrics['node_mae']
            node_mse = val_metrics['node_loss']  # node_loss即为MSE损失（模型优化目标）
            node_r2 = val_metrics['node_r2']
            global_mae = val_metrics['global_mae']
            global_mse = val_metrics['global_loss']  # global_loss即为MSE损失（模型优化目标）
            global_r2 = val_metrics['global_r2']
            consistency_loss = val_metrics['consistency_loss']  # 新增：一致性损失
            consistency_enabled = val_metrics['consistency_enabled']  # 新增：启用状态
            avg_r2 = val_metrics['avg_r2']
            
            # 格式化输出，层次清晰
            print(f"   - 节点级指标:")
            print(f"     · 节点MAE（平均绝对误差）: {node_mae:.6f}")
            print(f"     · 节点MSE（均方误差）: {node_mse:.6f}")
            print(f"     · 节点R²（决定系数）: {node_r2:.6f}")
            print(f"   - 全局级指标:")
            print(f"     · 全局MAE（平均绝对误差）: {global_mae:.6f}")
            print(f"     · 全局MSE（均方误差）: {global_mse:.6f}")
            print(f"     · 全局R²（决定系数）: {global_r2:.6f}")
            print(f"   - 一致性指标:")
            print(f"     · 节点求和-全局预测MSE（一致性损失）: {consistency_loss:.6f}")
            print(f"     · 一致性损失启用状态: {'Yes' if consistency_enabled else 'No'}")
            print(f"   - 综合指标:")
            print(f"     · 验证集平均R²（节点R²+全局R²取平均）: {avg_r2:.6f}")
        else:
            # 兜底：无日志时仅显示最佳平均R²
            print(f"   - 验证集平均R²（节点R²+全局R²取平均）: {self.best_val_r2:.6f}")
            print(f"   - 提示：未找到完整指标日志，或未达到{self.best_model_start_epoch}轮最优模型记录阈值")
        
        # 保存路径
        print(f"\n5. 统计结果保存路径")
        print(f"   - 显存/时间统计: {self.resource_log_path}")
        print(f"   - 最佳模型: {self.best_model_path}")
        print("\n" + "="*80)

    def run(self):
        """运行完整训练流程（根据学习率策略动态调用scheduler，记录显存/时间，50轮后启用一致性损失）"""
        print(f"\n===== 开始训练循环（共 {self.config['epochs']} 个 Epoch） =====")
        print(f"===== 训练结果将归档至：{self.exp_dir} =====")
        print(f"===== 全局池化类型：{self.config.get('pool_type', 'add')} =====")
        print(f"===== 最优模型判断依据：验证集平均R²（节点R²+全局R²）最大化（≥{self.best_model_start_epoch}轮生效） =====")
        print(f"===== 一致性损失配置：50轮后启用，权重 {self.config.get('consistency_weight', 1.0)} =====")
        
        # 记录总训练开始时间
        self.total_training_start = time.time()
        
        for self.epoch in range(1, self.config['epochs'] + 1):
            # 记录单轮开始时间
            epoch_start = time.time()
            
            # 训练单轮
            train_metrics = self.train_one_epoch()
            
            # 验证单轮
            val_metrics = self.evaluate(mode='val')
            
            # 【核心判断】仅ReduceLROnPlateau在验证后调用（基于验证集平均R²）
            if self.lr_scheduler_type == 'ReduceLROnPlateau' and self.scheduler:
                self.scheduler.step(val_metrics['avg_r2'])
            
            # 记录单轮结束时间和峰值显存
            epoch_end = time.time()
            epoch_elapsed = epoch_end - epoch_start
            epoch_max_gpu_mem = get_max_gpu_memory_usage(self.device)
            self.epoch_train_times.append(epoch_elapsed)
            self.epoch_gpu_memories.append(epoch_max_gpu_mem)
            if epoch_max_gpu_mem > self.max_gpu_memory_used:
                self.max_gpu_memory_used = epoch_max_gpu_mem
            
            # 记录日志（新增：一致性损失 + 启用状态）
            log_entry = {
                'epoch': self.epoch,
                'time': epoch_elapsed,
                'lr': self.optimizer.param_groups[0]['lr'],
                'epoch_max_gpu_mem_mb': epoch_max_gpu_mem,
                'train': train_metrics,
                'val': val_metrics
            }
            self.train_log.append(log_entry)
            self._save_train_log()
            
            # 【核心修改】50轮之后才开始判断并更新最优模型
            current_val_avg_r2 = val_metrics['avg_r2']
            if self.epoch >= self.best_model_start_epoch:
                # 达到阈值，正常判断最优模型
                if current_val_avg_r2 > self.best_val_r2:
                    self.best_val_r2 = current_val_avg_r2
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                    print(f"🎉 验证集平均R²提升至：{self.best_val_r2:.6f}（更新最佳模型，≥{self.best_model_start_epoch}轮）")
                else:
                    self.patience_counter += 1
                    print(f"⚠️  验证集R²未提升，耐心值：{self.patience_counter}/{self.config['patience']}（当前最优：{self.best_val_r2:.6f}，≥{self.best_model_start_epoch}轮）")
                
                # 早停判断（仅在达到最优模型记录阈值后生效）
                if self.patience_counter >= self.config['patience']:
                    print(f"\n===== 早停触发（耐心值耗尽，≥{self.best_model_start_epoch}轮）=====")
                    print(f"最优验证集平均R²：{self.best_val_r2:.6f}")
                    break
            else:
                # 未达到50轮，不更新最优模型，不触发早停
                self.patience_counter = 0  # 重置耐心值，避免累积
                print(f"ℹ️  当前轮数 {self.epoch} < {self.best_model_start_epoch} 轮，暂不记录最优模型，不触发早停")
        
        # 计算总训练时间
        self.total_training_time = time.time() - self.total_training_start
        
        # 保存资源统计 + 打印汇总信息（仅最后显示）
        self._save_resource_stats()
        self._print_resource_summary()
        
        # 训练结束提示
        print(f"\n===== 训练流程全部结束 =====")
        if self.best_val_r2 != -float('inf'):
            print(f"✅ 最优验证集平均R²：{self.best_val_r2:.6f}（≥{self.best_model_start_epoch}轮）")
        else:
            print(f"✅ 训练完成，但未达到{self.best_model_start_epoch}轮，无最优模型记录")
        print(f"✅ 本次训练使用全局池化类型：{self.config.get('pool_type', 'add')}")
        print(f"✅ 本次训练一致性损失：50轮后启用，权重 {self.config.get('consistency_weight', 1.0)}")
        print(f"✅ 资源消耗统计已保存至：{self.log_dir}")


# ==================== 主函数（简洁封装，加载配置并运行） ====================
def main(args):
    """主函数（加载YAML配置，初始化训练环境并运行）"""
    print("=" * 60)
    print("初始化 GraphGDP 训练环境...")
    print("=" * 60)
    
    # 加载YAML配置（含学习率策略与预训练参数 + 一致性损失权重）
    config = load_yaml_config(args.config_path)
    
    # 初始化设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    print(f"加载配置文件：{args.config_path}")
    print(f"输出目录（YAML指定）：{config['output_dir']}")
    print(f"验证集比例：{config['val_size']}")
    print(f"全局池化类型：{config['pool_type']}")
    print(f"学习率策略：{config['lr_scheduler_type']}")
    print(f"一致性损失配置：50轮后启用，权重 {config['consistency_weight']}")
    
    # 初始化训练器并运行
    trainer = GraphGDPTrainer(config, device)
    trainer.run()

# ==================== 命令行参数解析 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphGDP 双回归模型训练（配置化学习率策略 + MAE/R² 指标 + 显存/时间统计 + 50轮后启用一致性损失）')
    parser.add_argument('--config_path', type=str, default='config/GraphGDP_config.yaml',
                        help='YAML配置文件路径（默认：config/GraphGDP_config.yaml）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 运行主函数
    main(args)
