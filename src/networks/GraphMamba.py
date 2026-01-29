
# -*- coding: utf-8 -*-
import sys
import os

# 1. 获取当前脚本（GraphMamba.py）的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 向上追溯找到项目根目录（e:\ruralincome\，即 src 目录的父目录）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
# 3. 将项目根目录加入 sys.path（让 Python 能找到 cross_atten）
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from inspect import signature
import inspect

# PyTorch 核心层导入（统一规范，避免冗余）
from torch import Tensor

from torch.nn import (
    BatchNorm1d,
    Dropout,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    GELU,  # 新增：GELU 激活函数，更适合回归任务
    Sequential,
    BatchNorm1d
)

# PyTorch Geometric 导入
from torch_geometric.typing import Adj
from torch_geometric.nn import GINEConv, global_add_pool, GlobalAttention  # 新增：GlobalAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.utils import (
    degree,
    sort_edge_index,
    to_dense_batch,
    scatter,
)

# Mamba 相关导入
from cross_atten.mamba import Mamba, MambaConfig

# ------------------------------------------------------------------------------
# 工具函数：批次内节点随机排列
# ------------------------------------------------------------------------------
def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

# ------------------------------------------------------------------------------
# GPSConv 层（修复 reset_parameters 方法，保持原有功能）
# ------------------------------------------------------------------------------
class GPSConv(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            conv: Optional[MessagePassing],
            heads: int = 1,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            act: str = 'relu',
            att_type: str = 'transformer',
            order_by_degree: bool = False,
            shuffle_ind: int = 0,
            d_state: int = 16,
            d_conv: int = 4,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = 'batch_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert (self.order_by_degree == True and self.shuffle_ind == 0) or (
                self.order_by_degree == False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'

        if self.att_type == 'mamba':
            # 初始化 Mamba 模型（使用你的配置）
            config = MambaConfig(d_model=channels, n_layers=2, use_cuda=True)
            self.self_attn = Mamba(config)

        # MLP 层（保持原有结构）
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        # 归一化层（保持原有逻辑）
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        # 修复：Mamba 无 _reset_parameters 方法，直接重置 self.self_attn（若支持）
        if hasattr(self.self_attn, 'reset_parameters'):
            self.self_attn.reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
        
        # ------------ Mamba 全局注意力计算 ------------
        if self.att_type == 'mamba':
            # 如果启用按节点度排序（提升重要节点的上下文访问）
            if self.order_by_degree and batch is not None:
                # 计算每个节点的度（基于边索引）
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                # 按 [batch, degree] 排序节点（重要节点排在序列末尾）
                order_tensor = torch.stack([batch, deg], 1).T
                _, x_sorted = sort_edge_index(order_tensor, edge_attr=x)
                x = x_sorted

            # 单次排列推理（默认模式）
            if self.shuffle_ind == 0 and batch is not None:
                # 将稀疏图数据转换为密集批次格式（处理变长序列）
                h, mask = to_dense_batch(x, batch)
                # Mamba 处理密集序列后恢复稀疏格式
                h = self.self_attn(h)[mask]
            # 多次排列平均（增强稳定性）
            elif self.shuffle_ind > 0 and batch is not None:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    # 在批次内随机排列节点顺序（减少顺序偏差）
                    h_ind_perm = permute_within_batch(x, batch)
                    # 转换为密集格式并通过 Mamba 处理
                    h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                    h_i = self.self_attn(h_i)[mask][h_ind_perm]  # 恢复原始顺序
                    mamba_arr.append(h_i)
                # 对多次排列结果取平均
                h = sum(mamba_arr) / self.shuffle_ind
            else:
                # 无批次信息时，直接处理
                h = self.self_attn(x.unsqueeze(0)).squeeze(0)

        # ------------ 残差连接与归一化 ------------
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # 残差连接：Mamba 输出与原始输入相加
        if self.norm2 is not None:
            if self.norm_with_batch and batch is not None:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)  # 保存全局 Mamba 的输出

        # ------------ 合并局部和全局输出 ------------
        out = sum(hs)  # 简单相加（或可自定义加权方式）

        # ------------ 最终 MLP 处理 ------------
        if self.norm3 is not None:
            if self.norm_with_batch and batch is not None:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')

# ------------------------------------------------------------------------------
# Graphblock 层（保持原有结构，无修改）
# ------------------------------------------------------------------------------
class Graphblock(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int, model_type: str, shuffle_ind: int, d_state: int,
                 d_conv: int, order_by_degree: bool = False, if_pool=False):
        super().__init__()

        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.if_pool = if_pool
        self.convs = ModuleList()
        
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            
            if self.model_type == 'gine':
                conv = GINEConv(nn)
            elif self.model_type == 'mamba':
                # 输入先经过第二个参数(GINEConv(nn))再进入 mamba
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
            elif self.model_type == "only_mamba":
                conv = GPSConv(channels, None, heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
            else:
                raise ValueError(f"不支持的模型类型：{self.model_type}")
            
            self.convs.append(conv)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
        
        if self.if_pool:
            x = global_add_pool(x, batch)
        
        return x

# ------------------------------------------------------------------------------
# GraphModel 层（保持原有结构，无修改）
# ------------------------------------------------------------------------------
class GraphModel(torch.nn.Module):
    """
    图神经网络模型，支持多种图卷积类型（GINE、Mamba、纯Mamba）
    
    主要功能：
    1. 处理节点特征、位置编码和边特征
    2. 使用多种图卷积层进行特征提取
    3. 支持全局池化操作
    4. 适用于从医学影像构建的图结构数据
    """
    
    def __init__(
        self, 
        channels: int,               # 输出通道数（特征维度）
        pe_dim: int,                 # 位置编码的维度
        num_layers: int,             # 图卷积层的数量
        model_type: str,             # 模型类型：'gine'、'mamba'、'only_mamba'
        shuffle_ind: int,            # 随机排序索引（用于Mamba模型）
        d_state: int,                # Mamba模型的状态维度
        d_conv: int,                 # Mamba模型的卷积维度
        order_by_degree: bool,       # 是否按节点度排序（用于Mamba模型）
        node_dim: int = 48,          # 节点特征维度（默认48）
        edge_dim: int = 1,           # 边特征维度（默认1）
        if_pool: bool = False,       # 是否使用全局池化
        drop: float = 0.2            # Dropout率
    ):
        super().__init__()
        
        # ========== 特征嵌入层 ==========
        
        # 节点特征嵌入：将原始节点特征映射到(channels - pe_dim)维度
        # 留出pe_dim的维度给位置编码
        self.node_emb = Linear(node_dim, channels - pe_dim)
        
        # 位置编码处理：包含归一化和线性变换
        # 输入：随机游走位置编码（walk_length=20），输出维度为20
        self.pe_norm = BatchNorm1d(20)      # 对位置编码进行批归一化
        self.pe_lin = Linear(20, pe_dim)    # 将位置编码映射到pe_dim维度
        
        # 边特征嵌入：将边特征映射到channels维度
        self.edge_emb = Linear(edge_dim, channels)
        
        # ========== 模型配置 ==========
        
        self.if_pool = if_pool          # 是否使用全局池化
        self.model_type = model_type    # 模型类型
        self.shuffle_ind = shuffle_ind  # 随机排序索引
        self.order_by_degree = order_by_degree  # 是否按节点度排序
        
        # Dropout层，防止过拟合
        self.drop = Dropout(drop)
        
        # ========== 图卷积层构建 ==========
        
        self.convs = ModuleList()  # 存储多个图卷积层
        
        for _ in range(num_layers):
            # 构建GINE卷积中的多层感知机（MLP）
            nn = Sequential(
                Linear(channels, channels),  # 第一层线性变换
                ReLU(),                      # 激活函数
                Linear(channels, channels),  # 第二层线性变换
            )
            
            # 根据模型类型选择不同的卷积层
            if self.model_type == 'gine':
                # GINE（Graph Isomorphism Network with Edge features）卷积
                # 支持边特征的图卷积网络
                conv = GINEConv(nn)
                
            elif self.model_type == 'mamba':
                # Mamba图卷积：结合GINE局部卷积和Mamba全局注意力
                # GPSConv = Graph + Positional + Structural Convolution
                conv = GPSConv(
                    channels,          # 特征维度
                    GINEConv(nn),      # 局部卷积层（GINE）
                    heads=4,           # 注意力头数
                    attn_dropout=0.5,  # 注意力dropout率
                    att_type='mamba',  # 注意力类型为Mamba
                    shuffle_ind=self.shuffle_ind,      # 随机排序索引
                    order_by_degree=self.order_by_degree,  # 按节点度排序
                    d_state=d_state,   # Mamba状态维度
                    d_conv=d_conv      # Mamba卷积维度
                )
                
            elif self.model_type == "only_mamba":
                # 纯Mamba模型：不使用局部卷积，仅用Mamba全局注意力
                conv = GPSConv(
                    channels, 
                    None,             # 不使用局部卷积层
                    heads=4, 
                    attn_dropout=0.5,
                    att_type='mamba',
                    shuffle_ind=self.shuffle_ind,
                    order_by_degree=self.order_by_degree,
                    d_state=d_state, 
                    d_conv=d_conv
                )
            
            # 将卷积层添加到模块列表中
            self.convs.append(conv)

# ------------------------------------------------------------------------------
# GraphGDP 主模型（保持原始结构，无任何修改）
# ------------------------------------------------------------------------------
class GraphGDP(torch.nn.Module):
    """
    图神经网络模型，支持多种图卷积类型（GINE、Mamba、纯Mamba）
    核心修改1：双回归任务，同时输出「节点级GDP标签」和「全局级图块GDP标签」
    核心修改2：全局池化方法可选，支持「加池化」「可学习注意力池化」「均值聚合」
    
    主要功能：
    1. 处理节点特征、三种位置编码（LapPE/随机游走/网格坐标）和边特征
    2. 支持位置编码可选启用，兼容你的 LapPE 数据集（维度 12）
    3. 双回归输出：节点标签（每个网格GDP）、全局标签（图块总GDP）
    4. 灵活配置全局池化，适配夜间灯光/人口密度等空间异质性数据
    """
    
    def __init__(
        self, 
        channels: int,               # 输出通道数（特征维度）
        pe_dim: int,                 # 单种位置编码的维度（默认 12，适配 LapPE）
        num_layers: int,             # 图卷积层的数量
        model_type: str,             # 模型类型：'gine'、'mamba'、'only_mamba'
        shuffle_ind: int,            # 随机排序索引（用于 Mamba 模型）
        d_state: int,                # Mamba 模型的状态维度
        d_conv: int,                 # Mamba 模型的卷积维度
        order_by_degree: bool,       # 是否按节点度排序（用于 Mamba 模型）
        # 新增：PE 启用控制（默认仅启用 LapPE，关闭无意义的随机游走 PE）
        use_rw_pe: bool = False,     # 是否启用随机游走 PE
        use_lap_pe: bool = True,     # 是否启用 LapPE（默认启用，适配你的数据集）
        use_grid_pe: bool = False,   # 是否启用网格坐标 PE
        # 基础参数（适配你的数据集）
        node_dim: int = 422,         # 节点特征维度（修改为 422，与你的数据集一致）
        edge_dim: int = 1,           # 边特征维度（默认 1）
        if_pool: bool = False,       # 是否使用全局池化（为全局标签服务）
        pool_type: str = 'add',      # 新增：全局池化类型，可选 'add'/'attention'/'mean'
        drop: float = 0.2,           # Dropout 率
        # 双回归标签配置（适配你的数据集：原始GDP + log(1+GDP)）
        node_label_dim: int = 2,     # 节点级标签维度（grid_gdp + grid_gdp_log）
        global_label_dim: int = 2,   # 全局级标签维度（patch_gdp + patch_log_gdp）
        # 其他 PE 输入维度配置
        rw_pe_input_dim: int = 20,   # 随机游走 PE 输入维度（默认 20 步）
        grid_pe_input_dim: int = 2   # 网格坐标 PE 输入维度（row, col，默认 2）
    ):
        super().__init__()
        
        # ========== 核心配置：记录 PE 启用状态与总维度 ==========
        self.use_rw_pe = use_rw_pe
        self.use_lap_pe = use_lap_pe
        self.use_grid_pe = use_grid_pe
        self.pe_dim = pe_dim
        self.total_pe_dim = pe_dim * sum([use_rw_pe, use_lap_pe, use_grid_pe])
        self.if_pool = if_pool  # 保留池化标记，用于全局标签生成
        
        # 校验：总 PE 维度不超过输出通道数，避免维度浪费
        if self.total_pe_dim > 0 and self.total_pe_dim >= channels:
            raise ValueError(f"总位置编码维度({self.total_pe_dim})不能超过输出通道数({channels})")
        
        # ========== 特征嵌入层 ==========
        # 节点特征嵌入：留出总 PE 维度的空间，剩余维度给节点特征
        node_emb_out_dim = channels - self.total_pe_dim
        self.node_emb = Linear(node_dim, node_emb_out_dim)
        
        # ========== 三种位置编码独立处理层（按需初始化） ==========
        # 1. 随机游走 PE 处理（仅当启用时初始化）
        if self.use_rw_pe:
            self.rw_pe_norm = BatchNorm1d(rw_pe_input_dim)
            self.rw_pe_lin = Linear(rw_pe_input_dim, pe_dim)
        
        # 2. LapPE 处理（默认启用，适配你的数据集维度 12）
        if self.use_lap_pe:
            self.lap_pe_norm = BatchNorm1d(pe_dim)  # 输入维度 = pe_dim（12）
            self.lap_pe_lin = Linear(pe_dim, pe_dim)
        
        # 3. 网格坐标 PE 处理（仅当启用时初始化）
        if self.use_grid_pe:
            self.grid_pe_norm = BatchNorm1d(grid_pe_input_dim)
            self.grid_pe_lin = Linear(grid_pe_input_dim, pe_dim)
        
        # ========== 边特征嵌入 ==========
        self.edge_emb = Linear(edge_dim, channels)
        
        # ========== 模型配置 ==========
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.drop = Dropout(drop)
        
        # ========== 新增：全局池化配置（可选可学习注意力池化） ==========
        self.pool_type = pool_type
        # 验证池化类型合法性
        supported_pools = ['add', 'attention', 'mean']
        if self.pool_type not in supported_pools:
            raise ValueError(f"不支持的池化类型：{pool_type}，可选：{supported_pools}")
        
        # 初始化可学习注意力池化层（仅当启用池化且类型为 attention 时）
        if self.if_pool and self.pool_type == 'attention':
            # 评分网络（gate_nn）：可学习，用于计算节点注意力权重
            # 输入：节点特征（channels 维），输出：1 维注意力分数
            self.attention_pool = GlobalAttention(
                gate_nn=Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Dropout(drop),
                    Linear(channels, 1)  # 输出单个注意力权重分数
                )
            )
        
        # ========== 图卷积层构建 ==========
        self.convs = ModuleList()  # 存储多个图卷积层
        
        for _ in range(num_layers):
            # 构建 GINE 卷积中的多层感知机（MLP）
            nn_mlp = Sequential(
                Linear(channels, channels),  # 第一层线性变换
                ReLU(),                      # 激活函数
                Linear(channels, channels),  # 第二层线性变换
            )
            
            # 根据模型类型选择不同的卷积层
            if self.model_type == 'gine':
                conv = GINEConv(nn_mlp)
                
            elif self.model_type == 'mamba':
                conv = GPSConv(
                    channels,          # 特征维度
                    GINEConv(nn_mlp),  # 局部卷积层（GINE）
                    heads=4,           # 注意力头数
                    attn_dropout=0.5,  # 注意力 dropout 率
                    att_type='mamba',  # 注意力类型为 Mamba
                    shuffle_ind=self.shuffle_ind,
                    order_by_degree=self.order_by_degree,
                    d_state=d_state,
                    d_conv=d_conv
                )
                
            elif self.model_type == "only_mamba":
                conv = GPSConv(
                    channels, 
                    None,
                    heads=4, 
                    attn_dropout=0.5,
                    att_type='mamba',
                    shuffle_ind=self.shuffle_ind,
                    order_by_degree=self.order_by_degree,
                    d_state=d_state, 
                    d_conv=d_conv
                )
            
            else:
                raise ValueError(f"不支持的模型类型：{model_type}，可选：'gine'、'mamba'、'only_mamba'")
            
            # 将卷积层添加到模块列表中
            self.convs.append(conv)
        
        
        # ========== 核心修改：双回归输出头 ==========
        # ========== 核心修改：双回归输出头（BatchNorm1d + GELU 版本） ==========
        # 1. 节点级回归头（输出每个节点的 GDP 标签：grid_gdp + grid_gdp_log）
        self.node_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),  # 新增：BatchNorm1d 归一化，稳定训练
            GELU(),  # 替换 ReLU 为 GELU，更平滑的激活函数，缓解梯度消失
            Dropout(drop),
            Linear(channels // 2, node_label_dim)  # 输出 [num_nodes, 2]，保持原有输出维度不变
        )

        # 2. 全局级回归头（输出整个图块的 GDP 标签：patch_gdp + patch_log_gdp）
        self.global_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),  # 新增：BatchNorm1d 归一化，匹配节点头结构
            GELU(),  # 替换 ReLU 为 GELU，保持双回归头结构一致性
            Dropout(drop),
            Linear(channels // 2, global_label_dim)  # 输出 [batch_size, 2]，保持原有输出维度不变
        )
        
    def forward(
        self, 
        x,              # 节点特征，形状: [num_nodes, 422]（与你的数据集一致）
        edge_index,     # 边索引，形状: [2, num_edges]（8 邻域图）
        edge_attr,      # 边特征，形状: [num_edges, 1]
        batch,          # 批次索引，形状: [num_nodes]
        lap_pe=None,    # LapPE 编码，形状: [num_nodes, 12]（默认启用，适配你的数据集）
        rw_pe=None,     # 随机游走 PE，形状: [num_nodes, 20]（可选启用）
        grid_pe=None    # 网格坐标 PE，形状: [num_nodes, 2]（可选启用）
    ):
        """
        前向传播过程（修改：双回归输出 + 全局池化方法可选）
        Returns:
            node_pred: 节点级标签预测，形状 [num_nodes, node_label_dim]
            global_pred: 全局级标签预测，形状 [batch_size, global_label_dim]
        """
        
        # ========== 节点特征预处理 ==========
        x_node = self.node_emb(x)  # 形状: [num_nodes, node_emb_out_dim]
        
        # ========== 位置编码处理（按需启用，独立处理） ==========
        pe_list = []  # 存储启用的 PE 处理结果
        
        # 1. 处理随机游走 PE（若启用）
        if self.use_rw_pe and rw_pe is not None:
            rw_x_pe = self.rw_pe_norm(rw_pe)
            rw_x_pe = self.rw_pe_lin(rw_x_pe)
            pe_list.append(rw_x_pe)
        
        # 2. 处理 LapPE（默认启用，适配你的数据集）
        if self.use_lap_pe and lap_pe is not None:
            lap_x_pe = self.lap_pe_norm(lap_pe)
            lap_x_pe = self.lap_pe_lin(lap_x_pe)
            pe_list.append(lap_x_pe)
        
        # 3. 处理网格坐标 PE（若启用）
        if self.use_grid_pe and grid_pe is not None:
            grid_x_pe = self.grid_pe_norm(grid_pe)
            grid_x_pe = self.grid_pe_lin(grid_x_pe)
            pe_list.append(grid_x_pe)
        
        # ========== 拼接节点特征和启用的位置编码 ==========
        if len(pe_list) > 0:
            total_pe = torch.cat(pe_list, dim=1)  # 形状: [num_nodes, total_pe_dim]
            x = torch.cat((x_node, total_pe), dim=1)  # 形状: [num_nodes, channels]
        else:
            x = x_node  # 未启用任何 PE，直接使用节点特征
        
        # ========== 边特征预处理 ==========
        edge_attr = self.edge_emb(edge_attr)  # 形状: [num_edges, channels]
        
        # ========== 多层图卷积 ==========
        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
            
            # 应用 Dropout 防止过拟合
            x = self.drop(x)
        
        # ========== 核心修改：双回归预测（全局池化方法可选） ==========
        # 1. 节点级标签预测（直接基于卷积后的节点特征）
        node_pred = self.node_regression_head(x)  # 形状: [num_nodes, node_label_dim]
        
        # 2. 全局级标签预测（根据 pool_type 选择不同全局池化方法，得到图级特征）
        if self.if_pool:
            # 选择对应的全局池化方法
            if self.pool_type == 'add':
                # 方法1：简单加池化（无学习权重，保留总量信息）
                graph_feature = global_add_pool(x, batch)
            
            elif self.pool_type == 'attention':
                # 方法2：可学习注意力池化（加权求和，自动学习节点贡献权重）
                graph_feature = self.attention_pool(x, batch)
            
            elif self.pool_type == 'mean':
                # 方法3：均值聚合（兜底方案，抹平节点差异）
                unique_batch = torch.unique(batch)
                graph_feature_list = []
                for b in unique_batch:
                    batch_node_feat = x[batch == b]
                    mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                    graph_feature_list.append(mean_feat)
                graph_feature = torch.cat(graph_feature_list, dim=0)
        
        else:
            # 不启用池化，保留原有均值聚合兜底
            unique_batch = torch.unique(batch)
            graph_feature_list = []
            for b in unique_batch:
                batch_node_feat = x[batch == b]
                mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                graph_feature_list.append(mean_feat)
            graph_feature = torch.cat(graph_feature_list, dim=0)  # [batch_size, channels]
        
        global_pred = self.global_regression_head(graph_feature)  # [batch_size, global_label_dim]
        
        # 返回双输出：节点标签 + 全局标签
        return node_pred, global_pred


# ------------------------------------------------------------------------------
# 样例推理（if __name__ == '__main__'）：适配三个模型（原始+残差+注意力）
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. 设备配置（优先使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== 模型推理演示（原始+残差+注意力增强版） =====")
    print(f"使用设备：{device}")
    
    # 2. 通用模型配置（三个模型共用）
    common_model_config = {
        "channels": 128,
        "pe_dim": 12,  # LapPE 维度，与你的数据集一致
        "num_layers": 4,
        "model_type": "mamba",
        "shuffle_ind": 0,
        "d_state": 16,
        "d_conv": 4,
        "order_by_degree": True,
        "use_lap_pe": True,
        "use_rw_pe": False,
        "use_grid_pe": False,
        "node_dim": 422,  # 节点特征维度，与你的数据集一致
        "if_pool": True,  # 启用全局池化
        "pool_type": "attention",  # 可选：'add'/'attention'/'mean'
        "drop": 0.2,
        "node_label_dim": 2,  # 节点标签：grid_gdp + grid_gdp_log
        "global_label_dim": 2  # 全局标签：patch_gdp + patch_log_gdp
    }
    
    # 3. 初始化三个模型
    model_original = GraphGDP(**common_model_config).to(device)
    
    # 切换到推理模式
    for model, name in zip([model_original], ["原始版"]):
        model.eval()
        print(f"\n{name}模型初始化完成，LapPE 维度 {common_model_config['pe_dim']}")
    
    # 4. 生成模拟数据（与你的数据集格式一致）
    num_nodes = 25  # 单个图块的节点数
    num_edges = 124 # 单个图块的边数
    batch_size = 1  # 单个图块，批次大小为 1
    
    # 4.1 节点特征：[num_nodes, 422]
    x = torch.randn(num_nodes, common_model_config['node_dim'], device=device)
    
    # 4.2 LapPE 编码：[num_nodes, 12]
    lap_pe = torch.randn(num_nodes, common_model_config['pe_dim'], device=device)
    
    # 4.3 边索引：[2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    
    # 4.4 边特征：[num_edges, 1]
    edge_attr = torch.randn(num_edges, 1, device=device)
    
    # 4.5 批次索引：[num_nodes]
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    print(f"\n===== 输入数据信息 =====")
    print(f"节点特征形状：{x.shape}")
    print(f"LapPE 编码形状：{lap_pe.shape}")
    print(f"边索引形状：{edge_index.shape}")
    print(f"边特征形状：{edge_attr.shape}")
    print(f"批次索引形状：{batch.shape}")
    
    # 5. 三个模型分别推理
    with torch.no_grad():  # 关闭梯度计算
        # 原始版
        node_pred_ori, global_pred_ori = model_original(x=x, lap_pe=lap_pe, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
    
    # 6. 输出推理结果概要
    print(f"\n===== 推理结果概要（三个模型） =====")
    for pred, name in zip(
        [(node_pred_ori, global_pred_ori)],
        ["原始版"]
    ):
        node_pred, global_pred = pred
        print(f"\n{name}：")
        print(f"  节点标签形状：{node_pred.shape}，全局标签形状：{global_pred.shape}")
        print(f"  全局预测值（patch_gdp, patch_log_gdp）：\n{global_pred}")
    
    print(f"\n===== 三个模型推理演示完成 =====")
