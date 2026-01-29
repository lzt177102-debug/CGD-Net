import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    ModuleList,
    ReLU,
    GELU,
    Sequential,
    LayerNorm,
    MultiheadAttention,
)
from torch_geometric.typing import Adj
from torch_geometric.nn import global_add_pool, GlobalAttention
from torch_geometric.utils import to_dense_batch, degree, sort_edge_index

class GraphormerLayer(torch.nn.Module):
    """Graphormer核心层：空间编码 + 多头注意力 + FFN（适配GDP预测任务）"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: int = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or dim * 4  # FFN维度默认是特征维度的4倍

        # Graphormer核心：多头自注意力（PyTorch原生实现，无第三方依赖）
        self.self_attn = MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # 批次维度在前，适配图数据格式
        )

        # FFN层（Graphormer经典结构）
        self.ffn = Sequential(
            Linear(dim, self.ffn_dim),
            GELU(),  # 替换ReLU，更适配Transformer类模型
            Dropout(dropout),
            Linear(self.ffn_dim, dim),
            Dropout(dropout),
        )

        # 归一化层（Graphormer使用LayerNorm，而非BatchNorm）
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        前向传播：适配稀疏图数据 -> 密集批次 -> 注意力 -> 恢复稀疏格式
        """
        # 1. 将稀疏图数据转换为密集批次格式（处理变长序列）
        x_dense, mask = to_dense_batch(x, batch)
        # MultiheadAttention需要bool mask（True表示该位置被mask）
        padding_mask = ~mask  

        # 2. 多头自注意力（Graphormer核心）
        x_norm = self.norm1(x_dense)  # 前置LayerNorm
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_bias,       # 空间编码偏置（可选）
            key_padding_mask=padding_mask,  # 填充mask
        )
        # 残差连接 + Dropout
        x_dense = x_dense + self.dropout(attn_out)

        # 3. FFN前馈网络
        x_norm = self.norm2(x_dense)  # 前置LayerNorm
        ffn_out = self.ffn(x_norm)
        # 残差连接 + Dropout
        x_dense = x_dense + self.dropout(ffn_out)

        # 4. 恢复稀疏格式（适配图数据输出）
        x = x_dense[~padding_mask]
        return x

class GraphGDP_Graphormer(torch.nn.Module):
    """
    Graphormer版本的GraphGDP模型（无第三方依赖）
    核心：Graphormer的多头注意力 + 空间位置编码
    保留：原模型双回归任务、LapPE位置编码、全局池化等所有核心逻辑
    适配：针对网格GDP数据优化，移除冗余的图结构编码，提升效率
    """
    def __init__(
        self, 
        channels: int = 128,               # 特征维度（与原模型一致）
        pe_dim: int = 12,                 # LapPE维度（适配你的数据集）
        num_layers: int = 4,              # Graphormer层数
        use_rw_pe: bool = False,          # 是否启用随机游走PE
        use_lap_pe: bool = True,          # 是否启用LapPE（默认启用）
        use_grid_pe: bool = False,        # 是否启用网格坐标PE
        node_dim: int = 422,              # 节点特征维度（与你的数据集一致）
        edge_dim: int = 1,                # 仅兼容，Graphormer不使用边特征
        if_pool: bool = True,             # 是否使用全局池化（为全局GDP预测服务）
        pool_type: str = 'add',           # 全局池化类型：add/attention/mean
        drop: float = 0.2,                # Dropout率
        node_label_dim: int = 2,          # 节点级标签维度（grid_gdp + log_gdp）
        global_label_dim: int = 2,        # 全局级标签维度（patch_gdp + log_gdp）
        rw_pe_input_dim: int = 20,        # 随机游走PE输入维度
        grid_pe_input_dim: int = 2,       # 网格坐标PE输入维度（row/col）
        graphormer_heads: int = 8,        # Graphormer注意力头数（默认8）
    ):
        super().__init__()
        
        # ========== 核心配置：位置编码启用状态 ==========
        self.use_rw_pe = use_rw_pe
        self.use_lap_pe = use_lap_pe
        self.use_grid_pe = use_grid_pe
        self.pe_dim = pe_dim
        # 计算总位置编码维度
        self.total_pe_dim = pe_dim * sum([use_rw_pe, use_lap_pe, use_grid_pe])
        self.if_pool = if_pool
        self.pool_type = pool_type

        # 校验：总PE维度不能超过特征通道数
        if self.total_pe_dim > 0 and self.total_pe_dim >= channels:
            raise ValueError(f"总位置编码维度({self.total_pe_dim})不能超过输出通道数({channels})")
        
        # ========== 特征嵌入层（与原GraphGDP完全一致） ==========
        # 节点特征嵌入：留出PE维度的空间
        node_emb_out_dim = channels - self.total_pe_dim
        self.node_emb = Linear(node_dim, node_emb_out_dim)
        
        # 1. 随机游走PE处理层（仅启用时初始化）
        if self.use_rw_pe:
            self.rw_pe_norm = BatchNorm1d(rw_pe_input_dim)
            self.rw_pe_lin = Linear(rw_pe_input_dim, pe_dim)
        
        # 2. LapPE处理层（默认启用，适配你的数据集）
        if self.use_lap_pe:
            self.lap_pe_norm = BatchNorm1d(pe_dim)
            self.lap_pe_lin = Linear(pe_dim, pe_dim)
        
        # 3. 网格坐标PE处理层（仅启用时初始化）
        if self.use_grid_pe:
            self.grid_pe_norm = BatchNorm1d(grid_pe_input_dim)
            self.grid_pe_lin = Linear(grid_pe_input_dim, pe_dim)
        
        # 边特征嵌入（仅兼容，Graphormer不使用边特征）
        self.edge_emb = Linear(edge_dim, channels)
        self.drop = Dropout(drop)
        
        # ========== 全局池化配置（与原模型一致） ==========
        supported_pools = ['add', 'attention', 'mean']
        if self.pool_type not in supported_pools:
            raise ValueError(f"不支持的池化类型：{pool_type}，可选：{supported_pools}")
        
        # 可学习注意力池化（仅启用时初始化）
        if self.if_pool and self.pool_type == 'attention':
            self.attention_pool = GlobalAttention(
                gate_nn=Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Dropout(drop),
                    Linear(channels, 1)  # 输出单个注意力权重
                )
            )
        
        # ========== Graphormer核心层堆叠 ==========
        self.graphormer_layers = ModuleList()
        for _ in range(num_layers):
            layer = GraphormerLayer(
                dim=channels,
                num_heads=graphormer_heads,
                dropout=drop,
            )
            self.graphormer_layers.append(layer)
        
        # ========== 双回归输出头（与原GraphGDP一致） ==========
        # 1. 节点级回归头（预测每个网格的GDP）
        self.node_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),  # 批归一化稳定训练
            GELU(),                      # 更平滑的激活函数
            Dropout(drop),
            Linear(channels // 2, node_label_dim)
        )

        # 2. 全局级回归头（预测整个图块的GDP）
        self.global_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),
            GELU(),
            Dropout(drop),
            Linear(channels // 2, global_label_dim)
        )
        
    def forward(
        self, 
        x: Tensor,              # 节点特征: [num_nodes, 422]
        edge_index: Adj,        # 边索引: [2, num_edges]（仅兼容，不使用）
        edge_attr: Optional[Tensor] = None,  # 边特征（仅兼容）
        batch: Optional[Tensor] = None,      # 批次索引: [num_nodes]
        lap_pe: Optional[Tensor] = None,     # LapPE编码: [num_nodes, 12]
        rw_pe: Optional[Tensor] = None,      # 随机游走PE: [num_nodes, 20]
        grid_pe: Optional[Tensor] = None     # 网格坐标PE: [num_nodes, 2]
    ) -> tuple[Tensor, Tensor]:
        """
        前向传播（Graphormer版）
        Returns:
            node_pred: 节点级GDP预测 [num_nodes, node_label_dim]
            global_pred: 全局级GDP预测 [batch_size, global_label_dim]
        """
        # ========== 节点特征预处理 ==========
        x_node = self.node_emb(x)  # 节点特征嵌入
        
        # ========== 位置编码处理（与原模型一致） ==========
        pe_list = []
        # 1. 随机游走PE
        if self.use_rw_pe and rw_pe is not None:
            rw_x_pe = self.rw_pe_norm(rw_pe)
            rw_x_pe = self.rw_pe_lin(rw_x_pe)
            pe_list.append(rw_x_pe)
        
        # 2. LapPE（核心位置编码，默认启用）
        if self.use_lap_pe and lap_pe is not None:
            lap_x_pe = self.lap_pe_norm(lap_pe)
            lap_x_pe = self.lap_pe_lin(lap_x_pe)
            pe_list.append(lap_x_pe)
        
        # 3. 网格坐标PE
        if self.use_grid_pe and grid_pe is not None:
            grid_x_pe = self.grid_pe_norm(grid_pe)
            grid_x_pe = self.grid_pe_lin(grid_x_pe)
            pe_list.append(grid_x_pe)
        
        # 拼接节点特征和位置编码
        if len(pe_list) > 0:
            total_pe = torch.cat(pe_list, dim=1)
            x = torch.cat((x_node, total_pe), dim=1)  # [num_nodes, channels]
        else:
            x = x_node
        
        # 边特征预处理（仅兼容，Graphormer不使用）
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
        
        # ========== Graphormer多层注意力计算 ==========
        for layer in self.graphormer_layers:
            x = layer(x, batch=batch)  # 逐层计算注意力
            x = self.drop(x)           # Dropout防止过拟合
        
        # ========== 双回归预测（与原模型一致） ==========
        # 1. 节点级GDP预测（每个网格）
        node_pred = self.node_regression_head(x)
        
        # 2. 全局级GDP预测（整个图块，需全局池化）
        if self.if_pool:
            if self.pool_type == 'add':
                # 加池化：保留总量信息，适配GDP求和场景
                graph_feature = global_add_pool(x, batch)
            
            elif self.pool_type == 'attention':
                # 可学习注意力池化：自动关注重要节点
                graph_feature = self.attention_pool(x, batch)
            
            elif self.pool_type == 'mean':
                # 均值池化：兜底方案
                unique_batch = torch.unique(batch)
                graph_feature_list = []
                for b in unique_batch:
                    batch_node_feat = x[batch == b]
                    mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                    graph_feature_list.append(mean_feat)
                graph_feature = torch.cat(graph_feature_list, dim=0)
        else:
            # 未启用池化时，默认均值聚合
            unique_batch = torch.unique(batch)
            graph_feature_list = []
            for b in unique_batch:
                batch_node_feat = x[batch == b]
                mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                graph_feature_list.append(mean_feat)
            graph_feature = torch.cat(graph_feature_list, dim=0)
        
        # 全局GDP预测
        global_pred = self.global_regression_head(graph_feature)
        
        return node_pred, global_pred

# ------------------------------ 测试样例 ------------------------------
if __name__ == '__main__':
    # 设备配置（优先使用GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== Graphormer模型测试 =====")
    print(f"使用设备：{device}")
    
    # 初始化Graphormer模型
    model = GraphGDP_Graphormer(
        channels=128,
        pe_dim=12,          # LapPE维度与你的数据集一致
        num_layers=4,       # Graphormer层数
        use_lap_pe=True,    # 启用LapPE
        node_dim=422,       # 节点特征维度与你的数据集一致
        if_pool=True,       # 启用全局池化
        pool_type='add',    # 加池化适配GDP求和
        drop=0.2,
        graphormer_heads=8  # 8头注意力（Graphormer经典配置）
    ).to(device)
    
    # 模拟输入数据（与你的数据集格式完全一致）
    num_nodes = 25  # 单个图块的节点数
    x = torch.randn(num_nodes, 422, device=device)          # 节点特征
    lap_pe = torch.randn(num_nodes, 12, device=device)      # LapPE编码
    edge_index = torch.randint(0, num_nodes, (2, 124), device=device)  # 边索引
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)    # 批次索引
    
    # 推理测试（关闭梯度计算）
    model.eval()
    with torch.no_grad():
        node_pred, global_pred = model(
            x=x,
            edge_index=edge_index,
            batch=batch,
            lap_pe=lap_pe
        )
    
    # 输出结果验证
    print(f"\n✅ 模型推理成功！")
    print(f"节点级预测形状: {node_pred.shape} (预期: [25, 2])")
    print(f"全局级预测形状: {global_pred.shape} (预期: [1, 2])")
    print(f"\n全局GDP预测值示例:\n{global_pred.cpu().numpy()}")