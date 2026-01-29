import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    Linear,
    ModuleList,
    ReLU,
    GELU,
    Sequential,
)
from torch_geometric.typing import Adj
from torch_geometric.nn import GCNConv, global_add_pool, GlobalAttention

class GraphGDP_GCN(torch.nn.Module):
    """
    纯GCN版本的GraphGDP模型（对比实验基线）
    核心改动：移除所有Mamba相关逻辑，仅保留GCNConv作为核心卷积模块
    保留：双回归任务、位置编码（LapPE为主）、全局池化、残差/注意力回归头等原有设计
    """
    def __init__(
        self, 
        channels: int,               # 输出通道数（特征维度）
        pe_dim: int,                 # 位置编码维度（默认12，适配LapPE）
        num_layers: int,             # GCN卷积层数量
        # PE启用控制（与原模型保持一致）
        use_rw_pe: bool = False,     
        use_lap_pe: bool = True,     
        use_grid_pe: bool = False,
        # 基础参数（适配你的数据集）
        node_dim: int = 422,         # 节点特征维度（与你的数据集一致）
        edge_dim: int = 1,           # GCN不使用边特征，但保留参数兼容
        if_pool: bool = False,       # 是否使用全局池化（为全局标签服务）
        pool_type: str = 'add',      # 全局池化类型：'add'/'attention'/'mean'
        drop: float = 0.2,           # Dropout率
        # 双回归标签配置
        node_label_dim: int = 2,     # 节点级标签维度（grid_gdp + grid_gdp_log）
        global_label_dim: int = 2,   # 全局级标签维度（patch_gdp + patch_log_gdp）
        # 其他PE维度配置
        rw_pe_input_dim: int = 20,
        grid_pe_input_dim: int = 2
    ):
        super().__init__()
        
        # ========== 核心配置：PE启用状态与总维度 ==========
        self.use_rw_pe = use_rw_pe
        self.use_lap_pe = use_lap_pe
        self.use_grid_pe = use_grid_pe
        self.pe_dim = pe_dim
        self.total_pe_dim = pe_dim * sum([use_rw_pe, use_lap_pe, use_grid_pe])
        self.if_pool = if_pool
        self.pool_type = pool_type

        # 校验PE维度
        if self.total_pe_dim > 0 and self.total_pe_dim >= channels:
            raise ValueError(f"总位置编码维度({self.total_pe_dim})不能超过输出通道数({channels})")
        
        # ========== 特征嵌入层（与原模型一致） ==========
        # 节点特征嵌入：留出PE维度空间
        node_emb_out_dim = channels - self.total_pe_dim
        self.node_emb = Linear(node_dim, node_emb_out_dim)
        
        # 三种位置编码处理层（按需初始化）
        if self.use_rw_pe:
            self.rw_pe_norm = BatchNorm1d(rw_pe_input_dim)
            self.rw_pe_lin = Linear(rw_pe_input_dim, pe_dim)
        
        if self.use_lap_pe:
            self.lap_pe_norm = BatchNorm1d(pe_dim)
            self.lap_pe_lin = Linear(pe_dim, pe_dim)
        
        if self.use_grid_pe:
            self.grid_pe_norm = BatchNorm1d(grid_pe_input_dim)
            self.grid_pe_lin = Linear(grid_pe_input_dim, pe_dim)
        
        # 边特征嵌入（GCN不使用，但保留以兼容输入）
        self.edge_emb = Linear(edge_dim, channels)
        
        # ========== Dropout层 ==========
        self.drop = Dropout(drop)
        
        # ========== 全局池化配置（与原模型一致） ==========
        supported_pools = ['add', 'attention', 'mean']
        if self.pool_type not in supported_pools:
            raise ValueError(f"不支持的池化类型：{pool_type}，可选：{supported_pools}")
        
        if self.if_pool and self.pool_type == 'attention':
            self.attention_pool = GlobalAttention(
                gate_nn=Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Dropout(drop),
                    Linear(channels, 1)
                )
            )
        
        # ========== 纯GCN卷积层构建 ==========
        self.convs = ModuleList()
        # 第一层GCN：输入维度=channels（节点+PE拼接后）
        self.convs.append(GCNConv(channels, channels))
        # 后续GCN层：保持通道数一致
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(channels, channels))
        
        # ========== 双回归输出头（与原模型一致的残差+GELU结构） ==========
        # 节点级回归头
        self.node_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),
            GELU(),
            Dropout(drop),
            Linear(channels // 2, node_label_dim)
        )

        # 全局级回归头
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
        edge_index: Adj,        # 边索引: [2, num_edges]
        edge_attr: Optional[Tensor] = None,  # 边特征（GCN不使用，仅兼容）
        batch: Optional[Tensor] = None,      # 批次索引: [num_nodes]
        lap_pe: Optional[Tensor] = None,     # LapPE: [num_nodes, 12]
        rw_pe: Optional[Tensor] = None,      # 随机游走PE: [num_nodes, 20]
        grid_pe: Optional[Tensor] = None     # 网格坐标PE: [num_nodes, 2]
    ) -> tuple[Tensor, Tensor]:
        """
        前向传播（纯GCN，无Mamba逻辑）
        Returns:
            node_pred: 节点级GDP预测 [num_nodes, node_label_dim]
            global_pred: 全局级GDP预测 [batch_size, global_label_dim]
        """
        # ========== 节点特征预处理 ==========
        x_node = self.node_emb(x)
        
        # ========== 位置编码处理（与原模型一致） ==========
        pe_list = []
        if self.use_rw_pe and rw_pe is not None:
            rw_x_pe = self.rw_pe_norm(rw_pe)
            rw_x_pe = self.rw_pe_lin(rw_x_pe)
            pe_list.append(rw_x_pe)
        
        if self.use_lap_pe and lap_pe is not None:
            lap_x_pe = self.lap_pe_norm(lap_pe)
            lap_x_pe = self.lap_pe_lin(lap_x_pe)
            pe_list.append(lap_x_pe)
        
        if self.use_grid_pe and grid_pe is not None:
            grid_x_pe = self.grid_pe_norm(grid_pe)
            grid_x_pe = self.grid_pe_lin(grid_x_pe)
            pe_list.append(grid_x_pe)
        
        # 拼接节点特征和PE
        if len(pe_list) > 0:
            total_pe = torch.cat(pe_list, dim=1)
            x = torch.cat((x_node, total_pe), dim=1)
        else:
            x = x_node
        
        # ========== 边特征预处理（仅兼容，GCN不使用） ==========
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
        
        # ========== 纯GCN多层卷积 ==========
        for conv in self.convs:
            # GCNConv仅需要x和edge_index，无需edge_attr
            x = conv(x, edge_index)
            x = F.relu(x)  # GCN经典ReLU激活
            x = self.drop(x)
        
        # ========== 双回归预测（与原模型一致） ==========
        # 1. 节点级预测
        node_pred = self.node_regression_head(x)
        
        # 2. 全局级预测（全局池化）
        if self.if_pool:
            if self.pool_type == 'add':
                graph_feature = global_add_pool(x, batch)
            elif self.pool_type == 'attention':
                graph_feature = self.attention_pool(x, batch)
            elif self.pool_type == 'mean':
                unique_batch = torch.unique(batch)
                graph_feature_list = []
                for b in unique_batch:
                    batch_node_feat = x[batch == b]
                    mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                    graph_feature_list.append(mean_feat)
                graph_feature = torch.cat(graph_feature_list, dim=0)
        else:
            # 兜底均值聚合
            unique_batch = torch.unique(batch)
            graph_feature_list = []
            for b in unique_batch:
                batch_node_feat = x[batch == b]
                mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                graph_feature_list.append(mean_feat)
            graph_feature = torch.cat(graph_feature_list, dim=0)
        
        global_pred = self.global_regression_head(graph_feature)
        
        return node_pred, global_pred

# ------------------------------ 测试样例 ------------------------------
if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化纯GCN模型
    model = GraphGDP_GCN(
        channels=128,
        pe_dim=12,
        num_layers=4,
        use_lap_pe=True,
        node_dim=422,
        if_pool=True,
        pool_type='add',
        drop=0.2
    ).to(device)
    
    # 模拟输入数据
    num_nodes = 25
    x = torch.randn(num_nodes, 422, device=device)
    edge_index = torch.randint(0, num_nodes, (2, 124), device=device)
    lap_pe = torch.randn(num_nodes, 12, device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    # 推理测试
    model.eval()
    with torch.no_grad():
        node_pred, global_pred = model(
            x=x,
            edge_index=edge_index,
            batch=batch,
            lap_pe=lap_pe
        )
    
    print(f"节点预测形状: {node_pred.shape}")  # [25, 2]
    print(f"全局预测形状: {global_pred.shape}")  # [1, 2]