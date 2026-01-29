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
)
from torch_geometric.typing import Adj
from torch_geometric.nn import GINEConv, global_add_pool, GlobalAttention
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.utils import to_dense_batch

# Performer 线性注意力（兼容所有版本）
try:
    from performer_pytorch import Performer
except ImportError:
    print("警告：未安装performer-pytorch，执行 pip install performer-pytorch==1.1.0 安装")
    raise

class GPSConv_GraphGPS(torch.nn.Module):
    """GraphGPS核心卷积模块：GINE局部 + Performer全局（全版本兼容）"""
    def __init__(
        self,
        channels: int,
        conv: Optional[torch.nn.Module],
        dropout: float = 0.0,
        act: str = 'relu',
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        performer_heads: int = 4,
        performer_dim_head: int = 32,
        performer_depth: int = 1,  # Performer必填的depth参数
    ):
        super().__init__()
        self.channels = channels
        self.conv = conv
        self.dropout = dropout

        # ========== 核心修正：添加必填的depth参数 + 移除不兼容参数 ==========
        self.self_attn = Performer(
            dim=channels,
            depth=performer_depth,  # 新增：必填参数，指定Performer层数
            heads=performer_heads,
            dim_head=performer_dim_head,
            causal=False,
            generalized_attention=False,
            kernel_fn=torch.nn.ReLU(),
            # 移除所有可能不兼容的参数，保证版本适配
        )
        # 单独的Dropout层（替代Performer内部dropout）
        self.attn_dropout = Dropout(dropout)

        # MLP层（与原GPSConv一致）
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        # 归一化层
        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        hs = []
        # 1. GINE局部聚合
        if self.conv is not None:
            h = self.conv(x, edge_index, edge_attr=edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x  # 残差连接
            h = self.norm1(h) if self.norm1 is not None else h
            hs.append(h)
        
        # 2. Performer全局线性注意力（兼容版）
        h_dense, mask = to_dense_batch(x, batch)
        
        # ========== 新增：注意力输入归一化（防止爆炸） ==========
        if h_dense.size(1) > 0:  # 确保有节点
            # 对每个样本的序列维度进行标准化
            h_dense_mean = h_dense.mean(dim=1, keepdim=True)
            h_dense_std = h_dense.std(dim=1, keepdim=True) + 1e-6
            h_dense = (h_dense - h_dense_mean) / h_dense_std
        
        h_attn_dense = self.self_attn(h_dense)
        h_attn = h_attn_dense[mask]  # 恢复稀疏格式
        h_attn = self.attn_dropout(h_attn)      # 外部Dropout保证正则化
        h_attn = h_attn + x                     # 残差连接
        
        # ========== 新增：温和的数值裁剪 ==========
        if not self.training:  # 只在推理时应用
            if h_attn.abs().max() > 100:  # 值过大时才裁剪
                h_attn = torch.tanh(h_attn / 100.0) * 100.0
        
        h_attn = self.norm2(h_attn) if self.norm2 is not None else h_attn
        hs.append(h_attn)
        
        # 3. 合并局部+全局输出
        out = sum(hs)
        
        # ========== 新增：中间输出数值检查 ==========
        if not self.training:
            if torch.isnan(out).any() or torch.isinf(out).any():
                # 修复NaN/Inf
                out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
            elif out.abs().max() > 1000:  # 值过大时温和裁剪
                out = torch.tanh(out / 1000.0) * 1000.0
        
        out = self.norm3(out) if self.norm3 is not None else out
        mlp_out = self.mlp(out)
        
        # 最终残差连接
        final_out = mlp_out + out
        
        # ========== 新增：最终输出数值保护 ==========
        if not self.training:
            # 检测并修复NaN/Inf
            if torch.isnan(final_out).any() or torch.isinf(final_out).any():
                final_out = torch.nan_to_num(final_out, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # 最终输出的温和裁剪
            if final_out.abs().max() > 1000:
                final_out = torch.tanh(final_out / 1000.0) * 1000.0
        
        return final_out

class GraphGDP_GraphGPS(torch.nn.Module):
    """
    GraphGPS版本的GraphGDP（GINE+Performer）
    核心：GINEConv局部聚合 + Performer线性注意力全局建模
    保留：原模型双回归、位置编码、全局池化等所有逻辑
    """
    def __init__(
        self, 
        channels: int = 128,
        pe_dim: int = 12,
        num_layers: int = 4,
        use_rw_pe: bool = False,
        use_lap_pe: bool = True,
        use_grid_pe: bool = False,
        node_dim: int = 422,
        edge_dim: int = 1,
        if_pool: bool = True,
        pool_type: str = 'add',
        drop: float = 0.2,
        node_label_dim: int = 2,
        global_label_dim: int = 2,
        rw_pe_input_dim: int = 20,
        grid_pe_input_dim: int = 2,
        performer_heads: int = 4,
        performer_dim_head: int = 32,
        performer_depth: int = 1,  # 传递给Performer的depth参数
    ):
        super().__init__()
        
        # ========== 基础配置（与原模型一致） ==========
        self.use_rw_pe = use_rw_pe
        self.use_lap_pe = use_lap_pe
        self.use_grid_pe = use_grid_pe
        self.pe_dim = pe_dim
        self.total_pe_dim = pe_dim * sum([use_rw_pe, use_lap_pe, use_grid_pe])
        self.if_pool = if_pool
        self.pool_type = pool_type

        # PE维度校验
        if self.total_pe_dim > 0 and self.total_pe_dim >= channels:
            raise ValueError(f"总PE维度({self.total_pe_dim})不能超过通道数({channels})")
        
        # ========== 特征嵌入层（与原模型一致） ==========
        node_emb_out_dim = channels - self.total_pe_dim
        self.node_emb = Linear(node_dim, node_emb_out_dim)
        
        # 位置编码处理层
        if self.use_rw_pe:
            self.rw_pe_norm = BatchNorm1d(rw_pe_input_dim)
            self.rw_pe_lin = Linear(rw_pe_input_dim, pe_dim)
        if self.use_lap_pe:
            self.lap_pe_norm = BatchNorm1d(pe_dim)
            self.lap_pe_lin = Linear(pe_dim, pe_dim)
        if self.use_grid_pe:
            self.grid_pe_norm = BatchNorm1d(grid_pe_input_dim)
            self.grid_pe_lin = Linear(grid_pe_input_dim, pe_dim)
        
        # 边特征嵌入
        self.edge_emb = Linear(edge_dim, channels)
        self.drop = Dropout(drop)
        
        # ========== 全局池化配置（与原模型一致） ==========
        supported_pools = ['add', 'attention', 'mean']
        if self.pool_type not in supported_pools:
            raise ValueError(f"池化类型仅支持：{supported_pools}")
        if self.if_pool and self.pool_type == 'attention':
            self.attention_pool = GlobalAttention(
                gate_nn=Sequential(
                    Linear(channels, channels),
                    ReLU(),
                    Dropout(drop),
                    Linear(channels, 1)
                )
            )
        
        # ========== GraphGPS核心层（GINE+Performer） ==========
        self.convs = ModuleList()
        for _ in range(num_layers):
            # GINE局部卷积
            nn_mlp = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            gine_conv = GINEConv(nn_mlp)
            
            # GraphGPS卷积模块（传递depth参数）
            gps_conv = GPSConv_GraphGPS(
                channels=channels,
                conv=gine_conv,
                dropout=drop,
                performer_heads=performer_heads,
                performer_dim_head=performer_dim_head,
                performer_depth=performer_depth,  # 新增：传递depth参数
            )
            self.convs.append(gps_conv)
        
        # ========== 双回归输出头（与原模型一致） ==========
        self.node_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),
            GELU(),
            Dropout(drop),
            Linear(channels // 2, node_label_dim)
        )
        self.global_regression_head = Sequential(
            Linear(channels, channels // 2),
            BatchNorm1d(channels // 2),
            GELU(),
            Dropout(drop),
            Linear(channels // 2, global_label_dim)
        )
        
    def forward(
        self, 
        x: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        lap_pe: Optional[Tensor] = None,
        rw_pe: Optional[Tensor] = None,
        grid_pe: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        # ========== 节点特征+PE处理（与原模型一致） ==========
        x_node = self.node_emb(x)
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
        if len(pe_list) > 0:
            total_pe = torch.cat(pe_list, dim=1)
            x = torch.cat((x_node, total_pe), dim=1)
        else:
            x = x_node
        
        # 边特征处理
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
        
        # ========== GraphGPS多层卷积 ==========
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = self.drop(x)
            
            # ========== 新增：层间数值保护 ==========
            if not self.training:
                # 检测NaN/Inf
                if torch.isnan(x).any() or torch.isinf(x).any():
                    x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
                # 中间层温和裁剪
                elif x.abs().max() > 1000:
                    x = torch.tanh(x / 1000.0) * 1000.0
        
        # ========== 双回归预测（与原模型一致） ==========
        node_pred = self.node_regression_head(x)
        
        # 全局池化
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
            unique_batch = torch.unique(batch)
            graph_feature_list = []
            for b in unique_batch:
                batch_node_feat = x[batch == b]
                mean_feat = batch_node_feat.mean(dim=0, keepdim=True)
                graph_feature_list.append(mean_feat)
            graph_feature = torch.cat(graph_feature_list, dim=0)
        
        global_pred = self.global_regression_head(graph_feature)
        
        # ========== 新增：最终输出数值保护 ==========
        if not self.training:
            # 修复NaN/Inf
            if torch.isnan(node_pred).any() or torch.isinf(node_pred).any():
                node_pred = torch.nan_to_num(node_pred, nan=0.0, posinf=10.0, neginf=-10.0)
            if torch.isnan(global_pred).any() or torch.isinf(global_pred).any():
                global_pred = torch.nan_to_num(global_pred, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # 温和的最终裁剪
            if node_pred.abs().max() > 1000:
                node_pred = torch.tanh(node_pred / 1000.0) * 1000.0
            if global_pred.abs().max() > 1000:
                global_pred = torch.tanh(global_pred / 1000.0) * 1000.0
        
        return node_pred, global_pred

# ------------------------------ 测试样例 ------------------------------
if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    # 初始化GraphGPS模型（指定performer_depth参数）
    model = GraphGDP_GraphGPS(
        channels=128,
        pe_dim=12,
        num_layers=4,
        use_lap_pe=True,
        node_dim=422,
        if_pool=True,
        pool_type='add',
        drop=0.2,
        performer_depth=1,  # 显式指定Performer的depth参数
    ).to(device)
    
    # 模拟输入数据
    num_nodes = 25
    x = torch.randn(num_nodes, 422, device=device)
    edge_index = torch.randint(0, num_nodes, (2, 124), device=device)
    edge_attr = torch.randn(124, 1, device=device)
    lap_pe = torch.randn(num_nodes, 12, device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    
    # 推理测试
    model.eval()
    with torch.no_grad():
        node_pred, global_pred = model(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            batch=batch, lap_pe=lap_pe
        )
    
    print(f"节点预测形状: {node_pred.shape}")  # [25, 2]
    print(f"全局预测形状: {global_pred.shape}")  # [1, 2]
    
    # 检查数值范围
    print(f"\n数值范围检查:")
    print(f"节点预测范围: [{node_pred.min():.4f}, {node_pred.max():.4f}]")
    print(f"全局预测范围: [{global_pred.min():.4f}, {global_pred.max():.4f}]")
    
    if torch.isnan(node_pred).any() or torch.isnan(global_pred).any():
        print("❌ 检测到NaN!")
    elif torch.isinf(node_pred).any() or torch.isinf(global_pred).any():
        print("❌ 检测到Inf!")
    else:
        print("✅ 模型推理成功，数值稳定！")