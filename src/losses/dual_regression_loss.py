import torch
import torch.nn as nn
from torch_scatter import scatter_add
from typing import Tuple


class DualRegressionLoss(nn.Module):
    """
    双回归损失：
    - 节点 MSE
    - 全局 MSE
    - 50轮后启用节点-全局一致性损失
    """
    def __init__(self, node_weight, global_weight, consistency_weight):
        super().__init__()
        self.node_weight = node_weight
        self.global_weight = global_weight
        self.consistency_weight = consistency_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        node_pred,
        node_target,
        global_pred,
        global_target,
        batch_idx,
        enable_consistency=False
    ) -> Tuple[torch.Tensor, ...]:

        node_loss = self.mse(node_pred, node_target)
        global_loss = self.mse(global_pred, global_target)

        consistency_loss = torch.zeros(1, device=node_pred.device)

        if enable_consistency:
            eps = 1e-8
            node_raw = torch.expm1(node_pred + eps)
            node_sum_raw = scatter_add(node_raw, batch_idx, dim=0)
            node_sum_log = torch.log1p(node_sum_raw + eps)
            consistency_loss = self.mse(node_sum_log, global_pred)

        total_loss = (
            self.node_weight * node_loss +
            self.global_weight * global_loss +
            (self.consistency_weight * consistency_loss if enable_consistency else 0.0)
        )

        return total_loss, node_loss, global_loss, consistency_loss