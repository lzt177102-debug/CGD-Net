# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add
import numpy as np
import os
import json
import yaml
import argparse
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
from osgeo import gdal, osr  # 确保导入osr模块
# 导入数据加载器与模型（复用训练时的组件）
from RSGCN_DataLoader import GraphDataBuilder, GraphDataLoaderManager, GraphPatchDataset
from src.networks.GraphMamba import GraphGDP
from src.networks.GCN import GraphGDP_GCN
from src.networks.GraphGPS import GraphGDP_GraphGPS
from src.networks.Graphormer import GraphGDP_Graphormer
from feature_engineering import extract_features_from_paired_dataset

# ==================== 工具函数（无新增，仅适配伪标签提取） ====================
def get_device():
    """获取推理设备（优先GPU）"""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def makedirs(path):
    """创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_model(model, checkpoint_path, device, multi_gpu=False):
    """加载推理模型（兼容多卡权重）"""
    pretrain = torch.load(checkpoint_path, map_location=device)
    state_dict = pretrain.get('model_state_dict', pretrain.get('state_dict', pretrain))
    
    # 移除多卡前缀
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def read_tif(file_path):
    """读取TIFF文件（日间遥感/夜间灯光等），返回数据和完整地理信息（以日间遥感为标准）"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"无法打开TIFF文件：{file_path}")
    data = dataset.ReadAsArray()
    
    # 严格读取日间遥感的完整地理信息（核心）
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    # 强制校验：确保读取的是有完整地理信息的日间遥感图
    if geo_transform is None or projection is None:
        raise ValueError(f"错误：日间遥感TIFF文件 {file_path} 缺少关键地理信息！")
    
    geo_info = {
        'transform': geo_transform,  # 完全复用日间遥感的地理变换
        'projection': projection,    # 完全复用日间遥感的投影信息
        'width': dataset.RasterXSize,
        'height': dataset.RasterYSize
    }
    dataset = None
    return data, geo_info

def write_tif(data, geo_info, output_path):
    """保存TIFF文件（严格以日间遥感的地理信息为标准，取消NoData设置）"""
    driver = gdal.GetDriverByName('GTiff')
    if len(data.shape) == 2:
        bands = 1
        data = data[np.newaxis, :, :]
    else:
        bands = data.shape[0]
    
    # 1. 先删除已有文件（避免GDAL缓存导致地理信息写入失败）
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 2. 严格按照日间遥感的尺寸和地理信息创建TIFF
    dataset = driver.Create(
        output_path,
        geo_info['width'],          # 复用日间遥感宽度
        geo_info['height'],         # 复用日间遥感高度
        bands,
        gdal.GDT_Float32            # 保持和你原有代码一致的精度
    )
    
    # 3. 强制写入日间遥感的地理信息（核心修复）
    dataset.SetGeoTransform(geo_info['transform'])  # 完全继承日间遥感的坐标变换
    dataset.SetProjection(geo_info['projection'])   # 完全继承日间遥感的投影
    
    # 4. 写入数据（取消NoData设置，完全按你的原始逻辑）
    for i in range(bands):
        dataset.GetRasterBand(i+1).WriteArray(data[i])
    
    dataset.FlushCache()
    dataset = None
    
    # 验证：确保地理信息写入成功
    verify_ds = gdal.Open(output_path)
    if verify_ds:
        if verify_ds.GetGeoTransform() == geo_info['transform'] and verify_ds.GetProjection() == geo_info['projection']:
            print(f"✅ TIFF文件已保存（地理信息与日间遥感完全一致）：{output_path}")
        else:
            print(f"⚠️ 警告：{output_path} 地理信息与日间遥感不一致！")
        verify_ds = None
    else:
        print(f"❌ 错误：{output_path} 保存失败！")

def restore_gdp(log_gdp: np.ndarray) -> np.ndarray:
    """从log(1+GDP)还原原始GDP"""
    raw_gdp = np.expm1(log_gdp)
    raw_gdp[raw_gdp < 0] = 0
    return raw_gdp

def normalize_to_255(data: np.ndarray) -> np.ndarray:
    """仅对GDP矩阵归一化到0-255（生成热力图TIFF）"""
    if np.max(data) == np.min(data):
        return np.zeros_like(data, dtype=np.float32)
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    norm_data = np.clip(norm_data, 0, 255)
    return norm_data.astype(np.float32)

# ==================== 核心推理类（提取图构建阶段的伪标签并保存） ====================
class GraphGDPInferencer:
    """GraphGDP模型推理类（提取图构建阶段生成的伪标签）"""
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.output_dir = makedirs(config['infer_output_dir'])
        
        # 执行特征提取并获取行列分开的分辨率比例
        self.features, self.county_node_sizes = extract_features_from_paired_dataset(
            model_path=self.config['model_path'],
            remote_sensing_dir=self.config['remote_sensing_dir'],
            nl_dir=self.config['nl_dir'],
            landuse_dir=self.config['landuse_dir'],
            population_dir=self.config['population_dir'],
            output_dir=self.config['output_dir'],
            model_name=self.config['model_name'],
            num_processes=1,
            poi_dir=self.config['poi_dir'],
            target_county=self.config['infer_county']
        )
        
        # 获取当前推理县的行列比例
        if self.config['infer_county'] in self.county_node_sizes:
            self.rs_to_nl_ratio_row, self.rs_to_nl_ratio_col = self.county_node_sizes[self.config['infer_county']]
        else:
            self.rs_to_nl_ratio_row = 30
            self.rs_to_nl_ratio_col = 30
        
        print(f"🔍 分辨率比例（行列分开）：{self.rs_to_nl_ratio_row}×{self.rs_to_nl_ratio_col}")

        # 加载模型、数据集、scaler
        self.model = self._build_model()
        self.infer_dataset, self.geo_info, self.rs_data = self._build_infer_dataset()
        self.scaler = self._load_scaler()
        
        # ========== 核心新增：提取图构建阶段生成的伪标签矩阵 ==========
        self.pseudo_label_matrix = self._extract_pseudo_label_from_dataset()
    
    def _build_model(self):
        """构建并加载推理模型"""
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
        return model
    
    def _load_scaler(self):
        """加载训练时的scaler"""
        scaler_path = self.config['scaler_path']
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"scaler文件不存在：{scaler_path}")
        scaler = torch.load(scaler_path, map_location='cpu', weights_only=False)
        # 兼容单独保存的scaler或数据集里的scaler
        scaler = scaler['scaler'] if 'scaler' in scaler else scaler
        print(f"✅ scaler加载完成：{scaler_path}")
        return scaler
    
    def _build_infer_dataset(self):
        """构建指定县的推理数据集（复用图构建逻辑）"""
        print(f"\n===== 加载推理数据（{self.config['infer_county']}） =====")
        # 读取原始遥感图（仅用于获取地理信息）
        rs_tif_path = self.config['rs_tif_path']
        rs_data, geo_info = read_tif(rs_tif_path)
        print(f"✅ 原始遥感图加载完成：{rs_tif_path} (尺寸：{geo_info['height']}×{geo_info['width']})")
        
        # 构建图数据集（复用训练时的GraphDataBuilder）
        builder = GraphDataBuilder(
            gdp_file_path=self.config['gdp_file_path'],
            patch_size=self.config['patch_size'],
            lap_pe_k=self.config['lap_pe_k']
        )
        
        # 构建仅包含目标县的推理数据集
        dataset = builder.build_graph_dataset(
            features_dir=self.config.get('features_dir', './dataset/extracted_features_90'),
            output_dir=None,  # 推理时不保存数据集
            stride=self.config.get('stride', 6),
            max_counties=None,
            random_patches=False,
            min_nodes_threshold=self.config.get('min_nodes_threshold', 5),
            target_county=self.config['infer_county']  # 仅处理目标县
        )
        
        if len(dataset) == 0:
            raise ValueError(f"未找到{self.config['infer_county']}的图数据")
        
        print(f"✅ 推理数据集加载完成：{len(dataset)} 个图块")
        return dataset, geo_info, rs_data
    
    def _extract_pseudo_label_from_dataset(self):
        """
        修复后：提取图构建阶段的伪标签矩阵（和预测值坐标映射逻辑完全对齐）
        """
        print(f"\n===== 提取图构建阶段的伪标签（{self.config['infer_county']}） =====")
        
        # 初始化伪标签矩阵（和遥感图尺寸一致）
        pseudo_label_matrix = np.zeros((self.geo_info['height'], self.geo_info['width']), dtype=np.float32)
        
        total_nodes = 0
        valid_fill = 0
        non_zero_label_count = 0
        
        # 遍历所有图块，提取伪标签并填充到矩阵
        for patch_idx, patch in enumerate(tqdm(self.infer_dataset, desc="提取伪标签")):
            # 1. 校验核心属性是否存在（和GraphDataBuilder输出对齐）
            if not hasattr(patch, 'pos') or patch.pos is None:
                print(f"⚠️ 图块 {patch_idx} 无pos属性，跳过")
                continue
            if not hasattr(patch, 'y_node') or patch.y_node is None:
                print(f"⚠️ 图块 {patch_idx} 无y_node属性，跳过")
                continue
            
            # 2. 获取节点位置和伪标签（和GraphDataBuilder生成的格式对齐）
            node_positions = patch.pos.numpy()  # [num_nodes, 2] (网格行, 网格列)
            node_pseudo_labels = patch.y_node[:, 0].numpy()  # 取第0列：原始grid_gdp（非log值）
            
            # 调试：打印前3个图块的关键信息
            if patch_idx < 3:
                print(f"\n📌 图块 {patch_idx} 调试信息：")
                print(f"   节点数：{len(node_positions)}")
                print(f"   伪标签范围：{node_pseudo_labels.min():.4f} ~ {node_pseudo_labels.max():.4f}")
                print(f"   非零伪标签数：{np.count_nonzero(node_pseudo_labels)}")
                print(f"   前5个节点位置：{node_positions[:5]}")
                print(f"   前5个伪标签值：{node_pseudo_labels[:5]}")
            
            total_nodes += len(node_positions)
            non_zero_label_count += np.count_nonzero(node_pseudo_labels)
            
            # 3. 填充矩阵（关键修复：和预测值用完全相同的坐标缩放逻辑）
            for i, (grid_row, grid_col) in enumerate(node_positions):
                label_value = node_pseudo_labels[i]
                if label_value <= 0:  # 过滤无效标签
                    continue
                
                # 核心修复：网格坐标 → 像素坐标（和预测值逻辑一致）
                pixel_row = int(grid_row * self.rs_to_nl_ratio_row)
                pixel_col = int(grid_col * self.rs_to_nl_ratio_col)
                
                # 计算像素块范围（覆盖整个网格对应的像素区域）
                row_end = min(pixel_row + self.rs_to_nl_ratio_row, self.geo_info['height'])
                col_end = min(pixel_col + self.rs_to_nl_ratio_col, self.geo_info['width'])
                
                # 校验坐标是否在矩阵范围内
                if 0 <= pixel_row < self.geo_info['height'] and 0 <= pixel_col < self.geo_info['width']:
                    # 填充整个像素块（而非单个像素），确保值能显示
                    pseudo_label_matrix[pixel_row:row_end, pixel_col:col_end] = label_value
                    valid_fill += 1
        
        # 最终统计
        print(f"\n✅ 伪标签提取完成：")
        print(f"   总处理节点数：{total_nodes}")
        print(f"   非零伪标签数：{non_zero_label_count}")
        print(f"   有效填充像素块数：{valid_fill}")
        print(f"   伪标签矩阵非零像素数：{np.count_nonzero(pseudo_label_matrix)}")
        print(f"   伪标签矩阵值范围：{pseudo_label_matrix.min():.4f} ~ {pseudo_label_matrix.max():.4f}")
        
        return pseudo_label_matrix
    
    def _process_batch(self, batch):
        """处理推理批次数据"""
        batch = batch.to(self.device)
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None and batch.edge_attr.nelement() > 0:
            edge_attr = batch.edge_attr
        else:
            edge_attr = torch.zeros((edge_index.shape[1], self.config['edge_dim']), device=self.device)
        
        if hasattr(batch, 'lap_pe') and batch.lap_pe is not None:
            lap_pe = batch.lap_pe
        else:
            lap_pe = torch.zeros((batch.num_nodes, self.config['pe_dim']), device=self.device)
        
        return x, edge_index, edge_attr, lap_pe, batch_idx, batch.pos
    
    def infer(self):
        """核心推理流程"""
        print("\n===== 开始推理 =====")
        self.model.eval()
        infer_loader = DataLoader(
            self.infer_dataset,
            batch_size=self.config['infer_batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        all_node_preds = []
        all_positions = []
        
        with torch.no_grad():
            for batch in tqdm(infer_loader, desc="推理进度"):
                x, edge_index, edge_attr, lap_pe, batch_idx, pos = self._process_batch(batch)
                
                # 前向推理
                node_pred, global_pred = self.model(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx, lap_pe=lap_pe
                )
                # node_pred = torch.clamp(node_pred, min=0.0, max=14.0)  # 关键！阻止数值爆炸
                # # 过滤NaN/Inf（Performer易出现，双重兜底）
                # node_pred = torch.nan_to_num(node_pred, nan=0.0, posinf=14.0, neginf=0.0)
                # 提取log(1+GDP)列
                node_log_gdp = node_pred[:, 1:2]
                
                all_node_preds.append(node_log_gdp.cpu().numpy())
                all_positions.append(pos.cpu().numpy())
        
        # 合并结果
        node_preds = np.vstack(all_node_preds)
        positions = np.vstack(all_positions)
        
        # 还原原始GDP（模型预测值）
        raw_gdp = restore_gdp(node_preds.squeeze())
        
        # 构建GDP分布矩阵（预测值）
        gdp_matrix = np.zeros((self.geo_info['height'], self.geo_info['width']), dtype=np.float32)
        

        
        # 行列分开填充矩阵
        for i, (row, col) in enumerate(positions):
            pixel_row = int(row * self.rs_to_nl_ratio_row)
            pixel_col = int(col * self.rs_to_nl_ratio_col)
            row_end = min(pixel_row + self.rs_to_nl_ratio_row, self.geo_info['height'])
            col_end = min(pixel_col + self.rs_to_nl_ratio_col, self.geo_info['width'])
            
            if pixel_row < self.geo_info['height'] and pixel_col < self.geo_info['width']:
                gdp_matrix[pixel_row:row_end, pixel_col:col_end] = raw_gdp[i]
        
        # 生成热力图矩阵（预测值）
        gdp_heatmap_matrix = normalize_to_255(gdp_matrix)
        
        # 生成伪标签热力图矩阵（用于对比）
        pseudo_label_heatmap_matrix = normalize_to_255(self.pseudo_label_matrix)
        
        # 保存结果（含预测值+伪标签TIFF）
        self._save_results(
            node_preds, raw_gdp, gdp_matrix, gdp_heatmap_matrix,
            self.pseudo_label_matrix, pseudo_label_heatmap_matrix
        )
        
        print("\n===== 推理完成 =====")
        print(f"📊 结果汇总：")
        print(f"   - 模型预测GDP范围：{np.min(raw_gdp):.2f} ~ {np.max(raw_gdp):.2f} 万元")
        print(f"   - 伪标签GDP范围：{np.min(self.pseudo_label_matrix):.2f} ~ {np.max(self.pseudo_label_matrix):.2f} 万元")
        print(f"   - 热力图范围：0.0 ~ 255.0（地理信息一致）")
        print(f"   - 结果保存目录：{self.output_dir}")
        print(f"   - 输出文件：")
        print(f"     ✔ {self.config['infer_county']}_raw_gdp.tif（模型预测）")
        print(f"     ✔ {self.config['infer_county']}_gdp_heatmap.tif（预测热力图）")
        print(f"     ✔ {self.config['infer_county']}_pseudo_label.tif（伪标签）")
        print(f"     ✔ {self.config['infer_county']}_pseudo_label_heatmap.tif（伪标签热力图）")
        
        return {
            'raw_gdp_matrix': gdp_matrix,
            'raw_gdp_tif_path': os.path.join(self.output_dir, f"{self.config['infer_county']}_raw_gdp.tif"),
            'gdp_heatmap_matrix': gdp_heatmap_matrix,
            'gdp_heatmap_tif_path': os.path.join(self.output_dir, f"{self.config['infer_county']}_gdp_heatmap.tif"),
            'pseudo_label_matrix': self.pseudo_label_matrix,
            'pseudo_label_tif_path': os.path.join(self.output_dir, f"{self.config['infer_county']}_pseudo_label.tif"),
            'pseudo_label_heatmap_tif_path': os.path.join(self.output_dir, f"{self.config['infer_county']}_pseudo_label_heatmap.tif")
        }
    
    def _save_results(self, node_preds, raw_gdp, gdp_matrix, gdp_heatmap_matrix,
                     pseudo_label_matrix, pseudo_label_heatmap_matrix):
        """保存所有推理结果（新增伪标签TIFF）"""
        # 1. 保存矩阵文件（npy）
        np.save(os.path.join(self.output_dir, f"{self.config['infer_county']}_log_gdp_node_preds.npy"), node_preds)
        np.save(os.path.join(self.output_dir, f"{self.config['infer_county']}_raw_gdp_matrix.npy"), gdp_matrix)
        np.save(os.path.join(self.output_dir, f"{self.config['infer_county']}_gdp_heatmap_matrix.npy"), gdp_heatmap_matrix)
        np.save(os.path.join(self.output_dir, f"{self.config['infer_county']}_pseudo_label_matrix.npy"), pseudo_label_matrix)
        
        # 2. 保存模型预测GDP TIFF
        write_tif(
            gdp_matrix,
            self.geo_info,
            os.path.join(self.output_dir, f"{self.config['infer_county']}_raw_gdp.tif")
        )
        
        # 3. 保存预测热力图 TIFF
        write_tif(
            gdp_heatmap_matrix,
            self.geo_info,
            os.path.join(self.output_dir, f"{self.config['infer_county']}_gdp_heatmap.tif")
        )
        
        # 4. 保存伪标签 TIFF（核心新增）
        write_tif(
            pseudo_label_matrix,
            self.geo_info,
            os.path.join(self.output_dir, f"{self.config['infer_county']}_pseudo_label.tif")
        )
        
        # 5. 保存伪标签热力图 TIFF（用于可视化对比）
        write_tif(
            pseudo_label_heatmap_matrix,
            self.geo_info,
            os.path.join(self.output_dir, f"{self.config['infer_county']}_pseudo_label_heatmap.tif")
        )
        
        # 6. 保存统计信息（新增伪标签统计）
        stats = {
            'county': self.config['infer_county'],
            'log_gdp_mean': float(np.mean(node_preds)),
            'log_gdp_std': float(np.std(node_preds)),
            # 模型预测统计
            'pred_raw_gdp_mean': float(np.mean(raw_gdp)),
            'pred_raw_gdp_std': float(np.std(raw_gdp)),
            'pred_raw_gdp_min': float(np.min(raw_gdp)),
            'pred_raw_gdp_max': float(np.max(raw_gdp)),
            # 伪标签统计
            'pseudo_label_mean': float(np.mean(pseudo_label_matrix[pseudo_label_matrix > 0])),
            'pseudo_label_std': float(np.std(pseudo_label_matrix[pseudo_label_matrix > 0])),
            'pseudo_label_min': float(np.min(pseudo_label_matrix)),
            'pseudo_label_max': float(np.max(pseudo_label_matrix)),
            'pseudo_label_nonzero_count': int(np.count_nonzero(pseudo_label_matrix)),
            # 其他信息
            'rs_to_nl_ratio_row': self.rs_to_nl_ratio_row,
            'rs_to_nl_ratio_col': self.rs_to_nl_ratio_col,
            'infer_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'note': 'pseudo_label是图构建阶段生成的原始伪标签，非模型预测值'
        }
        
        with open(os.path.join(self.output_dir, f"{self.config['infer_county']}_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 所有结果文件已保存（含伪标签TIFF）")

# ==================== 配置加载 + 主函数 ====================
def load_infer_config(config_path):
    """加载推理YAML配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 补充默认值
    default_config = {
        'infer_batch_size': 8,
        'multi_gpu': False,
        'infer_output_dir': './infer_results',
        'num_workers': 0,
        'patch_size': 12,
        'lap_pe_k': 12,
        'stride': 6,
        'min_nodes_threshold': 5
    }
    for k, v in default_config.items():
        if k not in config:
            config[k] = v
    
    # 校验必填参数（新增scaler_path）
    required = [
        'checkpoint_path', 'scaler_path', 'infer_county',
        'rs_tif_path', 'gdp_file_path', 'patch_size', 'lap_pe_k',
        'model_path', 'remote_sensing_dir', 'nl_dir', 'landuse_dir',
        'population_dir', 'output_dir', 'model_name', 'poi_dir',
        'features_dir'  # 新增：特征文件目录
    ]
    for k in required:
        if k not in config:
            raise ValueError(f"配置缺失必填项：{k}")
    
    return config

def main(args):
    """推理主函数"""
    config = load_infer_config(args.config_path)
    print("="*60)
    print(f"推理配置：")
    print(f"   - 推理县：{config['infer_county']}")
    print(f"   - 模型权重：{config['checkpoint_path']}")
    print(f"   - 遥感图路径：{config['rs_tif_path']}")
    print(f"   - 输出目录：{config['infer_output_dir']}")
    print(f"   - 输出格式：TIFF（模型预测 + 伪标签）")
    print("="*60)
    
    inferencer = GraphGDPInferencer(config)
    results = inferencer.infer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphGDP模型推理（提取图构建阶段伪标签）')
    parser.add_argument('--config_path', type=str, default='config/GraphGDP_infer.yaml',
                        help='推理YAML配置文件路径')
    args = parser.parse_args()
    main(args)