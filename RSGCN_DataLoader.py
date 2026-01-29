import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
# 优先使用PyG专用DataLoader，自动处理图批次拼接，避免tuple报错
from torch_geometric.loader import DataLoader as PyGDataLoader
# 单独导入完整版 get_laplacian（支持 normalized 参数）
from torch_geometric.utils.laplacian import get_laplacian
# 其他工具函数保持原有导入
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.linalg import eigs
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import random
from typing import List, Tuple, Dict, Optional
warnings.filterwarnings('ignore')


class GraphPatchDataset(Dataset):
    """图块数据集类，继承自PyG Dataset"""
    
    def __init__(self, patches: List[Data], patch_gdps: List[float], 
                 transform=None, pre_transform=None):
        """
        初始化图块数据集
        
        Args:
            patches: 图块数据列表
            patch_gdps: 图块GDP值列表（原始值）
            transform: 数据变换
            pre_transform: 预变换
        """
        super().__init__(transform=transform, pre_transform=pre_transform)
        self.patches = patches
        self.patch_gdps = patch_gdps  # 保存原始GDP值
        
        # 为每个图块设置两个标签：原始GDP和log(1+GDP)（图块级全局标签）
        # 同时保留节点级标签（patch.y_node 已在构建时存入）
        for patch, gdp in zip(self.patches, self.patch_gdps):
            # 计算log(1+GDP)
            log_gdp = np.log1p(gdp)
            
            # 设置图块级全局标签：原始值和log值（保持原有逻辑）
            patch.y = torch.tensor([[gdp, log_gdp]], dtype=torch.float)
        
        # 创建索引映射
        self._indices = list(range(len(self.patches)))
    
    def len(self):
        """PyG Dataset要求的len方法"""
        return len(self.patches)
    
    def __len__(self):
        """Python标准的__len__方法"""
        return len(self.patches)
    
    def get(self, idx):
        """PyG Dataset要求的get方法（核心：返回单个PyG Data对象，避免tuple）"""
        return self.patches[idx]
    
    def __getitem__(self, idx):
        """支持索引访问（返回单个PyG Data对象，兼容DataLoader）"""
        if isinstance(idx, slice):
            return GraphPatchDataset(self.patches[idx], self.patch_gdps[idx])
        return self.patches[idx]
    
    def indices(self):
        """PyG Dataset要求的indices方法"""
        return self._indices
    
    def split_by_county(self, county_names: List[str], test_size: float = 0.2, 
                       random_state: int = 42):
        """
        按县划分数据集（防止数据泄露）
        
        Args:
            county_names: 每个图块所属的县名列表（长度需与patches相同）
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            train_dataset, val_dataset, test_dataset: 划分后的数据集
        """
        # 校验输入长度
        if len(county_names) != len(self.patches):
            raise ValueError("县名列表长度必须与图块列表长度一致")
        
        # 获取所有唯一的县
        unique_counties = list(set(county_names))
        
        # 划分训练县和测试县
        train_counties, test_counties = train_test_split(
            unique_counties, test_size=test_size, random_state=random_state
        )
        
        # 从测试县中再划分验证县
        test_counties, val_counties = train_test_split(
            test_counties, test_size=0.5, random_state=random_state
        )
        
        # 根据县划分数据集
        train_indices = [i for i, county in enumerate(county_names) 
                        if county in train_counties]
        val_indices = [i for i, county in enumerate(county_names) 
                      if county in val_counties]
        test_indices = [i for i, county in enumerate(county_names) 
                       if county in test_counties]
        
        # 创建子数据集
        train_dataset = self.subset(train_indices)
        val_dataset = self.subset(val_indices)
        test_dataset = self.subset(test_indices)
        
        print(f"📊 按县划分数据集:")
        print(f"  训练县: {len(train_counties)} 个, 图块: {len(train_indices)} 个")
        print(f"  验证县: {len(val_counties)} 个, 图块: {len(val_indices)} 个")
        print(f"  测试县: {len(test_counties)} 个, 图块: {len(test_indices)} 个")
        
        return train_dataset, val_dataset, test_dataset
    
    def split_random(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                    test_ratio: float = 0.15, random_state: int = 42):
        """
        随机划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            train_dataset, val_dataset, test_dataset: 划分后的数据集
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        # 设置随机种子
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        # 打乱索引
        indices = self._indices.copy()
        random.shuffle(indices)
        
        # 计算划分点
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # 划分索引
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # 创建子数据集
        train_dataset = self.subset(train_indices)
        val_dataset = self.subset(val_indices)
        test_dataset = self.subset(test_indices)
        
        print(f"📊 随机划分数据集:")
        print(f"  训练集: {len(train_indices)} 个图块 ({train_ratio*100:.1f}%)")
        print(f"  验证集: {len(val_indices)} 个图块 ({val_ratio*100:.1f}%)")
        print(f"  测试集: {len(test_indices)} 个图块 ({test_ratio*100:.1f}%)")
        
        return train_dataset, val_dataset, test_dataset
    
    def subset(self, indices: List[int]):
        """创建子集（返回GraphPatchDataset实例，保持数据结构一致）"""
        if not indices:
            return GraphPatchDataset([], [])
        
        subset_patches = [self.patches[i] for i in indices]
        subset_gdps = [self.patch_gdps[i] for i in indices]
        
        return GraphPatchDataset(subset_patches, subset_gdps)
    
    def get_statistics(self):
        """获取数据集统计信息（新增节点级标签统计）"""
        if len(self.patches) == 0:
            return {}
        
        gdps = np.array(self.patch_gdps)
        log_gdps = np.log1p(gdps)  # 计算log(1+GDP)
        num_nodes = [patch.num_nodes for patch in self.patches]
        num_edges = [patch.edge_index.shape[1] for patch in self.patches]
        
        # 新增：节点级标签统计
        node_gdp_list = []
        node_log_gdp_list = []
        for patch in self.patches:
            if hasattr(patch, 'y_node') and patch.y_node is not None:
                # 提取该图块所有节点的GDP和logGDP
                node_gdps = patch.y_node[:, 0].numpy()
                node_log_gdps = patch.y_node[:, 1].numpy()
                node_gdp_list.extend(node_gdps)
                node_log_gdp_list.extend(node_log_gdps)
        
        # 修复原代码中的typo（max_log_gdp误写为max_node_log_gdp）
        min_log_gdp = float(np.min(log_gdps))
        max_log_gdp = float(np.max(log_gdps))
        
        # 将所有numpy类型转换为Python原生类型
        stats = {
            'num_patches': int(len(self.patches)),
            'avg_nodes': float(np.mean(num_nodes)),
            'std_nodes': float(np.std(num_nodes)),
            'min_nodes': int(np.min(num_nodes)),
            'max_nodes': int(np.max(num_nodes)),
            'avg_edges': float(np.mean(num_edges)),
            'avg_gdp': float(np.mean(gdps)),
            'std_gdp': float(np.std(gdps)),
            'min_gdp': float(np.min(gdps)),
            'max_gdp': float(np.max(gdps)),
            'avg_log_gdp': float(np.mean(log_gdps)),
            'std_log_gdp': float(np.std(log_gdps)),
            'min_log_gdp': min_log_gdp,
            'max_log_gdp': max_log_gdp,
            'feature_dim': int(self.patches[0].x.shape[1] if self.patches else 0),
            'has_lappe': bool(hasattr(self.patches[0], 'lap_pe') and self.patches[0].lap_pe is not None) if self.patches else False,
            # 新增：节点级标签统计
            'has_node_labels': bool(len(node_gdp_list) > 0),
            'avg_node_gdp': float(np.mean(node_gdp_list)) if node_gdp_list else 0.0,
            'std_node_gdp': float(np.std(node_gdp_list)) if node_gdp_list else 0.0,
            'min_node_gdp': float(np.min(node_gdp_list)) if node_gdp_list else 0.0,
            'max_node_gdp': float(np.max(node_gdp_list)) if node_gdp_list else 0.0,
            'avg_node_log_gdp': float(np.mean(node_log_gdp_list)) if node_log_gdp_list else 0.0,
            'std_node_log_gdp': float(np.std(node_log_gdp_list)) if node_log_gdp_list else 0.0,
        }
        
        return stats


class GraphDataBuilder:
    """图数据构建器：整合图构建、图块生成和伪标签生成（新增节点级标签）"""
    
    def __init__(self, gdp_file_path, patch_size=16, lap_pe_k=10):
        """
        初始化图数据构建器
        
        Args:
            gdp_file_path: GDP数据文件路径
            patch_size: 图块尺寸（默认16×16）
            lap_pe_k: LapPE编码的特征维度（取前k个最小非零特征值对应的特征向量）
        """
        self.gdp_file_path = gdp_file_path
        self.patch_size = patch_size
        self.lap_pe_k = lap_pe_k  # LapPE 特征维度
        
        # 修复原代码typo：is_fitted（原代码为is_fitted）
        self.is_fitted = False
        self.feature_columns_to_scale = None
        self.all_feature_columns = None
        self.cnn_feature_columns = None
        
        # 存储数据
        self.all_dataframes = {}
        self.patch_county_mapping = []  # 初始化县映射（修复原有未定义问题）
        self.scaler = StandardScaler()  # 显式初始化标准化器，避免属性不存在报错
    
    # ==================== LapPE 编码部分（保持原有逻辑，无修改） ====================
    def compute_laplacian_positional_encoding(self, graph_data: Data) -> torch.Tensor:
        """
        计算图的拉普拉斯位置编码（LapPE）- 正确使用 normalization 参数
        """
        num_nodes = graph_data.num_nodes
        
        # 处理空图或单节点图（无法计算拉普拉斯矩阵）
        if num_nodes <= 1 or graph_data.edge_index.shape[1] == 0:
            return torch.zeros((num_nodes, self.lap_pe_k), dtype=torch.float)
        
        # 步骤1：计算对称归一化拉普拉斯矩阵（正确使用 normalization="sym"）
        laplacian_edge_index, laplacian_edge_weight = get_laplacian(
            edge_index=graph_data.edge_index,
            normalization="sym",  # 替换 normalized=True 为 normalization="sym"，实现对称归一化
            num_nodes=num_nodes   # 传入节点数提高鲁棒性
        )
        
        # 转换为Scipy稀疏矩阵格式（核心修改：位置参数传入边权重）
        laplacian_sparse = to_scipy_sparse_matrix(
            laplacian_edge_index,  # 第1个位置参数：边索引（必传）
            laplacian_edge_weight, # 第2个位置参数：边权重（去掉 edge_weight= 关键字）
            num_nodes=num_nodes    # 第3个参数：节点数（合法关键字参数，保留）
        )
        
        try:
            eigvals, eigvecs = eigs(
                laplacian_sparse,
                k=self.lap_pe_k + 1,
                which='SM',
                return_eigenvectors=True
            )
            
            eigvals_real = np.real(eigvals)
            eigvecs_real = np.real(eigvecs)
            
            non_zero_mask = eigvals_real > 1e-8
            non_zero_eigvals = eigvals_real[non_zero_mask]
            non_zero_eigvecs = eigvecs_real[:, non_zero_mask]
            
            if len(non_zero_eigvals) >= self.lap_pe_k:
                sorted_indices = np.argsort(non_zero_eigvals)[:self.lap_pe_k]
                lap_pe = non_zero_eigvecs[:, sorted_indices]
            else:
                lap_pe = np.zeros((num_nodes, self.lap_pe_k), dtype=np.float64)
                lap_pe[:, :len(non_zero_eigvals)] = non_zero_eigvecs[:, :len(non_zero_eigvals)]
            
            return torch.tensor(lap_pe, dtype=torch.float)
        
        except Exception as e:
            print(f"⚠️ 计算LapPE编码失败: {e}，返回零矩阵")
            return torch.zeros((num_nodes, self.lap_pe_k), dtype=torch.float)
    
    def merge_lap_pe_with_node_features(self, graph_data: Data) -> Data:
        """
        将LapPE编码与原始节点特征合并（可选：替换/拼接）
        此处采用拼接方式，保持原始特征不变，新增位置编码信息
        """
        # 计算LapPE编码
        lap_pe = self.compute_laplacian_positional_encoding(graph_data)
        
        # 存储LapPE编码作为图的独立属性（推荐，不破坏原始特征）
        graph_data.lap_pe = lap_pe
        
        return graph_data
    
    # ==================== GDP数据加载部分（保持原有逻辑，无修改） ====================
    def load_county_gdp_dict(self):
        """加载县GDP数据字典"""
        try:
            gdp_df = pd.read_excel(self.gdp_file_path)
            
            gdp_dict = {}
            for _, row in gdp_df.iterrows():
                region_code = row.iloc[0]
                gdp_2020 = row.iloc[1]
                
                if pd.notna(region_code) and pd.notna(gdp_2020) and gdp_2020 > 0:
                    try:
                        code_int = int(region_code)
                        gdp_dict[code_int] = float(gdp_2020)
                    except (ValueError, TypeError):
                        continue
            
            print(f"✅ GDP数据加载完成: {len(gdp_dict)} 个县")
            return gdp_dict
            
        except Exception as e:
            print(f"❌ 读取GDP文件失败: {e}")
            return {}
    
    def match_county_names(self, feature_files, gdp_dict):
        """匹配特征文件名和行政区域代码，输出未匹配成功的县名"""
        matched_data = {}
        unmatched_counties = []  # 新增：收集未匹配成功的县名
        
        for feature_file in feature_files:
            county_name = feature_file.replace('_features.csv', '')
            
            try:
                code_int = int(county_name)
                if code_int in gdp_dict:
                    matched_data[county_name] = gdp_dict[code_int]
                else:
                    # 新增：县名可转为数字，但不在gdp_dict中（未匹配）
                    unmatched_counties.append(county_name)
            except ValueError:
                # 新增：县名无法转为数字（格式错误，未匹配）
                unmatched_counties.append(county_name)
                continue
        
        # 原有：打印匹配成功信息
        print(f"✅ 县名匹配完成: 成功 {len(matched_data)} 个县")
        
        # 新增：打印未匹配信息（分情况，更友好）
        if unmatched_counties:
            # 去重（避免重复文件导致重复记录），排序（方便查看）
            unique_unmatched = sorted(list(set(unmatched_counties)))
            print(f"❌ 未匹配成功的县名共 {len(unique_unmatched)} 个：")
            # 格式化输出，每行显示5个，避免过长刷屏
            for i in range(0, len(unique_unmatched), 5):
                batch = unique_unmatched[i:i+5]
                print(f"   {' | '.join(batch)}")
        else:
            print(f"🎉 所有县名都匹配成功，无遗漏！")
        
        return matched_data
    
    # ==================== 特征处理部分（保持原有逻辑，无修改） ====================
    def _get_feature_columns(self, df):
        # """获取所有特征列，并区分需要标准化的列和CNN特征列"""
        # exclude_cols = ['county_name', 'position_row', 'position_col', 
        #                'grid_gdp', 'grid_gdp_log', 'weight', 'nl_pop_product']
        # exclude_cols = ['county_name', 'position_row', 'position_col', 
        #            'grid_gdp', 'grid_gdp_log', 'weight', 'nl_pop_product',
        #            'nl_intensity', 'population_density']

        # 1. 定义基础排除列（你原本指定的）
        exclude_cols = [
            'county_name', 'position_row', 'position_col', 
            'grid_gdp', 'grid_gdp_log', 'weight', 'nl_pop_product'
        ]

        # 2. 手动列出所有POI列（适配你给出的列名，确保无遗漏）
        poi_cols = [
            'poi_total_count', 'poi_餐饮美食', 'poi_公司企业', 'poi_购物消费',
            'poi_交通设施', 'poi_金融机构', 'poi_酒店住宿', 'poi_科教文化',
            'poi_旅游景点', 'poi_汽车相关', 'poi_商务住宅', 'poi_生活服务',
            'poi_休闲娱乐', 'poi_医疗保健', 'poi_运动健身'
        ]

        # 3. 手动列出所有土地利用列（适配你给出的列名，无遗漏）
        landuse_cols = [
            'landuse_11', 'landuse_12', 'landuse_21', 'landuse_22', 'landuse_23', 
            'landuse_24', 'landuse_31', 'landuse_32', 'landuse_33', 'landuse_41', 
            'landuse_42', 'landuse_43', 'landuse_45', 'landuse_46', 'landuse_51', 
            'landuse_52', 'landuse_53', 'landuse_64', 'landuse_65', 'landuse_66', 
            'landuse_99'
        ]

        # 4. 合并所有需要排除的列（基础列+POI列+土地利用列），去重避免重复
        exclude_cols = list(set(exclude_cols + poi_cols))

        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in [np.int64, np.float64]]
        
        all_feature_cols = [col for col in numeric_cols 
                           if col not in exclude_cols]
        
        cnn_feature_cols = [col for col in all_feature_cols 
                           if col.startswith('rs_feature_')]
        
        features_to_scale = [col for col in all_feature_cols 
                            if not col.startswith('rs_feature_')]
        
        return all_feature_cols, features_to_scale, cnn_feature_cols
    
    def _calculate_grid_gdp_labels(self, df, county_total_gdp):
        """计算每个网格的GDP伪标签：GDP ∝ NL × POP（保持原有逻辑）"""
        df['weight'] = df['nl_intensity'] * df['population_density']
        total_weight = df['weight'].sum()
        
        if total_weight == 0:
            df['weight'] = 1 / len(df)
            total_weight = 1
        
        # 计算原始GDP
        df['grid_gdp'] = (df['weight'] / total_weight) * county_total_gdp
        
        # 计算log(1+GDP)标签
        df['grid_gdp_log'] = np.log1p(df['grid_gdp'])
        
        return df
    
    def load_and_preprocess_county(self, features_dir, county_name, county_gdp):
        """加载并预处理单个县的数据"""
        file_path = os.path.join(features_dir, f'{county_name}_features.csv')
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path)
        if len(df) == 0:
            return None
        
        # 计算GDP伪标签（包括原始值和log值）
        df = self._calculate_grid_gdp_labels(df, county_gdp)
        
        # 获取特征列
        if self.all_feature_columns is None:
            self.all_feature_columns, self.feature_columns_to_scale, self.cnn_feature_columns = self._get_feature_columns(df)
        
        # 存储原始数据
        self.all_dataframes[county_name] = df
        
        return df
    
    def fit_scaler(self, features_dir, county_gdp_dict):
        """拟合标准化器"""
        print("🔧 拟合特征标准化器...")
        
        all_features_to_scale = []
        
        for county_name, county_gdp in tqdm(county_gdp_dict.items(), desc="收集标准化数据"):
            df = self.load_and_preprocess_county(features_dir, county_name, county_gdp)
            if df is not None and self.feature_columns_to_scale:
                features = df[self.feature_columns_to_scale].values
                all_features_to_scale.append(features)
        
        if all_features_to_scale:
            all_features_to_scale = np.vstack(all_features_to_scale)
            self.scaler.fit(all_features_to_scale)
            self.is_fitted = True
            print(f"✅ 标准化器拟合完成，处理 {all_features_to_scale.shape[0]} 个样本")
        else:
            print("⚠️ 没有数据可用于拟合标准化器")
    
    # ==================== 图构建部分（核心修改：新增节点级标签） ====================
    def build_8_neighbor_edges(self, positions):
        """
        构建8-邻域连接的边（保持原有逻辑，无修改）
        """
        positions_dict = {pos: idx for idx, pos in enumerate(positions)}
        edges = []
        
        # 8个方向
        directions = [
            (0, 1),    # 东
            (1, 0),    # 南
            (0, -1),   # 西
            (-1, 0),   # 北
            (-1, 1),   # 东北
            (-1, -1),  # 西北
            (1, 1),    # 东南
            (1, -1)    # 西南（之前遗漏，补全8邻域）
        ]
        
        for pos_idx, (row, col) in enumerate(positions):
            for dr, dc in directions:
                neighbor_pos = (row + dr, col + dc)
                if neighbor_pos in positions_dict:
                    neighbor_idx = positions_dict[neighbor_pos]
                    edges.append([pos_idx, neighbor_idx])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            return edge_index
        else:
            return torch.tensor([[], []], dtype=torch.long)
    
    def build_graph_from_dataframe(self, df):
        """
        从DataFrame构建图（包含LapPE编码+节点级标签）- 核心修改新增节点标签
        """
        # 获取位置
        positions = list(zip(df['position_row'].values, df['position_col'].values))
        
        # 构建边
        edge_index = self.build_8_neighbor_edges(positions)
        
        # 准备节点特征
        if self.is_fitted and self.feature_columns_to_scale and self.all_feature_columns:
            # 对需要标准化的特征进行标准化
            features_to_scale = df[self.feature_columns_to_scale].values
            features_scaled = self.scaler.transform(features_to_scale)
            
            # 获取CNN特征
            cnn_features = df[self.cnn_feature_columns].values if self.cnn_feature_columns else np.array([])
            
            # 合并特征
            X_final = np.zeros((len(df), len(self.all_feature_columns)))
            
            # 找到特征位置
            scale_indices = [self.all_feature_columns.index(col) 
                           for col in self.feature_columns_to_scale]
            cnn_indices = [self.all_feature_columns.index(col) 
                         for col in self.cnn_feature_columns] if self.cnn_feature_columns else []
            
            # 填入标准化特征
            for i, idx in enumerate(scale_indices):
                X_final[:, idx] = features_scaled[:, i]
            
            # 填入CNN特征
            for i, idx in enumerate(cnn_indices):
                X_final[:, idx] = cnn_features[:, i]
        else:
            # 如果没有拟合标准化器，使用原始特征
            X_final = df[self.all_feature_columns].values if self.all_feature_columns else np.zeros((len(df), 0))
        
        # 正确转换为PyTorch张量
        x = torch.tensor(X_final, dtype=torch.float)
        pos = torch.tensor(positions, dtype=torch.long)
        
        # ========== 核心修改1：构建节点级标签（每个节点对应自身的grid_gdp和grid_gdp_log） ==========
        grid_gdp = df['grid_gdp'].values.reshape(-1, 1)  # [num_nodes, 1]
        grid_gdp_log = df['grid_gdp_log'].values.reshape(-1, 1)  # [num_nodes, 1]
        y_node = torch.tensor(np.hstack([grid_gdp, grid_gdp_log]), dtype=torch.float)  # [num_nodes, 2]
        
        # 原有：构建图级全局标签（整图GDP总和，此处暂不设置，留待图块生成时处理）
        y = torch.tensor([[0.0, 0.0]], dtype=torch.float)
        
        # 正确创建 Data 对象（新增 y_node 节点级标签）
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            y=y,  # 图级全局标签（后续更新）
            y_node=y_node,  # 新增：节点级标签 [num_nodes, 2]
            pos=pos,
            num_nodes=len(df)
        )
        
        # 合并LapPE编码
        graph_data = self.merge_lap_pe_with_node_features(graph_data)
        
        return graph_data
    
    # ==================== 图块生成部分（核心修改：保留图块内节点级标签） ====================
    def extract_patch_from_graph(self, graph_data, start_row, start_col, min_nodes_threshold):
        """
        从图中提取指定位置的图块（保留LapPE编码+节点级标签）- 优化LapPE提取
        """
        # 获取位置信息
        positions = graph_data.pos.numpy()
        
        # 选择在图块范围内的节点（基于实际坐标）
        mask = ((positions[:, 0] >= start_row) & 
                (positions[:, 0] < start_row + self.patch_size) &
                (positions[:, 1] >= start_col) & 
                (positions[:, 1] < start_col + self.patch_size))
        
        node_indices = np.where(mask)[0]
        
        # 过滤太小的图块
        if len(node_indices) < min_nodes_threshold:
            return None, 0.0, 0.0
        
        # 提取基础子图数据
        x_patch = graph_data.x[node_indices]
        pos_patch = graph_data.pos[node_indices]
        
        # 核心修改2：提取图块内的节点级标签
        y_node_patch = graph_data.y_node[node_indices]  # [num_patch_nodes, 2]
        
        # 重新计算子图的边连接
        positions_patch = [(int(pos[0]), int(pos[1])) for pos in pos_patch]
        edge_index_patch = self.build_8_neighbor_edges(positions_patch)
        
        # 计算图块总GDP（原始值）- 基于节点级标签求和（更准确）
        patch_gdp = float(y_node_patch[:, 0].sum().item())
        
        # 计算图块log(1+GDP) - 使用torch，避免类型混淆
        patch_log_gdp = float(torch.log1p(torch.tensor(patch_gdp, dtype=torch.float)).item())
        
        # ========== 优化点：直接提取大图的LapPE子集，无需重新计算 ==========
        lap_pe_patch = graph_data.lap_pe[node_indices] if hasattr(graph_data, 'lap_pe') else torch.zeros((len(node_indices), self.lap_pe_k), dtype=torch.float)
        
        # 直接创建图块Data对象，包含LapPE子集
        patch_graph = Data(
            x=x_patch,
            edge_index=edge_index_patch,
            pos=pos_patch,
            num_nodes=len(node_indices),
            lap_pe=lap_pe_patch,  # 直接赋值提取的LapPE子集
            y=torch.tensor([[patch_gdp, patch_log_gdp]], dtype=torch.float),
            y_node=y_node_patch
        )
        
        # 移除冗余的LapPE重新计算逻辑
        if patch_graph.num_nodes <= 0:
            return None, 0.0, 0.0
        
        return patch_graph, patch_gdp, patch_log_gdp
    
    def generate_patches_for_county(self, graph_data, county_name, stride=None, 
                                  min_nodes_threshold=5):
        """
        为县图生成滑动窗口图块（保留LapPE编码+节点级标签）
        """
        if stride is None:
            stride = self.patch_size
        
        patches = []
        patch_gdps = []
        
        # 获取图的边界（实际坐标范围）
        positions = graph_data.pos.numpy()
        if len(positions) == 0:
            return patches, patch_gdps
        
        # 计算实际坐标范围
        min_row = int(positions[:, 0].min())
        max_row = int(positions[:, 0].max())
        min_col = int(positions[:, 1].min())
        max_col = int(positions[:, 1].max())
        
        # print(f"  {county_name}: 坐标范围: 行[{min_row}-{max_row}], 列[{min_col}-{max_col}], 总节点数: {len(positions)}")
        
        # 计算图块数量估计
        total_rows = max_row - min_row + 1
        total_cols = max_col - min_col + 1
        
        # 计算图块数量估计
        num_patches_h = max(1, total_rows // stride)
        num_patches_w = max(1, total_cols // stride)
        estimated_patches = num_patches_h * num_patches_w
        
        # 动态阈值计算
        avg_nodes_in_patch = len(positions) / estimated_patches if estimated_patches > 0 else 0
        
        # 更合理的阈值计算：考虑不同大小的县
        if avg_nodes_in_patch > 100:
            # 大型县：阈值设为平均节点数的1/5
            dynamic_threshold = max(min_nodes_threshold, int(avg_nodes_in_patch / 5))
        elif avg_nodes_in_patch > 50:
            # 中型县：阈值设为平均节点数的1/3
            dynamic_threshold = max(min_nodes_threshold, int(avg_nodes_in_patch / 3))
        else:
            # 小型县：使用最小阈值
            dynamic_threshold = min_nodes_threshold
        
        # 限制最大阈值不超过30
        dynamic_threshold = min(dynamic_threshold, 30)
        
        # print(f"  {county_name}: 估计图块数: {estimated_patches}, 平均节点/图块: {avg_nodes_in_patch:.1f}, 动态阈值: {dynamic_threshold}")
        
        # 滑动窗口：从实际最小坐标开始，到最大坐标结束
        for start_row in range(min_row, max_row + 1, stride):
            for start_col in range(min_col, max_col + 1, stride):
                patch_graph, patch_gdp, patch_log_gdp = self.extract_patch_from_graph(
                    graph_data, start_row, start_col, dynamic_threshold
                )
                
                if patch_graph is not None and patch_graph.num_nodes > 0:
                    patches.append(patch_graph)
                    patch_gdps.append(patch_gdp)  # 保存原始GDP值
                    self.patch_county_mapping.append(county_name)
        
        return patches, patch_gdps
    
    # ==================== 主流程（保持原有逻辑，无修改） ====================
    def build_graph_dataset(self, features_dir, output_dir=None, stride=None, 
                       max_counties=None, random_patches=False, min_nodes_threshold=5,
                       target_county=None):  # 新增：target_county参数，指定要处理的县
        """
        构建图数据集主流程（包含LapPE编码+节点级标签）
        新增：target_county参数，支持只处理指定县（推理时用）
        """
        # 清空之前的映射
        self.patch_county_mapping = []
        
        # 1. 加载GDP数据
        gdp_dict = self.load_county_gdp_dict()
        
        # 2. 获取特征文件
        feature_files = [f for f in os.listdir(features_dir) 
                        if f.endswith('_features.csv')]
        print(f"找到特征文件: {len(feature_files)} 个")
        
        # 3. 匹配县名
        self.county_gdp_dict = self.match_county_names(feature_files, gdp_dict)
        
        if not self.county_gdp_dict:
            print("❌ 没有匹配的县数据")
            return None
        
        # ========== 新增：只保留指定县的数据（推理时用） ==========
        if target_county is not None:
            if target_county in self.county_gdp_dict:
                # 只保留指定县
                self.county_gdp_dict = {target_county: self.county_gdp_dict[target_county]}
                print(f"🔍 仅处理指定县: {target_county}")
            else:
                raise ValueError(f"❌ 指定的县 {target_county} 不在匹配的县列表中")
        
        # 4. 拟合标准化器
        self.fit_scaler(features_dir, self.county_gdp_dict)
        
        # 5. 构建图和图块
        print(f"\n🔨 开始构建图数据集 (图块尺寸: {self.patch_size}×{self.patch_size}, LapPE维度: {self.lap_pe_k})...")
        
        all_patches = []
        all_patch_gdps = []
        
        county_items = list(self.county_gdp_dict.items())
        if max_counties and target_county is None:  # 只有没指定县时，才限制max_counties
            county_items = county_items[:max_counties]
            print(f"仅处理前 {max_counties} 个县用于调试")
        
        for county_name, county_gdp in tqdm(county_items, desc="处理各县"):
            # 加载并预处理数据
            df = self.load_and_preprocess_county(features_dir, county_name, county_gdp)
            if df is None:
                continue
            
            # 构建完整图（包含LapPE+节点级标签）
            graph_data = self.build_graph_from_dataframe(df)
            
            # 生成图块（包含LapPE+节点级标签）
            if random_patches:
                # 随机采样图块
                patches, patch_gdps = self._generate_random_patches(graph_data, county_name)
            else:
                # 滑动窗口图块
                patches, patch_gdps = self.generate_patches_for_county(
                    graph_data, county_name, 
                    stride=stride, 
                    min_nodes_threshold=min_nodes_threshold
                )
            
            # 添加到总列表
            all_patches.extend(patches)
            all_patch_gdps.extend(patch_gdps)
            
            print(f"  {county_name}: 生成 {len(patches)} 个图块（均包含LapPE编码+节点级标签）")
        
        # 6. 创建数据集
        dataset = GraphPatchDataset(all_patches, all_patch_gdps)
        
        # 7. 汇总统计
        stats = dataset.get_statistics()
        if stats:
            print(f"\n📊 数据集统计:")
            print(f"   处理县数: {len(county_items)}")
            print(f"   总图块数: {stats['num_patches']}")
            print(f"   平均节点数: {stats['avg_nodes']:.1f} ± {stats['std_nodes']:.1f}")
            print(f"   节点范围: {stats['min_nodes']} ~ {stats['max_nodes']}")
            print(f"   平均边数: {stats['avg_edges']:.1f}")
            print(f"   GDP范围: {stats['min_gdp']:.2f} ~ {stats['max_gdp']:.2f} 万元")
            print(f"   log(1+GDP)范围: {stats['min_log_gdp']:.4f} ~ {stats['max_log_gdp']:.4f}")
            print(f"   特征维度: {stats['feature_dim']}")
            print(f"   包含LapPE编码: {stats['has_lappe']}（维度: {self.lap_pe_k}）")
            print(f"   包含节点级标签: {stats['has_node_labels']}（平均节点GDP: {stats['avg_node_gdp']:.2f}）")
        
        # 8. 保存数据（可选，保留LapPE编码+节点级标签）
        if output_dir and stats:
            self._save_dataset(dataset, output_dir)
        
        return dataset
    
    def _generate_random_patches(self, graph_data, county_name, num_patches=10):
        """随机采样图块（用于调试，保留LapPE编码+节点级标签）- 修复语法错误"""
        patches = []
        patch_gdps = []
        
        positions = graph_data.pos.numpy()
        if len(positions) == 0:
            return patches, patch_gdps
        
        min_row = int(positions[:, 0].min())
        max_row = int(positions[:, 0].max())
        min_col = int(positions[:, 1].min())
        max_col = int(positions[:, 1].max())
        
        for _ in range(num_patches):
            # 随机起始位置（确保不超出边界）
            start_row = np.random.randint(min_row, max(max_row - self.patch_size + 1, min_row + 1))
            start_col = np.random.randint(min_col, max(max_col - self.patch_size + 1, min_col + 1))
            
            # 使用最小阈值进行随机采样
            patch_graph, patch_gdp, patch_log_gdp = self.extract_patch_from_graph(
                graph_data, start_row, start_col, min_nodes_threshold=3
            )
            
            if patch_graph is not None and patch_graph.num_nodes > 0:
                patches.append(patch_graph)
                patch_gdps.append(patch_gdp)
                self.patch_county_mapping.append(county_name)
        
        return patches, patch_gdps
    
    def _save_dataset(self, dataset, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # 修复核心：先判断scaler是否有mean_属性，没有则赋值None
        scaler_mean = self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None
        scaler_scale = self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None
        
        # 保存为PyG Dataset格式
        dataset_path = os.path.join(output_dir, 'graph_patches_with_lappe_and_node_labels.pt')
        torch.save({
            'dataset': dataset,
            'patch_county_mapping': self.patch_county_mapping,
            'patch_size': self.patch_size,
            'lap_pe_k': self.lap_pe_k,
            'scaler': self.scaler,
            'feature_columns': {
                'all': self.all_feature_columns,
                'to_scale': self.feature_columns_to_scale,
                'cnn': self.cnn_feature_columns
            },
            # 用判断后的变量，避免报错
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale,
            'scaler_var': self.scaler.var_ if hasattr(self.scaler, 'var_') else None,
            'scaler_n_samples_seen': self.scaler.n_samples_seen_ if hasattr(self.scaler, 'n_samples_seen_') else None
        }, dataset_path)
        
        # 单独保存scaler的部分也加判断
        scaler_path = os.path.join(output_dir, 'scaler.pth')
        torch.save({
            'scaler': self.scaler,
            'feature_columns_to_scale': self.feature_columns_to_scale,
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale,
            'scaler_var': self.scaler.var_ if hasattr(self.scaler, 'var_') else None,
            'scaler_n_samples_seen': self.scaler.n_samples_seen_ if hasattr(self.scaler, 'n_samples_seen_') else None
        }, scaler_path)
        
        # 统计信息保存不变
        stats = dataset.get_statistics()
        import json
        stats_path = os.path.join(output_dir, 'dataset_statistics_with_node_labels.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据集已保存到: {dataset_path}")
        print(f"✅ 标准化器已单独保存到: {scaler_path}")
        print(f"✅ 统计信息已保存到: {stats_path}")


class GraphDataLoaderManager:
    """数据加载器管理器（支持包含LapPE编码+节点级标签的图数据）"""
    
    def __init__(self, batch_size=32, num_workers=0):
        """
        初始化数据加载器管理器
        
        Args:
            batch_size: 批大小
            num_workers: 数据加载工作进程数
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def create_data_loaders(self, train_dataset, val_dataset=None, test_dataset=None, 
                      shuffle_train=True):
        """
        创建数据加载器（支持LapPE编码+节点级标签数据）
        核心修改：使用PyG DataLoader，自动拼接Batch对象，避免tuple报错
        """
        data_loaders = {}
        
        # 创建训练集加载器（使用PyG DataLoader，自动处理图批次）
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()  # 自动判断是否使用pin_memory
        )
        data_loaders['train'] = train_loader
        print(f"✅ 训练集加载器: {len(train_loader)} 批")
        
        # 创建验证集加载器（如果有）
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = PyGDataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            data_loaders['val'] = val_loader
            print(f"✅ 验证集加载器: {len(val_loader)} 批")
        
        # 创建测试集加载器（如果有）
        if test_dataset is not None and len(test_dataset) > 0:
            test_loader = PyGDataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            data_loaders['test'] = test_loader
            print(f"✅ 测试集加载器: {len(test_loader)} 批")
        
        return data_loaders
    
    def get_batch_statistics(self, data_loader, max_batches=10):
        """
        获取批次统计信息 - 正确处理PyG Batch对象（包含LapPE编码+节点级标签）
        """
        batch_sizes = []
        node_counts_per_graph = []
        edge_counts_per_graph = []
        gdp_values = []
        log_gdp_values = []
        lap_pe_dims = []
        
        # 新增：节点级标签统计
        node_gdp_batch_list = []
        
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            
            # PyG DataLoader返回Batch对象，自动拼接多个图
            if hasattr(batch, 'batch') and batch.batch is not None:
                # 获取批次中的图数量
                batch_size = int(batch.batch.max().item()) + 1 if len(batch.batch) > 0 else 1
                batch_sizes.append(batch_size)
                
                # 从Batch对象中提取每个图的节点数
                if hasattr(batch, '__num_nodes__') and batch.__num_nodes__:
                    # 新版本PyG
                    num_nodes_list = batch.__num_nodes__
                    if isinstance(num_nodes_list, list):
                        node_counts_per_graph.extend(num_nodes_list)
                else:
                    # 旧版本或手动计算
                    for graph_idx in range(batch_size):
                        node_count = (batch.batch == graph_idx).sum().item() if len(batch.batch) > 0 else 0
                        node_counts_per_graph.append(node_count)
                
                # 从Batch对象中提取每个图的边数
                if hasattr(batch, 'edge_index') and batch.edge_index is not None and batch.edge_index.shape[1] > 0:
                    # 需要根据batch.batch分离每个图的边
                    edge_batch = batch.batch[batch.edge_index[0]]  # 每条边的起点属于哪个图
                    for graph_idx in range(batch_size):
                        edge_count = (edge_batch == graph_idx).sum().item()
                        edge_counts_per_graph.append(edge_count)
                
                # 获取GDP值
                if hasattr(batch, 'y') and batch.y is not None and batch.y.dim() >= 2:
                    if batch.y.dim() == 2:  # [batch_size, 2]
                        gdp_values.extend(batch.y[:, 0].tolist())
                        log_gdp_values.extend(batch.y[:, 1].tolist())
                
                # 记录LapPE维度
                if hasattr(batch, 'lap_pe') and batch.lap_pe is not None:
                    lap_pe_dims.append(batch.lap_pe.shape[-1])
                
                # 新增：记录节点级GDP
                if hasattr(batch, 'y_node') and batch.y_node is not None:
                    node_gdp_batch_list.extend(batch.y_node[:, 0].tolist())
        
        # 计算统计信息（增加容错，避免空列表报错）
        def safe_mean(arr):
            return float(np.mean(arr)) if arr and not np.isnan(np.mean(arr)) else 0.0
        
        def safe_std(arr):
            return float(np.std(arr)) if arr and not np.isnan(np.std(arr)) else 0.0
        
        def safe_min(arr):
            return float(np.min(arr)) if arr else 0.0
        
        def safe_max(arr):
            return float(np.max(arr)) if arr else 0.0
        
        stats = {
            'avg_batch_size': safe_mean(batch_sizes),
            'std_batch_size': safe_std(batch_sizes),
            'avg_nodes': safe_mean(node_counts_per_graph),
            'std_nodes': safe_std(node_counts_per_graph),
            'avg_edges': safe_mean(edge_counts_per_graph),
            'std_edges': safe_std(edge_counts_per_graph),
            'avg_gdp': safe_mean(gdp_values),
            'std_gdp': safe_std(gdp_values),
            'avg_log_gdp': safe_mean(log_gdp_values),
            'std_log_gdp': safe_std(log_gdp_values),
            'min_gdp': safe_min(gdp_values),
            'max_gdp': safe_max(gdp_values),
            'avg_lap_pe_dim': safe_mean(lap_pe_dims),
            'has_lappe': bool(len(lap_pe_dims) > 0),
            # 新增：节点级标签统计
            'avg_node_gdp_in_batch': safe_mean(node_gdp_batch_list),
            'has_node_labels': bool(len(node_gdp_batch_list) > 0)
        }
        
        return stats


# ========== 主程序运行 ==========
if __name__ == "__main__":
    

    # 初始化图数据构建器（指定LapPE维度）
    builder = GraphDataBuilder(
        gdp_file_path='./dataset/分县GDP统计.xlsx',
        patch_size=12,  
        lap_pe_k=12  # LapPE编码维度为12（可根据需求调整）
    )
    
    # 检查数据集目录是否存在
    features_dir = './dataset/extracted_features_90'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir, exist_ok=True)
        print(f"⚠️  特征目录不存在，已创建: {features_dir}")
    
    # 构建数据集
    output_dir = './dataset/graph_data_with_lappe_and_node_labels'
    
    # 构建完整数据集（包含LapPE编码+节点级标签）
    dataset = builder.build_graph_dataset(
        features_dir=features_dir,
        output_dir=output_dir,
        stride=6,  # 无重叠
        max_counties=90,  # 仅处理前10个县用于测试
        random_patches=False,
        min_nodes_threshold=30  # 最小节点数阈值
    )
    
    if dataset is not None and len(dataset) > 0:
        
        # ========== 原有逻辑：数据集划分 ==========
        print("\n" + "="*50)
        print("数据集划分（保留LapPE编码+节点级标签）")
        print("="*50)
        
        # 方式1: 按县划分（推荐，防止数据泄露）
        train_dataset, val_dataset, test_dataset = dataset.split_by_county(
            builder.patch_county_mapping,
            test_size=0.2,
            random_state=42
        )
        
        # ========== 原有逻辑：创建数据加载器 ==========
        print("\n" + "="*50)
        print("创建数据加载器（支持LapPE编码+节点级标签）")
        print("="*50)
        
        loader_manager = GraphDataLoaderManager(
            batch_size=8,  # 较小的批大小，因为图的大小不同
            num_workers=0  # 在Windows上设置为0，Linux上可以设为2-4
        )
        
        data_loaders = loader_manager.create_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            shuffle_train=True
        )
        
        # ========== 原有逻辑：检查批次统计 ==========
        print("\n" + "="*50)
        print("批次统计信息（包含LapPE编码+节点级标签）")
        print("="*50)
        
        for split, loader in data_loaders.items():
            stats = loader_manager.get_batch_statistics(loader, max_batches=5)
            print(f"\n{split} 集:")
            print(f"  平均批次大小: {stats['avg_batch_size']:.1f} ± {stats['std_batch_size']:.1f}")
            print(f"  平均节点数: {stats['avg_nodes']:.1f} ± {stats['std_nodes']:.1f}")
            print(f"  平均边数: {stats['avg_edges']:.1f} ± {stats['std_edges']:.1f}")
            print(f"  GDP范围: {stats['min_gdp']:.2f} ~ {stats['max_gdp']:.2f} 万元")
            print(f"  平均log(1+GDP): {stats['avg_log_gdp']:.4f} ± {stats['std_log_gdp']:.4f}")
            print(f"  包含LapPE编码: {stats['has_lappe']}，平均维度: {stats['avg_lap_pe_dim']:.1f}")
            print(f"  包含节点级标签: {stats['has_node_labels']}，平均节点GDP: {stats['avg_node_gdp_in_batch']:.2f}")
        
        # ========== 原有逻辑：检查一个批次的数据（验证节点级标签） ==========
        print("\n" + "="*50)
        print("检查一个训练批次（验证节点级标签）")
        print("="*50)
        
        train_loader = data_loaders.get('train')
        if train_loader:
            for batch in train_loader:
                # PyG Batch对象自动拼接，获取批次信息
                print(f"批次类型: {type(batch)}")
                print(f"批次节点特征形状: {batch.x.shape}")
                print(f"批次边索引形状: {batch.edge_index.shape}")
                print(f"批次图级标签形状: {batch.y.shape}")
                if hasattr(batch, 'y_node'):
                    print(f"批次节点级标签形状: {batch.y_node.shape}")
                if hasattr(batch, 'lap_pe'):
                    print(f"批次LapPE编码形状: {batch.lap_pe.shape}")
                print(f"批次包含图数量: {int(batch.batch.max().item()) + 1 if len(batch.batch) > 0 else 1}")
                
                # 验证第一个图的节点级标签求和
                if hasattr(batch, 'y_node') and hasattr(batch, 'batch') and len(batch.batch) > 0:
                    first_graph_mask = (batch.batch == 0)
                    first_graph_node_gdp = batch.y_node[first_graph_mask, 0]
                    first_graph_node_gdp_sum = first_graph_node_gdp.sum().item()
                    first_graph_patch_gdp = batch.y[0, 0].item()
                    print(f"  第一个图节点级GDP总和: {first_graph_node_gdp_sum:.2f}")
                    print(f"  第一个图图块级GDP: {first_graph_patch_gdp:.2f}（误差: {abs(first_graph_node_gdp_sum - first_graph_patch_gdp):.4f}）")
                break
        
        # # ========== 新增：演示如何加载保存的scaler.pth ==========
        # print("\n" + "="*50)
        # print("演示加载保存的scaler.pth")
        # print("="*50)
        # scaler_path = os.path.join(output_dir, 'scaler.pth')
        # if os.path.exists(scaler_path):
        #     scaler_data = torch.load(scaler_path, weights_only=False)
        #     loaded_scaler = scaler_data['scaler']
        #     print(f"✅ 成功加载scaler.pth")
        #     print(f"  scaler均值: {loaded_scaler.mean_[:5]}...")  # 打印前5个均值
        #     print(f"  scaler缩放: {loaded_scaler.scale_[:5]}...")  # 打印前5个缩放值
        #     print(f"  标准化特征列: {scaler_data['feature_columns_to_scale'][:5]}...")  # 打印前5个特征列
        # else:
        #     print(f"❌ 未找到scaler.pth文件")