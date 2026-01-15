#!/usr/bin/env python3
"""
完整的 ClimaX Baseline for CNP
实现 ClimaX 的三大核心创新：
1. Variable Tokenization - 每个变量/模态独立处理
2. Variable Embedding - 为每个变量学习身份标识
3. Variable Aggregation - 使用交叉注意力聚合变量信息
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cnp_combined_model import (
    ForcingTemporalEncoder,
    StaticVariableEncoder,
    get_1d_sincos_pos_embed_from_grid
)
from config.training_config import ModelConfig

# 导入 ClimaX 的 ViT Block
from timm.models.vision_transformer import Block

logger = logging.getLogger(__name__)


class ClimaXFullCNP(nn.Module):
    """
    完整的 ClimaX Baseline for CNP
    
    架构:
    1. 编码器: 复用原 CNP 编码器（但每个模态独立）
    2. Variable Embedding: 为每个模态/变量学习身份标识
    3. Variable Aggregation: ClimaX 的核心 - 交叉注意力聚合 ⭐
    4. ViT Transformer: ClimaX 的 ViT Blocks
    5. 输出头: 复用原 CNP 输出头
    
    关键创新:
    - 将不同模态视为不同的 "变量"
    - 使用可学习的 query 聚合跨变量信息（ClimaX 的核心思想）
    - 完整实现 ClimaX 的 Variable-Centric 设计
    """
    
    def __init__(
        self, 
        model_config: ModelConfig, 
        data_info: Dict[str, Any],
        include_water: bool = True,
        use_learnable_loss_weights: bool = False
    ):
        super().__init__()
        
        self.model_config = model_config
        self.data_info = data_info
        self.include_water = include_water
        self.use_learnable_loss_weights = use_learnable_loss_weights
        
        # 统一维度
        self.embed_dim = getattr(self.model_config, 'embed_dim', 256)
        self.token_dim = self.embed_dim
        self.dropout_p = getattr(self.model_config, 'dropout_p', 0.1)
        
        logger.info("=" * 80)
        logger.info("构建完整 ClimaX Baseline (with Variable Aggregation)")
        logger.info("=" * 80)
        
        # 1. 计算维度
        self._calculate_input_dimensions()
        
        # 2. 构建编码器（每个模态独立编码）
        logger.info(">> 构建模态编码器（Variable Tokenization）")
        self._build_encoders()
        
        # 3. ClimaX 核心组件
        logger.info(">> 构建 ClimaX 核心组件")
        self._build_climax_components()
        
        # 4. 构建输出头
        logger.info(">> 使用原 CNP 输出头")
        self._build_output_heads()
        
        # 5. Loss weights
        self._setup_loss_weights()
        self._initialize_weights()
        
        # 统计参数
        total_params = self._count_parameters()
        logger.info(f">> 总参数量: {total_params / 1e6:.2f}M")
        logger.info("=" * 80)
    
    def _calculate_input_dimensions(self):
        """计算输入维度"""
        self.lstm_input_size = len(self.data_info['time_series_columns'])
        self.time_series_length = self.data_info.get('time_series_length', 240)
        self.surface_input_size = len(self.data_info['static_columns'])
        self.pft_param_input_size = len(self.data_info.get('pft_param_columns', []))
        
        num_pfts = getattr(self.model_config, 'num_pfts', 17)
        self.actual_pft_param_size = self.pft_param_input_size * num_pfts
        self.pft_1d_input_size = len(self.data_info.get('variables_1d_pft', []))
        self.vector_length = getattr(self.model_config, 'vector_length', 16)
        self.actual_1d_size = self.pft_1d_input_size * self.vector_length
        self.water_input_size = len(self.data_info.get('x_list_water_columns', [])) if self.include_water else 0
        self.scalar_input_size = len(self.data_info.get('x_list_scalar_columns', []))
        self.actual_2d_channels = len(self.data_info.get('x_list_columns_2d', []))
        
        logger.info(f"   时序变量: {self.lstm_input_size} vars × {self.time_series_length} steps")
        logger.info(f"   静态变量: {self.surface_input_size}")
        logger.info(f"   PFT参数: {self.pft_param_input_size} params × {num_pfts} PFTs")
        logger.info(f"   标量变量: {self.scalar_input_size}")
        logger.info(f"   1D PFT变量: {self.pft_1d_input_size}")
        logger.info(f"   2D 土壤变量: {self.actual_2d_channels}")
    
    def _build_encoders(self):
        """
        构建编码器 - Variable Tokenization
        
        关键思想: 每个模态独立编码，生成该模态的 "变量表示"
        """
        # 1. 时序编码器
        patch_size = getattr(self.model_config, 'patch_size', 60)
        if self.time_series_length % patch_size != 0:
            patch_size = 12
        
        self.temporal_encoder = ForcingTemporalEncoder(
            num_vars=self.lstm_input_size,
            embed_dim=self.embed_dim,
            total_time_steps=self.time_series_length,
            patch_size=patch_size,
            lat_lon_dim=2
        )
        
        # 2. 静态编码器（简化版，输出单个 token）
        self.static_encoder = nn.Sequential(
            nn.Linear(self.surface_input_size, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        # 3. PFT 参数编码器（输出单个 token）
        self.pft_param_encoder = nn.Sequential(
            nn.Linear(self.actual_pft_param_size, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        # 4. 标量编码器（输出单个 token）
        if self.scalar_input_size > 0:
            self.scalar_encoder = nn.Sequential(
                nn.Linear(self.scalar_input_size, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_p)
            )
        
        # 5. 1D PFT 编码器（输出单个 token）
        if self.pft_1d_input_size > 0:
            self.pft_1d_encoder = nn.Sequential(
                nn.Linear(self.actual_1d_size, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.embed_dim * 2, self.embed_dim)
            )
        
        # 6. 2D 土壤编码器（输出单个 token）
        if self.actual_2d_channels > 0:
            matrix_size = self.actual_2d_channels * self.model_config.matrix_rows * self.model_config.matrix_cols
            self.soil_2d_encoder = nn.Sequential(
                nn.Linear(matrix_size, self.embed_dim * 2),
                nn.LayerNorm(self.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.embed_dim * 2, self.embed_dim)
            )
        
        logger.info(f"   构建了 6 个独立模态编码器")
    
    def _build_climax_components(self):
        """
        构建 ClimaX 核心组件
        
        1. Variable Embedding - 每个变量的身份标识
        2. Variable Aggregation - 跨变量交叉注意力（核心！）
        3. ViT Blocks - Transformer 骨干
        """
        num_layers = getattr(self.model_config, 'transformer_layers', 6)
        num_heads = getattr(self.model_config, 'transformer_heads', 4)
        
        # 定义变量数量（模态数量）
        # 1. 时序 2. 静态 3. PFT参数 4. 标量 5. 1D PFT 6. 2D 土壤
        self.num_variables = 6
        
        # 定义变量名称（用于创建变量映射）
        self.variable_names = [
            'time_series', 'static', 'pft_param', 
            'scalar', 'pft_1d', 'soil_2d'
        ]
        
        # ===== ClimaX 核心1: Variable Embedding =====
        # 使用原始 ClimaX 的方法创建 Variable Embedding
        self.var_embed, self.var_map = self.create_var_embedding(self.embed_dim)
        
        # ===== ClimaX 核心2: Variable Aggregation =====
        # 可学习的 query 用于聚合所有变量的信息
        self.var_query = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim),
            requires_grad=True
        )
        
        # 交叉注意力层：Query 是全局的，Key/Value 是所有变量
        self.var_agg = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=self.dropout_p
        )
        
        # Position Embedding（用于聚合后的 token）
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim),
            requires_grad=True
        )
        
        # ===== ClimaX 核心3: ViT Transformer Blocks =====
        dpr = [x.item() for x in torch.linspace(0, self.dropout_p, num_layers)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm,
                attn_drop=self.dropout_p,
                proj_drop=self.dropout_p
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        
        logger.info(f"   Variable Embedding: {self.num_variables} 个变量")
        logger.info(f"   Variable Aggregation: 交叉注意力机制（使用原始 ClimaX 方法）")
        logger.info(f"   ViT Blocks: {num_layers} layers, {num_heads} heads")
    
    def create_var_embedding(self, dim):
        """
        创建 Variable Embedding（来自原始 ClimaX 实现）
        
        Args:
            dim: embedding dimension
            
        Returns:
            var_embed: [1, num_variables, dim] 可学习的变量嵌入
            var_map: dict 变量名到索引的映射
        """
        var_embed = nn.Parameter(
            torch.zeros(1, self.num_variables, dim), 
            requires_grad=True
        )
        var_map = {}
        for idx, var_name in enumerate(self.variable_names):
            var_map[var_name] = idx
        return var_embed, var_map
    
    def aggregate_variables(self, x: torch.Tensor):
        """
        聚合变量 tokens（来自原始 ClimaX 实现）
        
        Args:
            x: [B, V, L, D] - V个变量，每个有L个tokens（在我们的实现中L=1）
            
        Returns:
            aggregated: [B, L, D] - 聚合后的tokens
        """
        b, _, l, _ = x.shape
        # 重排维度: [B, V, L, D] -> [B, L, V, D]
        x = torch.einsum("bvld->blvd", x)
        # 展平: [B, L, V, D] -> [B*L, V, D]
        x = x.flatten(0, 1)  # BxL, V, D
        
        # 扩展 query: [1, 1, D] -> [B*L, 1, D]
        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        
        # 交叉注意力聚合
        x, _ = self.var_agg(var_query, x, x)  # [B*L, 1, D]
        
        # 处理维度：确保是 [B*L, 1, D] 或 [B*L, D]
        if x.dim() == 3:
            x = x.squeeze(1)  # [B*L, 1, D] -> [B*L, D]
        elif x.dim() == 2 and x.shape[1] == 1:
            x = x.squeeze(1)  # [B*L, 1] -> [B*L] (不应该发生，但保险起见)
        
        # 恢复维度: [B*L, D] -> [B, L, D]
        # 如果 L=1，需要添加维度
        if l == 1:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
        else:
            x = x.unflatten(dim=0, sizes=(b, l))  # [B, L, D]
        
        return x
    
    def _build_output_heads(self):
        """构建输出头（与原模型完全相同）"""
        self.scalar_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, self.model_config.scalar_output_size)
        )
        
        n_2d_vars = len(self.data_info.get('y_list_columns_2d', []))
        self.matrix_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, n_2d_vars * self.model_config.matrix_rows * self.model_config.matrix_cols)
        )
        
        self.pft_1d_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(128, self.pft_1d_input_size * self.model_config.vector_length)
        )
        
        if self.include_water:
            self.water_head = nn.Sequential(
                nn.Linear(self.embed_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(64, 6)
            )
        else:
            self.water_head = None
    
    def _setup_loss_weights(self):
        """Loss weights 设置"""
        if self.use_learnable_loss_weights:
            self.log_sigma_scalar = nn.Parameter(torch.zeros(1))
            self.log_sigma_soil_2d = nn.Parameter(torch.zeros(1))
            self.log_sigma_pft_1d = nn.Parameter(torch.zeros(1))
            self.log_sigma_water = nn.Parameter(torch.zeros(1)) if self.include_water else None
        else:
            self.log_sigma_scalar = None
            self.log_sigma_soil_2d = None
            self.log_sigma_pft_1d = None
            self.log_sigma_water = None
    
    def _initialize_weights(self):
        """
        权重初始化（使用原始 ClimaX 的方法）
        """
        # Variable Embedding 使用 sincos 初始化（来自原始 ClimaX）
        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.embed_dim,
            np.arange(self.num_variables, dtype=np.float32)
        )
        # get_1d_sincos_pos_embed_from_grid 已经返回 Tensor，不需要 torch.from_numpy()
        self.var_embed.data.copy_(var_embed.unsqueeze(0))
        
        # Position Embedding（注意：我们使用 1D pos_embed，因为只有1个token）
        # 原始 ClimaX 使用 2D pos_embed（对应图像patches），我们适配为 1D
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.embed_dim,
            np.array([0], dtype=np.float32)
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
        # 使用原始 ClimaX 的权重初始化方法（trunc_normal_）
        # 注意：需要导入 trunc_normal_ 从 timm
        from timm.models.vision_transformer import trunc_normal_
        
        # 初始化 Linear 和 LayerNorm 层
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """
        权重初始化辅助方法（来自原始 ClimaX）
        """
        from timm.models.vision_transformer import trunc_normal_
        
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取 loss weights"""
        if self.use_learnable_loss_weights:
            weights = {}
            if self.log_sigma_scalar is not None:
                weights['scalar'] = (1 / (2 * torch.exp(self.log_sigma_scalar) ** 2)).item()
            if self.log_sigma_soil_2d is not None:
                weights['soil_2d'] = (1 / (2 * torch.exp(self.log_sigma_soil_2d) ** 2)).item()
            if self.include_water and self.log_sigma_water is not None:
                weights['water'] = (1 / (2 * torch.exp(self.log_sigma_water) ** 2)).item()
            if self.log_sigma_pft_1d is not None:
                weights['pft_1d'] = (1 / (2 * torch.exp(self.log_sigma_pft_1d) ** 2)).item()
            return weights
        else:
            weights = {
                'scalar': 1.0,
                'soil_2d': 1.0,
                'pft_1d': 1.0
            }
            if self.include_water:
                weights['water'] = 1.0
            return weights
    
    def forward(self, time_series_data, static_data, pft_param_data, 
                scalar, variables_1d_pft, variables_2d_soil):
        """
        前向传播 - 完整的 ClimaX 流程
        
        关键步骤:
        1. Variable Tokenization: 每个模态独立编码
        2. Variable Embedding: 添加变量身份标识
        3. Variable Aggregation: 交叉注意力聚合（ClimaX 核心！）
        4. ViT Transformer: 特征提取
        5. 输出: 多任务预测
        """
        B = static_data.shape[0]
        
        # ====== 步骤1: Variable Tokenization（每个模态独立编码）======
        variable_tokens = []
        
        # 1. 时序变量
        lat_lon = static_data[:, :2]
        dyn_tokens = self.temporal_encoder(time_series_data, lat_lon)
        # 取时序 tokens 的平均作为该"变量"的表示
        time_var_token = dyn_tokens.mean(dim=1)  # [B, D]
        variable_tokens.append(time_var_token)
        
        # 2. 静态变量
        static_var_token = self.static_encoder(static_data)  # [B, D]
        variable_tokens.append(static_var_token)
        
        # 3. PFT 参数变量
        pft_param_flat = pft_param_data.reshape(B, -1)  # [B, 748]
        pft_param_var_token = self.pft_param_encoder(pft_param_flat)  # [B, D]
        variable_tokens.append(pft_param_var_token)
        
        # 4. 标量变量
        if self.scalar_input_size > 0:
            scalar_var_token = self.scalar_encoder(scalar)  # [B, D]
            variable_tokens.append(scalar_var_token)
        else:
            # 如果没有标量，用零向量占位
            variable_tokens.append(torch.zeros(B, self.embed_dim, device=static_data.device))
        
        # 5. 1D PFT 变量
        if self.pft_1d_input_size > 0:
            pft_1d_flat = variables_1d_pft.reshape(B, -1)  # [B, 656]
            pft_1d_var_token = self.pft_1d_encoder(pft_1d_flat)  # [B, D]
            variable_tokens.append(pft_1d_var_token)
        else:
            variable_tokens.append(torch.zeros(B, self.embed_dim, device=static_data.device))
        
        # 6. 2D 土壤变量
        if self.actual_2d_channels > 0:
            soil_2d_flat = variables_2d_soil.reshape(B, -1)  # [B, 250]
            soil_2d_var_token = self.soil_2d_encoder(soil_2d_flat)  # [B, D]
            variable_tokens.append(soil_2d_var_token)
        else:
            variable_tokens.append(torch.zeros(B, self.embed_dim, device=static_data.device))
        
        # 堆叠所有变量 tokens
        var_tokens = torch.stack(variable_tokens, dim=1)  # [B, V, D] V=6
        
        # ====== 步骤2: Variable Embedding（添加变量身份）======
        var_tokens = var_tokens + self.var_embed  # [B, V, D]
        
        # ====== 步骤3: Variable Aggregation（ClimaX 核心！）======
        # 使用原始 ClimaX 的 aggregate_variables 方法
        # 需要将 [B, V, D] 扩展为 [B, V, 1, D] 以匹配原始接口
        var_tokens_expanded = var_tokens.unsqueeze(2)  # [B, V, 1, D]
        
        # 使用原始 ClimaX 的聚合方法
        aggregated = self.aggregate_variables(var_tokens_expanded)  # [B, 1, D]
        
        # 获取注意力权重（用于分析，可选）
        # 注意：aggregate_variables 内部已经计算了注意力，但不返回
        # 如果需要注意力权重，可以单独计算一次
        var_query = self.var_query.expand(B, -1, -1)  # [B, 1, D]
        _, attn_weights = self.var_agg(
            var_query,      # Query:  [B, 1, D]
            var_tokens,     # Key:    [B, V, D]
            var_tokens      # Value:  [B, V, D]
        )  # 用于获取注意力权重
        
        # ====== 步骤4: Position Embedding ======
        x = aggregated + self.pos_embed  # [B, 1, D]
        
        # ====== 步骤5: ViT Transformer Blocks ======
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, 1, D]
        
        # ====== 步骤6: 输出预测 ======
        pooled = x.squeeze(1)  # [B, D]
        
        scalar_out = self.scalar_head(pooled)
        
        pft_1d_out = self.pft_1d_head(pooled)
        pft_1d_out = pft_1d_out.reshape(B, self.pft_1d_input_size, self.model_config.vector_length)
        
        soil_2d_out = self.matrix_head(pooled)
        n_2d_vars = len(self.data_info.get('y_list_columns_2d', []))
        soil_2d_out = soil_2d_out.reshape(
            B, n_2d_vars, 
            self.model_config.matrix_rows, 
            self.model_config.matrix_cols
        )
        
        water_out = self.water_head(pooled) if self.include_water else None
        
        # 返回 trainer 期望的键名（注意：不是 'y_scalar' 而是 'scalar'）
        return {
            'scalar': scalar_out,
            'pft_1d': pft_1d_out,
            'soil_2d': soil_2d_out,
            'water': water_out,
            'attention_weights': attn_weights  # 可选：返回注意力权重用于分析
        }


# 测试代码
if __name__ == '__main__':
    print("测试完整 ClimaX Baseline Model")
    
    from types import SimpleNamespace
    
    model_config = SimpleNamespace(
        embed_dim=128,
        dropout_p=0.1,
        patch_size=20,
        transformer_layers=6,
        transformer_heads=4,
        num_pfts=17,
        vector_length=16,
        matrix_rows=1,
        matrix_cols=10,
        scalar_output_size=4
    )
    
    data_info = {
        'time_series_columns': ['var1', 'var2', 'var3', 'var4', 'var5', 'var6'],
        'time_series_length': 240,
        'static_columns': ['lat', 'lon'] + [f'static_{i}' for i in range(47)],
        'pft_param_columns': [f'pft_param_{i}' for i in range(44)],
        'x_list_scalar_columns': [f'scalar_{i}' for i in range(4)],
        'variables_1d_pft': [f'pft1d_{i}' for i in range(41)],
        'x_list_columns_2d': [f'soil2d_{i}' for i in range(25)],
        'y_list_columns_2d': [f'Y_soil2d_{i}' for i in range(25)],
        'x_list_water_columns': []
    }
    
    # 创建模型
    model = ClimaXFullCNP(model_config, data_info, include_water=False)
    
    print(f"\n总参数量: {model._count_parameters() / 1e6:.2f}M")
    
    # 测试前向传播
    B = 4
    time_series = torch.randn(B, 240, 6)
    static = torch.randn(B, 49)
    pft_param = torch.randn(B, 44, 17)
    scalar = torch.randn(B, 4)
    pft_1d = torch.randn(B, 41, 16)
    soil_2d = torch.randn(B, 25, 1, 10)
    
    outputs = model(time_series, static, pft_param, scalar, pft_1d, soil_2d)
    
    print("\n输出形状:")
    for k, v in outputs.items():
        if v is not None and k != 'attention_weights':
            print(f"  {k}: {v.shape}")
    
    if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
        print(f"  attention_weights: {outputs['attention_weights'].shape}")
        print(f"    解释: [B, num_heads, 1, V] = [{B}, 4, 1, 6]")
        print(f"    含义: 每个样本的注意力权重分布在 6 个变量上")
    
    print("\n✅ 完整 ClimaX 模型测试通过!")

