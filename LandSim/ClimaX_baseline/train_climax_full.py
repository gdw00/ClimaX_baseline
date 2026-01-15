#!/usr/bin/env python3
"""
训练完整的 ClimaX Baseline 模型（包含 Variable Aggregation）
"""

import sys
import os
import json
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import get_cnp_combined_config
from data.data_loader_individual import DataLoaderIndividual
from training.trainer import ModelTrainer
from scripts.run_inference_all import verify_locations

# 导入完整的 ClimaX 模型
from model_climax_full import ClimaXFullCNP


def setup_logging(log_file: str, level: str = 'INFO') -> None:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='完整 ClimaX Baseline Training for CNP')
    
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--output-dir', default='climax_full_results')
    parser.add_argument('--epochs', '--epoch', dest='epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--dropout-p', type=float, default=None)
    
    parser.add_argument('--use-trendy1', action='store_true')
    parser.add_argument('--use-trendy05', action='store_true')
    parser.add_argument('--use-tva4km', action='store_true')
    
    parser.add_argument('--variable-list', type=str, default=None)
    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--normalization', choices=['group', 'individual', 'hybrid'], 
                       default='individual')
    
    parser.add_argument('--strict-determinism', action='store_true')
    parser.add_argument('--mask-absent-pfts', dest='mask_absent_pfts', 
                       action='store_true', default=True)
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / f"climax_full_training_{timestamp}.log"
    setup_logging(str(log_file), args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("完整 ClimaX Baseline Training (with Variable Aggregation)")
    logger.info("=" * 80)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"标准化方法: {args.normalization}")
    
    # 默认使用 trendy1
    if not args.use_trendy1 and not args.use_trendy05 and not args.use_tva4km:
        args.use_trendy1 = True
        args.use_trendy05 = False
    
    try:
        include_water = False
        logger.info(f"Water variables included: {include_water}")
        
        # 获取配置
        from config.training_config import get_cnp_combined_config
        config = get_cnp_combined_config(
            use_trendy1=args.use_trendy1,
            use_trendy05=args.use_trendy05,
            use_tva4km=args.use_tva4km,
            max_files=args.max_files,
            include_water=include_water,
            variable_list_path=args.variable_list,
            model_config_path=args.model_config
        )
        
        if args.variable_list is not None:
            logger.info(f"使用变量列表文件: {args.variable_list}")
        if args.model_config is not None:
            logger.info(f"使用模型配置文件: {args.model_config}")
        
        # 设置训练配置
        config.update_data_config(train_split=0.8)
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.update_training_config(device=device_str)
        config.update_data_config(max_files=args.max_files)
        config.update_training_config(log_gpu_memory=False, log_gpu_utilization=False)
        
        effective_lr = args.learning_rate if args.learning_rate is not None else config.training_config.learning_rate
        config.update_training_config(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=effective_lr,
            model_save_path=str(output_dir / "climax_full_model.pt"),
            losses_save_path=str(output_dir / "climax_full_losses.csv"),
            predictions_dir=str(output_dir / "climax_full_predictions"),
            use_early_stopping=False
        )
        
        if args.mask_absent_pfts:
            try:
                config.update_training_config(mask_absent_pfts=True)
                logger.info("Masking absent PFTs enabled")
            except Exception as e:
                logger.warning(f"Failed to enable mask_absent_pfts: {e}")
        
        logger.info(f"有效学习率: {effective_lr}")
        
        # 严格确定性
        if args.strict_determinism:
            from train_cnp_model import set_global_determinism
            seed = getattr(config.training_config, 'random_seed', 42)
            set_global_determinism(seed)
            logger.info(f"Strict determinism enabled with seed={seed}")
        
        # Dropout 覆盖
        if args.dropout_p is not None:
            try:
                config.update_model_config(dropout_p=float(args.dropout_p))
                logger.info(f"Dropout 覆盖: {config.model_config.dropout_p}")
            except Exception as e:
                logger.warning(f"Failed to apply dropout override: {e}")
        
        # 加载数据
        logger.info("=" * 80)
        logger.info("加载数据...")
        logger.info("=" * 80)
        
        data_loader = DataLoaderIndividual(
            config.data_config,
            config.preprocessing_config
        )
        
        raw_data = data_loader.load_data()
        
        if hasattr(data_loader, 'df'):
            logger.info(f"数据集变量数: {len(data_loader.df.columns)}")
            logger.info(f"数据集形状: {data_loader.df.shape}")
        
        logger.info("预处理数据...")
        preprocessed_data = data_loader.preprocess_data()
        data_info = data_loader.get_data_info()
        
        logger.info(f"标准化数据 (方法: {args.normalization})...")
        if args.normalization == 'group':
            normalized_data = data_loader.normalize_data()
        elif args.normalization == 'individual':
            normalized_data = data_loader.normalize_data_individual()
        else:
            normalized_data = data_loader.normalize_data_hybrid(
                use_individual_for=['scalar', 'pft_1d', 'soil_2d', 'y_scalar', 'y_pft_1d', 'y_soil_2d'],
                group_soil_vars=['sminn_vr', 'smin_no3_vr', 'smin_nh4_vr']
            )
        
        logger.info("分割训练/验证集...")
        split_data = data_loader.split_data(normalized_data)
        
        if hasattr(data_loader, 'df'):
            train_indices = data_loader.train_indices
            test_indices = data_loader.test_indices
            
            train_df = data_loader.df.iloc[train_indices]
            val_df = data_loader.df.iloc[test_indices]
            
            verify_locations(train_df, "Training data")
            verify_locations(val_df, "Validation data")
        
        # 创建完整 ClimaX 模型
        logger.info("=" * 80)
        logger.info("创建完整 ClimaX Baseline 模型...")
        logger.info("=" * 80)
        
        model = ClimaXFullCNP(
            config.model_config,
            data_info,
            include_water=include_water,
            use_learnable_loss_weights=config.training_config.use_learnable_loss_weights
        )
        
        # 初始化 Trainer
        trainer = ModelTrainer(
            config.training_config,
            model,
            split_data['train'],
            split_data['test'],
            normalized_data['scalers'],
            data_info
        )
        
        # 开始训练
        logger.info("=" * 80)
        logger.info("开始训练完整 ClimaX Baseline...")
        logger.info("=" * 80)
        
        results = trainer.run_training_pipeline()
        
        logger.info("=" * 80)
        logger.info("训练完成!")
        logger.info("=" * 80)
        logger.info(f"最终指标: {results['metrics']}")
        
        # 保存结果
        with open(output_dir / "climax_full_metrics.json", "w") as f:
            json.dump(results['metrics'], f, indent=2)
        
        # 保存配置
        with open(output_dir / "climax_full_config.json", "w") as f:
            config_dict = {
                'model_type': 'ClimaX_Full_with_Variable_Aggregation',
                'include_water': include_water,
                'normalization_method': args.normalization,
                'data_info': data_info,
                'model_config': config.model_config.__dict__,
                'training_config': config.training_config.__dict__,
                'climax_components': {
                    'variable_tokenization': True,
                    'variable_embedding': True,
                    'variable_aggregation': True,
                    'num_variables': 6
                }
            }
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"结果保存到: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

