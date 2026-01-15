#!/bin/bash
#SBATCH -A m4814_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00   
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=climax_full
#SBATCH -D /global/cfs/cdirs/m4814/daweigao/15_code_Landsim/LandSim
#SBATCH --output=ClimaX_baseline/logs/climax_full-%j.out
#SBATCH --error=ClimaX_baseline/logs/climax_full-%j.err

set -euo pipefail

echo "=========================================="
echo "完整 ClimaX Baseline Training"
echo "(with Variable Aggregation)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 激活 conda 环境（根据 read_me_gdw.md 中的配置）
module load conda
conda activate /global/cfs/cdirs/m4814/daweigao/14_Code/000_LandSim-LandSim_dynamic_configuration_pft_mask/env

PYTHON="${PYTHON:-python}"

# # 进入 LandSim 目录（由 #SBATCH -D 保证，但这里显式写清楚）
# cd /global/cfs/cdirs/m4814/daweigao/15_code_Landsim/LandSim

# 进入 LandSim 目录（由 #SBATCH -D 保证，但这里显式写清楚）
cd /global/cfs/cdirs/m4814/daweigao/15_code_Landsim/LandSim

# 创建日志目录（与 --output/--error 对齐）
mkdir -p ClimaX_baseline/logs

# 打印环境信息
echo "Python Path: $PYTHONPATH"
echo "Working Directory: $(pwd)"
echo "Python Executable: $(command -v $PYTHON)"
echo "Python Version: $($PYTHON --version)"
echo "timm Version: $($PYTHON -c 'import timm; print(timm.__version__)')"
echo "PyTorch Version: $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
if $PYTHON -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "GPU Device: $($PYTHON -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "=========================================="

# 运行训练脚本
echo "开始训练完整 ClimaX Baseline..."
echo "=========================================="

srun -u $PYTHON ClimaX_baseline/train_climax_full.py \
--model-config CNP_model_config_v01.txt \
--variable-list CNP_IO_updated9_dev.txt \
--use-trendy1 \
--epochs 150 \
--batch-size 128 \
--learning-rate 1e-4 \
--normalization individual \
--strict-determinism \
--mask-absent-pfts \
--output-dir climax_full_results

exit_code=$?

echo "=========================================="
echo "训练完成 (Exit Code: $exit_code)"
echo "End Time: $(date)"
echo "=========================================="

if [ $exit_code -eq 0 ]; then
    echo "✅ 训练成功完成"
else
    echo "❌ 训练失败"
fi

exit $exit_code

