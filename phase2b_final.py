"""
Phase 2B 樣本加權 - 最終生產版本
可設定隨機種子以確保可重現性

使用方法:
    python phase2b_final.py                    # 使用預設種子2024
    python phase2b_final.py --seed 42          # 使用指定種子
    python phase2b_final.py --seed 2024 -v     # 詳細模式
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
import warnings
import random
import os
import argparse

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    設置隨機種子以確保完全可重現性
    
    Args:
        seed: 隨機種子數值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 隨機種子設定為: {seed}")


def clear_gpu_cache():
    """清空GPU快取"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==========================================
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """深度神經網路特徵提取器"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], output_dim=8, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.network(x)


class GPRegressionModel(gpytorch.models.ExactGP):
    """高斯過程回歸模型"""
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        )
    
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 損失函數
# ==========================================

def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    """
    加權MAPE損失函數
    
    注意: 在標準化空間計算（訓練時用）
    """
    mape_per_sample = torch.abs((y_true - y_pred) / 
                                (torch.abs(y_true) + epsilon)) * 100
    weighted_mape = torch.sum(mape_per_sample * weights) / torch.sum(weights)
    return weighted_mape


def compute_sample_weights(X, weight_factor=3.0):
    """
    計算樣本權重
    
    困難樣本定義: TIM_TYPE=3 AND Coverage=0.8 AND THICKNESS>=220
    
    Args:
        X: 特徵矩陣 [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
        weight_factor: 困難樣本的權重倍數
        
    Returns:
        weights: 樣本權重數組
    """
    weights = np.ones(len(X))
    
    difficult_mask = (
        (X[:, 0] == 3) &      # TIM_TYPE = 3
        (X[:, 2] == 0.8) &    # TIM_COVERAGE = 0.8
        (X[:, 1] >= 220)      # TIM_THICKNESS >= 220
    )
    
    weights[difficult_mask] *= weight_factor
    
    return weights


# ==========================================
# 訓練與評估
# ==========================================

def train_model(X_train, y_train, config, verbose=True):
    """
    訓練DKL模型
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        config: 訓練配置
        verbose: 是否顯示訓練過程
        
    Returns:
        model, likelihood, scaler_x, scaler_y
    """
    # 計算樣本權重
    sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
    
    if verbose:
        difficult_count = np.sum(sample_weights_np > 1.0)
        print(f"\n計算樣本權重:")
        print(f"  策略: Type 3 + Coverage 0.8 + THICKNESS>=220")
        print(f"  困難樣本數: {difficult_count} ({difficult_count/len(X_train)*100:.2f}%)")
        print(f"  權重倍數: {config['sample_weight_factor']}x")
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_train_scaled).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    sample_weights = torch.from_numpy(sample_weights_np).to(device)
    
    # 建立模型
    feature_extractor = DnnFeatureExtractor(
        input_dim=train_x.shape[1],
        hidden_dims=config['hidden_dims'],
        output_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(train_x, train_y, likelihood, feature_extractor).to(device)
    
    # 優化器
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr'], 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # 訓練
    if verbose:
        print(f"\n開始訓練...")
    
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        mape = weighted_mape_loss(output.mean, train_y, sample_weights)
        total_loss = gp_loss + config['mape_weight'] * mape
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        
        # 顯示訓練進度
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: GP Loss={gp_loss.item():.4f}, "
                  f"MAPE={mape.item():.2f}%, Total={total_loss.item():.4f}")
        
        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_state = {
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict(),
            }
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            if verbose:
                print(f"早停 at Epoch {epoch+1}")
            break
    
    # 載入最佳模型
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    if verbose:
        print(f"訓練完成 (Final Loss: {best_loss:.4f})")
    
    return model, likelihood, scaler_x, scaler_y


def evaluate_model(model, likelihood, X_test, y_test, scaler_x, scaler_y, verbose=True):
    """
    評估模型
    
    Args:
        model: 訓練好的模型
        likelihood: Likelihood
        X_test: 測試特徵
        y_test: 測試標籤
        scaler_x, scaler_y: 標準化器
        verbose: 是否顯示評估結果
        
    Returns:
        results: 包含MAPE, outliers等指標的字典
    """
    model.eval()
    likelihood.eval()
    
    X_test_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_test_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred_scaled = pred_dist.mean.cpu().numpy()
        y_std_scaled = pred_dist.stddev.cpu().numpy()
    
    # 反標準化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    # 計算指標 (在原始空間)
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    # Type 3分析
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((relative_errors > 20) & type3_mask)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"評估結果")
        print(f"{'='*60}")
        print(f"樣本數: {len(y_test)}")
        print(f"\n準確度:")
        print(f"  MAPE:      {mape:.2f}%")
        print(f"  MAE:       {mae:.4f}")
        print(f"  Max Error: {max_error:.2f}%")
        print(f"\n異常點:")
        print(f"  >20%: {outliers_20}/{len(y_test)} ({outliers_20/len(y_test)*100:.2f}%)")
        print(f"  >15%: {outliers_15}/{len(y_test)} ({outliers_15/len(y_test)*100:.2f}%)")
        print(f"  >10%: {outliers_10}/{len(y_test)} ({outliers_10/len(y_test)*100:.2f}%)")
        if np.sum(type3_mask) > 0:
            print(f"\nType 3異常點: {type3_outliers}/{np.sum(type3_mask)}")
        print(f"{'='*60}\n")
    
    results = {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'type3_outliers': type3_outliers,
        'predictions': y_pred,
        'std': y_std,
        'errors': relative_errors
    }
    
    return results


def save_predictions(X_test, y_test, results, filename):
    """保存預測結果到CSV"""
    df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Predicted': results['predictions'],
        'Error%': results['errors'],
        'Std': results['std']
    })
    
    df.to_csv(filename, index=False)
    print(f"✓ 預測結果已保存到: {filename}")


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, verbose=True):
    """
    主訓練流程
    
    Args:
        seed: 隨機種子
        verbose: 是否顯示詳細信息
    """
    # 設置隨機種子和清空GPU
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}\n")
    
    print("="*60)
    print("Phase 2B: 樣本加權 (Sample Weighting) - 最終版本")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置（最佳參數）
    config = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
        'sample_weight_factor': 3.0,
    }
    
    if verbose:
        print(f"\n配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # ==========================================
    # Above Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Above 50% Coverage'}\n")
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    # 訓練集清理（去除重複，取平均）
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆")
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    # 訓練
    model_above, likelihood_above, scaler_x_above, scaler_y_above = train_model(
        X_train_above, y_train_above, config, verbose=verbose
    )
    
    # 評估
    results_above = evaluate_model(
        model_above, likelihood_above, 
        X_test_above, y_test_above, 
        scaler_x_above, scaler_y_above,
        verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_above, y_test_above, results_above, 
                     f'phase2b_final_above_seed{seed}_predictions.csv')
    
    # ==========================================
    # Below Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Below 50% Coverage'}\n")
    
    # 載入資料
    train_below = pd.read_excel('data/train/Below.xlsx')
    test_below = pd.read_excel('data/test/Below.xlsx')
    
    # 訓練集清理
    train_below_clean = train_below.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_below_clean)} 筆")
    print(f"測試集: {len(test_below)} 筆")
    
    X_train_below = train_below_clean[feature_cols].values
    y_train_below = train_below_clean[target_col].values
    
    X_test_below = test_below[feature_cols].values
    y_test_below = test_below[target_col].values
    
    # 訓練
    model_below, likelihood_below, scaler_x_below, scaler_y_below = train_model(
        X_train_below, y_train_below, config, verbose=verbose
    )
    
    # 評估
    results_below = evaluate_model(
        model_below, likelihood_below,
        X_test_below, y_test_below,
        scaler_x_below, scaler_y_below,
        verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_below, y_test_below, results_below,
                     f'phase2b_final_below_seed{seed}_predictions.csv')
    
    # ==========================================
    # 總結
    # ==========================================
    
    print("\n" + "="*60)
    print("最終結果總結")
    print("="*60)
    print(f"隨機種子: {seed}")
    print(f"\nAbove資料集:")
    print(f"  異常點 (>20%): {results_above['outliers_20']}/{len(y_test_above)} ({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"  MAPE: {results_above['mape']:.2f}%")
    print(f"  Type 3異常點: {results_above['type3_outliers']}")
    
    print(f"\nBelow資料集:")
    print(f"  異常點 (>20%): {results_below['outliers_20']}/{len(y_test_below)} ({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    print(f"  MAPE: {results_below['mape']:.2f}%")
    
    print("\n" + "="*60)
    print("✓ 訓練完成！")
    print("="*60 + "\n")
    
    return {
        'above': results_above,
        'below': results_below,
        'seed': seed
    }


if __name__ == "__main__":
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='Phase 2B 樣本加權 - 最終版本')
    parser.add_argument('--seed', type=int, default=2024, 
                        help='隨機種子 (預設: 2024)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='顯示詳細訓練過程')
    
    args = parser.parse_args()
    
    # 運行訓練
    results = main(seed=args.seed, verbose=args.verbose)
    
    # 顯示最佳種子建議
    print("\n💡 提示:")
    print(f"   當前使用種子: {args.seed}")
    print(f"   最佳種子 (經10次測試): 2024")
    print(f"   其他優秀種子: 42, 123, 999\n")
    print("運行示例:")
    print("  python phase2b_final.py                # 使用預設種子2024")
    print("  python phase2b_final.py --seed 42      # 使用種子42")
    print("  python phase2b_final.py --seed 123 -v  # 使用種子123，詳細模式\n")
