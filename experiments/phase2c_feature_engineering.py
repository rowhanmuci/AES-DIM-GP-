"""
Phase 2C: 特徵工程改進版 DKL
基於數據分析的發現，針對 Type 3 異常點進行優化

關鍵改進:
1. 物理意義特徵: 1/Coverage, Thickness/Coverage
2. Type-specific 交互特徵
3. Log-space 建模選項
4. 針對小值的加權策略
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
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_gpu_cache():
    """清空GPU快取"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==========================================
# 特徵工程
# ==========================================

def create_features(X_raw, feature_set='full'):
    """
    創建擴展特徵
    
    Args:
        X_raw: 原始特徵 [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
        feature_set: 'basic', 'physics', 'full'
        
    Returns:
        X_extended: 擴展後的特徵
        feature_names: 特徵名稱列表
    """
    TYPE = X_raw[:, 0]
    THICKNESS = X_raw[:, 1]
    COVERAGE = X_raw[:, 2]
    
    features = []
    names = []
    
    # 基本特徵
    features.append(TYPE)
    names.append('TYPE')
    
    features.append(THICKNESS)
    names.append('THICKNESS')
    
    features.append(COVERAGE)
    names.append('COVERAGE')
    
    if feature_set in ['physics', 'full']:
        # 物理意義特徵
        features.append(1.0 / COVERAGE)
        names.append('1/COVERAGE')
        
        features.append(THICKNESS / COVERAGE)
        names.append('THICK/COV')
        
        features.append(np.log(THICKNESS + 1))
        names.append('log(THICK)')
        
        features.append(THICKNESS * COVERAGE)
        names.append('THICK*COV')
        
    if feature_set == 'full':
        # Type-specific 特徵
        type1_flag = (TYPE == 1).astype(float)
        type2_flag = (TYPE == 2).astype(float)
        type3_flag = (TYPE == 3).astype(float)
        
        features.append(type1_flag)
        names.append('TYPE1_flag')
        
        features.append(type2_flag)
        names.append('TYPE2_flag')
        
        features.append(type3_flag)
        names.append('TYPE3_flag')
        
        # Type 交互特徵
        features.append(type3_flag * THICKNESS)
        names.append('TYPE3*THICK')
        
        features.append(type3_flag * COVERAGE)
        names.append('TYPE3*COV')
        
        features.append(type3_flag / COVERAGE)
        names.append('TYPE3/COV')
        
        # Type 1&2 的 thickness 效應 (他們有正相關)
        features.append((type1_flag + type2_flag) * THICKNESS)
        names.append('TYPE12*THICK')
        
        # 二次項
        features.append(COVERAGE ** 2)
        names.append('COV^2')
        
        features.append(THICKNESS ** 2 / 10000)  # 縮放
        names.append('THICK^2')
    
    X_extended = np.column_stack(features)
    
    return X_extended, names


# ==========================================
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """深度神經網路特徵提取器"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=16, dropout=0.15):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),  # LeakyReLU 對小值更友好
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.network(x)


class GPRegressionModel(gpytorch.models.ExactGP):
    """高斯過程回歸模型 - 使用 Matérn kernel"""
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor, kernel_type='matern'):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 選擇 kernel
        if kernel_type == 'rbf':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        elif kernel_type == 'matern':
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, 
                ard_num_dims=feature_extractor.output_dim
            )
        elif kernel_type == 'combined':
            # RBF + Matérn 組合
            rbf = gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
            matern = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_extractor.output_dim)
            base_kernel = rbf + matern
        else:
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 損失函數與權重
# ==========================================

def compute_sample_weights(X_raw, y, weight_config):
    """
    計算樣本權重 - 多重策略
    
    Args:
        X_raw: 原始特徵 [TYPE, THICKNESS, COVERAGE]
        y: 目標值
        weight_config: 權重配置字典
    """
    weights = np.ones(len(X_raw))
    
    TYPE = X_raw[:, 0]
    THICKNESS = X_raw[:, 1]
    COVERAGE = X_raw[:, 2]
    
    # 策略1: Type 3 加權
    if weight_config.get('type3_weight', 1.0) > 1.0:
        type3_mask = (TYPE == 3)
        weights[type3_mask] *= weight_config['type3_weight']
    
    # 策略2: 高 Coverage + Type 3 加權
    if weight_config.get('type3_high_cov_weight', 1.0) > 1.0:
        mask = (TYPE == 3) & (COVERAGE >= 0.8)
        weights[mask] *= weight_config['type3_high_cov_weight']
    
    # 策略3: 小值樣本加權 (MAPE 對小值敏感)
    if weight_config.get('small_value_weight', 1.0) > 1.0:
        small_mask = (y < 0.03)  # Theta.JC < 0.03
        weights[small_mask] *= weight_config['small_value_weight']
    
    # 策略4: 邊界區域加權 (Coverage = 0.8 是邊界)
    if weight_config.get('boundary_weight', 1.0) > 1.0:
        boundary_mask = (COVERAGE >= 0.75) & (COVERAGE <= 0.85)
        weights[boundary_mask] *= weight_config['boundary_weight']
    
    return weights


def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    """加權 MAPE 損失"""
    mape = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    return torch.sum(mape * weights) / torch.sum(weights)


def log_cosh_loss(y_pred, y_true, weights):
    """Log-Cosh 損失 - 對異常值更魯棒"""
    diff = y_pred - y_true
    loss = torch.log(torch.cosh(diff + 1e-12))
    return torch.sum(loss * weights) / torch.sum(weights)


# ==========================================
# 訓練
# ==========================================

def train_model(X_train_raw, y_train, config, verbose=True):
    """訓練模型"""
    
    # 特徵工程
    X_train, feature_names = create_features(X_train_raw, config['feature_set'])
    
    if verbose:
        print(f"\n特徵工程: {config['feature_set']}")
        print(f"特徵數: {len(feature_names)}")
        print(f"特徵: {feature_names}")
    
    # 計算樣本權重
    sample_weights_np = compute_sample_weights(X_train_raw, y_train, config['weight_config'])
    
    if verbose:
        print(f"\n樣本權重統計:")
        print(f"  權重 > 1: {np.sum(sample_weights_np > 1)} 筆")
        print(f"  最大權重: {sample_weights_np.max():.1f}")
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    
    # 選擇是否在 log 空間建模
    if config.get('log_target', False):
        y_train_transformed = np.log(y_train + 1e-6)
    else:
        y_train_transformed = y_train
    
    y_train_scaled = scaler_y.fit_transform(y_train_transformed.reshape(-1, 1)).flatten()
    
    # 轉換為 tensor
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
    model = GPRegressionModel(
        train_x, train_y, likelihood, feature_extractor,
        kernel_type=config.get('kernel_type', 'matern')
    ).to(device)
    
    # 優化器
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr'], 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.mean_module.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.likelihood.parameters(), 'lr': config['lr'] * 0.1},
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # 訓練
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        
        # 混合損失
        mape = weighted_mape_loss(output.mean, train_y, sample_weights)
        lc_loss = log_cosh_loss(output.mean, train_y, sample_weights)
        
        total_loss = gp_loss + config['mape_weight'] * mape + config.get('lc_weight', 0) * lc_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: GP={gp_loss.item():.4f}, MAPE={mape.item():.2f}%, Total={current_loss:.4f}")
        
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_state = {
                'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'likelihood': {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
            }
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            if verbose:
                print(f"早停 at Epoch {epoch+1}")
            break
    
    # 載入最佳模型
    model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
    likelihood.load_state_dict({k: v.to(device) for k, v in best_state['likelihood'].items()})
    
    return model, likelihood, scaler_x, scaler_y, feature_names, config.get('log_target', False)


def evaluate_model(model, likelihood, X_test_raw, y_test, 
                   scaler_x, scaler_y, feature_names, log_target, 
                   feature_set, verbose=True):
    """評估模型"""
    
    model.eval()
    likelihood.eval()
    
    # 特徵工程
    X_test, _ = create_features(X_test_raw, feature_set)
    X_test_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_test_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred_scaled = pred_dist.mean.cpu().numpy()
        y_std_scaled = pred_dist.stddev.cpu().numpy()
    
    # 反標準化
    y_pred_transformed = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    if log_target:
        y_pred = np.exp(y_pred_transformed) - 1e-6
        y_pred = np.maximum(y_pred, 0.001)  # 確保正值
    else:
        y_pred = y_pred_transformed
    
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    # 計算指標
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    # Type 分析
    type3_mask = X_test_raw[:, 0] == 3
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
        print(f"\n異常點 (>20%):")
        print(f"  總數: {outliers_20}/{len(y_test)} ({outliers_20/len(y_test)*100:.2f}%)")
        print(f"  Type 3: {type3_outliers}/{np.sum(type3_mask)}")
        print(f"\n其他閾值:")
        print(f"  >15%: {outliers_15}/{len(y_test)}")
        print(f"  >10%: {outliers_10}/{len(y_test)}")
        
        # 顯示異常點詳情
        if outliers_20 > 0:
            print(f"\n異常點詳情:")
            outlier_idx = np.where(relative_errors > 20)[0]
            for idx in outlier_idx:
                print(f"  Type={X_test_raw[idx, 0]:.0f}, Thick={X_test_raw[idx, 1]:.0f}, "
                      f"Cov={X_test_raw[idx, 2]:.1f}: True={y_test[idx]:.4f}, "
                      f"Pred={y_pred[idx]:.4f}, Err={relative_errors[idx]:.1f}%")
    
    return {
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


def save_predictions(X_test, y_test, results, filename):
    """保存預測結果"""
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
    print(f"✓ 預測結果已保存: {filename}")


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, verbose=True):
    """主訓練流程"""
    
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}")
    print(f"隨機種子: {seed}\n")
    
    print("="*60)
    print("Phase 2C: 特徵工程改進版 DKL")
    print("="*60)
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 訓練集清理
    train_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_above[feature_cols].values
    y_test = test_above[target_col].values
    
    print(f"\n訓練集: {len(train_clean)} 筆 (去重後)")
    print(f"測試集: {len(test_above)} 筆")
    
    # 配置
    config = {
        # 特徵工程
        'feature_set': 'full',  # 'basic', 'physics', 'full'
        
        # 模型架構
        'hidden_dims': [128, 64, 32],
        'feature_dim': 16,
        'dropout': 0.15,
        'kernel_type': 'matern',  # 'rbf', 'matern', 'combined'
        
        # 訓練參數
        'lr': 0.005,
        'epochs': 600,
        'patience': 80,
        'mape_weight': 0.15,
        'lc_weight': 0.05,
        
        # 是否在 log 空間建模
        'log_target': True,
        
        # 樣本權重配置
        'weight_config': {
            'type3_weight': 1.5,
            'type3_high_cov_weight': 2.0,
            'small_value_weight': 2.5,
            'boundary_weight': 1.5,
        }
    }
    
    print(f"\n配置:")
    print(f"  Feature set: {config['feature_set']}")
    print(f"  Kernel: {config['kernel_type']}")
    print(f"  Hidden dims: {config['hidden_dims']}")
    print(f"  Log target: {config['log_target']}")
    print(f"  Weight config: {config['weight_config']}")
    
    # 訓練
    print(f"\n{'='*60}")
    print("開始訓練...")
    print(f"{'='*60}")
    
    model, likelihood, scaler_x, scaler_y, feature_names, log_target = train_model(
        X_train, y_train, config, verbose=verbose
    )
    
    # 評估
    results = evaluate_model(
        model, likelihood, X_test, y_test,
        scaler_x, scaler_y, feature_names, log_target,
        config['feature_set'], verbose=verbose
    )
    
    # 保存
    save_predictions(X_test, y_test, results, 
                     f'phase2c_above_seed{seed}_predictions.csv')
    
    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
