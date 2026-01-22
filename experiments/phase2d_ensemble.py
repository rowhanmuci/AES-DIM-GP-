"""
Phase 2D Ensemble 版本 - 多模型集成
整合以下策略:
1. 標準 DKL 模型
2. Type 3 專家模型
3. 對數空間模型
4. 高權重困難樣本模型
5. 動態加權集成

使用方法:
    python phase2d_ensemble.py --seed 2024 --n-models 3
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
from tqdm import tqdm

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
# 特徵工程 (從 phase2c 導入)
# ==========================================

def create_enhanced_features(X):
    """增強特徵"""
    tim_type = X[:, 0:1]
    thickness = X[:, 1:2]
    coverage = X[:, 2:3]
    
    features = [X]
    
    # Coverage 非線性
    features.append(coverage ** 2)
    features.append(coverage ** 3)
    features.append(np.sqrt(coverage))
    
    # Coverage 臨界值指示器
    features.append((np.abs(coverage - 0.8) < 0.1).astype(float))
    features.append((np.abs(coverage - 1.0) < 0.1).astype(float))
    features.append((coverage >= 0.75).astype(float))
    features.append((coverage >= 0.9).astype(float))
    
    # 交互作用
    features.append(thickness * coverage)
    features.append(thickness * coverage ** 2)
    features.append(thickness ** 2 * coverage)
    features.append(thickness / (1.01 - coverage + 1e-8))
    
    # 對數特徵
    features.append(np.log(thickness + 1))
    features.append(np.log(coverage + 0.01))
    features.append(np.exp(coverage))
    
    # Type-specific
    features.append(tim_type * thickness)
    features.append(tim_type * coverage)
    features.append(tim_type * thickness * coverage)
    
    return np.hstack(features)


def compute_advanced_weights(X, y=None, weight_config=None):
    """進階樣本加權"""
    if weight_config is None:
        weight_config = {
            'type3_base': 2.0,
            'high_coverage': 5.0,
            'small_value': 3.0,
        }
    
    weights = np.ones(len(X))
    
    tim_type = X[:, 0]
    coverage = X[:, 2]
    
    # Type 3 基礎權重
    type3_mask = tim_type == 3
    weights[type3_mask] *= weight_config['type3_base']
    
    # 高 Coverage 區域
    high_cov_mask = (
        ((coverage >= 0.75) & (coverage <= 0.85)) |
        (coverage >= 0.95)
    )
    weights[type3_mask & high_cov_mask] *= weight_config['high_coverage']
    
    # 極小真實值
    if y is not None:
        small_value_mask = y < 0.03
        weights[small_value_mask] *= weight_config['small_value']
    
    return weights


# ==========================================
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """深度神經網路特徵提取器"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=16, dropout=0.2):
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
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim) +
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_extractor.output_dim) +
            gpytorch.kernels.LinearKernel()
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
    """加權MAPE損失"""
    mape_per_sample = torch.abs((y_true - y_pred) / 
                                (torch.abs(y_true) + epsilon)) * 100
    weighted_mape = torch.sum(mape_per_sample * weights) / torch.sum(weights)
    return weighted_mape


# ==========================================
# 訓練函數
# ==========================================

def train_single_model(X_train, y_train, config, sample_weights=None, 
                      model_type='standard', verbose=False):
    """
    訓練單個模型
    
    Args:
        model_type: 'standard', 'type3_specialist', 'log_space', 'high_weight'
    """
    # 根據模型類型調整資料和權重
    if model_type == 'type3_specialist':
        # Type 3 專家模型 - 只用 Type 3 資料
        type3_mask = X_train[:, 0] == 3
        X_train_use = X_train[type3_mask]
        y_train_use = y_train[type3_mask]
        if sample_weights is not None:
            sample_weights = sample_weights[type3_mask]
        if verbose:
            print(f"  Type 3 專家模型: 使用 {len(X_train_use)} 筆 Type 3 樣本")
    
    elif model_type == 'log_space':
        # 對數空間模型
        X_train_use = X_train
        y_train_use = np.log(y_train + 1e-6)  # 轉換到對數空間
        if verbose:
            print(f"  對數空間模型: 在 log 空間訓練")
    
    elif model_type == 'high_weight':
        # 超高權重困難樣本
        X_train_use = X_train
        y_train_use = y_train
        if sample_weights is None:
            sample_weights = compute_advanced_weights(X_train, y_train)
        sample_weights = sample_weights.copy()
        
        # 進一步增加困難樣本權重
        type3_mask = X_train[:, 0] == 3
        high_cov_mask = (X_train[:, 2] >= 0.75)
        sample_weights[type3_mask & high_cov_mask] *= 3.0
        
        if verbose:
            print(f"  高權重模型: 困難樣本權重 × 15")
    
    else:  # standard
        X_train_use = X_train
        y_train_use = y_train
    
    # 特徵增強
    X_train_enhanced = create_enhanced_features(X_train_use)
    
    # 計算樣本權重 (如果還沒有)
    if sample_weights is None:
        sample_weights = compute_advanced_weights(X_train_use, y_train_use)
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train_enhanced)
    y_train_scaled = scaler_y.fit_transform(y_train_use.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_train_scaled).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    weights_tensor = torch.from_numpy(sample_weights).to(device)
    
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
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr'], 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters(), 'lr': config['lr']},
        {'params': model.mean_module.parameters(), 'lr': config['lr']},
        {'params': model.likelihood.parameters(), 'lr': config['lr']},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # 訓練
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        mape = weighted_mape_loss(output.mean, train_y, weights_tensor)
        total_loss = gp_loss + config['mape_weight'] * mape
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        
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
            break
    
    # 載入最佳模型
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'model_type': model_type,
        'is_log_space': (model_type == 'log_space')
    }


def predict_single_model(model_dict, X_test):
    """單個模型預測"""
    model = model_dict['model']
    likelihood = model_dict['likelihood']
    scaler_x = model_dict['scaler_x']
    scaler_y = model_dict['scaler_y']
    is_log_space = model_dict['is_log_space']
    
    model.eval()
    likelihood.eval()
    
    # 特徵增強
    X_test_enhanced = create_enhanced_features(X_test)
    X_test_scaled = scaler_x.transform(X_test_enhanced)
    test_x = torch.from_numpy(X_test_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred_scaled = pred_dist.mean.cpu().numpy()
        y_std_scaled = pred_dist.stddev.cpu().numpy()
    
    # 反標準化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    # 如果是對數空間，轉回原始空間
    if is_log_space:
        y_pred = np.exp(y_pred) - 1e-6
        y_pred = np.maximum(y_pred, 0.001)  # 確保非負
    
    return y_pred, y_std


# ==========================================
# Ensemble 集成
# ==========================================

def train_ensemble(X_train, y_train, config, n_models=3, verbose=True):
    """
    訓練多模型集成
    
    Returns:
        models: 模型列表
    """
    models = []
    
    model_types = ['standard', 'type3_specialist', 'high_weight']
    if n_models > 3:
        model_types.extend(['log_space'] * (n_models - 3))
    
    print(f"\n訓練 {n_models} 個子模型...")
    
    for i in range(n_models):
        model_type = model_types[i % len(model_types)]
        
        if verbose:
            print(f"\n模型 {i+1}/{n_models}: {model_type}")
        
        # 每個模型使用不同種子
        set_seed(config['seed'] + i * 100)
        
        model_dict = train_single_model(
            X_train, y_train, config,
            model_type=model_type,
            verbose=verbose
        )
        
        models.append(model_dict)
        
        clear_gpu_cache()
    
    print(f"\n✓ {n_models} 個子模型訓練完成")
    
    return models


def ensemble_predict(models, X_test):
    """
    集成預測 - 動態加權
    
    Args:
        models: 模型列表
        X_test: 測試特徵
    
    Returns:
        y_pred: 集成預測結果
        y_std: 不確定性
    """
    n_models = len(models)
    all_preds = []
    all_stds = []
    
    # 獲取所有模型預測
    for model_dict in models:
        y_pred, y_std = predict_single_model(model_dict, X_test)
        all_preds.append(y_pred)
        all_stds.append(y_std)
    
    all_preds = np.array(all_preds)  # (n_models, n_samples)
    all_stds = np.array(all_stds)
    
    # 計算動態權重
    weights = np.ones((len(X_test), n_models))
    
    # Type 3 + 高 coverage → 增加專家模型權重
    for i, model_dict in enumerate(models):
        if model_dict['model_type'] == 'type3_specialist':
            type3_high_cov = (X_test[:, 0] == 3) & (X_test[:, 2] >= 0.75)
            weights[type3_high_cov, i] *= 3.0
        
        elif model_dict['model_type'] == 'log_space':
            # 對數空間模型適合極小值
            type3_high_cov = (X_test[:, 0] == 3) & (X_test[:, 2] >= 0.75)
            weights[type3_high_cov, i] *= 2.0
    
    # 歸一化權重
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # 加權平均
    y_pred_ensemble = np.sum(all_preds.T * weights, axis=1)
    
    # 集成不確定性 (考慮模型間差異)
    y_std_ensemble = np.sqrt(
        np.mean(all_stds ** 2, axis=0) +  # 平均方差
        np.var(all_preds, axis=0)  # 模型間方差
    )
    
    return y_pred_ensemble, y_std_ensemble


# ==========================================
# 評估函數
# ==========================================

def evaluate_ensemble(models, X_test, y_test, verbose=True):
    """評估集成模型"""
    
    # 集成預測
    y_pred, y_std = ensemble_predict(models, X_test)
    
    # 計算指標
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    # Type 3 分析
    type3_mask = X_test[:, 0] == 3
    if np.sum(type3_mask) > 0:
        type3_errors = relative_errors[type3_mask]
        type3_mape = np.mean(type3_errors)
        type3_outliers = np.sum(type3_errors > 20)
        
        # Coverage 0.8 分析
        cov08_mask = type3_mask & (X_test[:, 2] == 0.8)
        if np.sum(cov08_mask) > 0:
            cov08_errors = relative_errors[cov08_mask]
            cov08_mape = np.mean(cov08_errors)
            cov08_outliers = np.sum(cov08_errors > 20)
        else:
            cov08_mape = 0
            cov08_outliers = 0
    else:
        type3_mape = 0
        type3_outliers = 0
        cov08_mape = 0
        cov08_outliers = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Ensemble 評估結果 ({len(models)} 個子模型)")
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
            print(f"\nType 3 詳細:")
            print(f"  樣本數: {np.sum(type3_mask)}")
            print(f"  MAPE: {type3_mape:.2f}%")
            print(f"  異常點: {type3_outliers}/{np.sum(type3_mask)}")
            if np.sum(cov08_mask) > 0:
                print(f"\n  Coverage 0.8 子集:")
                print(f"    MAPE: {cov08_mape:.2f}%")
                print(f"    異常點: {cov08_outliers}/{np.sum(cov08_mask)}")
        print(f"{'='*60}\n")
    
    results = {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'type3_mape': type3_mape,
        'type3_outliers': type3_outliers,
        'cov08_mape': cov08_mape,
        'cov08_outliers': cov08_outliers,
        'predictions': y_pred,
        'std': y_std,
        'errors': relative_errors
    }
    
    return results


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
    print(f"✓ 預測結果已保存到: {filename}")


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, n_models=3, verbose=True):
    """主訓練流程"""
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}\n")
    
    print("="*60)
    print(f"Phase 2D: Ensemble 集成版本 ({n_models} 個子模型)")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置
    config = {
        'hidden_dims': [128, 64, 32],
        'feature_dim': 16,
        'dropout': 0.2,
        'lr': 0.008,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.15,
        'seed': seed,
    }
    
    if verbose:
        print(f"\n配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # ==========================================
    # Above Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Above 50% Coverage (Ensemble)'}\n")
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    # 訓練集清理
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆")
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    # 訓練 Ensemble
    models = train_ensemble(
        X_train_above, y_train_above, config,
        n_models=n_models, verbose=verbose
    )
    
    # 評估
    results_above = evaluate_ensemble(
        models, X_test_above, y_test_above, verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_above, y_test_above, results_above, 
                     f'phase2d_ensemble_{n_models}models_seed{seed}_predictions.csv')
    
    # ==========================================
    # 總結
    # ==========================================
    
    print("\n" + "="*60)
    print(f"最終結果總結 (Phase 2D Ensemble - {n_models} 個子模型)")
    print("="*60)
    print(f"隨機種子: {seed}")
    print(f"\nAbove資料集:")
    print(f"  總體 MAPE: {results_above['mape']:.2f}%")
    print(f"  異常點 (>20%): {results_above['outliers_20']}/{len(y_test_above)} ({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"\n  Type 3 MAPE: {results_above['type3_mape']:.2f}%")
    print(f"  Type 3 異常點: {results_above['type3_outliers']}")
    print(f"\n  Coverage 0.8 (Type 3) MAPE: {results_above['cov08_mape']:.2f}%")
    print(f"  Coverage 0.8 異常點: {results_above['cov08_outliers']}")
    
    print("\nEnsemble 策略:")
    print(f"  ✓ {n_models} 個子模型動態加權集成")
    print("  ✓ Type 3 專家模型")
    print("  ✓ 對數空間模型")
    print("  ✓ 高權重困難樣本模型")
    print("  ✓ 不確定性量化")
    
    print("\n" + "="*60)
    print("✓ 訓練完成！")
    print("="*60 + "\n")
    
    return results_above


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2D Ensemble 版本')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('--n-models', type=int, default=3, help='子模型數量 (建議 3-5)')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細模式')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, n_models=args.n_models, verbose=args.verbose)
    
    print("\n💡 使用說明:")
    print(f"  當前: {args.n_models} 個子模型集成")
    print("  建議嘗試: 3-5 個子模型達到最佳效果")
    print("  預期改善: Type 3 異常點 5/18 → 2-3/18")
    print("           Coverage 0.8 MAPE 26.92% → <12%\n")
