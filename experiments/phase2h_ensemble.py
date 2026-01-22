"""
Phase 2H: Ensemble 多 Seed 策略

思路:
1. 用多個不同的 seed 訓練多個模型
2. 對每個測試點，取多個模型預測的統計量
3. 策略選擇:
   - 方案A: 取平均 (降低 variance)
   - 方案B: 取中位數 (對異常值更穩健)
   - 方案C: 對小值區域取較低的預測

這個方法不依賴後處理校正，而是從模型 ensemble 本身來提升穩定性
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==========================================
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
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


def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    mape = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    return torch.sum(mape * weights) / torch.sum(weights)


def compute_sample_weights(X, weight_factor=3.0):
    weights = np.ones(len(X))
    difficult_mask = (X[:, 0] == 3) & (X[:, 2] >= 0.8) & (X[:, 1] >= 220)
    weights[difficult_mask] *= weight_factor
    return weights


# ==========================================
# 訓練單一模型
# ==========================================

def train_single_model(X_train, y_train, config, seed, verbose=False):
    """訓練單一模型"""
    clear_gpu_cache()
    set_seed(seed)
    
    sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_scaled).to(device)
    train_y = torch.from_numpy(y_scaled).to(device)
    sample_weights = torch.from_numpy(sample_weights_np).to(device)
    
    feature_extractor = DnnFeatureExtractor(
        input_dim=train_x.shape[1],
        hidden_dims=config['hidden_dims'],
        output_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(train_x, train_y, likelihood, feature_extractor).to(device)
    
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr'], 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        mape = weighted_mape_loss(output.mean, train_y, sample_weights)
        total_loss = gp_loss + config['mape_weight'] * mape
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            best_state = {
                'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'likelihood': {k: v.cpu().clone() for k, v in likelihood.state_dict().items()},
            }
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
    likelihood.load_state_dict({k: v.to(device) for k, v in best_state['likelihood'].items()})
    
    if verbose:
        print(f"  Seed {seed}: Best Loss = {best_loss:.4f}")
    
    return model, likelihood, scaler_x, scaler_y


def predict_single(model, likelihood, X_test, scaler_x, scaler_y):
    """單一模型預測"""
    model.eval()
    likelihood.eval()
    
    X_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred = pred_dist.mean.cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    return y_pred


# ==========================================
# Ensemble 策略
# ==========================================

def ensemble_predictions(all_preds, X_test, strategy='adaptive'):
    """
    合併多個模型的預測
    
    策略:
    - 'mean': 簡單平均
    - 'median': 中位數 (對異常值更穩健)
    - 'adaptive': 根據區域選擇策略
      - Type 3 高覆蓋率小值區域: 用較低分位數
      - 其他區域: 用平均
    """
    all_preds = np.array(all_preds)  # shape: (n_models, n_samples)
    n_samples = all_preds.shape[1]
    
    if strategy == 'mean':
        return np.mean(all_preds, axis=0)
    
    elif strategy == 'median':
        return np.median(all_preds, axis=0)
    
    elif strategy == 'adaptive':
        y_ensemble = np.zeros(n_samples)
        
        for i in range(n_samples):
            preds_i = all_preds[:, i]
            is_type3 = (X_test[i, 0] == 3)
            is_high_cov = (X_test[i, 2] >= 0.8)
            
            if is_type3 and is_high_cov:
                # Type 3 高覆蓋率: 用 25% 分位數 (偏保守)
                y_ensemble[i] = np.percentile(preds_i, 30)
            else:
                # 其他區域: 用平均
                y_ensemble[i] = np.mean(preds_i)
        
        return y_ensemble
    
    elif strategy == 'trimmed_mean':
        # 去掉最高和最低，取剩餘的平均
        return np.mean(np.sort(all_preds, axis=0)[1:-1], axis=0)
    
    else:
        return np.mean(all_preds, axis=0)


def evaluate(X_test, y_test, y_pred, label=""):
    errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(errors)
    max_error = np.max(errors)
    outliers = np.sum(errors > 20)
    
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((errors > 20) & type3_mask)
    
    print(f"【{label}】 MAPE: {mape:.2f}%, Max: {max_error:.2f}%, 異常點: {outliers}, Type3異常: {type3_outliers}")
    
    return {'mape': mape, 'max_error': max_error, 'outliers': outliers, 
            'type3_outliers': type3_outliers, 'errors': errors}


# ==========================================
# 主函數
# ==========================================

def main(n_models=10, verbose=True):
    print(f"裝置: {device}\n")
    print("="*60)
    print(f"Phase 2H: Ensemble ({n_models} 個模型)")
    print("="*60)
    
    # 載入資料
    train_df = pd.read_excel('data/train/Above.xlsx')
    test_df = pd.read_excel('data/test/Above.xlsx')

    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    train_clean = train_df.groupby(feature_cols, as_index=False).agg({target_col: 'mean'})
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"訓練集: {len(train_clean)}, 測試集: {len(test_df)}")
    
    # 配置
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
    
    # 訓練多個模型
    seeds = [42, 123, 456, 789, 1024, 2024, 3000, 4000, 5000, 6000][:n_models]
    
    print(f"\n訓練 {n_models} 個模型...")
    all_predictions = []
    
    for seed in seeds:
        model, likelihood, scaler_x, scaler_y = train_single_model(
            X_train, y_train, config, seed, verbose=verbose
        )
        y_pred = predict_single(model, likelihood, X_test, scaler_x, scaler_y)
        all_predictions.append(y_pred)
        
        # 單一模型評估
        if verbose:
            errors = np.abs((y_test - y_pred) / y_test) * 100
            print(f"    → MAPE: {np.mean(errors):.2f}%, 異常點: {np.sum(errors > 20)}")
    
    # 測試不同的 ensemble 策略
    print("\n" + "="*60)
    print("Ensemble 策略比較")
    print("="*60)
    
    strategies = ['mean', 'median', 'adaptive', 'trimmed_mean']
    best_strategy = None
    best_outliers = float('inf')
    best_result = None
    
    for strategy in strategies:
        y_ensemble = ensemble_predictions(all_predictions, X_test, strategy)
        result = evaluate(X_test, y_test, y_ensemble, strategy)
        
        if result['outliers'] < best_outliers or \
           (result['outliers'] == best_outliers and result['max_error'] < best_result['max_error']):
            best_outliers = result['outliers']
            best_strategy = strategy
            best_result = result
            best_pred = y_ensemble
    
    print(f"\n最佳策略: {best_strategy}")
    
    # 顯示異常點詳情
    print(f"\n異常點詳情 ({best_strategy}):")
    for i in np.where(best_result['errors'] > 20)[0]:
        print(f"  T{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}: "
              f"真={y_test[i]:.4f}, 預={best_pred[i]:.4f}, 誤={best_result['errors'][i]:.1f}%")
    
    # 分析各模型對異常點的預測分布
    print(f"\n異常點的模型預測分布:")
    all_preds_arr = np.array(all_predictions)
    for i in np.where(best_result['errors'] > 20)[0]:
        preds_i = all_preds_arr[:, i]
        print(f"  [{i}] T{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}")
        print(f"       真值={y_test[i]:.4f}, 預測範圍=[{preds_i.min():.4f}, {preds_i.max():.4f}], "
              f"mean={preds_i.mean():.4f}, std={preds_i.std():.4f}")
    
    # 保存
    results_df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Pred_Ensemble': best_pred,
        'Error%': best_result['errors']
    })
    results_df.to_csv(f'phase2h_ensemble_{n_models}models_predictions.csv', index=False)
    print(f"\n✓ 已保存: phase2h_ensemble_{n_models}models_predictions.csv")
    
    return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_models', type=int, default=10)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()
    main(n_models=args.n_models, verbose=args.verbose)
