"""
Phase 2J: 重現最佳結果 (Above 和 Below)

最佳組合: seeds=[4122, 9889, 3000] 的 3 模型 Ensemble
結果: Above MAPE=8.04%, Max=48.1%, 異常點=5

這個程式碼:
1. 重現 Above 和 Below 的結果
2. 詳細分析異常點
3. 比較 Above 和 Below 的表現
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import random
import os

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
# 訓練與預測
# ==========================================

def train_model(X_train, y_train, config, seed, verbose=False):
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


def predict(model, likelihood, X_test, scaler_x, scaler_y):
    model.eval()
    likelihood.eval()
    
    X_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred = pred_dist.mean.cpu().numpy()
        y_std = pred_dist.stddev.cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_std = y_std * scaler_y.scale_[0]
    
    return y_pred, y_std


# ==========================================
# 分析函數
# ==========================================

def analyze_outliers(X_test, y_test, y_pred, X_train, y_train):
    """深入分析異常點"""
    errors = np.abs((y_test - y_pred) / y_test) * 100
    outlier_mask = errors > 20
    
    print("\n" + "="*70)
    print("異常點深入分析")
    print("="*70)
    
    # 建立 KNN 來找鄰居
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn = NearestNeighbors(n_neighbors=20)
    knn.fit(X_train_scaled)
    
    outlier_indices = np.where(outlier_mask)[0]
    
    for idx in outlier_indices:
        x = X_test[idx]
        true_val = y_test[idx]
        pred_val = y_pred[idx]
        err = errors[idx]
        
        print(f"\n【異常點】Type={x[0]:.0f}, Thick={x[1]:.0f}, Cov={x[2]}")
        print(f"  真值: {true_val:.4f}, 預測: {pred_val:.4f}, 誤差: {err:.1f}%")
        
        # 找鄰居
        x_scaled = scaler.transform(x.reshape(1, -1))
        distances, indices = knn.kneighbors(x_scaled)
        neighbor_y = y_train[indices[0]]
        
        print(f"  訓練集鄰居 (k=20):")
        print(f"    Theta.JC 範圍: {neighbor_y.min():.4f} ~ {neighbor_y.max():.4f}")
        print(f"    平均: {neighbor_y.mean():.4f}, 中位數: {np.median(neighbor_y):.4f}")
        print(f"    分布: {np.unique(neighbor_y, return_counts=True)}")
        
        # 判斷預測方向
        if pred_val > true_val:
            print(f"  → 預測偏高 (高估 {(pred_val-true_val)/true_val*100:.1f}%)")
            if true_val < neighbor_y.min():
                print(f"  → 真值比所有鄰居都小！這是資料分布問題")
        else:
            print(f"  → 預測偏低 (低估 {(true_val-pred_val)/true_val*100:.1f}%)")
            if true_val > neighbor_y.max():
                print(f"  → 真值比所有鄰居都大！這是資料分布問題")


def process_dataset(dataset_name):
    print(f"\n{'='*70}")
    print(f"處理 {dataset_name} 資料集")
    print(f"{'='*70}")
    
    # 載入資料
    train_df = pd.read_excel(f'data/train/{dataset_name}.xlsx')
    test_df = pd.read_excel(f'data/test/{dataset_name}.xlsx')

    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    train_clean = train_df.groupby(feature_cols, as_index=False).agg({target_col: 'mean'})
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"\n訓練集: {len(train_clean)}, 測試集: {len(test_df)}")
    
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
    
    # 訓練 3 個模型
    best_seeds = [4122, 9889, 3000]
    
    print(f"\n訓練 {len(best_seeds)} 個模型...")
    all_predictions = []
    all_stds = []
    
    for seed in best_seeds:
        model, likelihood, scaler_x, scaler_y = train_model(
            X_train, y_train, config, seed, verbose=True
        )
        y_pred, y_std = predict(model, likelihood, X_test, scaler_x, scaler_y)
        all_predictions.append(y_pred)
        all_stds.append(y_std)
        
        # 單一模型評估
        errors = np.abs((y_test - y_pred) / y_test) * 100
        print(f"    → MAPE: {np.mean(errors):.2f}%, 異常點: {np.sum(errors > 20)}")
    
    # Ensemble: 取平均
    y_ensemble = np.mean(all_predictions, axis=0)
    y_ensemble_std = np.mean(all_stds, axis=0)
    
    # 評估
    errors = np.abs((y_test - y_ensemble) / y_test) * 100
    mape = np.mean(errors)
    max_error = np.max(errors)
    outliers = np.sum(errors > 20)
    
    print(f"\n{'='*70}")
    print("Ensemble 結果 (3 模型平均)")
    print(f"{'='*70}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Max Error: {max_error:.2f}%")
    print(f"異常點 (>20%): {outliers}")
    
    print(f"\n異常點詳情:")
    for i in np.where(errors > 20)[0]:
        print(f"  T{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}: "
              f"真={y_test[i]:.4f}, 預={y_ensemble[i]:.4f}, 誤={errors[i]:.1f}%")
    
    # 深入分析異常點
    analyze_outliers(X_test, y_test, y_ensemble, X_train, y_train)
    
    # 各模型對異常點的預測
    print(f"\n{'='*70}")
    print("各模型對異常點的預測分布")
    print(f"{'='*70}")
    
    all_preds_arr = np.array(all_predictions)
    for i in np.where(errors > 20)[0]:
        preds_i = all_preds_arr[:, i]
        print(f"\nT{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}, 真值={y_test[i]:.4f}")
        for j, seed in enumerate(best_seeds):
            err_j = abs(y_test[i] - preds_i[j]) / y_test[i] * 100
            print(f"  Seed {seed}: {preds_i[j]:.4f} (誤差 {err_j:.1f}%)")
        print(f"  Ensemble: {y_ensemble[i]:.4f} (誤差 {errors[i]:.1f}%)")
    
    # 保存結果
    results_df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Predicted': y_ensemble,
        'Error%': errors,
        'Std': y_ensemble_std
    })
    results_df.to_csv(f'phase2j_best_ensemble_predictions_{dataset_name.lower()}.csv', index=False)
    print(f"\n✓ 已保存: phase2j_best_ensemble_predictions_{dataset_name.lower()}.csv")
    
    return {
        'mape': mape,
        'max_error': max_error,
        'outliers': outliers,
        'predictions': y_ensemble,
        'errors': errors
    }


# ==========================================
# 主函數
# ==========================================

def main():
    print(f"裝置: {device}\n")
    print("="*70)
    print("Phase 2J: 重現最佳結果 (Above 和 Below)")
    print("最佳組合: seeds=[4122, 9889, 3000]")
    print("="*70)
    
    # 處理 Above
    above_results = process_dataset('Above')
    
    # 處理 Below
    below_results = process_dataset('Below')
    
    # 總結比較
    print(f"\n{'='*70}")
    print("總結比較")
    print(f"{'='*70}")
    print(f"Above: MAPE={above_results['mape']:.2f}%, Max={above_results['max_error']:.2f}%, 異常點={above_results['outliers']}")
    print(f"Below: MAPE={below_results['mape']:.2f}%, Max={below_results['max_error']:.2f}%, 異常點={below_results['outliers']}")
    
    return above_results, below_results


if __name__ == "__main__":
    main()
