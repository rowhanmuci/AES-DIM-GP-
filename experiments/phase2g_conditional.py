"""
Phase 2G: 條件式保守校正 (Conditional Conservative Correction)

改進自 Phase 2F:
- 只對「預測明顯偏高」的點做校正
- 避免把原本OK的預測弄壞

校正條件:
1. 必須是 Type 3 + Coverage >= 0.8 (高變異區域)
2. 模型預測 > 鄰居的某個閾值 (Q50 或 mean)
3. 鄰居變異係數 CV > 0.2
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
    difficult_mask = (
        (X[:, 0] == 3) & (X[:, 2] >= 0.8) & (X[:, 1] >= 220)
    )
    weights[difficult_mask] *= weight_factor
    return weights


# ==========================================
# 條件式保守校正
# ==========================================

class ConditionalCorrector:
    """條件式保守校正器"""
    
    def __init__(self, X_train, y_train, k=20):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.knn = NearestNeighbors(n_neighbors=k)
        self.knn.fit(X_scaled)
    
    def get_neighbor_stats(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        distances, indices = self.knn.kneighbors(X_scaled)
        
        stats = []
        for idx in indices:
            neighbor_y = self.y_train[idx]
            stats.append({
                'mean': np.mean(neighbor_y),
                'std': np.std(neighbor_y),
                'min': np.min(neighbor_y),
                'max': np.max(neighbor_y),
                'q10': np.percentile(neighbor_y, 10),
                'q25': np.percentile(neighbor_y, 25),
                'q50': np.percentile(neighbor_y, 50),
                'q75': np.percentile(neighbor_y, 75),
                'cv': np.std(neighbor_y) / (np.mean(neighbor_y) + 1e-8),
            })
        return stats
    
    def correct(self, X_test, y_pred):
        """
        條件式校正
        
        只有滿足以下所有條件才校正:
        1. Type 3 且 Coverage >= 0.8
        2. CV > 0.2 (高變異區域)
        3. 預測值 > 鄰居 Q50 (預測偏高)
        """
        stats = self.get_neighbor_stats(X_test)
        y_corrected = np.copy(y_pred)
        info = []
        
        for i in range(len(X_test)):
            s = stats[i]
            is_type3 = (X_test[i, 0] == 3)
            cov = X_test[i, 2]
            
            corrected = False
            reason = "不需校正"
            
            # 條件 1: Type 3 且高覆蓋率
            if is_type3 and cov >= 0.8:
                # 條件 2: 高變異區域
                if s['cv'] > 0.2:
                    # 條件 3: 預測偏高
                    # 使用不同的閾值: Coverage=1.0 用 Q50, Coverage=0.8 用更高的閾值
                    if cov >= 0.95:
                        threshold = s['q50']
                        target = s['q25']
                    else:
                        threshold = s['q75']
                        target = s['q50']
                    
                    if y_pred[i] > threshold:
                        # 校正: 往 target 方向拉
                        alpha = 0.5  # 50% 混合
                        y_corrected[i] = alpha * y_pred[i] + (1 - alpha) * target
                        corrected = True
                        reason = f"預測({y_pred[i]:.4f}) > 閾值({threshold:.4f})"
            
            # 額外保護: 不要校正到太低
            # 如果校正後比 Q10 還低，就拉回來一點
            if y_corrected[i] < s['q10']:
                y_corrected[i] = s['q10']
            
            info.append({
                'original': y_pred[i],
                'corrected': y_corrected[i],
                'was_corrected': corrected,
                'reason': reason,
                'cv': s['cv'],
                'q25': s['q25'],
                'q50': s['q50'],
                'q75': s['q75'],
            })
        
        return y_corrected, info


# ==========================================
# 訓練
# ==========================================

def train_model(X_train, y_train, config, verbose=True):
    sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
    
    if verbose:
        print(f"困難樣本: {np.sum(sample_weights_np > 1)}")
    
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
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss.item():.4f}")
        
        if patience_counter >= config['patience']:
            if verbose:
                print(f"  早停 at Epoch {epoch+1}")
            break
    
    model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
    likelihood.load_state_dict({k: v.to(device) for k, v in best_state['likelihood'].items()})
    
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


def evaluate(X_test, y_test, y_pred, label="", show_details=True):
    errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(errors)
    max_error = np.max(errors)
    outliers = np.sum(errors > 20)
    
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((errors > 20) & type3_mask)
    
    print(f"\n【{label}】")
    print(f"  MAPE: {mape:.2f}%, Max: {max_error:.2f}%")
    print(f"  異常點(>20%): {outliers}/138, Type3: {type3_outliers}/18")
    
    if show_details and outliers > 0:
        print(f"  詳情:")
        for i in np.where(errors > 20)[0]:
            print(f"    T{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}: "
                  f"真={y_test[i]:.4f}, 預={y_pred[i]:.4f}, 誤={errors[i]:.1f}%")
    
    return {'mape': mape, 'max_error': max_error, 'outliers': outliers, 
            'type3_outliers': type3_outliers, 'errors': errors}


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, verbose=True):
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"裝置: {device}, Seed: {seed}\n")
    print("="*60)
    print("Phase 2G: 條件式保守校正")
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
    
    # 訓練
    print("\n訓練模型...")
    model, likelihood, scaler_x, scaler_y = train_model(X_train, y_train, config, verbose)
    
    # 原始預測
    y_pred_raw, y_std = predict(model, likelihood, X_test, scaler_x, scaler_y)
    results_raw = evaluate(X_test, y_test, y_pred_raw, "原始預測")
    
    # 條件式校正
    print("\n應用條件式校正...")
    corrector = ConditionalCorrector(X_train, y_train, k=20)
    y_pred_corrected, correction_info = corrector.correct(X_test, y_pred_raw)
    results_corrected = evaluate(X_test, y_test, y_pred_corrected, "條件式校正後")
    
    # 比較
    print("\n" + "="*60)
    print("校正效果")
    print("="*60)
    print(f"{'指標':<15} {'原始':<10} {'校正後':<10} {'改善':<10}")
    print("-"*45)
    print(f"{'MAPE':<15} {results_raw['mape']:<10.2f} {results_corrected['mape']:<10.2f} "
          f"{results_raw['mape']-results_corrected['mape']:+.2f}")
    print(f"{'Max Error':<15} {results_raw['max_error']:<10.2f} {results_corrected['max_error']:<10.2f} "
          f"{results_raw['max_error']-results_corrected['max_error']:+.2f}")
    print(f"{'異常點':<15} {results_raw['outliers']:<10} {results_corrected['outliers']:<10} "
          f"{results_raw['outliers']-results_corrected['outliers']:+d}")
    
    # 顯示被校正的樣本
    print("\n被校正的樣本:")
    for i, info in enumerate(correction_info):
        if info['was_corrected']:
            err_before = abs(y_test[i] - info['original']) / y_test[i] * 100
            err_after = abs(y_test[i] - info['corrected']) / y_test[i] * 100
            print(f"  [{i}] T{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}")
            print(f"       {info['original']:.4f} → {info['corrected']:.4f} (真值={y_test[i]:.4f})")
            print(f"       誤差: {err_before:.1f}% → {err_after:.1f}% ({info['reason']})")
    
    # 保存
    results_df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Pred_Raw': y_pred_raw,
        'Pred_Corrected': y_pred_corrected,
        'Error_Raw': results_raw['errors'],
        'Error_Corrected': results_corrected['errors']
    })
    results_df.to_csv(f'phase2g_seed{seed}_predictions.csv', index=False)
    print(f"\n✓ 已保存: phase2g_seed{seed}_predictions.csv")
    
    return results_corrected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()
    main(seed=args.seed, verbose=args.verbose)
