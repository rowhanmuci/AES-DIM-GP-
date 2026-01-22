"""
Phase 2E: Phase 2B 基礎 + KNN 後處理校正

策略:
1. 使用 Phase 2B 的 DKL 模型作為基礎預測
2. 對預測結果進行 KNN 鄰近校正
3. 根據模型不確定性 (std) 動態調整校正強度

KNN 校正邏輯:
- 找訓練集中特徵最相似的 K 個樣本
- 計算這些樣本的 Theta.JC 統計量
- 當模型不確定性高時，更依賴鄰近樣本
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
# 模型定義 (與 Phase 2B 相同)
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


# ==========================================
# 損失函數
# ==========================================

def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    mape = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    return torch.sum(mape * weights) / torch.sum(weights)


def compute_sample_weights(X, weight_factor=3.0):
    """Phase 2B 的樣本權重策略"""
    weights = np.ones(len(X))
    
    difficult_mask = (
        (X[:, 0] == 3) &      # TIM_TYPE = 3
        (X[:, 2] == 0.8) &    # TIM_COVERAGE = 0.8
        (X[:, 1] >= 220)      # TIM_THICKNESS >= 220
    )
    
    weights[difficult_mask] *= weight_factor
    return weights


# ==========================================
# KNN 後處理校正
# ==========================================

class KNNPostProcessor:
    """KNN 後處理校正器"""
    
    def __init__(self, X_train, y_train, k=10, distance_metric='euclidean'):
        """
        初始化 KNN 校正器
        
        Args:
            X_train: 訓練集特徵 (原始空間，未標準化)
            y_train: 訓練集目標值
            k: 鄰居數量
            distance_metric: 距離度量
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        
        # 對特徵進行標準化以計算距離
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 建立 KNN 索引
        self.knn = NearestNeighbors(n_neighbors=k, metric=distance_metric)
        self.knn.fit(X_train_scaled)
        
        print(f"KNN 校正器初始化: k={k}, 訓練樣本={len(X_train)}")
    
    def get_neighbors_stats(self, X_test):
        """
        獲取測試樣本的鄰居統計量
        
        Returns:
            neighbor_mean: 鄰居 y 的平均值
            neighbor_std: 鄰居 y 的標準差
            neighbor_median: 鄰居 y 的中位數
            distances: 到鄰居的距離
        """
        X_test_scaled = self.scaler.transform(X_test)
        distances, indices = self.knn.kneighbors(X_test_scaled)
        
        neighbor_mean = np.array([self.y_train[idx].mean() for idx in indices])
        neighbor_std = np.array([self.y_train[idx].std() for idx in indices])
        neighbor_median = np.array([np.median(self.y_train[idx]) for idx in indices])
        neighbor_min = np.array([self.y_train[idx].min() for idx in indices])
        neighbor_max = np.array([self.y_train[idx].max() for idx in indices])
        
        return {
            'mean': neighbor_mean,
            'std': neighbor_std,
            'median': neighbor_median,
            'min': neighbor_min,
            'max': neighbor_max,
            'distances': distances,
            'indices': indices
        }
    
    def correct_predictions(self, X_test, y_pred, y_model_std, config=None):
        """
        校正預測值
        
        Args:
            X_test: 測試集特徵
            y_pred: 模型預測值
            y_model_std: 模型預測的標準差 (不確定性)
            config: 校正配置
            
        Returns:
            y_corrected: 校正後的預測值
            correction_info: 校正資訊
        """
        if config is None:
            config = {
                'base_alpha': 0.7,        # 基礎模型權重
                'std_sensitivity': 15.0,   # 不確定性敏感度
                'use_median': False,       # 是否使用中位數
                'clip_to_range': True,     # 是否限制在鄰居範圍內
                'type3_special': True,     # Type 3 特殊處理
            }
        
        # 獲取鄰居統計量
        neighbor_stats = self.get_neighbors_stats(X_test)
        
        y_corrected = np.copy(y_pred)
        correction_info = []
        
        for i in range(len(X_test)):
            # 選擇鄰居的代表值
            if config['use_median']:
                y_neighbor = neighbor_stats['median'][i]
            else:
                y_neighbor = neighbor_stats['mean'][i]
            
            # 計算動態 alpha (模型權重)
            # 不確定性越大，alpha 越小，越依賴鄰居
            model_std = y_model_std[i]
            neighbor_std = neighbor_stats['std'][i]
            
            # 基於模型不確定性調整
            uncertainty_factor = 1.0 / (1.0 + config['std_sensitivity'] * model_std)
            
            # 基於鄰居一致性調整 (鄰居 std 小 = 更可信)
            consistency_factor = 1.0 / (1.0 + 5.0 * neighbor_std)
            
            # Type 3 特殊處理
            is_type3 = (X_test[i, 0] == 3)
            is_high_cov = (X_test[i, 2] >= 0.8)
            
            if config['type3_special'] and is_type3 and is_high_cov:
                # Type 3 高覆蓋率區域，更依賴鄰居
                alpha = config['base_alpha'] * 0.6 * uncertainty_factor
            else:
                alpha = config['base_alpha'] * uncertainty_factor
            
            # 混合預測
            y_mixed = alpha * y_pred[i] + (1 - alpha) * y_neighbor
            
            # 可選：限制在鄰居範圍內
            if config['clip_to_range']:
                margin = 0.2  # 允許 20% 的範圍擴展
                y_min = neighbor_stats['min'][i] * (1 - margin)
                y_max = neighbor_stats['max'][i] * (1 + margin)
                y_mixed = np.clip(y_mixed, y_min, y_max)
            
            y_corrected[i] = y_mixed
            
            correction_info.append({
                'original': y_pred[i],
                'corrected': y_mixed,
                'neighbor_mean': neighbor_stats['mean'][i],
                'neighbor_std': neighbor_stats['std'][i],
                'model_std': model_std,
                'alpha': alpha,
                'is_type3_high_cov': is_type3 and is_high_cov
            })
        
        return y_corrected, correction_info


# ==========================================
# 訓練函數
# ==========================================

def train_model(X_train, y_train, config, verbose=True):
    """訓練 DKL 模型 (Phase 2B)"""
    
    sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
    
    if verbose:
        difficult_count = np.sum(sample_weights_np > 1.0)
        print(f"\n樣本權重: {difficult_count} 個困難樣本 (權重 {config['sample_weight_factor']}x)")
    
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
        
        current_loss = total_loss.item()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: Loss={current_loss:.4f}, MAPE={mape.item():.2f}%")
        
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
                print(f"  早停 at Epoch {epoch+1}")
            break
    
    model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
    likelihood.load_state_dict({k: v.to(device) for k, v in best_state['likelihood'].items()})
    
    if verbose:
        print(f"  訓練完成 (Best Loss: {best_loss:.4f})")
    
    return model, likelihood, scaler_x, scaler_y


def predict_with_uncertainty(model, likelihood, X_test, scaler_x, scaler_y):
    """預測並返回不確定性"""
    model.eval()
    likelihood.eval()
    
    X_test_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_test_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred_scaled = pred_dist.mean.cpu().numpy()
        y_std_scaled = pred_dist.stddev.cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    return y_pred, y_std


# ==========================================
# 評估函數
# ==========================================

def evaluate(X_test, y_test, y_pred, y_std, label=""):
    """評估預測結果"""
    errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(errors)
    
    outliers_20 = np.sum(errors > 20)
    outliers_15 = np.sum(errors > 15)
    outliers_10 = np.sum(errors > 10)
    
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((errors > 20) & type3_mask)
    
    print(f"\n{'='*60}")
    print(f"評估結果 {label}")
    print(f"{'='*60}")
    print(f"樣本數: {len(y_test)}")
    print(f"\n準確度:")
    print(f"  MAPE:      {mape:.2f}%")
    print(f"  MAE:       {mae:.4f}")
    print(f"  Max Error: {max_error:.2f}%")
    print(f"\n異常點:")
    print(f"  >20%: {outliers_20}/{len(y_test)} ({outliers_20/len(y_test)*100:.2f}%)")
    print(f"  >15%: {outliers_15}/{len(y_test)}")
    print(f"  >10%: {outliers_10}/{len(y_test)}")
    print(f"  Type 3 (>20%): {type3_outliers}/{np.sum(type3_mask)}")
    
    # 顯示異常點詳情
    if outliers_20 > 0:
        print(f"\n異常點詳情:")
        outlier_indices = np.where(errors > 20)[0]
        for idx in outlier_indices:
            print(f"  Type={X_test[idx,0]:.0f}, Thick={X_test[idx,1]:.0f}, "
                  f"Cov={X_test[idx,2]:.1f}: True={y_test[idx]:.4f}, "
                  f"Pred={y_pred[idx]:.4f}, Err={errors[idx]:.1f}%")
    
    return {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'type3_outliers': type3_outliers,
        'errors': errors
    }


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, verbose=True):
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}")
    print(f"隨機種子: {seed}\n")
    
    print("="*60)
    print("Phase 2E: DKL + KNN 後處理校正")
    print("="*60)
    
    # 載入資料
    train_df = pd.read_excel('data/train/Above.xlsx')
    test_df = pd.read_excel('data/test/Above.xlsx')

    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 去重
    train_clean = train_df.groupby(feature_cols, as_index=False).agg({target_col: 'mean'})
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"\n訓練集: {len(train_clean)} 筆")
    print(f"測試集: {len(test_df)} 筆")
    
    # ==========================================
    # Step 1: 訓練 DKL 模型
    # ==========================================
    
    print(f"\n{'='*60}")
    print("Step 1: 訓練 DKL 模型 (Phase 2B)")
    print(f"{'='*60}")
    
    model_config = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
        'sample_weight_factor': 3.0,
    }
    
    model, likelihood, scaler_x, scaler_y = train_model(
        X_train, y_train, model_config, verbose=verbose
    )
    
    # ==========================================
    # Step 2: 模型預測
    # ==========================================
    
    print(f"\n{'='*60}")
    print("Step 2: 模型預測")
    print(f"{'='*60}")
    
    y_pred_raw, y_std = predict_with_uncertainty(
        model, likelihood, X_test, scaler_x, scaler_y
    )
    
    # 評估原始預測
    results_raw = evaluate(X_test, y_test, y_pred_raw, y_std, "(原始模型)")
    
    # ==========================================
    # Step 3: KNN 後處理校正
    # ==========================================
    
    print(f"\n{'='*60}")
    print("Step 3: KNN 後處理校正")
    print(f"{'='*60}")
    
    # 初始化 KNN 校正器
    knn_processor = KNNPostProcessor(X_train, y_train, k=10)
    
    # 校正配置
    knn_config = {
        'base_alpha': 0.6,        # 基礎模型權重 (0.6 = 模型佔 60%)
        'std_sensitivity': 20.0,   # 不確定性敏感度
        'use_median': False,       # 使用平均值
        'clip_to_range': True,     # 限制在鄰居範圍內
        'type3_special': True,     # Type 3 特殊處理
    }
    
    print(f"\nKNN 校正配置:")
    for k, v in knn_config.items():
        print(f"  {k}: {v}")
    
    # 執行校正
    y_pred_corrected, correction_info = knn_processor.correct_predictions(
        X_test, y_pred_raw, y_std, knn_config
    )
    
    # 評估校正後預測
    results_corrected = evaluate(X_test, y_test, y_pred_corrected, y_std, "(KNN 校正後)")
    
    # ==========================================
    # 比較結果
    # ==========================================
    
    print(f"\n{'='*60}")
    print("校正效果比較")
    print(f"{'='*60}")
    print(f"{'指標':<15} {'原始模型':<12} {'KNN校正後':<12} {'改善':<10}")
    print("-" * 50)
    print(f"{'MAPE':<15} {results_raw['mape']:<12.2f} {results_corrected['mape']:<12.2f} "
          f"{results_raw['mape'] - results_corrected['mape']:+.2f}")
    print(f"{'異常點(>20%)':<15} {results_raw['outliers_20']:<12} {results_corrected['outliers_20']:<12} "
          f"{results_raw['outliers_20'] - results_corrected['outliers_20']:+d}")
    print(f"{'Type3異常點':<15} {results_raw['type3_outliers']:<12} {results_corrected['type3_outliers']:<12} "
          f"{results_raw['type3_outliers'] - results_corrected['type3_outliers']:+d}")
    print(f"{'Max Error':<15} {results_raw['max_error']:<12.2f} {results_corrected['max_error']:<12.2f} "
          f"{results_raw['max_error'] - results_corrected['max_error']:+.2f}")
    
    # 顯示校正詳情
    print(f"\n校正詳情 (變化較大的樣本):")
    for i, info in enumerate(correction_info):
        change = abs(info['corrected'] - info['original'])
        if change > 0.002:  # 變化超過 0.002
            print(f"  [{i}] Type={X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]:.1f}")
            print(f"       原始: {info['original']:.4f} → 校正: {info['corrected']:.4f} "
                  f"(α={info['alpha']:.2f}, 鄰居={info['neighbor_mean']:.4f}±{info['neighbor_std']:.4f})")
    
    # 保存結果
    results_df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Pred_Raw': y_pred_raw,
        'Pred_Corrected': y_pred_corrected,
        'Error_Raw%': results_raw['errors'],
        'Error_Corrected%': results_corrected['errors'],
        'Model_Std': y_std
    })
    
    output_file = f'phase2e_above_seed{seed}_predictions.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 預測結果已保存: {output_file}")
    
    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")
    
    return {
        'raw': results_raw,
        'corrected': results_corrected
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
