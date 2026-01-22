"""
Phase 2D: 分模型策略 (Separate Models)
Type 1&2 和 Type 3 分開訓練，因為它們的物理行為完全不同

關鍵發現:
1. Type 1&2: THICKNESS 20~200, Thickness 與 Theta.JC 正相關 (~0.35)
2. Type 3: THICKNESS 200~300, Thickness 與 Theta.JC 幾乎無關 (0.04)
3. 分開訓練可以讓每個模型專注於更簡單的關係
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
# 特徵工程 (簡化版，針對各自的 Type)
# ==========================================

def create_features_type12(X_raw):
    """Type 1 & 2 的特徵 - THICKNESS 有影響"""
    TYPE = X_raw[:, 0]
    THICKNESS = X_raw[:, 1]
    COVERAGE = X_raw[:, 2]
    
    features = [
        TYPE,
        THICKNESS,
        COVERAGE,
        1.0 / COVERAGE,
        THICKNESS / COVERAGE,
        THICKNESS * COVERAGE,
        np.log(THICKNESS + 1),
        COVERAGE ** 2,
        (TYPE == 1).astype(float),  # Type 1 flag
    ]
    
    names = ['TYPE', 'THICK', 'COV', '1/COV', 'THICK/COV', 
             'THICK*COV', 'log(THICK)', 'COV^2', 'TYPE1_flag']
    
    return np.column_stack(features), names


def create_features_type3(X_raw):
    """Type 3 的特徵 - THICKNESS 幾乎沒影響，專注 COVERAGE"""
    THICKNESS = X_raw[:, 1]
    COVERAGE = X_raw[:, 2]
    
    features = [
        COVERAGE,
        1.0 / COVERAGE,
        COVERAGE ** 2,
        COVERAGE ** 3,
        np.log(COVERAGE + 0.1),
        THICKNESS / 100,  # 縮放後的 thickness，影響很小
        (COVERAGE - 0.8),  # 以 0.8 為中心的偏移
        (COVERAGE >= 0.8).astype(float),  # 高覆蓋率 flag
    ]
    
    names = ['COV', '1/COV', 'COV^2', 'COV^3', 'log(COV)', 
             'THICK_scaled', 'COV_offset', 'HIGH_COV_flag']
    
    return np.column_stack(features), names


# ==========================================
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
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

def compute_sample_weights_type12(X_raw, y):
    """Type 1&2 的樣本權重"""
    weights = np.ones(len(X_raw))
    
    THICKNESS = X_raw[:, 1]
    COVERAGE = X_raw[:, 2]
    
    # 小 Thickness 樣本較少，加權
    thin_mask = (THICKNESS <= 40)
    weights[thin_mask] *= 2.0
    
    # 高 Coverage 加權
    high_cov_mask = (COVERAGE >= 0.8)
    weights[high_cov_mask] *= 1.5
    
    # 小值加權
    small_y_mask = (y < 0.04)
    weights[small_y_mask] *= 2.0
    
    return weights


def compute_sample_weights_type3(X_raw, y):
    """Type 3 的樣本權重"""
    weights = np.ones(len(X_raw))
    
    COVERAGE = X_raw[:, 2]
    
    # Coverage = 0.8 是問題區域
    boundary_mask = (COVERAGE >= 0.78) & (COVERAGE <= 0.82)
    weights[boundary_mask] *= 3.0
    
    # 小值加權 (Type 3 的值普遍較小)
    small_y_mask = (y < 0.03)
    weights[small_y_mask] *= 2.5
    
    return weights


def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    mape = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    return torch.sum(mape * weights) / torch.sum(weights)


# ==========================================
# 訓練函數
# ==========================================

def train_single_model(X_train, y_train, feature_fn, weight_fn, config, model_name, verbose=True):
    """訓練單一模型"""
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"訓練 {model_name}")
        print(f"{'='*50}")
    
    # 特徵工程
    X_features, feature_names = feature_fn(X_train)
    
    if verbose:
        print(f"樣本數: {len(X_train)}")
        print(f"特徵數: {len(feature_names)}")
        print(f"特徵: {feature_names}")
    
    # 樣本權重
    sample_weights_np = weight_fn(X_train, y_train)
    
    if verbose:
        print(f"加權樣本數: {np.sum(sample_weights_np > 1)}")
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X_features)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_scaled).to(device)
    train_y = torch.from_numpy(y_scaled).to(device)
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
    
    # 載入最佳模型
    model.load_state_dict({k: v.to(device) for k, v in best_state['model'].items()})
    likelihood.load_state_dict({k: v.to(device) for k, v in best_state['likelihood'].items()})
    
    if verbose:
        print(f"  訓練完成 (Best Loss: {best_loss:.4f})")
    
    return model, likelihood, scaler_x, scaler_y, feature_fn


def predict_single_model(model, likelihood, X_test, scaler_x, scaler_y, feature_fn):
    """單一模型預測"""
    model.eval()
    likelihood.eval()
    
    X_features, _ = feature_fn(X_test)
    X_scaled = scaler_x.transform(X_features)
    test_x = torch.from_numpy(X_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred_scaled = pred_dist.mean.cpu().numpy()
        y_std_scaled = pred_dist.stddev.cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    return y_pred, y_std


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, verbose=True):
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}")
    print(f"隨機種子: {seed}\n")
    
    print("="*60)
    print("Phase 2D: 分模型策略")
    print("Type 1&2 和 Type 3 分開訓練")
    print("="*60)
    
    # 載入資料
    train_df = pd.read_excel('data/train/Above.xlsx')
    test_df = pd.read_excel('data/test/Above.xlsx')
    
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 去重
    train_clean = train_df.groupby(feature_cols, as_index=False).agg({target_col: 'mean'})
    
    # 分割資料
    train_type12 = train_clean[train_clean['TIM_TYPE'].isin([1, 2])]
    train_type3 = train_clean[train_clean['TIM_TYPE'] == 3]
    
    test_type12 = test_df[test_df['TIM_TYPE'].isin([1, 2])]
    test_type3 = test_df[test_df['TIM_TYPE'] == 3]
    
    print(f"\n資料分割:")
    print(f"  Type 1&2 訓練: {len(train_type12)} 筆, 測試: {len(test_type12)} 筆")
    print(f"  Type 3 訓練: {len(train_type3)} 筆, 測試: {len(test_type3)} 筆")
    
    X_train_12 = train_type12[feature_cols].values
    y_train_12 = train_type12[target_col].values
    X_test_12 = test_type12[feature_cols].values
    y_test_12 = test_type12[target_col].values
    
    X_train_3 = train_type3[feature_cols].values
    y_train_3 = train_type3[target_col].values
    X_test_3 = test_type3[feature_cols].values
    y_test_3 = test_type3[target_col].values
    
    # 配置
    config_type12 = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
    }
    
    config_type3 = {
        'hidden_dims': [32, 16],  # Type 3 關係更簡單，用更小的網路
        'feature_dim': 6,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.15,  # Type 3 更強調 MAPE
    }
    
    # ==========================================
    # 訓練 Type 1&2 模型
    # ==========================================
    
    model_12, likelihood_12, scaler_x_12, scaler_y_12, feat_fn_12 = train_single_model(
        X_train_12, y_train_12,
        create_features_type12,
        compute_sample_weights_type12,
        config_type12,
        "Model Type 1&2",
        verbose=verbose
    )
    
    # ==========================================
    # 訓練 Type 3 模型
    # ==========================================
    
    model_3, likelihood_3, scaler_x_3, scaler_y_3, feat_fn_3 = train_single_model(
        X_train_3, y_train_3,
        create_features_type3,
        compute_sample_weights_type3,
        config_type3,
        "Model Type 3",
        verbose=verbose
    )
    
    # ==========================================
    # 評估
    # ==========================================
    
    print(f"\n{'='*60}")
    print("評估結果")
    print(f"{'='*60}")
    
    # Type 1&2 預測
    y_pred_12, y_std_12 = predict_single_model(
        model_12, likelihood_12, X_test_12, scaler_x_12, scaler_y_12, feat_fn_12
    )
    
    # Type 3 預測
    y_pred_3, y_std_3 = predict_single_model(
        model_3, likelihood_3, X_test_3, scaler_x_3, scaler_y_3, feat_fn_3
    )
    
    # 合併結果
    all_y_test = np.concatenate([y_test_12, y_test_3])
    all_y_pred = np.concatenate([y_pred_12, y_pred_3])
    all_X_test = np.concatenate([X_test_12, X_test_3])
    all_std = np.concatenate([y_std_12, y_std_3])
    
    # 計算指標
    errors = np.abs((all_y_test - all_y_pred) / all_y_test) * 100
    
    mape = np.mean(errors)
    mae = np.mean(np.abs(all_y_test - all_y_pred))
    max_error = np.max(errors)
    
    outliers_20 = np.sum(errors > 20)
    outliers_15 = np.sum(errors > 15)
    outliers_10 = np.sum(errors > 10)
    
    print(f"\n【整體】 {len(all_y_test)} 筆")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  MAE: {mae:.4f}")
    print(f"  Max Error: {max_error:.2f}%")
    print(f"  >20%: {outliers_20} ({outliers_20/len(all_y_test)*100:.2f}%)")
    print(f"  >15%: {outliers_15}")
    print(f"  >10%: {outliers_10}")
    
    # Type 1&2 分析
    errors_12 = np.abs((y_test_12 - y_pred_12) / y_test_12) * 100
    print(f"\n【Type 1&2】 {len(y_test_12)} 筆")
    print(f"  MAPE: {np.mean(errors_12):.2f}%")
    print(f"  >20%: {np.sum(errors_12 > 20)}")
    
    # Type 3 分析
    errors_3 = np.abs((y_test_3 - y_pred_3) / y_test_3) * 100
    print(f"\n【Type 3】 {len(y_test_3)} 筆")
    print(f"  MAPE: {np.mean(errors_3):.2f}%")
    print(f"  >20%: {np.sum(errors_3 > 20)}")
    
    # 顯示異常點
    print(f"\n異常點詳情 (>20%):")
    
    # Type 1&2 異常點
    for i in range(len(y_test_12)):
        if errors_12[i] > 20:
            print(f"  Type={X_test_12[i,0]:.0f}, Thick={X_test_12[i,1]:.0f}, "
                  f"Cov={X_test_12[i,2]:.1f}: True={y_test_12[i]:.4f}, "
                  f"Pred={y_pred_12[i]:.4f}, Err={errors_12[i]:.1f}%")
    
    # Type 3 異常點
    for i in range(len(y_test_3)):
        if errors_3[i] > 20:
            print(f"  Type=3, Thick={X_test_3[i,1]:.0f}, "
                  f"Cov={X_test_3[i,2]:.1f}: True={y_test_3[i]:.4f}, "
                  f"Pred={y_pred_3[i]:.4f}, Err={errors_3[i]:.1f}%")
    
    # 保存結果
    # 重建原始順序
    results_12 = pd.DataFrame({
        'TIM_TYPE': X_test_12[:, 0],
        'TIM_THICKNESS': X_test_12[:, 1],
        'TIM_COVERAGE': X_test_12[:, 2],
        'True': y_test_12,
        'Predicted': y_pred_12,
        'Error%': errors_12,
        'Std': y_std_12
    })
    
    results_3 = pd.DataFrame({
        'TIM_TYPE': X_test_3[:, 0],
        'TIM_THICKNESS': X_test_3[:, 1],
        'TIM_COVERAGE': X_test_3[:, 2],
        'True': y_test_3,
        'Predicted': y_pred_3,
        'Error%': errors_3,
        'Std': y_std_3
    })
    
    results_all = pd.concat([results_12, results_3], ignore_index=True)
    results_all.to_csv(f'phase2d_above_seed{seed}_predictions.csv', index=False)
    print(f"\n✓ 預測結果已保存: phase2d_above_seed{seed}_predictions.csv")
    
    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")
    
    return {
        'mape': mape,
        'outliers_20': outliers_20,
        'type12_outliers': np.sum(errors_12 > 20),
        'type3_outliers': np.sum(errors_3 > 20),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
