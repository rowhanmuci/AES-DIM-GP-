"""
Phase 1 修正版: 
1. 只清洗訓練集，測試集保持原樣
2. 評估Above和Below
3. MAPE Loss優化
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, max_error
import warnings
warnings.filterwarnings('ignore')

# 設定
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}\n")


# ==========================================
# 資料清理 (只清洗訓練集)
# ==========================================

def clean_training_data(train_df, feature_cols):
    """只清洗訓練集"""
    print("="*60)
    print("資料清理: 訓練集重複樣本檢測")
    print("="*60 + "\n")
    
    original_len = len(train_df)
    
    # 檢查完全重複
    full_dup = train_df.duplicated(subset=feature_cols + ['Theta.JC'], keep='first')
    n_full_dup = full_dup.sum()
    
    # 檢查特徵重複但目標不同
    feature_dup = train_df.duplicated(subset=feature_cols, keep=False)
    ambiguous = train_df[feature_dup & ~full_dup]
    
    print(f"📊 重複樣本統計:")
    print(f"  完全重複: {n_full_dup} 筆")
    print(f"  特徵相同但目標不同: {len(ambiguous)} 筆\n")
    
    # 清理策略: 對相同特徵取平均
    train_clean = train_df.groupby(feature_cols, as_index=False).agg({
        'Theta.JC': 'mean'
    })
    
    cleaned_len = len(train_clean)
    removed = original_len - cleaned_len
    
    print(f"✓ 清理完成:")
    print(f"  原始: {original_len} 筆")
    print(f"  清理後: {cleaned_len} 筆")
    print(f"  移除: {removed} 筆 ({removed/original_len*100:.2f}%)\n")
    
    return train_clean


# ==========================================
# 改進的DKL模型
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """DNN特徵提取器"""
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
    """GP回歸模型"""
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        )
    
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def mape_loss(y_pred, y_true, epsilon=1e-8):
    """MAPE Loss"""
    return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon))) * 100


def train_improved_dkl(X_train, y_train, config=None):
    """訓練DKL"""
    if config is None:
        config = {
            'hidden_dims': [64, 32, 16],
            'feature_dim': 8,
            'dropout': 0.1,
            'lr': 0.01,
            'epochs': 500,
            'patience': 50,
            'mape_weight': 0.1,
        }
    
    print("訓練配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_train_scaled).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    
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
    
    print("開始訓練...")
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        mape = mape_loss(output.mean, train_y)
        total_loss = gp_loss + config['mape_weight'] * mape
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        
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
            print(f"早停 (Epoch {epoch+1}), Best Loss: {best_loss:.4f}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: GP Loss={gp_loss.item():.4f}, MAPE={mape.item():.2f}%, Total={total_loss.item():.4f}")
    
    # 載入最佳模型
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    print(f"訓練完成 (Final Loss: {best_loss:.4f})\n")
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'config': config
    }


def evaluate_model(model_dict, X_test, y_test, dataset_name="Test"):
    """評估模型"""
    model = model_dict['model']
    likelihood = model_dict['likelihood']
    scaler_x = model_dict['scaler_x']
    scaler_y = model_dict['scaler_y']
    
    # 預測
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
    
    # 計算指標
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = mean_absolute_error(y_test, y_pred)
    max_err = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    # CI覆蓋率
    ci_lower = y_pred - 1.96 * y_std
    ci_upper = y_pred + 1.96 * y_std
    ci_coverage = np.mean((y_test >= ci_lower) & (y_test <= ci_upper)) * 100
    ci_width = np.mean(ci_upper - ci_lower)
    
    print(f"="*60)
    print(f"{dataset_name} 評估結果")
    print(f"="*60)
    print(f"樣本數: {len(y_test)}")
    print(f"\n準確度:")
    print(f"  MAPE:      {mape:.2f}%")
    print(f"  MAE:       {mae:.4f}")
    print(f"  Max Error: {max_err:.2f}%")
    print(f"\n異常點:")
    print(f"  >20%: {outliers_20}/{len(y_test)} ({outliers_20/len(y_test)*100:.2f}%)")
    print(f"  >15%: {outliers_15}/{len(y_test)} ({outliers_15/len(y_test)*100:.2f}%)")
    print(f"  >10%: {outliers_10}/{len(y_test)} ({outliers_10/len(y_test)*100:.2f}%)")
    print(f"\n不確定性:")
    print(f"  CI Coverage: {ci_coverage:.2f}%")
    print(f"  CI Width:    {ci_width:.4f}")
    print(f"="*60 + "\n")
    
    return {
        'mape': mape,
        'mae': mae,
        'max_error': max_err,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'ci_coverage': ci_coverage,
        'ci_width': ci_width,
        'predictions': y_pred,
        'std': y_std,
        'relative_errors': relative_errors
    }


def main_phase1_corrected():
    """Phase 1 修正版主流程"""
    
    print("\n" + "="*60)
    print("Phase 1 修正版: Above + Below評估")
    print("="*60 + "\n")
    
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    config = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
    }
    
    results_summary = []
    
    # ==========================================
    # Above資料集
    # ==========================================
    print("\n" + "🔵 "*20)
    print("Above 50% Coverage")
    print("🔵 "*20 + "\n")
    
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    print(f"原始資料:")
    print(f"  訓練集: {len(train_above)} 筆")
    print(f"  測試集: {len(test_above)} 筆\n")
    
    # 清洗訓練集 (測試集不動!)
    train_above_clean = clean_training_data(train_above, feature_cols)
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    # 訓練
    print("\n訓練Above模型...\n")
    model_above = train_improved_dkl(X_train_above, y_train_above, config)
    
    # 評估
    results_above = evaluate_model(model_above, X_test_above, y_test_above, "Above")
    
    # 保存預測
    test_above_pred = test_above.copy()
    test_above_pred['Prediction'] = results_above['predictions']
    test_above_pred['Std'] = results_above['std']
    test_above_pred['Error%'] = results_above['relative_errors']
    test_above_pred.to_csv('phase1_above_predictions.csv', index=False)
    
    results_summary.append({
        'Dataset': 'Above',
        'MAPE': results_above['mape'],
        'Outliers_20': f"{results_above['outliers_20']}/{len(y_test_above)}",
        'Max_Error': results_above['max_error']
    })
    
    # ==========================================
    # Below資料集
    # ==========================================
    print("\n" + "🟢 "*20)
    print("Below 50% Coverage")
    print("🟢 "*20 + "\n")
    
    train_below = pd.read_excel('data/train/Below.xlsx')
    test_below = pd.read_excel('data/test/Below.xlsx')
    
    print(f"原始資料:")
    print(f"  訓練集: {len(train_below)} 筆")
    print(f"  測試集: {len(test_below)} 筆\n")
    
    # 清洗訓練集
    train_below_clean = clean_training_data(train_below, feature_cols)
    
    X_train_below = train_below_clean[feature_cols].values
    y_train_below = train_below_clean[target_col].values
    
    X_test_below = test_below[feature_cols].values
    y_test_below = test_below[target_col].values
    
    # 訓練
    print("\n訓練Below模型...\n")
    model_below = train_improved_dkl(X_train_below, y_train_below, config)
    
    # 評估
    results_below = evaluate_model(model_below, X_test_below, y_test_below, "Below")
    
    # 保存預測
    test_below_pred = test_below.copy()
    test_below_pred['Prediction'] = results_below['predictions']
    test_below_pred['Std'] = results_below['std']
    test_below_pred['Error%'] = results_below['relative_errors']
    test_below_pred.to_csv('phase1_below_predictions.csv', index=False)
    
    results_summary.append({
        'Dataset': 'Below',
        'MAPE': results_below['mape'],
        'Outliers_20': f"{results_below['outliers_20']}/{len(y_test_below)}",
        'Max_Error': results_below['max_error']
    })
    
    # ==========================================
    # 總結比較
    # ==========================================
    print("\n" + "="*60)
    print("📊 與Baseline比較")
    print("="*60 + "\n")
    
    print("Baseline (組員):")
    print("  Above: MAPE=8.89%, 異常點=16/138 (11.59%)")
    print("  Below: MAPE=3.76%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 1 (改進版):")
    print(f"  Above: MAPE={results_above['mape']:.2f}%, "
          f"異常點={results_above['outliers_20']}/{len(y_test_above)} "
          f"({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"  Below: MAPE={results_below['mape']:.2f}%, "
          f"異常點={results_below['outliers_20']}/{len(y_test_below)} "
          f"({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    
    # 計算改進
    above_mape_diff = results_above['mape'] - 8.89
    above_outlier_diff = results_above['outliers_20'] - 16
    
    below_mape_diff = results_below['mape'] - 3.76
    below_outlier_diff = results_below['outliers_20'] - 0
    
    print(f"\n改進:")
    print(f"  Above: MAPE {above_mape_diff:+.2f}%, 異常點 {above_outlier_diff:+d}")
    print(f"  Below: MAPE {below_mape_diff:+.2f}%, 異常點 {below_outlier_diff:+d}")
    
    print(f"\n{'='*60}\n")
    
    # 保存總結
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('phase1_summary.csv', index=False)
    print("✓ 結果已保存")
    print("  - phase1_above_predictions.csv")
    print("  - phase1_below_predictions.csv")
    print("  - phase1_summary.csv\n")
    
    return {
        'above': (model_above, results_above, test_above_pred),
        'below': (model_below, results_below, test_below_pred)
    }


if __name__ == "__main__":
    results = main_phase1_corrected()