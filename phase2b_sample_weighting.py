"""
Phase 2B: 樣本加權 (Sample Weighting)
核心策略: 對困難樣本（Type 3 + Coverage 0.8 + 大THICKNESS）加大權重
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

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}\n")


# ==========================================
# 模型定義 (與Phase 1相同)
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


def mape_loss(y_pred, y_true, epsilon=1e-8):
    """MAPE Loss"""
    return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon))) * 100


def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    """加權MAPE Loss"""
    mape_per_sample = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    weighted_mape = torch.sum(mape_per_sample * weights) / torch.sum(weights)
    return weighted_mape


# ==========================================
# 樣本權重計算 (核心改進!)
# ==========================================

def compute_sample_weights(X, y, strategy='difficult_samples', weight_factor=3.0):
    """
    計算樣本權重
    
    Parameters:
    -----------
    X : numpy array [N, 3]
        特徵矩陣 [TIM_TYPE, THICKNESS, COVERAGE]
    y : numpy array [N]
        目標變量 (用於識別小值樣本)
    strategy : str
        權重策略:
        - 'difficult_samples': 對Type 3 + Coverage 0.8 + 大THICKNESS加權
        - 'type3_only': 只對Type 3加權
        - 'small_values': 對小Theta.JC值加權
        - 'combined': 組合策略
    weight_factor : float
        權重倍數 (預設3倍)
    
    Returns:
    --------
    weights : numpy array [N]
        樣本權重
    """
    
    weights = np.ones(len(X))
    
    if strategy == 'difficult_samples':
        # 策略1: Type 3 + Coverage 0.8 + 大THICKNESS
        difficult_mask = (
            (X[:, 0] == 3) &              # Type 3
            (X[:, 2] == 0.8) &            # Coverage 0.8
            (X[:, 1] >= 220)              # THICKNESS >= 220
        )
        weights[difficult_mask] *= weight_factor
        
        n_difficult = difficult_mask.sum()
        print(f"  策略: Type 3 + Coverage 0.8 + THICKNESS>=220")
        print(f"  困難樣本數: {n_difficult} ({n_difficult/len(X)*100:.2f}%)")
        print(f"  權重倍數: {weight_factor}x\n")
    
    elif strategy == 'type3_only':
        # 策略2: 只對Type 3加權
        type3_mask = (X[:, 0] == 3)
        weights[type3_mask] *= weight_factor
        
        n_type3 = type3_mask.sum()
        print(f"  策略: Type 3全部樣本")
        print(f"  Type 3樣本數: {n_type3} ({n_type3/len(X)*100:.2f}%)")
        print(f"  權重倍數: {weight_factor}x\n")
    
    elif strategy == 'small_values':
        # 策略3: 對小Theta.JC值加權 (相對誤差容易大)
        small_value_mask = (y < np.percentile(y, 25))  # 最小的25%
        weights[small_value_mask] *= weight_factor
        
        n_small = small_value_mask.sum()
        print(f"  策略: Theta.JC < {np.percentile(y, 25):.4f} (最小25%)")
        print(f"  小值樣本數: {n_small} ({n_small/len(X)*100:.2f}%)")
        print(f"  權重倍數: {weight_factor}x\n")
    
    elif strategy == 'combined':
        # 策略4: 組合 (Type 3困難樣本 + 小值)
        difficult_mask = (
            (X[:, 0] == 3) & 
            (X[:, 2] == 0.8) & 
            (X[:, 1] >= 220)
        )
        small_value_mask = (y < np.percentile(y, 25))
        
        # Type 3困難樣本: 3倍權重
        weights[difficult_mask] *= weight_factor
        
        # 小值樣本: 額外1.5倍 (如果不是困難樣本)
        weights[small_value_mask & ~difficult_mask] *= 1.5
        
        # 如果既是困難樣本又是小值: 總共4.5倍
        weights[difficult_mask & small_value_mask] *= 1.5
        
        print(f"  策略: 組合 (Type 3困難樣本 + 小值)")
        print(f"  Type 3困難: {difficult_mask.sum()} ({difficult_mask.sum()/len(X)*100:.2f}%)")
        print(f"  小值: {small_value_mask.sum()} ({small_value_mask.sum()/len(X)*100:.2f}%)")
        print(f"  重疊: {(difficult_mask & small_value_mask).sum()}")
        print(f"  權重倍數: Type 3困難 {weight_factor}x, 小值 1.5x, 重疊 {weight_factor*1.5}x\n")
    
    return weights


# ==========================================
# 訓練函數 (加入樣本加權)
# ==========================================

def train_dkl_with_weighting(X_train, y_train, config=None):
    """訓練加權版DKL"""
    
    if config is None:
        config = {
            'hidden_dims': [64, 32, 16],
            'feature_dim': 8,
            'dropout': 0.1,
            'lr': 0.01,
            'epochs': 500,
            'patience': 50,
            'mape_weight': 0.1,
            'sample_weight_strategy': 'difficult_samples',  # 權重策略
            'sample_weight_factor': 3.0,                     # 權重倍數
        }
    
    print("="*60)
    print("訓練樣本加權版DKL")
    print("="*60 + "\n")
    
    print("配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 計算樣本權重 (在標準化前!)
    print("計算樣本權重:")
    sample_weights_np = compute_sample_weights(
        X_train, 
        y_train,
        strategy=config['sample_weight_strategy'],
        weight_factor=config['sample_weight_factor']
    )
    
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
    
    print("開始訓練...")
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        
        # 加權MAPE loss
        mape = weighted_mape_loss(output.mean, train_y, sample_weights)
        
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
            print(f"Epoch {epoch+1}: GP Loss={gp_loss.item():.4f}, "
                  f"MAPE={mape.item():.2f}%, Total={total_loss.item():.4f}")
    
    # 載入最佳模型
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    print(f"訓練完成 (Final Loss: {best_loss:.4f})\n")
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'config': config,
        'sample_weights': sample_weights_np
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
    
    # 分析異常點類型
    if outliers_20 > 0:
        print(f"\n異常點分析:")
        outlier_mask = relative_errors > 20
        outlier_types = X_test[outlier_mask, 0]
        
        for tim_type in [1, 2, 3]:
            n_type = np.sum(outlier_types == tim_type)
            if n_type > 0:
                print(f"  Type {tim_type}: {n_type}個異常點")
                
                # Type 3的詳細分析
                if tim_type == 3:
                    type3_outliers = X_test[outlier_mask & (X_test[:, 0] == 3)]
                    if len(type3_outliers) > 0:
                        print(f"    Coverage 0.8: {np.sum(type3_outliers[:, 2] == 0.8)}個")
                        print(f"    Coverage 1.0: {np.sum(type3_outliers[:, 2] == 1.0)}個")
                        print(f"    THICKNESS>=220: {np.sum(type3_outliers[:, 1] >= 220)}個")
    
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


# ==========================================
# 主函數
# ==========================================

def main_weighted_experiment():
    """樣本加權實驗主流程"""
    
    print("\n" + "="*60)
    print("Phase 2B: 樣本加權 (Sample Weighting)")
    print("="*60 + "\n")
    
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置 - 可以調整策略和權重倍數
    config = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
        'sample_weight_strategy': 'difficult_samples',  # 試試這個！
        'sample_weight_factor': 3.0,                     # 3倍權重
    }
    
    results_summary = []
    
    # ==========================================
    # Above
    # ==========================================
    print("\nbove 50% Coverage\n")
    
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆\n")
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    # 訓練加權版
    model_above = train_dkl_with_weighting(X_train_above, y_train_above, config)
    
    # 評估
    results_above = evaluate_model(model_above, X_test_above, y_test_above, "Above")
    
    # 保存
    test_above_pred = test_above.copy()
    test_above_pred['Prediction'] = results_above['predictions']
    test_above_pred['Std'] = results_above['std']
    test_above_pred['Error%'] = results_above['relative_errors']
    test_above_pred.to_csv('phase2b_above_predictions.csv', index=False)
    
    results_summary.append({
        'Dataset': 'Above',
        'Method': 'Sample Weighting',
        'MAPE': results_above['mape'],
        'Outliers_20': f"{results_above['outliers_20']}/{len(y_test_above)}",
        'Max_Error': results_above['max_error']
    })
    
    # ==========================================
    # Below
    # ==========================================
    print("\nBelow 50% Coverage\n")
    
    train_below = pd.read_excel('data/train/Below.xlsx')
    test_below = pd.read_excel('data/test/Below.xlsx')
    
    train_below_clean = train_below.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_below_clean)} 筆")
    print(f"測試集: {len(test_below)} 筆\n")
    
    X_train_below = train_below_clean[feature_cols].values
    y_train_below = train_below_clean[target_col].values
    
    X_test_below = test_below[feature_cols].values
    y_test_below = test_below[target_col].values
    
    # 訓練加權版
    model_below = train_dkl_with_weighting(X_train_below, y_train_below, config)
    
    # 評估
    results_below = evaluate_model(model_below, X_test_below, y_test_below, "Below")
    
    # 保存
    test_below_pred = test_below.copy()
    test_below_pred['Prediction'] = results_below['predictions']
    test_below_pred['Std'] = results_below['std']
    test_below_pred['Error%'] = results_below['relative_errors']
    test_below_pred.to_csv('phase2b_below_predictions.csv', index=False)
    
    results_summary.append({
        'Dataset': 'Below',
        'Method': 'Sample Weighting',
        'MAPE': results_below['mape'],
        'Outliers_20': f"{results_below['outliers_20']}/{len(y_test_below)}",
        'Max_Error': results_below['max_error']
    })
    
    # ==========================================
    # 比較
    # ==========================================
    print("\n" + "="*60)
    print("結果比較")
    print("="*60 + "\n")
    
    print("Baseline (組員):")
    print("  Above: MAPE=8.89%, 異常點=16/138 (11.59%)")
    print("  Below: MAPE=3.76%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 1 (MAPE Loss):")
    print("  Above: MAPE=8.63%, 異常點=10/138 (7.25%)")
    print("  Below: MAPE=3.88%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 2A (Entity Embedding):")
    print("  Above: MAPE=8.83%, 異常點=10/138 (7.25%)")
    print("  Below: MAPE=3.90%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 2B (樣本加權):")
    print(f"  Above: MAPE={results_above['mape']:.2f}%, "
          f"異常點={results_above['outliers_20']}/{len(y_test_above)} "
          f"({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"  Below: MAPE={results_below['mape']:.2f}%, "
          f"異常點={results_below['outliers_20']}/{len(y_test_below)} "
          f"({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    
    # 計算改進
    improvement = 10 - results_above['outliers_20']
    mape_improvement = 8.63 - results_above['mape']
    
    if improvement > 0:
        print(f"\n相比Phase 1改進:")
        print(f"  異常點: -{improvement} ({improvement/10*100:.1f}% reduction)")
        print(f"  MAPE: {mape_improvement:+.2f}%")
    elif improvement == 0:
        print(f"\n與Phase 1持平")
    else:
        print(f"\n相比Phase 1退步: +{-improvement}個異常點")
    
    print(f"\n{'='*60}\n")
    
    # 保存
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('phase2b_summary.csv', index=False)
    print("結果已保存\n")
    
    return {
        'above': (model_above, results_above, test_above_pred),
        'below': (model_below, results_below, test_below_pred)
    }


if __name__ == "__main__":
    results = main_weighted_experiment()