"""
Phase 2H: 對抗 Variance Collapse
針對 GP 過度壓縮變異性的問題

關鍵策略：
1. 降低 GP likelihood 噪音
2. 增加 feature extractor 的表達能力
3. 針對 Type 3 使用更激進的訓練策略

使用方法:
    python phase2h_anti_collapse.py --seed 2024
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
    print(f"✓ 隨機種子設定為: {seed}")


def clear_gpu_cache():
    """清空GPU快取"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==========================================
# 增強的特徵提取器 (更大容量)
# ==========================================

class EnhancedDnnFeatureExtractor(nn.Module):
    """
    增強的特徵提取器
    
    關鍵改進：
    1. 更深的網路 (捕捉複雜模式)
    2. 殘差連接 (防止梯度消失)
    3. 更大的特徵空間 (保留變異性)
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 128, 64, 32], output_dim=16, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 輸入層
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 隱藏層（帶殘差連接）
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # 輸出層
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x_new = layer(x)
            # 如果維度相同，加入殘差
            if x_new.shape[1] == x.shape[1]:
                x = x + x_new
            else:
                x = x_new
        
        x = self.output_layer(x)
        return x


class AntiCollapseGPModel(gpytorch.models.ExactGP):
    """
    對抗 Variance Collapse 的 GP 模型
    
    關鍵改進：
    1. 更小的 likelihood 噪音（強制模型擬合細節）
    2. 更敏感的 kernel（小 lengthscale）
    3. 使用 FixedNoiseGaussianLikelihood（手動控制噪音）
    """
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 敏感的 RBF kernel（小 lengthscale）
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=feature_extractor.output_dim,
                lengthscale_constraint=gpytorch.constraints.Interval(0.01, 1.0)  # 限制 lengthscale 更小
            )
        )
    
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 損失函數（加入方差懲罰）
# ==========================================

def variance_aware_loss(y_pred, y_true, weights, target_std, epsilon=1e-8):
    """
    加入方差懲罰的損失函數
    
    Args:
        y_pred: 預測值
        y_true: 真實值
        weights: 樣本權重
        target_std: 目標標準差（真實值的 std）
    """
    # MAPE 損失
    mape = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    weighted_mape = torch.sum(mape * weights) / torch.sum(weights)
    
    # 方差懲罰（鼓勵預測保持變異性）
    pred_std = torch.std(y_pred)
    std_penalty = torch.abs(pred_std - target_std) / target_std
    
    return weighted_mape + 10.0 * std_penalty  # 權重 10.0


# ==========================================
# 樣本權重
# ==========================================

def compute_advanced_weights(X, y=None):
    """計算樣本權重"""
    weights = np.ones(len(X))
    
    tim_type = X[:, 0]
    coverage = X[:, 2]
    
    # Type 3 基礎權重
    type3_mask = tim_type == 3
    weights[type3_mask] *= 3.0
    
    # 高 Coverage 區域
    high_cov_mask = (coverage >= 0.75)
    weights[type3_mask & high_cov_mask] *= 5.0
    
    # 極小真實值
    if y is not None:
        small_value_mask = y < 0.03
        weights[small_value_mask] *= 3.0
    
    return weights


# ==========================================
# 訓練函數
# ==========================================

def train_anti_collapse_model(X_train, y_train, config, verbose=True):
    """
    訓練對抗 Variance Collapse 的模型
    """
    # 計算樣本權重
    sample_weights = compute_advanced_weights(X_train, y_train)
    
    if verbose:
        print(f"\n高權重樣本數: {np.sum(sample_weights > 5.0)}")
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_train_scaled).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    sample_weights_tensor = torch.from_numpy(sample_weights).to(device)
    
    # 計算目標標準差
    target_std = torch.std(train_y)
    
    # 建立模型
    feature_extractor = EnhancedDnnFeatureExtractor(
        input_dim=train_x.shape[1],
        hidden_dims=config['hidden_dims'],
        output_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # 使用固定噪音（非常小的噪音）
    noise = torch.ones(len(train_x)) * 1e-4  # 非常小的噪音
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise.to(device),
        learn_additional_noise=False  # 不學習噪音
    ).to(device)
    
    model = AntiCollapseGPModel(train_x, train_y, likelihood, feature_extractor).to(device)
    
    # 優化器
    optimizer = optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr'], 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.mean_module.parameters(), 'lr': config['lr'] * 0.1},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # 訓練
    if verbose:
        print(f"\n開始訓練 (對抗 Variance Collapse)...")
    
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        
        # 加入方差懲罰
        variance_loss = variance_aware_loss(
            output.mean, train_y, sample_weights_tensor, target_std
        )
        
        total_loss = gp_loss + config['variance_weight'] * variance_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        current_loss = total_loss.item()
        
        if verbose and (epoch + 1) % 100 == 0:
            pred_std = torch.std(output.mean).item()
            print(f"Epoch {epoch+1}: GP Loss={gp_loss.item():.4f}, "
                  f"Var Loss={variance_loss.item():.4f}, "
                  f"Pred Std={pred_std:.4f}, Target Std={target_std.item():.4f}")
        
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
        print(f"訓練完成")
    
    return model, likelihood, scaler_x, scaler_y


def evaluate_model(model, likelihood, X_test, y_test, scaler_x, scaler_y, verbose=True):
    """評估模型"""
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
            
            # 檢查 variance collapse
            cov08_pred = y_pred[cov08_mask]
            cov08_true = y_test[cov08_mask]
            
            pred_std = np.std(cov08_pred)
            true_std = np.std(cov08_true)
            compression_ratio = pred_std / true_std if true_std > 0 else 0
            
            if verbose:
                print(f"\n{'='*60}")
                print("Coverage 0.8 詳細分析")
                print(f"{'='*60}")
                print(f"  真實值 Std: {true_std:.4f}")
                print(f"  預測值 Std: {pred_std:.4f}")
                print(f"  壓縮比: {compression_ratio:.3f} {'❌ 過度壓縮' if compression_ratio < 0.5 else '✓ 合理'}")
                print(f"\n  詳細預測:")
                for i in range(len(cov08_true)):
                    marker = "❌" if cov08_errors[i] > 20 else "✓"
                    print(f"    {marker} Thick={X_test[cov08_mask][i, 1]:.0f}, "
                          f"True={cov08_true[i]:.3f}, Pred={cov08_pred[i]:.3f}, "
                          f"Error={cov08_errors[i]:.1f}%")
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

def main(seed=2024, verbose=True):
    """主訓練流程"""
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}\n")
    
    print("="*60)
    print("Phase 2H: 對抗 Variance Collapse")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置
    config = {
        'hidden_dims': [128, 128, 64, 32],
        'feature_dim': 16,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 600,
        'patience': 60,
        'variance_weight': 0.5,  # 方差懲罰權重
    }
    
    if verbose:
        print(f"\n配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    # 訓練集清理
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"\n訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆")
    
    X_train = train_above_clean[feature_cols].values
    y_train = train_above_clean[target_col].values
    
    X_test = test_above[feature_cols].values
    y_test = test_above[target_col].values
    
    # 訓練
    model, likelihood, scaler_x, scaler_y = train_anti_collapse_model(
        X_train, y_train, config, verbose=verbose
    )
    
    # 評估
    results = evaluate_model(
        model, likelihood, X_test, y_test, 
        scaler_x, scaler_y, verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test, y_test, results,
                     f'phase2h_anti_collapse_seed{seed}_predictions.csv')
    
    # 總結
    print("\n" + "="*60)
    print("最終結果總結 (Phase 2H)")
    print("="*60)
    print(f"策略:")
    print(f"  ✓ 固定極小噪音 (1e-4)")
    print(f"  ✓ 敏感 kernel (小 lengthscale)")
    print(f"  ✓ 方差懲罰 (保持變異性)")
    print(f"  ✓ 更深網路 (128-128-64-32)")
    print(f"\n結果:")
    print(f"  總體 MAPE: {results['mape']:.2f}%")
    print(f"  Type 3 MAPE: {results['type3_mape']:.2f}%")
    print(f"  Coverage 0.8 MAPE: {results['cov08_mape']:.2f}%")
    print(f"  異常點: {results['outliers_20']}/{len(y_test)}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2H 對抗 Variance Collapse')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細模式')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
    
    print("\n💡 說明:")
    print("  此版本針對 Variance Collapse 問題")
    print("  目標：壓縮比從 0.29 提高到 >0.6\n")
