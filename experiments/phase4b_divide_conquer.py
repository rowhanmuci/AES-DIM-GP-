"""
Phase 4B - 分治訓練策略 (Divide and Conquer)
分別訓練 Type 1&2 和 Type 3 模型
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
import time

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """設置隨機種子以確保完全可重現性"""
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
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """深度神經網路特徵提取器"""
    
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
    """高斯過程回歸模型"""
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
        )
        '''
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feature_extractor.output_dim)
        )
        '''        
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 損失函數與權重
# ==========================================

def weighted_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    """加權MAPE損失函數"""
    mape_per_sample = torch.abs((y_true - y_pred) / 
                                (torch.abs(y_true) + epsilon)) * 100
    weighted_mape = torch.sum(mape_per_sample * weights) / torch.sum(weights)
    return weighted_mape


def compute_sample_weights_type3(X, weight_factor=10.0):
    """
    Type 3 專用權重計算
    針對高 Coverage (≥0.8) 和大 Thickness (≥220) 給予更高權重
    """
    weights = np.ones(len(X))
    
    # Type 3 的困難樣本: Coverage >= 0.8 AND Thickness >= 220
    difficult_mask = (
        (X[:, 1] >= 0.8) &    # Coverage >= 0.8 (注意這裡 X 是 [THICKNESS, COVERAGE])
        (X[:, 0] >= 220)      # THICKNESS >= 220
    )
    
    weights[difficult_mask] *= weight_factor
    
    return weights


# ==========================================
# 訓練與評估
# ==========================================

def train_model(X_train, y_train, config, model_name="Model", verbose=True):
    """
    訓練 DKL 模型
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        config: 訓練配置
        model_name: 模型名稱 (用於顯示)
        verbose: 是否顯示訓練過程
        
    Returns:
        model, likelihood, scaler_x, scaler_y
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"訓練 {model_name}")
        print(f"{'='*60}")
        print(f"訓練樣本數: {len(X_train)}")
    
    # 計算樣本權重 (只有 Type 3 模型用特殊權重)
    if 'Type 3' in model_name:
        sample_weights_np = compute_sample_weights_type3(X_train, config['type3_weight_factor'])
        if verbose:
            difficult_count = np.sum(sample_weights_np > 1.0)
            print(f"困難樣本 (Coverage≥0.8, Thickness≥220): {difficult_count} ({difficult_count/len(X_train)*100:.2f}%)")
            print(f"權重倍數: {config['type3_weight_factor']}x")
    else:
        sample_weights_np = np.ones(len(X_train))
        if verbose:
            print(f"使用均勻權重")
    
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
    if verbose:
        print(f"\n開始訓練...")
    
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        optimizer.zero_grad()
        
        output = model(train_x)
        gp_loss = -mll(output, train_y)
        mape = weighted_mape_loss(output.mean, train_y, sample_weights)
        total_loss = gp_loss + config['mape_weight'] * mape
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        epoch_time = time.time() - epoch_start
        
        # 顯示訓練進度
        if verbose and (epoch + 1) % 50 == 0:
            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / (epoch + 1)
            eta = avg_epoch_time * (config['epochs'] - epoch - 1)
            
            print(f"Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"GP Loss={gp_loss.item():7.4f}, "
                  f"MAPE={mape.item():6.2f}%, "
                  f"Total={total_loss.item():7.4f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min")
        
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
                print(f"\n早停 at Epoch {epoch+1}")
            break
    
    # 載入最佳模型
    model.load_state_dict(best_state['model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    if verbose:
        total_time = time.time() - start_time
        print(f"訓練完成 (Final Loss: {best_loss:.4f}, 時間: {total_time/60:.2f} 分鐘)")
    
    return model, likelihood, scaler_x, scaler_y


def evaluate_model(model, likelihood, X_test, y_test, scaler_x, scaler_y, 
                   model_name="Model", verbose=True):
    """
    評估模型
    
    Returns:
        results: 包含 MAPE, outliers 等指標的字典
    """
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
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{model_name} 評估結果")
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
        print(f"{'='*60}\n")
    
    results = {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'predictions': y_pred,
        'std': y_std,
        'errors': relative_errors
    }
    
    return results


def save_predictions(X_test, y_test, results, filename, include_type=False):
    """保存預測結果到CSV"""
    if include_type:
        df = pd.DataFrame({
            'TIM_TYPE': X_test[:, 0],
            'TIM_THICKNESS': X_test[:, 1],
            'TIM_COVERAGE': X_test[:, 2],
            'True': y_test,
            'Predicted': results['predictions'],
            'Error%': results['errors'],
            'Std': results['std']
        })
    else:
        df = pd.DataFrame({
            'TIM_THICKNESS': X_test[:, 0],
            'TIM_COVERAGE': X_test[:, 1],
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
    """
    主訓練流程：分治訓練 Type 1&2 和 Type 3
    """
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}\n")
    
    print("="*60)
    print("Phase 4B: 分治訓練策略 (Divide and Conquer)")
    print("="*60)
    print("策略: 分別訓練 Type 1&2 和 Type 3 模型")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置
    config = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
        'type3_weight_factor': 3.0,  # Type 3 困難樣本權重
    }
    
    if verbose:
        print(f"\n配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # ==========================================
    # Above Dataset
    # ==========================================
    
    print(f"\n\n{'='*80}")
    print(f"🔵 Above 50% Coverage")
    print(f"{'='*80}\n")
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    # 訓練集清理
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"總訓練集: {len(train_above)} 筆")
    print(f"總測試集: {len(test_above)} 筆")
    
    # ========== 分割訓練集 ==========
    
    # Type 1 & 2
    train_type12 = train_above[train_above['TIM_TYPE'].isin([1, 2])]
    # 移除 TIM_TYPE 欄位 (不需要作為特徵)
    X_train_type12 = train_type12[['TIM_THICKNESS', 'TIM_COVERAGE']].values
    y_train_type12 = train_type12[target_col].values
    
    # Type 3
    train_type3 = train_above[train_above['TIM_TYPE'] == 3]
    X_train_type3 = train_type3[['TIM_THICKNESS', 'TIM_COVERAGE']].values
    y_train_type3 = train_type3[target_col].values
    
    print(f"\n訓練集分割:")
    print(f"  Type 1 & 2: {len(train_type12)} 筆")
    print(f"  Type 3:     {len(train_type3)} 筆")
    
    # ========== 分割測試集 ==========
    
    test_type12 = test_above[test_above['TIM_TYPE'].isin([1, 2])]
    X_test_type12 = test_type12[['TIM_THICKNESS', 'TIM_COVERAGE']].values
    y_test_type12 = test_type12[target_col].values
    
    test_type3 = test_above[test_above['TIM_TYPE'] == 3]
    X_test_type3 = test_type3[['TIM_THICKNESS', 'TIM_COVERAGE']].values
    y_test_type3 = test_type3[target_col].values
    
    print(f"\n測試集分割:")
    print(f"  Type 1 & 2: {len(test_type12)} 筆")
    print(f"  Type 3:     {len(test_type3)} 筆")
    
    # ========== 訓練模型 A: Type 1 & 2 ==========
    
    model_type12, likelihood_type12, scaler_x_type12, scaler_y_type12 = train_model(
        X_train_type12, y_train_type12, config, 
        model_name="模型 A (Type 1 & 2)", 
        verbose=verbose
    )
    
    # 評估模型 A
    results_type12 = evaluate_model(
        model_type12, likelihood_type12,
        X_test_type12, y_test_type12,
        scaler_x_type12, scaler_y_type12,
        model_name="模型 A (Type 1 & 2)",
        verbose=verbose
    )
    
    save_predictions(X_test_type12, y_test_type12, results_type12,
                     f'phase4b_type12_above_seed{seed}_predictions.csv')
    
    clear_gpu_cache()
    
    # ========== 訓練模型 B: Type 3 ==========
    
    model_type3, likelihood_type3, scaler_x_type3, scaler_y_type3 = train_model(
        X_train_type3, y_train_type3, config,
        model_name="模型 B (Type 3)",
        verbose=verbose
    )
    
    # 評估模型 B
    results_type3 = evaluate_model(
        model_type3, likelihood_type3,
        X_test_type3, y_test_type3,
        scaler_x_type3, scaler_y_type3,
        model_name="模型 B (Type 3)",
        verbose=verbose
    )
    
    save_predictions(X_test_type3, y_test_type3, results_type3,
                     f'phase4b_type3_above_seed{seed}_predictions.csv')
    
    # ========== 合併結果 ==========
    
    # 合併預測
    all_predictions = np.zeros(len(test_above))
    all_errors = np.zeros(len(test_above))
    
    type12_indices = test_above[test_above['TIM_TYPE'].isin([1, 2])].index
    type3_indices = test_above[test_above['TIM_TYPE'] == 3].index
    
    all_predictions[type12_indices] = results_type12['predictions']
    all_predictions[type3_indices] = results_type3['predictions']
    
    all_errors[type12_indices] = results_type12['errors']
    all_errors[type3_indices] = results_type3['errors']
    
    # 計算整體指標
    overall_mape = np.mean(all_errors)
    overall_outliers_20 = np.sum(all_errors > 20)
    overall_outliers_15 = np.sum(all_errors > 15)
    overall_outliers_10 = np.sum(all_errors > 10)
    
    # Type 3 的異常點
    type3_outliers = np.sum(all_errors[type3_indices] > 20)
    
    # 保存合併結果
    combined_results = {
        'predictions': all_predictions,
        'errors': all_errors,
        'std': np.concatenate([results_type12['std'], results_type3['std']])
    }
    save_predictions(
        test_above[feature_cols].values,
        test_above[target_col].values,
        combined_results,
        f'phase4b_combined_above_seed{seed}_predictions.csv',
        include_type=True
    )
    
    # ==========================================
    # Below Dataset (保持原樣，不分治)
    # ==========================================
    
    print(f"\n\n{'='*80}")
    print(f"🔵 Below 50% Coverage (統一模型)")
    print(f"{'='*80}\n")
    
    train_below = pd.read_excel('data/train/Below.xlsx')
    test_below = pd.read_excel('data/test/Below.xlsx')
    
    train_below_clean = train_below.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_below_clean)} 筆")
    print(f"測試集: {len(test_below)} 筆")
    
    X_train_below = train_below_clean[feature_cols].values
    y_train_below = train_below_clean[target_col].values
    
    X_test_below = test_below[feature_cols].values
    y_test_below = test_below[target_col].values
    
    # 訓練 (使用原始方法，因為 Below 問題不大)
    # 暫時用 Type 1&2 的權重策略 (均勻權重)
    config_below = config.copy()
    config_below['type3_weight_factor'] = 3.0  # 降低權重
    
    model_below, likelihood_below, scaler_x_below, scaler_y_below = train_model(
        X_train_below, y_train_below, config_below,
        model_name="模型 (Below 統一)",
        verbose=verbose
    )
    
    # 評估
    results_below = evaluate_model(
        model_below, likelihood_below,
        X_test_below, y_test_below,
        scaler_x_below, scaler_y_below,
        model_name="模型 (Below 統一)",
        verbose=verbose
    )
    
    save_predictions(X_test_below, y_test_below, results_below,
                     f'phase4b_below_seed{seed}_predictions.csv',
                     include_type=True)
    
    # ==========================================
    # 總結
    # ==========================================
    
    print("\n" + "="*80)
    print("Phase 4B 最終結果總結")
    print("="*80)
    print(f"隨機種子: {seed}")
    
    print(f"\nAbove 資料集 (分治策略):")
    print(f"  模型 A (Type 1 & 2):")
    print(f"    樣本數: {len(test_type12)}")
    print(f"    MAPE: {results_type12['mape']:.2f}%")
    print(f"    異常點 >20%: {results_type12['outliers_20']}/{len(test_type12)} ({results_type12['outliers_20']/len(test_type12)*100:.2f}%)")
    
    print(f"\n  模型 B (Type 3):")
    print(f"    樣本數: {len(test_type3)}")
    print(f"    MAPE: {results_type3['mape']:.2f}%")
    print(f"    異常點 >20%: {results_type3['outliers_20']}/{len(test_type3)} ({results_type3['outliers_20']/len(test_type3)*100:.2f}%)")
    
    print(f"\n  整體 (合併):")
    print(f"    總樣本數: {len(test_above)}")
    print(f"    整體 MAPE: {overall_mape:.2f}%")
    print(f"    異常點 >20%: {overall_outliers_20}/{len(test_above)} ({overall_outliers_20/len(test_above)*100:.2f}%)")
    print(f"    異常點 >15%: {overall_outliers_15}/{len(test_above)} ({overall_outliers_15/len(test_above)*100:.2f}%)")
    print(f"    異常點 >10%: {overall_outliers_10}/{len(test_above)} ({overall_outliers_10/len(test_above)*100:.2f}%)")
    print(f"    Type 3 異常點: {type3_outliers}/{len(test_type3)}")
    
    print(f"\nBelow 資料集 (統一模型):")
    print(f"  異常點 (>20%): {results_below['outliers_20']}/{len(y_test_below)} ({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    print(f"  MAPE: {results_below['mape']:.2f}%")
    
    print("\n" + "="*80)
    print("✓ 分治訓練完成！")
    print("="*80)
    
    # 與 Phase 2B 比較
    print(f"\n💡 與 Phase 2B 比較:")
    print(f"   Phase 2B (統一模型): MAPE ~48%, Type 3 異常點較多")
    print(f"   Phase 4B (分治策略): MAPE {overall_mape:.2f}%, Type 3 異常點 {type3_outliers}/{len(test_type3)}")
    print(f"   改善: {'✓ 有改善' if overall_mape < 48 else '⚠️ 需要調整'}")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'above_type12': results_type12,
        'above_type3': results_type3,
        'above_overall_mape': overall_mape,
        'above_overall_outliers': overall_outliers_20,
        'below': results_below,
        'seed': seed
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 4B - 分治訓練策略')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子 (預設: 2024)')
    parser.add_argument('-v', '--verbose', action='store_true', help='顯示詳細訓練過程')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
    
    print("\n💡 使用範例:")
    print("  python phase4b_divide_conquer.py --seed 2024 -v")
    print("  python phase4b_divide_conquer.py --seed 42")
    print("\n🎯 優勢:")
    print("  1. Type 3 獨立建模，不受 Type 1&2 干擾")
    print("  2. Type 3 可用 10x 權重強化困難樣本")
    print("  3. 模型更專注，學習更精準")
    print("  4. 預測時根據 Type 自動選擇模型\n")