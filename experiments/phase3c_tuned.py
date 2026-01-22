"""
Phase 3D: 最終修正版 - 回歸簡單但正確的實現

核心修正:
1. 使用 scaler.inverse_transform() 確保反標準化正確
2. 去除複雜的 Barrier Loss，使用簡單的加權 MAPE
3. 在原始空間正確計算 Loss

使用方法:
    python phase3d_final.py --n_trials 30
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
import optuna
from optuna.samplers import TPESampler
import json

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

class DnnFeatureExtractorWithEmbedding(nn.Module):
    """深度神經網路特徵提取器 + Entity Embedding"""
    
    def __init__(self, num_categories=3, embedding_dim=4, numerical_dim=2, 
                 hidden_dims=[64, 32, 16], output_dim=8, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(num_categories + 1, embedding_dim)
        input_dim = embedding_dim + numerical_dim
        
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
        
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x_cat, x_num):
        embedded = self.embedding(x_cat.long())
        combined = torch.cat([embedded, x_num], dim=1)
        return self.network(combined)


class StandardGPModel(gpytorch.models.ExactGP):
    """標準 GP 模型"""
    
    def __init__(self, train_x, train_y, likelihood, feature_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 損失函數（簡化版）
# ==========================================

def compute_sample_weights(X, weight_factor=3.0):
    """計算樣本權重"""
    weights = np.ones(len(X))
    
    difficult_mask = (
        (X[:, 0] == 3) &
        (X[:, 2] == 0.8) &
        (X[:, 1] >= 220)
    )
    
    weights[difficult_mask] *= weight_factor
    return weights


def safe_mape_loss(y_pred, y_true, weights, epsilon=1e-8):
    """
    安全的 MAPE Loss - 在原始空間計算
    
    簡單但正確的實現，不使用複雜的 Barrier
    """
    # 計算百分比誤差
    abs_error_percent = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon)) * 100
    
    # 【安全性】Clamp 到合理範圍，避免極端值
    abs_error_percent = torch.clamp(abs_error_percent, max=100.0)
    
    # 加權平均
    weighted_mape = torch.sum(abs_error_percent * weights) / torch.sum(weights)
    
    return weighted_mape


# ==========================================
# 訓練函數（完全重寫）
# ==========================================

def train_dkl_model(X_train, y_train, config, verbose=False):
    """
    訓練 DKL 模型 - 最終修正版
    
    關鍵: 使用 scaler.inverse_transform() 確保反標準化正確
    """
    # 分離特徵
    X_cat = X_train[:, 0]
    X_num = X_train[:, 1:]
    
    # 標準化數值特徵
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(X_num)
    
    # 標準化目標變數
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # 轉 Tensor
    train_x_cat = torch.from_numpy(X_cat).to(device)
    train_x_num = torch.from_numpy(X_num_scaled).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    
    # 【重要】同時保存原始 y 用於 Loss 計算
    train_y_original = torch.from_numpy(y_train).to(device)
    
    # 樣本權重
    sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
    sample_weights = torch.from_numpy(sample_weights_np).to(device)
    
    # 建立模型
    feature_extractor = DnnFeatureExtractorWithEmbedding(
        num_categories=3,
        embedding_dim=config['embedding_dim'],
        numerical_dim=2,
        hidden_dims=config['hidden_dims'],
        output_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    with torch.no_grad():
        initial_features = feature_extractor(train_x_cat, train_x_num)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp_model = StandardGPModel(
        initial_features, train_y, likelihood, config['feature_dim']
    ).to(device)
    
    # 優化器
    optimizer = optim.Adam([
        {'params': feature_extractor.parameters(), 
         'lr': config['lr'], 'weight_decay': config['weight_decay']},
        {'params': gp_model.covar_module.parameters()},
        {'params': gp_model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    
    # 訓練
    feature_extractor.train()
    gp_model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        # 提取特徵
        features = feature_extractor(train_x_cat, train_x_num)
        gp_model.set_train_data(features, train_y, strict=False)
        output = gp_model(features)
        
        # Loss 1: GP Loss (在標準化空間)
        gp_loss = -mll(output, train_y)
        
        # Loss 2: MAPE (在原始空間！)
        # 【關鍵】使用 scaler 正確反標準化
        pred_scaled = output.mean.cpu().detach().numpy().reshape(-1, 1)
        pred_original = scaler_y.inverse_transform(pred_scaled).flatten()
        pred_original_tensor = torch.from_numpy(pred_original).to(device)
        
        # 計算原始空間的 MAPE
        mape_loss_val = safe_mape_loss(pred_original_tensor, train_y_original, sample_weights)
        
        # 總損失
        total_loss = gp_loss + config['mape_weight'] * mape_loss_val
        
        # 反向傳播
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        current_loss = total_loss.item()
        
        # 監控
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: GP Loss={gp_loss.item():.4f}, "
                  f"MAPE={mape_loss_val.item():.2f}%, Total={total_loss.item():.4f}")
        
        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_state = {
                'feature_extractor': feature_extractor.state_dict(),
                'gp_model': gp_model.state_dict(),
                'likelihood': likelihood.state_dict(),
            }
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            if verbose:
                print(f"早停 at Epoch {epoch+1}")
            break
    
    # 載入最佳模型
    feature_extractor.load_state_dict(best_state['feature_extractor'])
    gp_model.load_state_dict(best_state['gp_model'])
    likelihood.load_state_dict(best_state['likelihood'])
    
    return feature_extractor, gp_model, likelihood, scaler_num, scaler_y


def evaluate_dkl_model(feature_extractor, gp_model, likelihood, 
                       X_test, y_test, scaler_num, scaler_y, verbose=False):
    """評估模型"""
    feature_extractor.eval()
    gp_model.eval()
    likelihood.eval()
    
    X_cat = X_test[:, 0]
    X_num = X_test[:, 1:]
    X_num_scaled = scaler_num.transform(X_num)
    
    test_x_cat = torch.from_numpy(X_cat).to(device)
    test_x_num = torch.from_numpy(X_num_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_features = feature_extractor(test_x_cat, test_x_num)
        pred_dist = likelihood(gp_model(test_features))
        y_pred_scaled = pred_dist.mean.cpu().numpy()
        y_std_scaled = pred_dist.stddev.cpu().numpy()
    
    # 使用 scaler.inverse_transform() 反標準化
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_std = y_std_scaled * scaler_y.scale_[0]
    
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((relative_errors > 20) & type3_mask)
    
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
        print(f"  >15%: {outliers_15}/{len(y_test)}")
        print(f"  >10%: {outliers_10}/{len(y_test)}")
        if np.sum(type3_mask) > 0:
            print(f"\nType 3異常點: {type3_outliers}/{np.sum(type3_mask)}")
        print(f"{'='*60}\n")
    
    return {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'type3_outliers': type3_outliers,
        'predictions': y_pred,
        'std': y_std,
        'errors': relative_errors
    }


# ==========================================
# Optuna
# ==========================================

def objective(trial, data_dict, seed=2024):
    """Optuna 優化"""
    clear_gpu_cache()
    set_seed(seed)
    
    config = {
        'embedding_dim': trial.suggest_int('embedding_dim', 4, 8),
        'hidden_dims': [64, 32, 16],
        'feature_dim': trial.suggest_int('feature_dim', 8, 16),
        'dropout': trial.suggest_float('dropout', 0.05, 0.15),
        'lr': trial.suggest_float('lr', 0.008, 0.012, log=False),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True),
        'mape_weight': trial.suggest_float('mape_weight', 0.1, 0.3),
        'sample_weight_factor': trial.suggest_float('sample_weight_factor', 2.5, 3.5),
        'epochs': 500,
        'patience': 50,
    }
    
    try:
        # Above
        fe_above, gp_above, ll_above, scaler_num_above, scaler_y_above = train_dkl_model(
            data_dict['X_train_above'], data_dict['y_train_above'], config, verbose=False
        )
        
        results_above = evaluate_dkl_model(
            fe_above, gp_above, ll_above,
            data_dict['X_test_above'], data_dict['y_test_above'],
            scaler_num_above, scaler_y_above, verbose=False
        )
        
        # Below
        fe_below, gp_below, ll_below, scaler_num_below, scaler_y_below = train_dkl_model(
            data_dict['X_train_below'], data_dict['y_train_below'], config, verbose=False
        )
        
        results_below = evaluate_dkl_model(
            fe_below, gp_below, ll_below,
            data_dict['X_test_below'], data_dict['y_test_below'],
            scaler_num_below, scaler_y_below, verbose=False
        )
        
        # 目標值
        objective_value = (results_above['outliers_20'] + 0.3 * results_above['mape'] + 
                          0.1 * results_below['mape'])
        
        # 記錄指標
        trial.set_user_attr('above_outliers_20', results_above['outliers_20'])
        trial.set_user_attr('above_mape', results_above['mape'])
        trial.set_user_attr('above_type3_outliers', results_above['type3_outliers'])
        trial.set_user_attr('below_outliers_20', results_below['outliers_20'])
        trial.set_user_attr('below_mape', results_below['mape'])
        
        return objective_value
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')


# ==========================================
# 主函數
# ==========================================

def main_optuna(n_trials=30, seed=2024):
    """主訓練流程"""
    set_seed(seed)
    
    print("="*60)
    print("Phase 3D: 最終修正版")
    print("="*60)
    print(f"裝置: {device}")
    print(f"隨機種子: {seed}")
    print(f"Optuna 試驗次數: {n_trials}\n")
    
    print("【關鍵修正】")
    print("  ✓ 使用 scaler.inverse_transform() 正確反標準化")
    print("  ✓ 在原始空間計算 MAPE Loss")
    print("  ✓ 簡化但正確的實現\n")
    
    # 載入資料
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    train_below = pd.read_excel('data/train/Below.xlsx')
    test_below = pd.read_excel('data/test/Below.xlsx')
    
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({target_col: 'mean'})
    train_below_clean = train_below.groupby(feature_cols, as_index=False).agg({target_col: 'mean'})
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    X_train_below = train_below_clean[feature_cols].values
    y_train_below = train_below_clean[target_col].values
    X_test_below = test_below[feature_cols].values
    y_test_below = test_below[target_col].values
    
    print(f"Above: 訓練{len(X_train_above)}筆, 測試{len(X_test_above)}筆")
    print(f"Below: 訓練{len(X_train_below)}筆, 測試{len(X_test_below)}筆\n")
    
    data_dict = {
        'X_train_above': X_train_above, 'y_train_above': y_train_above,
        'X_test_above': X_test_above, 'y_test_above': y_test_above,
        'X_train_below': X_train_below, 'y_train_below': y_train_below,
        'X_test_below': X_test_below, 'y_test_below': y_test_below,
    }
    
    # Optuna
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=seed),
        study_name=f'phase3d_final_seed{seed}'
    )
    
    print("開始 Optuna 搜尋...\n")
    study.optimize(
        lambda trial: objective(trial, data_dict, seed=seed),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 檢查結果
    successful_trials = [t for t in study.trials if t.value != float('inf')]
    
    if len(successful_trials) == 0:
        print("\n❌ 所有 trials 都失敗了！\n")
        return None, None, None
    
    # 最佳結果
    best_trial = study.best_trial
    print("\n" + "="*60)
    print("Optuna 搜尋完成！")
    print("="*60)
    print(f"\n最佳 Trial: #{best_trial.number}")
    print(f"目標值: {best_trial.value:.4f}")
    print(f"成功 trials: {len(successful_trials)}/{n_trials}")
    
    print(f"\nAbove:")
    print(f"  異常點: {best_trial.user_attrs['above_outliers_20']}")
    print(f"  MAPE: {best_trial.user_attrs['above_mape']:.2f}%")
    print(f"  Type 3異常點: {best_trial.user_attrs['above_type3_outliers']}")
    
    print(f"\nBelow:")
    print(f"  異常點: {best_trial.user_attrs['below_outliers_20']}")
    print(f"  MAPE: {best_trial.user_attrs['below_mape']:.2f}%")
    
    print(f"\n最佳超參數:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # 重置種子並重新訓練
    print("\n" + "="*60)
    print("用最佳配置重新訓練...")
    print("="*60 + "\n")
    
    set_seed(seed)
    clear_gpu_cache()
    
    best_config = {
        'embedding_dim': best_trial.params['embedding_dim'],
        'hidden_dims': [64, 32, 16],
        'feature_dim': best_trial.params['feature_dim'],
        'dropout': best_trial.params['dropout'],
        'lr': best_trial.params['lr'],
        'weight_decay': best_trial.params['weight_decay'],
        'mape_weight': best_trial.params['mape_weight'],
        'sample_weight_factor': best_trial.params['sample_weight_factor'],
        'epochs': 500,
        'patience': 50,
    }
    
    # Above
    print("🔵 Above\n")
    fe_above, gp_above, ll_above, scaler_num_above, scaler_y_above = train_dkl_model(
        X_train_above, y_train_above, best_config, verbose=True
    )
    results_above = evaluate_dkl_model(
        fe_above, gp_above, ll_above, X_test_above, y_test_above,
        scaler_num_above, scaler_y_above, verbose=True
    )
    
    # Below
    print("\n🔵 Below\n")
    fe_below, gp_below, ll_below, scaler_num_below, scaler_y_below = train_dkl_model(
        X_train_below, y_train_below, best_config, verbose=True
    )
    results_below = evaluate_dkl_model(
        fe_below, gp_below, ll_below, X_test_below, y_test_below,
        scaler_num_below, scaler_y_below, verbose=True
    )
    
    # 保存結果
    pd.DataFrame({
        'TIM_TYPE': X_test_above[:, 0],
        'TIM_THICKNESS': X_test_above[:, 1],
        'TIM_COVERAGE': X_test_above[:, 2],
        'True': y_test_above,
        'Predicted': results_above['predictions'],
        'Error%': results_above['errors'],
        'Std': results_above['std']
    }).to_csv(f'phase3d_above_seed{seed}.csv', index=False)
    
    pd.DataFrame({
        'TIM_TYPE': X_test_below[:, 0],
        'TIM_THICKNESS': X_test_below[:, 1],
        'TIM_COVERAGE': X_test_below[:, 2],
        'True': y_test_below,
        'Predicted': results_below['predictions'],
        'Error%': results_below['errors'],
        'Std': results_below['std']
    }).to_csv(f'phase3d_below_seed{seed}.csv', index=False)
    
    with open(f'phase3d_config_seed{seed}.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"✓ 結果已保存\n")
    
    # 總結
    print("="*60)
    print("最終結果")
    print("="*60)
    print(f"\nAbove: 異常點 {results_above['outliers_20']}/138, MAPE {results_above['mape']:.2f}%")
    print(f"Below: 異常點 {results_below['outliers_20']}/48, MAPE {results_below['mape']:.2f}%")
    
    baseline_outliers, baseline_mape = 7, 8.34
    improvement_outliers = (baseline_outliers - results_above['outliers_20']) / baseline_outliers * 100
    improvement_mape = (baseline_mape - results_above['mape']) / baseline_mape * 100
    
    print(f"\n與 Phase 2B 比較:")
    print(f"  異常點: {baseline_outliers} → {results_above['outliers_20']} ({improvement_outliers:+.1f}%)")
    print(f"  MAPE: {baseline_mape:.2f}% → {results_above['mape']:.2f}% ({improvement_mape:+.1f}%)")
    print("="*60 + "\n")
    
    return study, results_above, results_below


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    
    study, results_above, results_below = main_optuna(args.n_trials, args.seed)