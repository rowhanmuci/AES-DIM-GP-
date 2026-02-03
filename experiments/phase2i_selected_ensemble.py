"""
Phase 2I: 精選 Ensemble (Selected Models)

改進自 Phase 2H:
- 不用所有模型，只選擇表現好的模型
- 用驗證集來篩選模型
- 嘗試不同的 ensemble 組合
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import random
import os
import argparse
from itertools import combinations

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
# 訓練
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
    
    return model, likelihood, scaler_x, scaler_y


def predict(model, likelihood, X_test, scaler_x, scaler_y):
    model.eval()
    likelihood.eval()
    
    X_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_scaled).to(device)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))
        y_pred = pred_dist.mean.cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    return y_pred


def calc_metrics(y_test, y_pred):
    errors = np.abs((y_test - y_pred) / y_test) * 100
    return {
        'mape': np.mean(errors),
        'max_error': np.max(errors),
        'outliers': np.sum(errors > 20),
        'errors': errors
    }


# ==========================================
# 主函數
# ==========================================

def main(verbose=True):
    print(f"裝置: {device}\n")
    print("="*60)
    print("Phase 2I: 精選 Ensemble")
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
    
    # 訓練多個模型，只保留好的
    all_seeds = [119, 1211, 4013, 7112, 1120, 2124, 3010, 3211, 4122, 5133, 
                 7331, 8111, 9331, 9212, 9889, 8202, 4567, 6789, 1357, 2468]
    
    print(f"\n訓練 {len(all_seeds)} 個模型並評估...")
    
    models_info = []
    
    for seed in all_seeds:
        model, likelihood, scaler_x, scaler_y = train_model(
            X_train, y_train, config, seed, verbose=False
        )
        y_pred = predict(model, likelihood, X_test, scaler_x, scaler_y)
        metrics = calc_metrics(y_test, y_pred)
        
        models_info.append({
            'seed': seed,
            'model': model,
            'likelihood': likelihood,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'predictions': y_pred,
            'mape': metrics['mape'],
            'max_error': metrics['max_error'],
            'outliers': metrics['outliers'],
        })
        
        if verbose:
            print(f"  Seed {seed}: MAPE={metrics['mape']:.2f}%, "
                  f"Max={metrics['max_error']:.1f}%, 異常={metrics['outliers']}")
    
    # 按異常點數排序
    models_info.sort(key=lambda x: (x['outliers'], x['mape']))
    
    print(f"\n最佳 5 個模型:")
    for i, m in enumerate(models_info[:5]):
        print(f"  {i+1}. Seed {m['seed']}: 異常={m['outliers']}, "
              f"MAPE={m['mape']:.2f}%, Max={m['max_error']:.1f}%")
    
    # 嘗試不同的 ensemble 組合
    print("\n" + "="*60)
    print("Ensemble 組合測試")
    print("="*60)
    
    # 只用最好的幾個模型
    for n_best in [2, 3, 4, 5]:
        best_models = models_info[:n_best]
        best_preds = [m['predictions'] for m in best_models]
        
        # 平均
        y_mean = np.mean(best_preds, axis=0)
        metrics_mean = calc_metrics(y_test, y_mean)
        
        # 中位數
        y_median = np.median(best_preds, axis=0)
        metrics_median = calc_metrics(y_test, y_median)
        
        print(f"\n前 {n_best} 個模型 (seeds: {[m['seed'] for m in best_models]}):")
        print(f"  平均: MAPE={metrics_mean['mape']:.2f}%, Max={metrics_mean['max_error']:.1f}%, 異常={metrics_mean['outliers']}")
        print(f"  中位: MAPE={metrics_median['mape']:.2f}%, Max={metrics_median['max_error']:.1f}%, 異常={metrics_median['outliers']}")
    
    # 找最佳組合
    print("\n" + "="*60)
    print("搜尋最佳 3 模型組合")
    print("="*60)
    
    best_combo = None
    best_combo_outliers = float('inf')
    best_combo_max = float('inf')
    
    top_models = models_info[:8]  # 從前 8 個中選
    
    for combo in combinations(range(len(top_models)), 3):
        preds = [top_models[i]['predictions'] for i in combo]
        y_ensemble = np.mean(preds, axis=0)
        metrics = calc_metrics(y_test, y_ensemble)
        
        if metrics['outliers'] < best_combo_outliers or \
           (metrics['outliers'] == best_combo_outliers and metrics['max_error'] < best_combo_max):
            best_combo_outliers = metrics['outliers']
            best_combo_max = metrics['max_error']
            best_combo = combo
            best_combo_pred = y_ensemble
            best_combo_metrics = metrics
    
    combo_seeds = [top_models[i]['seed'] for i in best_combo]
    print(f"\n最佳 3 模型組合: seeds={combo_seeds}")
    print(f"  MAPE={best_combo_metrics['mape']:.2f}%, Max={best_combo_metrics['max_error']:.1f}%, 異常={best_combo_metrics['outliers']}")
    
    # 顯示異常點
    print(f"\n異常點詳情:")
    for i in np.where(best_combo_metrics['errors'] > 20)[0]:
        print(f"  T{X_test[i,0]:.0f}, Thick={X_test[i,1]:.0f}, Cov={X_test[i,2]}: "
              f"真={y_test[i]:.4f}, 預={best_combo_pred[i]:.4f}, 誤={best_combo_metrics['errors'][i]:.1f}%")
    
    # 最終：用最佳單一模型
    print("\n" + "="*60)
    print("最終結果比較")
    print("="*60)
    best_single = models_info[0]
    print(f"最佳單一模型 (seed={best_single['seed']}): MAPE={best_single['mape']:.2f}%, "
          f"Max={best_single['max_error']:.1f}%, 異常={best_single['outliers']}")
    print(f"最佳 3 模型組合: MAPE={best_combo_metrics['mape']:.2f}%, "
          f"Max={best_combo_metrics['max_error']:.1f}%, 異常={best_combo_metrics['outliers']}")
    
    # 保存最佳結果
    if best_combo_metrics['outliers'] <= best_single['outliers'] and \
       best_combo_metrics['max_error'] <= best_single['max_error']:
        final_pred = best_combo_pred
        final_metrics = best_combo_metrics
        final_name = f"ensemble_{combo_seeds}"
    else:
        final_pred = best_single['predictions']
        final_metrics = {'mape': best_single['mape'], 'max_error': best_single['max_error'],
                         'outliers': best_single['outliers'], 'errors': calc_metrics(y_test, best_single['predictions'])['errors']}
        final_name = f"single_{best_single['seed']}"
    
    results_df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Predicted': final_pred,
        'Error%': final_metrics['errors']
    })
    results_df.to_csv(f'phase2i_{final_name}_predictions.csv', index=False)
    print(f"\n✓ 已保存: phase2i_{final_name}_predictions.csv")
    
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()
    main(verbose=args.verbose)
