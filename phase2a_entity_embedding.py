"""
Phase 2A 修正版: 修復MAPE loss計算bug
問題: MAPE在標準化空間計算導致訓練異常
解決: 在原始空間計算MAPE
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
# Entity Embedding (與之前相同)
# ==========================================

class TIMTypeEmbedding(nn.Module):
    def __init__(self, n_types=3, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(n_types, embedding_dim)
        self.embedding_dim = embedding_dim
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
    
    def forward(self, type_indices):
        return self.embedding(type_indices)


class DnnFeatureExtractorWithEmbedding(nn.Module):
    def __init__(self, continuous_dim=2, type_embed_dim=4, 
                 hidden_dims=[64, 32, 16], output_dim=8, dropout=0.1):
        super().__init__()
        
        self.type_embedding = TIMTypeEmbedding(n_types=3, embedding_dim=type_embed_dim)
        
        input_dim = type_embed_dim + continuous_dim
        
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
        type_indices = x[:, 0].long()
        continuous_features = x[:, 1:]
        
        type_embed = self.type_embedding(type_indices)
        combined = torch.cat([type_embed, continuous_features], dim=1)
        
        return self.network(combined)


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


def mape_loss_original_space(y_pred_scaled, y_true_scaled, scaler_y, epsilon=1e-8):
    """
    在原始空間計算MAPE loss (關鍵修正!)
    """
    # 反標準化到原始空間
    y_pred_original = scaler_y.inverse_transform(
        y_pred_scaled.detach().cpu().numpy().reshape(-1, 1)
    ).flatten()
    y_true_original = scaler_y.inverse_transform(
        y_true_scaled.detach().cpu().numpy().reshape(-1, 1)
    ).flatten()
    
    # 在原始空間計算MAPE
    y_pred_original = torch.from_numpy(y_pred_original).to(device)
    y_true_original = torch.from_numpy(y_true_original).to(device)
    
    mape = torch.mean(torch.abs((y_true_original - y_pred_original) / 
                                 (torch.abs(y_true_original) + epsilon))) * 100
    
    return mape


def preprocess_data(X, y=None):
    """資料預處理"""
    X_processed = X.copy()
    X_processed[:, 0] = X[:, 0] - 1  # TIM_TYPE: 1,2,3 → 0,1,2
    
    scaler_continuous = StandardScaler()
    X_processed[:, 1:] = scaler_continuous.fit_transform(X[:, 1:])
    
    if y is not None:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_processed, y_scaled, scaler_continuous, scaler_y
    else:
        return X_processed


# ==========================================
# 修正版訓練函數
# ==========================================

def train_dkl_with_embedding_fixed(X_train, y_train, config=None):
    """訓練Entity Embedding版DKL (修正版)"""
    
    if config is None:
        config = {
            'type_embed_dim': 4,
            'hidden_dims': [64, 32, 16],
            'feature_dim': 8,
            'dropout': 0.1,
            'lr': 0.01,
            'epochs': 500,
            'patience': 50,
            'mape_weight': 0.1,
        }
    
    print("="*60)
    print("訓練Entity Embedding版DKL (修正版)")
    print("="*60 + "\n")
    
    print("配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 資料預處理
    X_train_processed, y_train_scaled, scaler_continuous, scaler_y = preprocess_data(X_train, y_train)
    
    train_x = torch.from_numpy(X_train_processed).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    
    # 建立模型
    feature_extractor = DnnFeatureExtractorWithEmbedding(
        continuous_dim=2,
        type_embed_dim=config['type_embed_dim'],
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
        
        # 關鍵修正: 在原始空間計算MAPE
        mape = mape_loss_original_space(output.mean, train_y, scaler_y)
        
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
    
    # 檢查embedding
    print("="*60)
    print("學到的TIM_TYPE Embedding:")
    print("="*60)
    with torch.no_grad():
        embeddings = model.feature_extractor.type_embedding.embedding.weight.cpu().numpy()
        for i, emb in enumerate(embeddings):
            print(f"  Type {i+1}: [{', '.join([f'{x:.3f}' for x in emb])}]")
    
    print("\n類型間的歐式距離:")
    for i in range(3):
        for j in range(i+1, 3):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            print(f"  Type {i+1} ↔ Type {j+1}: {dist:.3f}")
    print("="*60 + "\n")
    
    return {
        'model': model,
        'likelihood': likelihood,
        'scaler_continuous': scaler_continuous,
        'scaler_y': scaler_y,
        'config': config,
        'embeddings': embeddings
    }


def evaluate_model(model_dict, X_test, y_test, dataset_name="Test"):
    """評估模型"""
    
    model = model_dict['model']
    likelihood = model_dict['likelihood']
    scaler_continuous = model_dict['scaler_continuous']
    scaler_y = model_dict['scaler_y']
    
    # 預處理
    X_test_processed = X_test.copy()
    X_test_processed[:, 0] = X_test[:, 0] - 1
    X_test_processed[:, 1:] = scaler_continuous.transform(X_test[:, 1:])
    
    # 預測
    model.eval()
    likelihood.eval()
    
    test_x = torch.from_numpy(X_test_processed).to(device)
    
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

def main_embedding_fixed():
    """修正版Entity Embedding實驗"""
    
    print("\n" + "="*60)
    print("Phase 2A 修正版: Entity Embedding (Fixed MAPE)")
    print("="*60 + "\n")
    
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    config = {
        'type_embed_dim': 4,
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
    # Above
    # ==========================================
    print("\n🔵 Above 50% Coverage\n")
    
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
    
    # 訓練修正版
    model_above = train_dkl_with_embedding_fixed(X_train_above, y_train_above, config)
    
    # 評估
    results_above = evaluate_model(model_above, X_test_above, y_test_above, "Above")
    
    # 保存
    test_above_pred = test_above.copy()
    test_above_pred['Prediction'] = results_above['predictions']
    test_above_pred['Std'] = results_above['std']
    test_above_pred['Error%'] = results_above['relative_errors']
    test_above_pred.to_csv('phase2a_fixed_above_predictions.csv', index=False)
    
    results_summary.append({
        'Dataset': 'Above',
        'MAPE': results_above['mape'],
        'Outliers_20': f"{results_above['outliers_20']}/{len(y_test_above)}",
        'Max_Error': results_above['max_error']
    })
    
    # ==========================================
    # Below
    # ==========================================
    print("\n🟢 Below 50% Coverage\n")
    
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
    
    # 訓練修正版
    model_below = train_dkl_with_embedding_fixed(X_train_below, y_train_below, config)
    
    # 評估
    results_below = evaluate_model(model_below, X_test_below, y_test_below, "Below")
    
    # 保存
    test_below_pred = test_below.copy()
    test_below_pred['Prediction'] = results_below['predictions']
    test_below_pred['Std'] = results_below['std']
    test_below_pred['Error%'] = results_below['relative_errors']
    test_below_pred.to_csv('phase2a_fixed_below_predictions.csv', index=False)
    
    results_summary.append({
        'Dataset': 'Below',
        'MAPE': results_below['mape'],
        'Outliers_20': f"{results_below['outliers_20']}/{len(y_test_below)}",
        'Max_Error': results_below['max_error']
    })
    
    # ==========================================
    # 比較
    # ==========================================
    print("\n" + "="*60)
    print("📊 結果比較")
    print("="*60 + "\n")
    
    print("Baseline (組員):")
    print("  Above: MAPE=8.89%, 異常點=16/138 (11.59%)")
    print("  Below: MAPE=3.76%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 1 (MAPE Loss):")
    print("  Above: MAPE=8.63%, 異常點=10/138 (7.25%)")
    print("  Below: MAPE=3.88%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 2A (原版Embedding - 有bug):")
    print("  Above: MAPE=8.92%, 異常點=10/138 (7.25%)")
    print("  Below: MAPE=3.82%, 異常點=0/48 (0.00%)")
    
    print("\nPhase 2A修正版 (Fixed MAPE):")
    print(f"  Above: MAPE={results_above['mape']:.2f}%, "
          f"異常點={results_above['outliers_20']}/{len(y_test_above)} "
          f"({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"  Below: MAPE={results_below['mape']:.2f}%, "
          f"異常點={results_below['outliers_20']}/{len(y_test_below)} "
          f"({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    
    # 計算改進
    if results_above['outliers_20'] < 10:
        improvement = 10 - results_above['outliers_20']
        print(f"\n✅ 相比Phase 1改進: 異常點 -{improvement}")
    elif results_above['outliers_20'] == 10:
        print(f"\n😐 與Phase 1持平")
    else:
        worsening = results_above['outliers_20'] - 10
        print(f"\n⚠️ 相比Phase 1退步: 異常點 +{worsening}")
    
    print(f"\n{'='*60}\n")
    
    # 保存
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv('phase2a_fixed_summary.csv', index=False)
    print("✓ 結果已保存\n")
    
    return {
        'above': (model_above, results_above, test_above_pred),
        'below': (model_below, results_below, test_below_pred)
    }


if __name__ == "__main__":
    results = main_embedding_fixed()