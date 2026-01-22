"""
Phase 4A - Spectral Mixture Kernel 實驗 (激進記憶體優化 v2)
簡化誘導點初始化，避免 BatchNorm 問題
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import random
import os
import argparse
import gc

warnings.filterwarnings('ignore')

# 使用 float32 來節省記憶體
torch.set_default_dtype(torch.float32)
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
    gc.collect()


# ==========================================
# 模型定義
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """輕量化特徵提取器 (專為 SM Kernel 設計)"""
    
    def __init__(self, input_dim, hidden_dims=[32, 16], output_dim=4, dropout=0.05):
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


class VariationalSMGP(gpytorch.models.ApproximateGP):
    """使用 SM Kernel 的 Variational GP"""
    
    def __init__(self, inducing_points, feature_extractor, kernel_type='sm', num_mixtures=2):
        """
        Args:
            inducing_points: 誘導點
            feature_extractor: DNN 特徵提取器
            kernel_type: 'sm', 'sm+rbf', 'sm+matern'
            num_mixtures: SM 混合數 (1-3 推薦)
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel_type = kernel_type
        
        # 根據 kernel_type 建立不同的 kernel
        if kernel_type == 'sm':
            # 純 SM
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(
                    num_mixtures=num_mixtures,
                    ard_num_dims=feature_extractor.output_dim
                )
            )
        
        elif kernel_type == 'sm+rbf':
            # SM + RBF
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(
                    num_mixtures=num_mixtures,
                    ard_num_dims=feature_extractor.output_dim
                ) + 
                gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
            )
        
        elif kernel_type == 'sm+matern':
            # SM + Matérn
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(
                    num_mixtures=num_mixtures,
                    ard_num_dims=feature_extractor.output_dim
                ) + 
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_extractor.output_dim)
            )
        
        elif kernel_type == 'rbf':
            # 純 RBF (對照組)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim)
            )
        
        elif kernel_type == 'matern':
            # 純 Matérn (對照組)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_extractor.output_dim)
            )
        
        elif kernel_type == 'rbf+matern':
            # RBF + Matérn
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=feature_extractor.output_dim) +
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=feature_extractor.output_dim)
            )
        
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
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


def compute_sample_weights(X, weight_factor=3.0):
    """計算樣本權重"""
    weights = np.ones(len(X))
    
    difficult_mask = (
        (X[:, 0] == 3) &      # TIM_TYPE = 3
        (X[:, 2] == 0.8) &    # TIM_COVERAGE = 0.8
        (X[:, 1] >= 220)      # TIM_THICKNESS >= 220
    )
    
    weights[difficult_mask] *= weight_factor
    
    return weights


# ==========================================
# 訓練與評估
# ==========================================

def train_model(X_train, y_train, config, verbose=True):
    """訓練 Variational DKL 模型"""
    
    # 計算樣本權重
    sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
    
    if verbose:
        difficult_count = np.sum(sample_weights_np > 1.0)
        print(f"\n計算樣本權重:")
        print(f"  困難樣本數: {difficult_count} ({difficult_count/len(X_train)*100:.2f}%)")
        print(f"  權重倍數: {config['sample_weight_factor']}x")
        print(f"  Kernel類型: {config['kernel_type']}")
        if 'sm' in config['kernel_type']:
            print(f"  SM Mixtures: {config['num_mixtures']}")
        print(f"  誘導點數量: {config['num_inducing']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  資料型別: float32 (記憶體優化)")
    
    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    train_x = torch.from_numpy(X_train_scaled).float().to(device)
    train_y = torch.from_numpy(y_train_scaled).float().to(device)
    sample_weights = torch.from_numpy(sample_weights_np).float().to(device)
    
    # 建立特徵提取器
    feature_extractor = DnnFeatureExtractor(
        input_dim=train_x.shape[1],
        hidden_dims=config['hidden_dims'],
        output_dim=config['feature_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # 選擇誘導點 (使用 k-means 在原始空間)
    num_inducing = min(config['num_inducing'], len(train_x))
    
    if verbose:
        print(f"\n初始化誘導點 (使用 k-means)...")
    
    # 在原始特徵空間做 k-means
    kmeans = KMeans(n_clusters=num_inducing, random_state=config.get('seed', 2024), n_init=10)
    kmeans.fit(X_train_scaled)
    
    # !!!關鍵修改：誘導點應該在輸入空間（標準化後的 X），不是特徵空間!!!
    inducing_points = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
    
    if verbose:
        print(f"✓ 誘導點初始化完成 (shape: {inducing_points.shape})")
    
    # 建立模型
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = VariationalSMGP(
        inducing_points, 
        feature_extractor,
        kernel_type=config['kernel_type'],
        num_mixtures=config.get('num_mixtures', 2)
    ).to(device)
    
    # 優化器
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': config['lr'], 'weight_decay': 1e-4},
        {'params': model.variational_parameters(), 'lr': config['lr'] * 0.5},
        {'params': model.covar_module.parameters(), 'lr': config['lr'] * 0.1},
        {'params': model.mean_module.parameters()},
        {'params': likelihood.parameters()},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # Variational ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))
    
    # 訓練
    if verbose:
        print(f"\n開始訓練...")
    
    model.train()
    likelihood.train()
    
    best_loss = float('inf')
    patience_counter = 0
    batch_size = config['batch_size']
    n_batches = (len(train_x) + batch_size - 1) // batch_size
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        epoch_elbo = 0.0
        epoch_mape = 0.0
        
        # Mini-batch 訓練
        indices_perm = torch.randperm(len(train_x))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(train_x))
            batch_indices = indices_perm[start_idx:end_idx]
            
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            batch_weights = sample_weights[batch_indices]
            
            optimizer.zero_grad()
            
            # 前向傳播
            output = model(batch_x)
            
            # ELBO loss
            elbo_loss = -mll(output, batch_y)
            
            # MAPE loss
            mape = weighted_mape_loss(output.mean, batch_y, batch_weights)
            
            # 總損失
            total_loss = elbo_loss + config['mape_weight'] * mape
            
            # 反向傳播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_elbo += elbo_loss.item()
            epoch_mape += mape.item()
            
            # 清理記憶體
            del output, elbo_loss, mape, total_loss
            
            # 每個 batch 都清理一次
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        scheduler.step()
        
        # 平均損失
        avg_loss = epoch_loss / n_batches
        avg_elbo = epoch_elbo / n_batches
        avg_mape = epoch_mape / n_batches
        
        # 顯示訓練進度
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: ELBO={avg_elbo:.4f}, "
                  f"MAPE={avg_mape:.2f}%, Total={avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
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
        print(f"訓練完成 (Final Loss: {best_loss:.4f})")
    
    clear_gpu_cache()
    
    return model, likelihood, scaler_x, scaler_y


def evaluate_model(model, likelihood, X_test, y_test, scaler_x, scaler_y, verbose=True):
    """評估模型"""
    model.eval()
    likelihood.eval()
    
    X_test_scaled = scaler_x.transform(X_test)
    test_x = torch.from_numpy(X_test_scaled).float().to(device)
    
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
    
    # Type 3分析
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
        print(f"  >15%: {outliers_15}/{len(y_test)} ({outliers_15/len(y_test)*100:.2f}%)")
        print(f"  >10%: {outliers_10}/{len(y_test)} ({outliers_10/len(y_test)*100:.2f}%)")
        if np.sum(type3_mask) > 0:
            print(f"\nType 3異常點: {type3_outliers}/{np.sum(type3_mask)}")
        print(f"{'='*60}\n")
    
    results = {
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
    
    return results


def save_predictions(X_test, y_test, results, filename):
    """保存預測結果到CSV"""
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

def main(seed=2024, kernel_type='sm', num_mixtures=2, verbose=True):
    """主訓練流程"""
    
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}\n")
    
    print("="*60)
    print(f"Phase 4A: SM Kernel 實驗 (激進記憶體優化 v2)")
    print(f"Kernel: {kernel_type}")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置 (激進記憶體優化)
    config = {
        'hidden_dims': [32, 16],       # 小網路
        'feature_dim': 4,               # 小特徵維度
        'dropout': 0.05,
        'lr': 0.01,
        'epochs': 500,
        'patience': 50,
        'mape_weight': 0.1,
        'sample_weight_factor': 3.0,
        'kernel_type': kernel_type,
        'num_mixtures': num_mixtures,
        'batch_size': 128,              # 小 batch
        'num_inducing': 256,            # 少量誘導點
        'seed': seed,
    }
    
    if verbose:
        print(f"\n配置:")
        for key, value in config.items():
            if key != 'seed':
                print(f"  {key}: {value}")
    
    # ==========================================
    # Above Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Above 50% Coverage'}\n")
    
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆")
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    # 訓練
    model_above, likelihood_above, scaler_x_above, scaler_y_above = train_model(
        X_train_above, y_train_above, config, verbose=verbose
    )
    
    # 評估
    results_above = evaluate_model(
        model_above, likelihood_above, 
        X_test_above, y_test_above, 
        scaler_x_above, scaler_y_above,
        verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_above, y_test_above, results_above, 
                     f'phase4a_{kernel_type}_m{num_mixtures}_above_seed{seed}_predictions.csv')
    
    clear_gpu_cache()
    
    # ==========================================
    # Below Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Below 50% Coverage'}\n")
    
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
    
    # 訓練
    model_below, likelihood_below, scaler_x_below, scaler_y_below = train_model(
        X_train_below, y_train_below, config, verbose=verbose
    )
    
    # 評估
    results_below = evaluate_model(
        model_below, likelihood_below,
        X_test_below, y_test_below,
        scaler_x_below, scaler_y_below,
        verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_below, y_test_below, results_below,
                     f'phase4a_{kernel_type}_m{num_mixtures}_below_seed{seed}_predictions.csv')
    
    # ==========================================
    # 總結
    # ==========================================
    
    print("\n" + "="*60)
    print(f"Phase 4A 結果總結 - Kernel: {kernel_type}")
    if 'sm' in kernel_type:
        print(f"Mixtures: {num_mixtures}")
    print("="*60)
    print(f"隨機種子: {seed}")
    
    print(f"\nAbove資料集:")
    print(f"  異常點 (>20%): {results_above['outliers_20']}/{len(y_test_above)} ({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"  MAPE: {results_above['mape']:.2f}%")
    print(f"  Type 3異常點: {results_above['type3_outliers']}")
    
    print(f"\nBelow資料集:")
    print(f"  異常點 (>20%): {results_below['outliers_20']}/{len(y_test_below)} ({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    print(f"  MAPE: {results_below['mape']:.2f}%")
    
    print("\n" + "="*60)
    print("✓ 訓練完成！")
    print("="*60 + "\n")
    
    return {
        'above': results_above,
        'below': results_below,
        'kernel_type': kernel_type,
        'seed': seed
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 4A - SM Kernel 實驗')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('--kernel', type=str, default='sm', 
                        choices=['sm', 'sm+rbf', 'sm+matern', 'rbf', 'matern', 'rbf+matern'],
                        help='Kernel類型')
    parser.add_argument('--mixtures', type=int, default=2, 
                        help='SM mixtures 數量 (1-3 推薦)')
    parser.add_argument('-v', '--verbose', action='store_true', help='顯示詳細訓練過程')
    
    args = parser.parse_args()
    
    results = main(
        seed=args.seed,
        kernel_type=args.kernel,
        num_mixtures=args.mixtures,
        verbose=args.verbose
    )
    
    print("\n💡 使用範例:")
    print("  python phase4a_sm_kernel.py --kernel sm --mixtures 2 -v         # 純 SM (2 mixtures)")
    print("  python phase4a_sm_kernel.py --kernel sm --mixtures 3 -v         # 純 SM (3 mixtures)")
    print("  python phase4a_sm_kernel.py --kernel sm+rbf --mixtures 2 -v     # SM + RBF")
    print("  python phase4a_sm_kernel.py --kernel sm+matern --mixtures 2 -v  # SM + Matérn")
    print("  python phase4a_sm_kernel.py --kernel rbf+matern -v              # RBF + Matérn (對照)\n")