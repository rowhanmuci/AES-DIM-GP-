"""
Phase 2E: 查表法 + 殘差學習
針對 Type 3 的特殊處理：
1. Type 3 只使用 Coverage 做主預測（Thickness 無關）
2. 用查表法獲取 Coverage 對應的平均值
3. 用輕量 GP 模型學習殘差（捕捉個體差異）

使用方法:
    python phase2e_lookup_residual.py --seed 2024
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
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
# Type 3 查表模型
# ==========================================

class Type3LookupModel:
    """
    Type 3 專用查表模型
    
    策略：
    1. 建立 Coverage → Theta.JC 的查找表（忽略 Thickness）
    2. 使用分位數回歸捕捉不確定性
    """
    
    def __init__(self, X_train, y_train):
        """
        Args:
            X_train: [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
            y_train: Theta.JC
        """
        # 只用 Type 3 資料
        type3_mask = X_train[:, 0] == 3
        coverage = X_train[type3_mask, 2]
        theta = y_train[type3_mask]
        
        # 按 Coverage 分組統計
        coverage_unique = np.unique(coverage)
        
        self.lookup_table = {}
        for cov in coverage_unique:
            cov_mask = coverage == cov
            cov_theta = theta[cov_mask]
            
            self.lookup_table[cov] = {
                'mean': np.mean(cov_theta),
                'median': np.median(cov_theta),
                'q25': np.percentile(cov_theta, 25),
                'q75': np.percentile(cov_theta, 75),
                'std': np.std(cov_theta),
                'min': np.min(cov_theta),
                'max': np.max(cov_theta),
                'count': len(cov_theta),
            }
        
        # 建立插值函數（用於未見過的 Coverage 值）
        coverages = sorted(self.lookup_table.keys())
        means = [self.lookup_table[c]['mean'] for c in coverages]
        medians = [self.lookup_table[c]['median'] for c in coverages]
        
        self.interp_mean = interp1d(coverages, means, kind='cubic', 
                                    fill_value='extrapolate')
        self.interp_median = interp1d(coverages, medians, kind='cubic',
                                      fill_value='extrapolate')
        
        print(f"✓ Type 3 查表模型已建立 ({len(coverages)} 個 Coverage 值)")
    
    def predict(self, X_test, use_median=False):
        """
        預測
        
        Args:
            X_test: [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
            use_median: 是否使用中位數（更穩健）
        
        Returns:
            predictions, std
        """
        type3_mask = X_test[:, 0] == 3
        coverage = X_test[type3_mask, 2]
        
        predictions = np.zeros(len(X_test))
        stds = np.zeros(len(X_test))
        
        for i, cov in enumerate(coverage):
            if cov in self.lookup_table:
                # 直接查表
                predictions[type3_mask][i] = (
                    self.lookup_table[cov]['median'] if use_median 
                    else self.lookup_table[cov]['mean']
                )
                stds[type3_mask][i] = self.lookup_table[cov]['std']
            else:
                # 插值
                predictions[type3_mask][i] = (
                    self.interp_median(cov) if use_median
                    else self.interp_mean(cov)
                )
                # 估計標準差（用最近的 Coverage）
                nearest_cov = min(self.lookup_table.keys(), 
                                 key=lambda x: abs(x - cov))
                stds[type3_mask][i] = self.lookup_table[nearest_cov]['std']
        
        return predictions, stds


# ==========================================
# 簡化的 DKL 模型（用於 Type 1, 2 和殘差學習）
# ==========================================

class SimpleDnnFeatureExtractor(nn.Module):
    """輕量 DNN 特徵提取器"""
    
    def __init__(self, input_dim, hidden_dims=[32, 16], output_dim=4, dropout=0.1):
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


class SimpleGPRegressionModel(gpytorch.models.ExactGP):
    """簡化 GP 模型"""
    
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 簡單 RBF kernel（高敏感度）
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=feature_extractor.output_dim,
                lengthscale_constraint=gpytorch.constraints.Interval(0.1, 2.0)  # 限制 lengthscale
            )
        )
    
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# Type 3 殘差模型
# ==========================================

class Type3ResidualModel:
    """
    Type 3 殘差學習模型
    
    步驟：
    1. 用查表法獲取基礎預測
    2. 計算殘差
    3. 用 GP 學習殘差模式（Coverage, Thickness 作為輔助特徵）
    """
    
    def __init__(self, X_train, y_train, lookup_model, config):
        """
        Args:
            X_train: Type 3 訓練資料
            y_train: Type 3 訓練標籤
            lookup_model: 已訓練的查表模型
            config: 配置
        """
        # 獲取查表基礎預測
        base_pred, _ = lookup_model.predict(X_train)
        
        # 計算殘差
        type3_mask = X_train[:, 0] == 3
        residuals = y_train[type3_mask] - base_pred[type3_mask]
        
        print(f"殘差統計: mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}")
        
        # 特徵：只用 Coverage 和 Thickness（編碼個體差異）
        X_residual = X_train[type3_mask][:, 1:]  # [Thickness, Coverage]
        
        # 標準化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_x.fit_transform(X_residual)
        y_scaled = self.scaler_y.fit_transform(residuals.reshape(-1, 1)).flatten()
        
        train_x = torch.from_numpy(X_scaled).to(device)
        train_y = torch.from_numpy(y_scaled).to(device)
        
        # 建立輕量模型
        feature_extractor = SimpleDnnFeatureExtractor(
            input_dim=2,
            hidden_dims=[16, 8],
            output_dim=4,
            dropout=0.05
        ).to(device)
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = SimpleGPRegressionModel(
            train_x, train_y, self.likelihood, feature_extractor
        ).to(device)
        
        # 訓練
        self._train(train_x, train_y, config)
    
    def _train(self, train_x, train_y, config):
        """訓練殘差模型"""
        optimizer = optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': 0.01},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=0.01)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        self.model.train()
        self.likelihood.train()
        
        for epoch in range(200):  # 少量 epoch
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        print(f"✓ 殘差模型訓練完成")
    
    def predict(self, X_test):
        """預測殘差"""
        self.model.eval()
        self.likelihood.eval()
        
        type3_mask = X_test[:, 0] == 3
        X_residual = X_test[type3_mask][:, 1:]  # [Thickness, Coverage]
        
        X_scaled = self.scaler_x.transform(X_residual)
        test_x = torch.from_numpy(X_scaled).to(device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(test_x))
            residual_scaled = pred_dist.mean.cpu().numpy()
        
        # 反標準化
        residual = self.scaler_y.inverse_transform(residual_scaled.reshape(-1, 1)).flatten()
        
        predictions = np.zeros(len(X_test))
        predictions[type3_mask] = residual
        
        return predictions


# ==========================================
# 混合模型
# ==========================================

class HybridModel:
    """
    混合模型：
    - Type 1, 2: 標準 DKL
    - Type 3: 查表法 + 殘差學習
    """
    
    def __init__(self, X_train, y_train, config):
        """初始化混合模型"""
        # Type 3 查表模型
        self.lookup_model = Type3LookupModel(X_train, y_train)
        
        # Type 3 殘差模型
        type3_mask = X_train[:, 0] == 3
        X_type3 = X_train[type3_mask]
        y_type3 = y_train[type3_mask]
        
        self.residual_model = Type3ResidualModel(
            X_train, y_train, self.lookup_model, config
        )
        
        # Type 1, 2 標準模型
        others_mask = ~type3_mask
        X_others = X_train[others_mask]
        y_others = y_train[others_mask]
        
        print(f"\n訓練 Type 1, 2 模型 ({len(X_others)} 筆)...")
        self.standard_model = self._train_standard_model(X_others, y_others, config)
    
    def _train_standard_model(self, X_train, y_train, config):
        """訓練 Type 1, 2 的標準模型"""
        # 標準化
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_x.fit_transform(X_train)
        y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        train_x = torch.from_numpy(X_scaled).to(device)
        train_y = torch.from_numpy(y_scaled).to(device)
        
        # 建立模型
        feature_extractor = SimpleDnnFeatureExtractor(
            input_dim=3,
            hidden_dims=[64, 32],
            output_dim=8,
            dropout=0.1
        ).to(device)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = SimpleGPRegressionModel(train_x, train_y, likelihood, feature_extractor).to(device)
        
        # 訓練
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        
        model.train()
        likelihood.train()
        
        for epoch in range(300):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}")
        
        print(f"✓ Type 1, 2 模型訓練完成")
        
        return {
            'model': model,
            'likelihood': likelihood,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y
        }
    
    def predict(self, X_test):
        """混合預測"""
        predictions = np.zeros(len(X_test))
        stds = np.zeros(len(X_test))
        
        # Type 3 預測
        type3_mask = X_test[:, 0] == 3
        if np.sum(type3_mask) > 0:
            # 查表基礎預測
            base_pred, base_std = self.lookup_model.predict(X_test, use_median=True)
            
            # 殘差預測
            residual_pred = self.residual_model.predict(X_test)
            
            # 組合
            predictions[type3_mask] = base_pred[type3_mask] + residual_pred[type3_mask]
            stds[type3_mask] = base_std[type3_mask]
        
        # Type 1, 2 預測
        others_mask = ~type3_mask
        if np.sum(others_mask) > 0:
            model_dict = self.standard_model
            model = model_dict['model']
            likelihood = model_dict['likelihood']
            scaler_x = model_dict['scaler_x']
            scaler_y = model_dict['scaler_y']
            
            model.eval()
            likelihood.eval()
            
            X_scaled = scaler_x.transform(X_test[others_mask])
            test_x = torch.from_numpy(X_scaled).to(device)
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_dist = likelihood(model(test_x))
                y_pred_scaled = pred_dist.mean.cpu().numpy()
                y_std_scaled = pred_dist.stddev.cpu().numpy()
            
            predictions[others_mask] = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            stds[others_mask] = y_std_scaled * scaler_y.scale_[0]
        
        return predictions, stds


# ==========================================
# 評估函數
# ==========================================

def evaluate_model(model, X_test, y_test, verbose=True):
    """評估模型"""
    y_pred, y_std = model.predict(X_test)
    
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
        print(f"評估結果 (查表法 + 殘差學習)")
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
    print("Phase 2E: 查表法 + 殘差學習")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置
    config = {
        'lr': 0.01,
        'epochs': 300,
        'seed': seed,
    }
    
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
    
    # 訓練混合模型
    print(f"\n{'='*60}")
    print("訓練混合模型")
    print(f"{'='*60}\n")
    
    model = HybridModel(X_train, y_train, config)
    
    # 評估
    results = evaluate_model(model, X_test, y_test, verbose=verbose)
    
    # 保存預測結果
    save_predictions(X_test, y_test, results,
                     f'phase2e_lookup_residual_seed{seed}_predictions.csv')
    
    # 總結
    print("\n" + "="*60)
    print("最終結果總結 (Phase 2E)")
    print("="*60)
    print(f"策略:")
    print(f"  ✓ Type 3: 查表法 (只用 Coverage)")
    print(f"  ✓ Type 3: GP 殘差學習 (捕捉個體差異)")
    print(f"  ✓ Type 1, 2: 標準 DKL")
    print(f"\n結果:")
    print(f"  總體 MAPE: {results['mape']:.2f}%")
    print(f"  Type 3 MAPE: {results['type3_mape']:.2f}%")
    print(f"  Coverage 0.8 MAPE: {results['cov08_mape']:.2f}%")
    print(f"  異常點: {results['outliers_20']}/{len(y_test)}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2E 查表法 + 殘差學習')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細模式')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
    
    print("\n💡 說明:")
    print("  此版本針對 Type 3 使用查表法 + 殘差學習")
    print("  預期: Coverage 0.8 MAPE < 20%")
    print("        Type 3 異常點 < 4/18\n")
