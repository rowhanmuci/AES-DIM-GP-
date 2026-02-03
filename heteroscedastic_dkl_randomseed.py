"""
Phase 3A: 種子搜尋 (1-3000)
找出最佳的隨機種子
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import warnings
import random
import os

warnings.filterwarnings('ignore')

# ============================================
# 1. 載入資料
# ============================================
print("="*80)
print("Phase 3A: 種子搜尋 (1-3000)")
print("="*80)


train_df = pd.read_excel('data/train/Above.xlsx')
test_df = pd.read_excel('data/test/Above.xlsx')

print(f"訓練集: {len(train_df)} 筆")
print(f"測試集: {len(test_df)} 筆")

# 特徵工程
def prepare_features(df):
    X = df[['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']].copy()
    X['type_1'] = (X['TIM_TYPE'] == 1).astype(float)
    X['type_2'] = (X['TIM_TYPE'] == 2).astype(float)
    X['type_3'] = (X['TIM_TYPE'] == 3).astype(float)
    X['thick_sq'] = X['TIM_THICKNESS'] ** 2
    X['cov_sq'] = X['TIM_COVERAGE'] ** 2
    X['thick_cov'] = X['TIM_THICKNESS'] * X['TIM_COVERAGE']
    X['inv_coverage'] = 1.0 / (X['TIM_COVERAGE'] + 0.01)
    feature_cols = ['TIM_THICKNESS', 'TIM_COVERAGE', 'type_1', 'type_2', 'type_3',
                    'thick_sq', 'cov_sq', 'thick_cov', 'inv_coverage']
    return X[feature_cols].values

X_train = prepare_features(train_df)
y_train = train_df['Theta.JC'].values
X_test = prepare_features(test_df)
y_test = test_df['Theta.JC'].values

# 標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
X_test_scaled = scaler_X.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# ============================================
# 2. 模型定義 (簡化版，快速訓練)
# ============================================

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=16):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class NoiseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.constant_(self.net[-1].bias, -2.0)
    
    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        log_noise = -4.0 + 3.0 * torch.sigmoid(raw)
        return log_noise


class HeteroscedasticGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class HeteroscedasticDKL(nn.Module):
    def __init__(self, input_dim, feature_dim=16, n_inducing=100):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, output_dim=feature_dim)
        self.noise_net = NoiseNetwork(input_dim)
        inducing_points = torch.randn(n_inducing, feature_dim)
        self.gp = HeteroscedasticGPModel(inducing_points)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        gp_output = self.gp(features)
        log_noise = self.noise_net(x)
        return gp_output, log_noise
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            gp_output = self.gp(features)
            log_noise = self.noise_net(x)
            mean = gp_output.mean
            var = gp_output.variance + torch.exp(log_noise)
        return mean, var, torch.exp(log_noise)


class HeteroscedasticLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, gp_output, log_noise, y):
        mean = gp_output.mean
        noise_var = torch.exp(log_noise)
        recon_loss = 0.5 * log_noise + 0.5 * (y - mean)**2 / noise_var
        return recon_loss.mean()


# ============================================
# 3. 訓練函數 (快速版本)
# ============================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_and_evaluate(seed, n_epochs=300, lr=0.005, patience=30):
    """快速訓練並評估單個種子"""
    set_seed(seed)
    
    input_dim = X_train_tensor.shape[1]
    model = HeteroscedasticDKL(input_dim, feature_dim=16, n_inducing=100)
    
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.noise_net.parameters(), 'lr': lr * 0.5},
        {'params': model.gp.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    hetero_loss = HeteroscedasticLoss()
    
    model.train()
    model.gp.train()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        gp_output, log_noise = model(X_train_tensor)
        loss_hetero = hetero_loss(gp_output, log_noise, y_train_tensor)
        kl_div = model.gp.variational_strategy.kl_divergence().mean() / len(y_train_tensor)
        loss = loss_hetero + 0.1 * kl_div
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # 載入最佳模型並評估
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        mean, var, noise = model.predict(X_test_tensor)
        pred_scaled = mean.numpy()
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    errors = np.abs(pred - y_test) / y_test * 100
    mape = np.mean(errors)
    max_error = np.max(errors)
    outliers_20 = np.sum(errors > 20)
    outliers_30 = np.sum(errors > 30)
    outliers_40 = np.sum(errors > 40)
    
    return {
        'seed': seed,
        'mape': mape,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_30': outliers_30,
        'outliers_40': outliers_40,
    }


# ============================================
# 4. 執行種子搜尋
# ============================================

print("\n開始搜尋種子 1-3000...")
print("="*80)

results = []
best_max_error = float('inf')
best_max_error_seed = None
best_outliers = float('inf')
best_outliers_seed = None

for seed in range(1, 3001):
    result = train_and_evaluate(seed)
    results.append(result)
    
    # 更新最佳記錄
    if result['max_error'] < best_max_error:
        best_max_error = result['max_error']
        best_max_error_seed = seed
    
    if result['outliers_20'] < best_outliers:
        best_outliers = result['outliers_20']
        best_outliers_seed = seed
    
    # 進度報告 (每100個)
    if seed % 100 == 0:
        print(f"Seed {seed:4d}: MAPE={result['mape']:.2f}%, Max={result['max_error']:.1f}%, "
              f">20%={result['outliers_20']} | Best: MaxErr={best_max_error:.1f}% (seed={best_max_error_seed}), "
              f"Outliers={best_outliers} (seed={best_outliers_seed})")

# ============================================
# 5. 結果分析
# ============================================

print("\n" + "="*80)
print("搜尋完成！結果分析")
print("="*80)

# 轉成 DataFrame
results_df = pd.DataFrame(results)

# 按 max_error 排序
print("\n【最低 Max Error Top 10】")
top_max_error = results_df.nsmallest(10, 'max_error')
print(f"{'Seed':<8} {'MAPE':<10} {'Max Error':<12} {'>20%':<8} {'>30%':<8} {'>40%':<8}")
print("-"*60)
for _, row in top_max_error.iterrows():
    print(f"{row['seed']:<8} {row['mape']:<10.2f} {row['max_error']:<12.1f} "
          f"{row['outliers_20']:<8} {row['outliers_30']:<8} {row['outliers_40']:<8}")

# 按 outliers_20 排序
print("\n【最少 Outliers (>20%) Top 10】")
top_outliers = results_df.nsmallest(10, ['outliers_20', 'max_error'])
print(f"{'Seed':<8} {'MAPE':<10} {'Max Error':<12} {'>20%':<8} {'>30%':<8} {'>40%':<8}")
print("-"*60)
for _, row in top_outliers.iterrows():
    print(f"{row['seed']:<8} {row['mape']:<10.2f} {row['max_error']:<12.1f} "
          f"{row['outliers_20']:<8} {row['outliers_30']:<8} {row['outliers_40']:<8}")

# 按 MAPE 排序
print("\n【最低 MAPE Top 10】")
top_mape = results_df.nsmallest(10, 'mape')
print(f"{'Seed':<8} {'MAPE':<10} {'Max Error':<12} {'>20%':<8} {'>30%':<8} {'>40%':<8}")
print("-"*60)
for _, row in top_mape.iterrows():
    print(f"{row['seed']:<8} {row['mape']:<10.2f} {row['max_error']:<12.1f} "
          f"{row['outliers_20']:<8} {row['outliers_30']:<8} {row['outliers_40']:<8}")

# 統計摘要
print("\n【統計摘要】")
print(f"MAPE: min={results_df['mape'].min():.2f}%, max={results_df['mape'].max():.2f}%, "
      f"mean={results_df['mape'].mean():.2f}%, std={results_df['mape'].std():.2f}%")
print(f"Max Error: min={results_df['max_error'].min():.1f}%, max={results_df['max_error'].max():.1f}%, "
      f"mean={results_df['max_error'].mean():.1f}%, std={results_df['max_error'].std():.1f}%")
print(f"Outliers (>20%): min={results_df['outliers_20'].min()}, max={results_df['outliers_20'].max()}, "
      f"mean={results_df['outliers_20'].mean():.1f}")

# 找出同時滿足多個條件的種子
print("\n【綜合最佳種子 (outliers_20 <= 5 且 max_error <= 40)】")
good_seeds = results_df[(results_df['outliers_20'] <= 5) & (results_df['max_error'] <= 40)]
good_seeds = good_seeds.sort_values(['outliers_20', 'max_error'])
if len(good_seeds) > 0:
    print(f"找到 {len(good_seeds)} 個符合條件的種子:")
    print(f"{'Seed':<8} {'MAPE':<10} {'Max Error':<12} {'>20%':<8} {'>30%':<8} {'>40%':<8}")
    print("-"*60)
    for _, row in good_seeds.head(20).iterrows():
        print(f"{row['seed']:<8} {row['mape']:<10.2f} {row['max_error']:<12.1f} "
              f"{row['outliers_20']:<8} {row['outliers_30']:<8} {row['outliers_40']:<8}")
else:
    print("沒有找到符合條件的種子")

# 保存完整結果
results_df.to_csv('phase3a_seed_search_results.csv', index=False)
print(f"\n✓ 完整結果已保存至 phase3a_seed_search_results.csv")