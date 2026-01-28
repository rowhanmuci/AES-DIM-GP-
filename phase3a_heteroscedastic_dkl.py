"""
Phase 3A: Heteroscedastic DKL 實驗
- 不去重，使用完整訓練資料
- 噪聲由網路預測，而非固定值
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 載入資料（不去重）
# ============================================
print("="*80)
print("Phase 3A: Heteroscedastic DKL")
print("="*80)

train_df = pd.read_excel('data/train/Above.xlsx')
test_df = pd.read_excel('data/test/Above.xlsx')

print(f"訓練集: {len(train_df)} 筆 (不去重)")
print(f"測試集: {len(test_df)} 筆")

# 特徵工程（與之前相同）
def prepare_features(df):
    X = df[['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']].copy()
    
    # Type encoding
    X['type_1'] = (X['TIM_TYPE'] == 1).astype(float)
    X['type_2'] = (X['TIM_TYPE'] == 2).astype(float)
    X['type_3'] = (X['TIM_TYPE'] == 3).astype(float)
    
    # Polynomial features
    X['thick_sq'] = X['TIM_THICKNESS'] ** 2
    X['cov_sq'] = X['TIM_COVERAGE'] ** 2
    X['thick_cov'] = X['TIM_THICKNESS'] * X['TIM_COVERAGE']
    
    # Physical feature
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

# 轉換為 tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

print(f"特徵維度: {X_train_tensor.shape[1]}")

# ============================================
# 2. Heteroscedastic DKL 模型定義
# ============================================

class FeatureExtractor(nn.Module):
    """特徵提取網路"""
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
    """預測 heteroscedastic noise 的網路"""
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
        
        # 初始化讓初始 noise 適中
        nn.init.constant_(self.net[-1].bias, -2.0)  # exp(-2) ≈ 0.135
    
    def forward(self, x):
        # 輸出 log(noise_variance)，用 softplus 確保正值
        raw = self.net(x).squeeze(-1)
        # noise variance 範圍約在 [0.01, 1.0]
        log_noise = -4.0 + 3.0 * torch.sigmoid(raw)  # 映射到 [-4, -1]
        return log_noise


class HeteroscedasticGPModel(ApproximateGP):
    """Heteroscedastic GP"""
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
    """完整的 Heteroscedastic DKL 模型"""
    def __init__(self, input_dim, feature_dim=16, n_inducing=100):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, output_dim=feature_dim)
        self.noise_net = NoiseNetwork(input_dim)
        
        # 初始化 inducing points
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
            # 總不確定性 = GP variance + heteroscedastic noise
            var = gp_output.variance + torch.exp(log_noise)
            
        return mean, var, torch.exp(log_noise)


# ============================================
# 3. 自定義 Loss（考慮 heteroscedastic noise）
# ============================================

class HeteroscedasticLoss(nn.Module):
    """
    Negative log-likelihood with heteroscedastic noise
    -log p(y|f, σ²(x)) = 0.5 * log(σ²) + 0.5 * (y-f)²/σ²
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, gp_output, log_noise, y):
        mean = gp_output.mean
        noise_var = torch.exp(log_noise)
        
        # Reconstruction loss
        recon_loss = 0.5 * log_noise + 0.5 * (y - mean)**2 / noise_var
        
        # GP prior (KL divergence from ELBO)
        # 這部分由 VariationalELBO 處理
        
        return recon_loss.mean()


# ============================================
# 4. 訓練
# ============================================

def train_heteroscedastic_dkl(X_train, y_train, X_test, y_test, 
                               scaler_y, seed=42, n_epochs=300, lr=0.005):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X_train.shape[1]
    model = HeteroscedasticDKL(input_dim, feature_dim=16, n_inducing=100)
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.noise_net.parameters(), 'lr': lr * 0.5},  # noise net 學慢一點
        {'params': model.gp.parameters(), 'lr': lr},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Loss
    mll = VariationalELBO(model.gp.likelihood if hasattr(model.gp, 'likelihood') 
                          else gpytorch.likelihoods.GaussianLikelihood(), 
                          model.gp, num_data=len(y_train))
    hetero_loss = HeteroscedasticLoss()
    
    # Training loop
    model.train()
    model.gp.train()
    
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward
        gp_output, log_noise = model(X_train)
        
        # Loss = heteroscedastic NLL + GP ELBO regularization
        loss_hetero = hetero_loss(gp_output, log_noise, y_train)
        
        # 加入 GP 的 KL divergence
        kl_div = model.gp.variational_strategy.kl_divergence().mean() / len(y_train)
        
        loss = loss_hetero + 0.1 * kl_div
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 50 == 0:
            # 計算當前性能
            model.eval()
            with torch.no_grad():
                mean, var, noise = model.predict(X_test)
                pred_scaled = mean.numpy()
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
                errors = np.abs(pred - y_test) / y_test * 100
                mape = np.mean(errors)
                max_err = np.max(errors)
                outliers = np.sum(errors > 20)
                
                avg_noise = noise.mean().item()
            
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, MAPE={mape:.2f}%, "
                  f"Max={max_err:.1f}%, Outliers={outliers}, AvgNoise={avg_noise:.4f}")
            model.train()
            model.gp.train()
    
    # Load best model
    model.load_state_dict(best_state)
    
    return model


# ============================================
# 5. 執行實驗
# ============================================

print("\n" + "="*80)
print("開始訓練 Heteroscedastic DKL...")
print("="*80)

model = train_heteroscedastic_dkl(
    X_train_tensor, y_train_tensor, 
    X_test_tensor, y_test,
    scaler_y, seed=42, n_epochs=300
)

# ============================================
# 6. 評估結果
# ============================================

print("\n" + "="*80)
print("評估結果")
print("="*80)

model.eval()
with torch.no_grad():
    mean, var, noise = model.predict(X_test_tensor)
    pred_scaled = mean.numpy()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    # Uncertainty
    std = np.sqrt(var.numpy())
    std_original = std * scaler_y.scale_[0]
    noise_original = np.sqrt(noise.numpy()) * scaler_y.scale_[0]

# 計算指標
errors = np.abs(pred - y_test) / y_test * 100

print(f"\n【整體指標】")
print(f"MAPE: {np.mean(errors):.2f}%")
print(f"Max Error: {np.max(errors):.1f}%")
print(f"Outliers (>20%): {np.sum(errors > 20)}")
print(f"Outliers (>30%): {np.sum(errors > 30)}")
print(f"Outliers (>40%): {np.sum(errors > 40)}")

# 分析 noise 分布
test_df_with_pred = test_df.copy()
test_df_with_pred['Predicted'] = pred
test_df_with_pred['Error_Pct'] = errors
test_df_with_pred['Noise'] = noise_original
test_df_with_pred['Std'] = std_original

print(f"\n【Learned Noise 分布】")
for typ in [1, 2, 3]:
    subset = test_df_with_pred[test_df_with_pred['TIM_TYPE'] == typ]
    print(f"Type {typ}: Avg Noise = {subset['Noise'].mean():.4f}")

print(f"\n【Type 3 按 Coverage 的 Noise】")
t3 = test_df_with_pred[test_df_with_pred['TIM_TYPE'] == 3]
for cov in [0.6, 0.8, 1.0]:
    subset = t3[t3['TIM_COVERAGE'] == cov]
    print(f"Cov={cov}: Avg Noise = {subset['Noise'].mean():.4f}, Avg Error = {subset['Error_Pct'].mean():.1f}%")

# 異常點詳情
print(f"\n【異常點詳情 (>20%)】")
outliers_df = test_df_with_pred[test_df_with_pred['Error_Pct'] > 20].sort_values('Error_Pct', ascending=False)
print(f"{'Type':<6} {'Thick':<8} {'Cov':<6} {'True':<10} {'Pred':<10} {'Error%':<10} {'Noise':<10}")
print("-"*70)
for _, row in outliers_df.iterrows():
    print(f"{row['TIM_TYPE']:<6} {row['TIM_THICKNESS']:<8.0f} {row['TIM_COVERAGE']:<6.1f} "
          f"{row['Theta.JC']:<10.4f} {row['Predicted']:<10.4f} {row['Error_Pct']:<10.1f} {row['Noise']:<10.4f}")

# 保存結果
output_df = test_df_with_pred[['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE', 
                                'Theta.JC', 'Predicted', 'Error_Pct', 'Noise', 'Std']]
output_df.to_csv('phase3a_heteroscedastic_results.csv', index=False)
print(f"\n✓ 結果已保存至 phase3a_heteroscedastic_results.csv")
