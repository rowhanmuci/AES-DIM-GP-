"""
Phase 3C: Mixture of Experts (MoE) DKL

設計原則：
1. 共享 Feature Extractor - 所有資料參與底層學習
2. 軟性 Gating - 不做 hard split
3. Gating 正則化 - 避免太極端的權重
4. Expert 差異化 - Expert 2 有更大的 noise tolerance
5. 監控各 Type 的 gating 分佈
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

# GPU 設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# ============================================
# 1. 載入資料
# ============================================
print("="*80)
print("Phase 3C: Mixture of Experts DKL")
print("="*80)

train_df = pd.read_excel('data/train/Above.xlsx')
test_df = pd.read_excel('data/test/Above.xlsx')


print(f"訓練集: {len(train_df)} 筆 (不去重)")
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

# 保留原始資訊用於分析
train_types = train_df['TIM_TYPE'].values
train_coverages = train_df['TIM_COVERAGE'].values
test_types = test_df['TIM_TYPE'].values
test_coverages = test_df['TIM_COVERAGE'].values

# 標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
X_test_scaled = scaler_X.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

print(f"特徵維度: {X_train_tensor.shape[1]}")

# ============================================
# 2. 模型定義
# ============================================

class SharedFeatureExtractor(nn.Module):
    """共享的特徵提取器 - 所有資料都參與訓練"""
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


class GatingNetwork(nn.Module):
    """
    Gating Network - 決定每個 Expert 的權重
    輸出 softmax 權重，確保是軟性混合
    """
    def __init__(self, input_dim, n_experts=2, hidden_dim=8, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )
        # 初始化：讓 Expert 1 一開始是主力
        # bias = [0.5, -0.5] 會讓 softmax 後 w1 ≈ 0.73, w2 ≈ 0.27
        nn.init.constant_(self.net[-1].bias, 0)
        self.net[-1].bias.data[0] = 0.5
        self.net[-1].bias.data[1] = -0.5
    
    def forward(self, x):
        logits = self.net(x) / self.temperature
        weights = torch.softmax(logits, dim=-1)
        return weights


class ExpertGP(ApproximateGP):
    """單個 Expert GP"""
    def __init__(self, inducing_points, lengthscale_prior=None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # RBF kernel with ARD
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1))
        if lengthscale_prior is not None:
            base_kernel.lengthscale = lengthscale_prior
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class NoiseNetwork(nn.Module):
    """Heteroscedastic Noise Network"""
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


class MoEDKL(nn.Module):
    """
    Mixture of Experts Deep Kernel Learning
    
    架構:
    Input → SharedFeatureExtractor → shared_features
                                          ↓
                    ┌─────────────────────┼─────────────────────┐
                    ↓                     ↓                     ↓
              GatingNetwork          Expert1 GP            Expert2 GP
                    ↓                     ↓                     ↓
              [w1, w2]                 mean1                  mean2
                    └─────────────────────┼─────────────────────┘
                                          ↓
                              Final = w1*mean1 + w2*mean2
    """
    def __init__(self, input_dim, feature_dim=16, n_inducing=50, temperature=1.0):
        super().__init__()
        
        # 共享特徵提取器
        self.feature_extractor = SharedFeatureExtractor(input_dim, output_dim=feature_dim)
        
        # Gating Network
        self.gating = GatingNetwork(feature_dim, n_experts=2, temperature=temperature)
        
        # Expert 1: General (處理大部分正常資料)
        inducing_1 = torch.randn(n_inducing, feature_dim)
        self.expert1 = ExpertGP(inducing_1)
        
        # Expert 2: Difficult (處理高變異區域)
        inducing_2 = torch.randn(n_inducing, feature_dim)
        self.expert2 = ExpertGP(inducing_2)
        
        # Noise Network (共用)
        self.noise_net = NoiseNetwork(feature_dim)
        
    def forward(self, x):
        # 共享特徵
        features = self.feature_extractor(x)
        
        # Gating 權重
        weights = self.gating(features)  # [N, 2]
        
        # Expert 預測
        expert1_output = self.expert1(features)
        expert2_output = self.expert2(features)
        
        # Noise
        log_noise = self.noise_net(features)
        
        return {
            'features': features,
            'weights': weights,
            'expert1_output': expert1_output,
            'expert2_output': expert2_output,
            'log_noise': log_noise
        }
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            weights = outputs['weights']
            mean1 = outputs['expert1_output'].mean
            mean2 = outputs['expert2_output'].mean
            var1 = outputs['expert1_output'].variance
            var2 = outputs['expert2_output'].variance
            log_noise = outputs['log_noise']
            
            # 加權平均
            w1, w2 = weights[:, 0], weights[:, 1]
            final_mean = w1 * mean1 + w2 * mean2
            
            # 混合方差 (包含 experts 之間的差異)
            # Var = E[Var] + Var[E]
            # = w1*var1 + w2*var2 + w1*w2*(mean1-mean2)^2
            final_var = (w1 * var1 + w2 * var2 + 
                        w1 * w2 * (mean1 - mean2)**2 +
                        torch.exp(log_noise))
            
        return final_mean, final_var, weights, torch.exp(log_noise)


class MoELoss(nn.Module):
    """
    MoE Loss = Heteroscedastic NLL + KL Divergence + Gating Regularization
    """
    def __init__(self, entropy_weight=0.1):
        super().__init__()
        self.entropy_weight = entropy_weight
    
    def forward(self, outputs, y):
        weights = outputs['weights']
        mean1 = outputs['expert1_output'].mean
        mean2 = outputs['expert2_output'].mean
        log_noise = outputs['log_noise']
        
        # 加權預測
        w1, w2 = weights[:, 0], weights[:, 1]
        pred_mean = w1 * mean1 + w2 * mean2
        
        # Heteroscedastic NLL
        noise_var = torch.exp(log_noise)
        nll = 0.5 * log_noise + 0.5 * (y - pred_mean)**2 / noise_var
        nll_loss = nll.mean()
        
        # Gating Entropy Regularization (鼓勵不要太極端)
        # entropy = -sum(w * log(w))，最大化 entropy 讓分佈更均勻
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        entropy_reg = -self.entropy_weight * entropy  # 負號因為要最大化 entropy
        
        return nll_loss, entropy_reg


# ============================================
# 3. 訓練
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


def train_moe_dkl(seed=42, n_epochs=300, lr=0.005, patience=30, 
                  temperature=1.0, entropy_weight=0.1):
    set_seed(seed)
    
    input_dim = X_train_tensor.shape[1]
    model = MoEDKL(input_dim, feature_dim=16, n_inducing=50, 
                   temperature=temperature).to(device)
    
    # Optimizer - 不同部分用不同 learning rate
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.gating.parameters(), 'lr': lr * 0.5},  # Gating 學慢一點
        {'params': model.expert1.parameters(), 'lr': lr},
        {'params': model.expert2.parameters(), 'lr': lr},
        {'params': model.noise_net.parameters(), 'lr': lr * 0.5},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = MoELoss(entropy_weight=entropy_weight)
    
    model.train()
    model.expert1.train()
    model.expert2.train()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    # 用於監控的變數
    train_types_tensor = torch.tensor(train_types, dtype=torch.float32).to(device)
    train_coverages_tensor = torch.tensor(train_coverages, dtype=torch.float32).to(device)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        
        # Main loss
        nll_loss, entropy_reg = loss_fn(outputs, y_train_tensor)
        
        # KL divergence for both experts
        kl_1 = model.expert1.variational_strategy.kl_divergence().mean() / len(y_train_tensor)
        kl_2 = model.expert2.variational_strategy.kl_divergence().mean() / len(y_train_tensor)
        
        total_loss = nll_loss + entropy_reg + 0.1 * (kl_1 + kl_2)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # 進度報告
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                mean, var, weights, noise = model.predict(X_test_tensor)
                pred = scaler_y.inverse_transform(mean.cpu().numpy().reshape(-1, 1)).flatten()
                errors = np.abs(pred - y_test) / y_test * 100
                
                # 分析 gating 權重分佈
                train_outputs = model(X_train_tensor)
                train_weights = train_outputs['weights']
                
                # 各 Type 的平均 Expert 2 權重
                type1_w2 = train_weights[train_types_tensor == 1, 1].mean().item()
                type2_w2 = train_weights[train_types_tensor == 2, 1].mean().item()
                type3_w2 = train_weights[train_types_tensor == 3, 1].mean().item()
                type3_high_cov_mask = (train_types_tensor == 3) & (train_coverages_tensor >= 0.8)
                type3_high_cov_w2 = train_weights[type3_high_cov_mask, 1].mean().item() if type3_high_cov_mask.sum() > 0 else 0
                
            print(f"Epoch {epoch+1}: Loss={total_loss.item():.4f}, NLL={nll_loss.item():.4f}, "
                  f"MAPE={np.mean(errors):.2f}%, Max={np.max(errors):.1f}%, >20%={np.sum(errors>20)}")
            print(f"  Gating (Expert2 weight): T1={type1_w2:.3f}, T2={type2_w2:.3f}, "
                  f"T3={type3_w2:.3f}, T3+HighCov={type3_high_cov_w2:.3f}")
            
            model.train()
            model.expert1.train()
            model.expert2.train()
    
    # 載入最佳模型
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    return model


# ============================================
# 4. 執行實驗
# ============================================

print("\n" + "="*80)
print("開始訓練 MoE DKL...")
print("="*80)

model = train_moe_dkl(
    seed=42, 
    n_epochs=300, 
    lr=0.005, 
    patience=30,
    temperature=1.0,      # Gating softmax 溫度
    entropy_weight=0.1    # Entropy 正則化權重
)

# ============================================
# 5. 評估結果
# ============================================

print("\n" + "="*80)
print("評估結果")
print("="*80)

model.eval()
with torch.no_grad():
    mean, var, weights, noise = model.predict(X_test_tensor)
    pred = scaler_y.inverse_transform(mean.cpu().numpy().reshape(-1, 1)).flatten()
    weights_np = weights.cpu().numpy()
    noise_np = noise.cpu().numpy()

# 計算指標
errors = np.abs(pred - y_test) / y_test * 100

print(f"\n【整體指標】")
print(f"MAPE: {np.mean(errors):.2f}%")
print(f"Max Error: {np.max(errors):.1f}%")
print(f"Outliers (>20%): {np.sum(errors > 20)}")
print(f"Outliers (>30%): {np.sum(errors > 30)}")
print(f"Outliers (>40%): {np.sum(errors > 40)}")

# Gating 權重分析
print(f"\n【Gating 權重分析 (測試集)】")
print(f"{'Type':<8} {'Cov':<8} {'Expert1 w':<12} {'Expert2 w':<12} {'Avg Error':<12}")
print("-"*55)

for typ in [1, 2, 3]:
    for cov in [0.6, 0.8, 1.0]:
        mask = (test_types == typ) & (np.abs(test_coverages - cov) < 0.01)
        if mask.sum() > 0:
            w1_avg = weights_np[mask, 0].mean()
            w2_avg = weights_np[mask, 1].mean()
            err_avg = errors[mask].mean()
            print(f"{typ:<8} {cov:<8.1f} {w1_avg:<12.3f} {w2_avg:<12.3f} {err_avg:<12.1f}%")

# 各 Type 的整體表現
print(f"\n【各 Type 整體表現】")
print(f"{'Type':<8} {'MAPE':<10} {'Max Err':<12} {'>20%':<8} {'Avg Expert2 w':<15}")
print("-"*55)
for typ in [1, 2, 3]:
    mask = test_types == typ
    if mask.sum() > 0:
        mape = errors[mask].mean()
        max_err = errors[mask].max()
        outliers = (errors[mask] > 20).sum()
        w2_avg = weights_np[mask, 1].mean()
        print(f"{typ:<8} {mape:<10.2f} {max_err:<12.1f} {outliers:<8} {w2_avg:<15.3f}")

# 異常點詳情
print(f"\n【異常點詳情 (>20%)】")
outliers_mask = errors > 20
outliers_df = pd.DataFrame({
    'Type': test_types[outliers_mask],
    'Thickness': test_df['TIM_THICKNESS'].values[outliers_mask],
    'Coverage': test_coverages[outliers_mask],
    'True': y_test[outliers_mask],
    'Pred': pred[outliers_mask],
    'Error%': errors[outliers_mask],
    'Expert1_w': weights_np[outliers_mask, 0],
    'Expert2_w': weights_np[outliers_mask, 1],
}).sort_values('Error%', ascending=False)

print(f"{'Type':<6} {'Thick':<8} {'Cov':<6} {'True':<10} {'Pred':<10} {'Error%':<10} {'E1_w':<8} {'E2_w':<8}")
print("-"*75)
for _, row in outliers_df.iterrows():
    print(f"{row['Type']:<6.0f} {row['Thickness']:<8.0f} {row['Coverage']:<6.1f} "
          f"{row['True']:<10.4f} {row['Pred']:<10.4f} {row['Error%']:<10.1f} "
          f"{row['Expert1_w']:<8.3f} {row['Expert2_w']:<8.3f}")

if len(outliers_df) == 0:
    print("無異常點！")

# 保存結果
results_df = pd.DataFrame({
    'TIM_TYPE': test_types,
    'TIM_THICKNESS': test_df['TIM_THICKNESS'].values,
    'TIM_COVERAGE': test_coverages,
    'Theta.JC': y_test,
    'Predicted': pred,
    'Error_Pct': errors,
    'Expert1_weight': weights_np[:, 0],
    'Expert2_weight': weights_np[:, 1],
})
results_df.to_csv('phase3c_moe_results.csv', index=False)
print(f"\n✓ 結果已保存至 phase3c_moe_results.csv")

# ============================================
# 6. 與 Phase 3A 比較
# ============================================

print("\n" + "="*80)
print("與 Phase 3A 比較")
print("="*80)
print("""
| 指標           | Phase 3A (Hetero) | Phase 3C (MoE) |
|----------------|-------------------|----------------|
| MAPE           | 7.53%             | {:.2f}%         |
| Max Error      | 36.3%             | {:.1f}%         |
| Outliers >20%  | 6                 | {}              |
| Outliers >40%  | 0                 | {}              |
""".format(np.mean(errors), np.max(errors), np.sum(errors > 20), np.sum(errors > 40)))
