"""
Phase 3D: 規則式雙 Expert MoE

改進自 Phase 3C:
- 不用學習的 Gating Network
- 用規則決定 Expert 權重（但仍是 soft mixture，不是 hard split）
- Expert 1: 處理正常區域
- Expert 2: 處理 Type 3 + 高 Coverage 區域

規則設計:
- Type 1, 2: w1=0.9, w2=0.1
- Type 3, Cov<0.8: w1=0.7, w2=0.3
- Type 3, Cov>=0.8: w1=0.3, w2=0.7
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
print("Phase 3D: 規則式雙 Expert MoE")
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

# 保留原始資訊
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

# 計算規則式權重
def compute_rule_based_weights(types, coverages):
    """
    規則式 Expert 權重
    返回 [w1, w2]，w1 + w2 = 1
    """
    n = len(types)
    weights = np.zeros((n, 2))
    
    for i in range(n):
        if types[i] in [1, 2]:
            # Type 1, 2: 主要用 Expert 1
            weights[i] = [0.9, 0.1]
        elif types[i] == 3 and coverages[i] < 0.8:
            # Type 3 低 Coverage: 混合使用
            weights[i] = [0.7, 0.3]
        else:
            # Type 3 高 Coverage: 主要用 Expert 2
            weights[i] = [0.3, 0.7]
    
    return weights

train_weights = torch.tensor(compute_rule_based_weights(train_types, train_coverages), 
                              dtype=torch.float32).to(device)
test_weights = torch.tensor(compute_rule_based_weights(test_types, test_coverages), 
                             dtype=torch.float32).to(device)

print(f"特徵維度: {X_train_tensor.shape[1]}")
print(f"\n規則式權重分佈:")
print(f"  Type 1, 2:        w1=0.9, w2=0.1")
print(f"  Type 3, Cov<0.8:  w1=0.7, w2=0.3")
print(f"  Type 3, Cov>=0.8: w1=0.3, w2=0.7")

# ============================================
# 2. 模型定義
# ============================================

class SharedFeatureExtractor(nn.Module):
    """共享的特徵提取器"""
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


class ExpertGP(ApproximateGP):
    """單個 Expert GP"""
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


class RuleBasedMoEDKL(nn.Module):
    """
    規則式 Mixture of Experts DKL
    
    權重由規則決定，不學習 Gating Network
    """
    def __init__(self, input_dim, feature_dim=16, n_inducing=50):
        super().__init__()
        
        # 共享特徵提取器
        self.feature_extractor = SharedFeatureExtractor(input_dim, output_dim=feature_dim)
        
        # Expert 1: General (處理正常資料)
        inducing_1 = torch.randn(n_inducing, feature_dim) * 0.1
        self.expert1 = ExpertGP(inducing_1)
        
        # Expert 2: Difficult (處理高變異區域)
        # 用不同的初始化讓兩個 Expert 有差異
        inducing_2 = torch.randn(n_inducing, feature_dim) * 0.1 + 0.5
        self.expert2 = ExpertGP(inducing_2)
        
        # Noise Network
        self.noise_net = NoiseNetwork(feature_dim)
        
    def forward(self, x, weights):
        """
        x: 輸入特徵
        weights: 規則式權重 [N, 2]
        """
        # 共享特徵
        features = self.feature_extractor(x)
        
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
    
    def predict(self, x, weights):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, weights)
            
            w1, w2 = weights[:, 0], weights[:, 1]
            mean1 = outputs['expert1_output'].mean
            mean2 = outputs['expert2_output'].mean
            var1 = outputs['expert1_output'].variance
            var2 = outputs['expert2_output'].variance
            log_noise = outputs['log_noise']
            
            # 加權平均
            final_mean = w1 * mean1 + w2 * mean2
            
            # 混合方差
            final_var = (w1 * var1 + w2 * var2 + 
                        w1 * w2 * (mean1 - mean2)**2 +
                        torch.exp(log_noise))
            
        return final_mean, final_var, torch.exp(log_noise)


class MoELoss(nn.Module):
    """MoE Loss"""
    def __init__(self):
        super().__init__()
    
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
        
        return nll.mean()


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


def train_rule_moe_dkl(seed=42, n_epochs=300, lr=0.005, patience=30):
    set_seed(seed)
    
    input_dim = X_train_tensor.shape[1]
    model = RuleBasedMoEDKL(input_dim, feature_dim=16, n_inducing=50).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.expert1.parameters(), 'lr': lr},
        {'params': model.expert2.parameters(), 'lr': lr},
        {'params': model.noise_net.parameters(), 'lr': lr * 0.5},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = MoELoss()
    
    model.train()
    model.expert1.train()
    model.expert2.train()
    
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor, train_weights)
        
        # Main loss
        nll_loss = loss_fn(outputs, y_train_tensor)
        
        # KL divergence
        kl_1 = model.expert1.variational_strategy.kl_divergence().mean() / len(y_train_tensor)
        kl_2 = model.expert2.variational_strategy.kl_divergence().mean() / len(y_train_tensor)
        
        total_loss = nll_loss + 0.1 * (kl_1 + kl_2)
        
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
                mean, var, noise = model.predict(X_test_tensor, test_weights)
                pred = scaler_y.inverse_transform(mean.cpu().numpy().reshape(-1, 1)).flatten()
                errors = np.abs(pred - y_test) / y_test * 100
                
            print(f"Epoch {epoch+1}: Loss={total_loss.item():.4f}, "
                  f"MAPE={np.mean(errors):.2f}%, Max={np.max(errors):.1f}%, >20%={np.sum(errors>20)}")
            
            model.train()
            model.expert1.train()
            model.expert2.train()
    
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


# ============================================
# 4. 執行實驗
# ============================================

print("\n" + "="*80)
print("開始訓練 Rule-based MoE DKL...")
print("="*80)

model = train_rule_moe_dkl(seed=42, n_epochs=300, lr=0.005, patience=30)

# ============================================
# 5. 評估結果
# ============================================

print("\n" + "="*80)
print("評估結果")
print("="*80)

model.eval()
with torch.no_grad():
    mean, var, noise = model.predict(X_test_tensor, test_weights)
    pred = scaler_y.inverse_transform(mean.cpu().numpy().reshape(-1, 1)).flatten()
    noise_np = noise.cpu().numpy()

errors = np.abs(pred - y_test) / y_test * 100
weights_np = test_weights.cpu().numpy()

print(f"\n【整體指標】")
print(f"MAPE: {np.mean(errors):.2f}%")
print(f"Max Error: {np.max(errors):.1f}%")
print(f"Outliers (>20%): {np.sum(errors > 20)}")
print(f"Outliers (>30%): {np.sum(errors > 30)}")
print(f"Outliers (>40%): {np.sum(errors > 40)}")

# 各 Type/Coverage 組合表現
print(f"\n【各 Type/Coverage 組合表現】")
print(f"{'Type':<8} {'Cov':<8} {'MAPE':<10} {'Max Err':<10} {'>20%':<8} {'w1':<8} {'w2':<8}")
print("-"*65)

for typ in [1, 2, 3]:
    for cov in [0.6, 0.8, 1.0]:
        mask = (test_types == typ) & (np.abs(test_coverages - cov) < 0.01)
        if mask.sum() > 0:
            mape = errors[mask].mean()
            max_err = errors[mask].max()
            outliers = (errors[mask] > 20).sum()
            w1 = weights_np[mask, 0][0]
            w2 = weights_np[mask, 1][0]
            print(f"{typ:<8} {cov:<8.1f} {mape:<10.2f} {max_err:<10.1f} {outliers:<8} {w1:<8.1f} {w2:<8.1f}")

# 各 Type 整體表現
print(f"\n【各 Type 整體表現】")
print(f"{'Type':<8} {'MAPE':<10} {'Max Err':<12} {'>20%':<8}")
print("-"*40)
for typ in [1, 2, 3]:
    mask = test_types == typ
    if mask.sum() > 0:
        mape = errors[mask].mean()
        max_err = errors[mask].max()
        outliers = (errors[mask] > 20).sum()
        print(f"{typ:<8} {mape:<10.2f} {max_err:<12.1f} {outliers:<8}")

# 異常點詳情
print(f"\n【異常點詳情 (>20%)】")
outliers_mask = errors > 20
if outliers_mask.sum() > 0:
    outliers_df = pd.DataFrame({
        'Type': test_types[outliers_mask],
        'Thickness': test_df['TIM_THICKNESS'].values[outliers_mask],
        'Coverage': test_coverages[outliers_mask],
        'True': y_test[outliers_mask],
        'Pred': pred[outliers_mask],
        'Error%': errors[outliers_mask],
        'w1': weights_np[outliers_mask, 0],
        'w2': weights_np[outliers_mask, 1],
    }).sort_values('Error%', ascending=False)

    print(f"{'Type':<6} {'Thick':<8} {'Cov':<6} {'True':<10} {'Pred':<10} {'Error%':<10} {'w1':<6} {'w2':<6}")
    print("-"*70)
    for _, row in outliers_df.iterrows():
        print(f"{row['Type']:<6.0f} {row['Thickness']:<8.0f} {row['Coverage']:<6.1f} "
              f"{row['True']:<10.4f} {row['Pred']:<10.4f} {row['Error%']:<10.1f} "
              f"{row['w1']:<6.1f} {row['w2']:<6.1f}")
else:
    print("無異常點！🎉")

# 保存結果
results_df = pd.DataFrame({
    'TIM_TYPE': test_types,
    'TIM_THICKNESS': test_df['TIM_THICKNESS'].values,
    'TIM_COVERAGE': test_coverages,
    'Theta.JC': y_test,
    'Predicted': pred,
    'Error_Pct': errors,
    'w1': weights_np[:, 0],
    'w2': weights_np[:, 1],
})
results_df.to_csv('phase3d_rule_moe_results.csv', index=False)
print(f"\n✓ 結果已保存至 phase3d_rule_moe_results.csv")

# ============================================
# 6. 比較
# ============================================

print("\n" + "="*80)
print("與之前版本比較")
print("="*80)
print(f"""
| 指標           | Phase 3A (Hetero) | Phase 3C (MoE) | Phase 3D (Rule MoE) |
|----------------|-------------------|----------------|---------------------|
| MAPE           | 7.53%             | 36.29%         | {np.mean(errors):.2f}%              |
| Max Error      | 36.3%             | 391.3%         | {np.max(errors):.1f}%              |
| Outliers >20%  | 6                 | 48             | {np.sum(errors > 20)}                 |
| Outliers >40%  | 0                 | 32             | {np.sum(errors > 40)}                 |
""")
