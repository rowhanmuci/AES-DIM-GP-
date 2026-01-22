"""
GDKL (Guided Deep Kernel Learning) V3 - 熱阻預測
加強數值穩定性版本

需要安裝:
    pip install torch gpytorch pandas numpy scikit-learn openpyxl

使用:
    python gdkl_thermal_v3.py --seed 2024 --beta 1.0
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings
import random
import os
import argparse
from typing import Tuple, Dict, Optional

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ 隨機種子: {seed}")


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# 穩定的矩陣運算
# ==========================================

def stable_cholesky(K: torch.Tensor, max_jitter: float = 1e-2) -> torch.Tensor:
    """
    數值穩定的 Cholesky 分解
    逐步增加 jitter 直到成功
    """
    n = K.shape[0]
    jitter = 1e-6
    
    for _ in range(10):
        try:
            K_jitter = K + jitter * torch.eye(n, device=K.device, dtype=K.dtype)
            L = torch.linalg.cholesky(K_jitter)
            return L
        except:
            jitter *= 10
            if jitter > max_jitter:
                break
    
    # Fallback: 使用 eigendecomposition 修復
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = torch.clamp(eigenvalues, min=1e-6)
    K_fixed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    K_fixed = K_fixed + 1e-5 * torch.eye(n, device=K.device, dtype=K.dtype)
    return torch.linalg.cholesky(K_fixed)


def stable_solve(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """穩定求解 K @ x = y"""
    try:
        L = stable_cholesky(K)
        return torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
    except:
        # Fallback: 使用 lstsq
        return torch.linalg.lstsq(K, y).solution


# ==========================================
# NNGP Kernel
# ==========================================

class NNGPKernel:
    """NNGP Kernel with ReLU activation"""
    
    def __init__(self, sigma_w: float = 1.5, sigma_b: float = 0.1, num_layers: int = 3):
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.num_layers = num_layers
    
    def _relu_expectation(self, k11: torch.Tensor, k12: torch.Tensor, 
                          k22: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        denom = torch.sqrt(torch.clamp(k11 * k22, min=eps))
        rho = torch.clamp(k12 / (denom + eps), -1 + eps, 1 - eps)
        theta = torch.acos(rho)
        return (1 / (2 * np.pi)) * denom * (torch.sin(theta) + (np.pi - theta) * rho)
    
    def compute(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if X2 is None:
            X2 = X1
        
        n1, d = X1.shape
        n2 = X2.shape[0]
        
        K12 = self.sigma_b**2 + self.sigma_w**2 * (X1 @ X2.T) / d
        K11_diag = self.sigma_b**2 + self.sigma_w**2 * (X1**2).sum(1) / d
        K22_diag = self.sigma_b**2 + self.sigma_w**2 * (X2**2).sum(1) / d
        
        for _ in range(1, self.num_layers):
            K11 = K11_diag.unsqueeze(1).expand(n1, n2)
            K22 = K22_diag.unsqueeze(0).expand(n1, n2)
            K12 = self.sigma_b**2 + self.sigma_w**2 * self._relu_expectation(K11, K12, K22)
            K11_diag = self.sigma_b**2 + self.sigma_w**2 * self._relu_expectation(K11_diag, K11_diag, K11_diag)
            K22_diag = self.sigma_b**2 + self.sigma_w**2 * self._relu_expectation(K22_diag, K22_diag, K22_diag)
        
        return K12


class NNGPPredictor:
    """NNGP 預測器"""
    
    def __init__(self, kernel: NNGPKernel, noise_var: float = 0.05):
        self.kernel = kernel
        self.noise_var = noise_var
    
    def predict(self, X_train: torch.Tensor, y_train: torch.Tensor,
                X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算 NNGP 預測"""
        n = X_train.shape[0]
        
        K_train = self.kernel.compute(X_train)
        K_train_noisy = K_train + self.noise_var * torch.eye(n, device=X_train.device, dtype=X_train.dtype)
        
        K_test_train = self.kernel.compute(X_test, X_train)
        K_test_diag = torch.diag(self.kernel.compute(X_test))
        
        alpha = stable_solve(K_train_noisy, y_train)
        mean = K_test_train @ alpha
        
        # 方差計算 (使用近似避免完整矩陣求逆)
        L = stable_cholesky(K_train_noisy)
        v = torch.linalg.solve_triangular(L, K_test_train.T, upper=False)
        var = K_test_diag - torch.sum(v**2, dim=0)
        var = torch.clamp(var, min=1e-6)
        
        return mean, var


# ==========================================
# Deep Kernel GP
# ==========================================

class FeatureExtractor(nn.Module):
    """特徵提取網路"""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),  # 用 LayerNorm 替代 BatchNorm，更穩定
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class DeepKernelGP(nn.Module):
    """Deep Kernel GP - 數值穩定版本"""
    
    def __init__(self, input_dim: int, hidden_dims: list, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dims, feature_dim, dropout)
        
        # Kernel 超參數 (使用合理的初始值)
        self.log_lengthscale = nn.Parameter(torch.zeros(feature_dim))
        self.log_outputscale = nn.Parameter(torch.tensor(0.0))
        self.log_noise = nn.Parameter(torch.tensor(-1.0))  # noise ~ 0.37
    
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale).clamp(min=0.01, max=10.0)
    
    @property
    def outputscale(self):
        return torch.exp(self.log_outputscale).clamp(min=0.01, max=10.0)
    
    @property
    def noise(self):
        return torch.exp(self.log_noise).clamp(min=1e-4, max=1.0)
    
    def compute_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """計算 Deep RBF Kernel"""
        Z1 = self.feature_extractor(X1)
        Z2 = self.feature_extractor(X2)
        
        # 正規化特徵 (重要！穩定kernel)
        Z1 = Z1 / (Z1.norm(dim=1, keepdim=True) + 1e-6)
        Z2 = Z2 / (Z2.norm(dim=1, keepdim=True) + 1e-6)
        
        # Scaled distance
        Z1_s = Z1 / self.lengthscale
        Z2_s = Z2 / self.lengthscale
        
        # RBF kernel
        dist_sq = torch.cdist(Z1_s, Z2_s, p=2)**2
        K = self.outputscale * torch.exp(-0.5 * dist_sq)
        
        return K
    
    def predict(self, X_train: torch.Tensor, y_train: torch.Tensor,
                X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """GP 預測"""
        n = X_train.shape[0]
        
        K_train = self.compute_kernel(X_train, X_train)
        K_train_noisy = K_train + self.noise * torch.eye(n, device=X_train.device, dtype=X_train.dtype)
        
        K_test_train = self.compute_kernel(X_test, X_train)
        
        # 預測
        alpha = stable_solve(K_train_noisy, y_train)
        mean = K_test_train @ alpha
        
        # 方差
        L = stable_cholesky(K_train_noisy)
        v = torch.linalg.solve_triangular(L, K_test_train.T, upper=False)
        var = self.outputscale - torch.sum(v**2, dim=0)
        var = torch.clamp(var, min=1e-6)
        
        return mean, var
    
    def compute_mll(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Marginal Log-Likelihood"""
        n = X.shape[0]
        K = self.compute_kernel(X, X)
        K_noisy = K + self.noise * torch.eye(n, device=X.device, dtype=X.dtype)
        
        L = stable_cholesky(K_noisy)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        
        log_det = 2 * torch.sum(torch.log(torch.diag(L) + 1e-8))
        quad_form = torch.dot(y, alpha)
        
        mll = -0.5 * (quad_form + log_det + n * np.log(2 * np.pi))
        return mll


# ==========================================
# GDKL Loss Functions
# ==========================================

def kl_divergence(mu_q: torch.Tensor, var_q: torch.Tensor,
                  mu_p: torch.Tensor, var_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) for Gaussians"""
    eps = 1e-8
    var_q = var_q.clamp(min=eps)
    var_p = var_p.clamp(min=eps)
    
    kl = 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_q - mu_p)**2) / var_p - 1)
    return kl


def ell_loss(y: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, 
             noise: torch.Tensor) -> torch.Tensor:
    """Expected Log-Likelihood loss"""
    return 0.5 * (np.log(2 * np.pi) + torch.log(noise + 1e-8) + 
                  ((y - mu)**2 + var) / (noise + 1e-8))


def get_sample_weights(X: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """計算樣本權重"""
    w = np.ones(len(X))
    mask = (X[:, 0] == 3) & (X[:, 2] >= 0.75) & (X[:, 1] >= 200)
    w[mask] *= factor
    return w


# ==========================================
# GDKL Trainer
# ==========================================

class GDKLTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.nngp = None
        self.scaler_x = None
        self.scaler_y = None
        self.train_x = None
        self.train_y = None
    
    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        cfg = self.config
        
        # 標準化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X_s = self.scaler_x.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.train_x = torch.from_numpy(X_s).to(device)
        self.train_y = torch.from_numpy(y_s).to(device)
        
        weights = torch.from_numpy(get_sample_weights(X, cfg['weight_factor'])).to(device)
        
        n = len(X)
        
        if verbose:
            n_hard = (weights > 1).sum().item()
            print(f"\n樣本: {n}, 困難樣本: {n_hard}")
        
        # 初始化 NNGP
        if verbose:
            print(f"\n[1] 初始化 NNGP...")
        
        nngp_kernel = NNGPKernel(cfg['nngp_sigma_w'], cfg['nngp_sigma_b'], cfg['nngp_layers'])
        self.nngp = NNGPPredictor(nngp_kernel, cfg['nngp_noise'])
        
        # 預計算 NNGP kernel
        self.nngp_K = nngp_kernel.compute(self.train_x)
        
        # 初始化 DKL
        if verbose:
            print(f"[2] 初始化 DKL...")
        
        self.model = DeepKernelGP(
            self.train_x.shape[1], cfg['hidden_dims'], cfg['feature_dim'], cfg['dropout']
        ).to(device)
        
        # 優化器
        optimizer = optim.AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        
        if verbose:
            print(f"[3] GDKL 訓練 (β={cfg['beta']})...\n")
        
        best_loss = float('inf')
        patience = 0
        best_state = None
        
        for epoch in range(cfg['epochs']):
            self.model.train()
            optimizer.zero_grad()
            
            # 隨機分割
            perm = torch.randperm(n)
            n1 = int(n * 0.7)
            idx1, idx2 = perm[:n1], perm[n1:]
            
            X1, y1 = self.train_x[idx1], self.train_y[idx1]
            X2, y2 = self.train_x[idx2], self.train_y[idx2]
            w2 = weights[idx2]
            
            # NNGP 預測 (使用預計算的 kernel)
            K11 = self.nngp_K[idx1][:, idx1]
            K21 = self.nngp_K[idx2][:, idx1]
            K22_diag = torch.diag(self.nngp_K)[idx2]
            
            K11_noisy = K11 + cfg['nngp_noise'] * torch.eye(n1, device=device, dtype=self.train_x.dtype)
            
            nngp_alpha = stable_solve(K11_noisy, y1)
            nngp_mean = K21 @ nngp_alpha
            
            L_nngp = stable_cholesky(K11_noisy)
            v_nngp = torch.linalg.solve_triangular(L_nngp, K21.T, upper=False)
            nngp_var = K22_diag - torch.sum(v_nngp**2, dim=0)
            nngp_var = nngp_var.clamp(min=1e-6)
            
            # DKL 預測
            dkl_mean, dkl_var = self.model.predict(X1, y1, X2)
            
            # GDKL Loss
            loss_ell = ell_loss(y2, dkl_mean, dkl_var, self.model.noise)
            loss_kl = kl_divergence(dkl_mean, dkl_var, nngp_mean, nngp_var)
            
            # 加權
            gdkl = (loss_ell * w2).sum() / w2.sum() + cfg['beta'] * (loss_kl * w2).sum() / w2.sum()
            
            # MLL 正則
            mll = -self.model.compute_mll(self.train_x, self.train_y) / n
            
            total = gdkl + cfg['mll_weight'] * mll
            
            # 檢查 NaN
            if torch.isnan(total):
                if verbose:
                    print(f"  Epoch {epoch+1}: NaN detected, skipping...")
                continue
            
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            loss_val = total.item()
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: ELL={loss_ell.mean().item():.4f}, "
                      f"KL={loss_kl.mean().item():.4f}, Total={loss_val:.4f}")
            
            if loss_val < best_loss:
                best_loss = loss_val
                patience = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
            
            if patience >= cfg['patience']:
                if verbose:
                    print(f"  早停 @ Epoch {epoch+1}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        if verbose:
            print(f"\n訓練完成 (Best Loss: {best_loss:.4f})")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        X_s = torch.from_numpy(self.scaler_x.transform(X)).to(device)
        
        with torch.no_grad():
            mean, var = self.model.predict(self.train_x, self.train_y, X_s)
        
        mean_np = mean.cpu().numpy()
        std_np = torch.sqrt(var).cpu().numpy()
        
        y_pred = self.scaler_y.inverse_transform(mean_np.reshape(-1, 1)).flatten()
        y_std = std_np * self.scaler_y.scale_[0]
        
        return y_pred, y_std


# ==========================================
# Evaluation
# ==========================================

def evaluate(trainer: GDKLTrainer, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
    pred, std = trainer.predict(X)
    
    err = np.abs((y - pred) / y) * 100
    
    mape = err.mean()
    mae = np.abs(y - pred).mean()
    
    out20 = (err > 20).sum()
    out15 = (err > 15).sum()
    out10 = (err > 10).sum()
    
    t3_mask = X[:, 0] == 3
    t3_out = ((err > 20) & t3_mask).sum()
    
    if verbose:
        print(f"\n{'='*55}")
        print(f"GDKL 結果")
        print(f"{'='*55}")
        print(f"MAPE: {mape:.2f}%")
        print(f"MAE:  {mae:.4f}")
        print(f"異常點 >20%: {out20}/{len(y)} ({out20/len(y)*100:.1f}%)")
        print(f"異常點 >15%: {out15}/{len(y)}")
        print(f"異常點 >10%: {out10}/{len(y)}")
        
        if t3_mask.sum() > 0:
            print(f"\nType 3: MAPE={err[t3_mask].mean():.2f}%, 異常點={t3_out}/{t3_mask.sum()}")
        
        print(f"\n最大誤差 Top 5:")
        for i in np.argsort(err)[-5:][::-1]:
            print(f"  T{X[i,0]:.0f} Th{X[i,1]:.0f} C{X[i,2]:.1f}: "
                  f"True={y[i]:.4f} Pred={pred[i]:.4f} Err={err[i]:.1f}%")
        print(f"{'='*55}")
    
    return {
        'mape': mape, 'mae': mae,
        'outliers_20': out20, 'outliers_15': out15, 'outliers_10': out10,
        'type3_outliers': t3_out,
        'predictions': pred, 'std': std, 'errors': err
    }


def save_results(X: np.ndarray, y: np.ndarray, res: Dict, path: str):
    df = pd.DataFrame({
        'TIM_TYPE': X[:, 0], 'TIM_THICKNESS': X[:, 1], 'TIM_COVERAGE': X[:, 2],
        'TRUE': y, 'Predicted': res['predictions'], 'Error%': res['errors'], 'Std': res['std']
    })
    df.to_csv(path, index=False)
    print(f"✓ 保存: {path}")


# ==========================================
# Main
# ==========================================

def main(seed: int = 2024, beta: float = 1.0, verbose: bool = True):
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n裝置: {device}")
    print("="*55)
    print("GDKL V3 - 熱阻預測")
    print("="*55)
    
    config = {
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        'lr': 0.003,
        'epochs': 800,
        'patience': 100,
        'beta': beta,
        'mll_weight': 0.2,
        'nngp_sigma_w': 1.5,
        'nngp_sigma_b': 0.1,
        'nngp_layers': 3,
        'nngp_noise': 0.05,
        'weight_factor': 3.0,
    }
    
    print(f"\nβ={beta}, lr={config['lr']}, epochs={config['epochs']}")
    
    # 載入資料
    train_df = pd.read_excel('data/train/Above.xlsx')
    test_df = pd.read_excel('data/test/Above.xlsx')
    
    cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target = 'Theta.JC'
    
    train_df = train_df.groupby(cols, as_index=False).agg({target: 'mean'})
    
    X_train = train_df[cols].values
    y_train = train_df[target].values
    X_test = test_df[cols].values
    y_test = test_df[target].values
    
    print(f"\n訓練: {len(X_train)}, 測試: {len(X_test)}")
    
    # 訓練
    trainer = GDKLTrainer(config)
    trainer.train(X_train, y_train, verbose)
    
    # 評估
    results = evaluate(trainer, X_test, y_test, verbose)
    
    # 保存
    save_results(X_test, y_test, results, f'gdkl_v3_seed{seed}_beta{beta}.csv')
    
    print(f"\n總結: 異常點={results['outliers_20']}, MAPE={results['mape']:.2f}%, "
          f"Type3異常={results['type3_outliers']}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    main(args.seed, args.beta, True)
    
    print("\n建議嘗試:")
    print("  python gdkl_thermal_v3.py --beta 0.5")
    print("  python gdkl_thermal_v3.py --beta 1.5")
    print("  python gdkl_thermal_v3.py --beta 2.0")
