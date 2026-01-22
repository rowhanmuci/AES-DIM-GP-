"""
GDKL V4 - 修復數值爆炸問題

主要改進:
1. 損失函數裁剪，避免極端值
2. 先預訓練 DKL (用標準 MLL)，再加入 NNGP guidance
3. 漸進式增加 beta
4. 更保守的初始化

使用:
    python gdkl_thermal_v4.py --seed 2024 --beta 1.0
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


# ==========================================
# 穩定的矩陣運算
# ==========================================

def safe_cholesky(K: torch.Tensor) -> torch.Tensor:
    """安全的 Cholesky 分解"""
    n = K.shape[0]
    jitter = 1e-6
    
    for _ in range(8):
        try:
            return torch.linalg.cholesky(K + jitter * torch.eye(n, device=K.device, dtype=K.dtype))
        except:
            jitter *= 10
    
    # 最後手段：eigenvalue 修復
    eigval, eigvec = torch.linalg.eigh(K)
    eigval = eigval.clamp(min=1e-5)
    K_fixed = eigvec @ torch.diag(eigval) @ eigvec.T
    return torch.linalg.cholesky(K_fixed + 1e-5 * torch.eye(n, device=K.device, dtype=K.dtype))


# ==========================================
# NNGP
# ==========================================

class NNGPKernel:
    def __init__(self, sigma_w: float = 1.0, sigma_b: float = 0.05, num_layers: int = 2):
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.num_layers = num_layers
    
    def _relu_exp(self, k11, k12, k22):
        eps = 1e-8
        denom = torch.sqrt((k11 * k22).clamp(min=eps))
        rho = (k12 / denom).clamp(-1 + eps, 1 - eps)
        theta = torch.acos(rho)
        return (1 / (2 * np.pi)) * denom * (torch.sin(theta) + (np.pi - theta) * rho)
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        n1, d = X1.shape
        n2 = X2.shape[0]
        
        K = self.sigma_b**2 + self.sigma_w**2 * (X1 @ X2.T) / d
        K11 = self.sigma_b**2 + self.sigma_w**2 * (X1**2).sum(1) / d
        K22 = self.sigma_b**2 + self.sigma_w**2 * (X2**2).sum(1) / d
        
        for _ in range(1, self.num_layers):
            K11_e = K11.unsqueeze(1).expand(n1, n2)
            K22_e = K22.unsqueeze(0).expand(n1, n2)
            K = self.sigma_b**2 + self.sigma_w**2 * self._relu_exp(K11_e, K, K22_e)
            K11 = self.sigma_b**2 + self.sigma_w**2 * self._relu_exp(K11, K11, K11)
            K22 = self.sigma_b**2 + self.sigma_w**2 * self._relu_exp(K22, K22, K22)
        
        return K


# ==========================================
# Deep Kernel GP
# ==========================================

class FeatureNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class DKLModel(nn.Module):
    def __init__(self, in_dim, hidden, feat_dim, dropout=0.1):
        super().__init__()
        self.feat = FeatureNet(in_dim, hidden, feat_dim, dropout)
        
        # 合理的初始值
        self.log_ls = nn.Parameter(torch.zeros(feat_dim))  # lengthscale ~ 1
        self.log_os = nn.Parameter(torch.tensor(-0.5))     # outputscale ~ 0.6
        self.log_noise = nn.Parameter(torch.tensor(-2.0))  # noise ~ 0.135
    
    @property
    def ls(self):
        return torch.exp(self.log_ls).clamp(0.1, 5.0)
    
    @property
    def os(self):
        return torch.exp(self.log_os).clamp(0.01, 5.0)
    
    @property
    def noise(self):
        return torch.exp(self.log_noise).clamp(0.01, 0.5)
    
    def kernel(self, X1, X2):
        Z1 = self.feat(X1)
        Z2 = self.feat(X2)
        
        # 標準化特徵
        Z1 = (Z1 - Z1.mean(0)) / (Z1.std(0) + 1e-6)
        Z2 = (Z2 - Z2.mean(0)) / (Z2.std(0) + 1e-6)
        
        dist = torch.cdist(Z1 / self.ls, Z2 / self.ls, p=2)**2
        return self.os * torch.exp(-0.5 * dist)
    
    def forward(self, X_tr, y_tr, X_te):
        """GP 預測，返回 mean 和 var"""
        n = X_tr.shape[0]
        
        K = self.kernel(X_tr, X_tr)
        K_noisy = K + self.noise * torch.eye(n, device=X_tr.device, dtype=X_tr.dtype)
        
        K_s = self.kernel(X_te, X_tr)
        
        L = safe_cholesky(K_noisy)
        alpha = torch.cholesky_solve(y_tr.unsqueeze(-1), L).squeeze(-1)
        
        mean = K_s @ alpha
        
        v = torch.linalg.solve_triangular(L, K_s.T, upper=False)
        var = self.os - (v**2).sum(0)
        var = var.clamp(min=1e-4)
        
        return mean, var
    
    def mll(self, X, y):
        """Marginal log-likelihood"""
        n = X.shape[0]
        K = self.kernel(X, X)
        K_n = K + self.noise * torch.eye(n, device=X.device, dtype=X.dtype)
        
        L = safe_cholesky(K_n)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        
        log_det = 2 * torch.log(torch.diag(L) + 1e-8).sum()
        quad = (y * alpha).sum()
        
        return -0.5 * (quad + log_det + n * np.log(2 * np.pi))


# ==========================================
# GDKL Losses (穩定版)
# ==========================================

def safe_kl(mu_q, var_q, mu_p, var_p, max_val=100.0):
    """
    穩定的 KL divergence，帶裁剪
    """
    eps = 1e-6
    var_q = var_q.clamp(min=eps)
    var_p = var_p.clamp(min=eps)
    
    # 標準 KL
    kl = 0.5 * (torch.log(var_p / var_q + eps) + 
                (var_q + (mu_q - mu_p)**2) / var_p - 1)
    
    # 裁剪極端值
    kl = kl.clamp(max=max_val)
    
    return kl


def safe_ell(y, mu, var, noise, max_val=100.0):
    """
    穩定的 Expected Log-Likelihood
    """
    eps = 1e-6
    var = var.clamp(min=eps)
    noise = noise.clamp(min=eps)
    
    ell = 0.5 * (np.log(2 * np.pi) + torch.log(noise) + 
                 ((y - mu)**2 + var) / noise)
    
    ell = ell.clamp(max=max_val)
    
    return ell


def sample_weights(X, factor=3.0):
    w = np.ones(len(X))
    mask = (X[:, 0] == 3) & (X[:, 2] >= 0.75) & (X[:, 1] >= 200)
    w[mask] *= factor
    return w


# ==========================================
# Trainer
# ==========================================

class GDKLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.X_tr = None
        self.y_tr = None
    
    def train(self, X, y, verbose=True):
        cfg = self.cfg
        
        # 標準化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        X_s = self.scaler_x.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.X_tr = torch.from_numpy(X_s).to(device)
        self.y_tr = torch.from_numpy(y_s).to(device)
        
        weights = torch.from_numpy(sample_weights(X, cfg['weight_factor'])).to(device)
        n = len(X)
        
        if verbose:
            print(f"\n樣本: {n}, 困難: {(weights > 1).sum().item()}")
        
        # NNGP kernel (預計算)
        if verbose:
            print(f"[1] 計算 NNGP kernel...")
        
        nngp = NNGPKernel(cfg['nngp_sw'], cfg['nngp_sb'], cfg['nngp_layers'])
        self.nngp_K = nngp(self.X_tr)
        
        # DKL 模型
        if verbose:
            print(f"[2] 初始化 DKL...")
        
        self.model = DKLModel(
            self.X_tr.shape[1], cfg['hidden'], cfg['feat_dim'], cfg['dropout']
        ).to(device)
        
        opt = optim.Adam(self.model.parameters(), lr=cfg['lr'])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
        
        # ==========================================
        # Phase 1: 預訓練 DKL (只用 MLL)
        # ==========================================
        if verbose:
            print(f"[3] Phase 1: DKL 預訓練 ({cfg['pretrain_epochs']} epochs)...")
        
        self.model.train()
        for ep in range(cfg['pretrain_epochs']):
            opt.zero_grad()
            loss = -self.model.mll(self.X_tr, self.y_tr) / n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            
            if verbose and (ep + 1) % 50 == 0:
                print(f"    Pretrain {ep+1}: MLL Loss = {loss.item():.4f}")
        
        # ==========================================
        # Phase 2: GDKL 訓練 (漸進式增加 beta)
        # ==========================================
        if verbose:
            print(f"[4] Phase 2: GDKL 訓練 (target β={cfg['beta']})...")
        
        best_loss = float('inf')
        patience = 0
        best_state = None
        
        for ep in range(cfg['epochs']):
            self.model.train()
            opt.zero_grad()
            
            # 漸進式 beta (前 100 epochs 慢慢增加)
            warmup = min(1.0, ep / 100.0)
            current_beta = cfg['beta'] * warmup
            
            # 隨機分割
            perm = torch.randperm(n)
            n1 = int(n * 0.7)
            i1, i2 = perm[:n1], perm[n1:]
            
            X1, y1 = self.X_tr[i1], self.y_tr[i1]
            X2, y2 = self.X_tr[i2], self.y_tr[i2]
            w2 = weights[i2]
            
            # NNGP 預測
            K11 = self.nngp_K[i1][:, i1]
            K21 = self.nngp_K[i2][:, i1]
            K22_d = torch.diag(self.nngp_K)[i2]
            
            K11_n = K11 + cfg['nngp_noise'] * torch.eye(n1, device=device, dtype=self.X_tr.dtype)
            L_nngp = safe_cholesky(K11_n)
            
            alpha_nngp = torch.cholesky_solve(y1.unsqueeze(-1), L_nngp).squeeze(-1)
            nngp_mu = K21 @ alpha_nngp
            
            v_nngp = torch.linalg.solve_triangular(L_nngp, K21.T, upper=False)
            nngp_var = (K22_d - (v_nngp**2).sum(0)).clamp(min=1e-4)
            
            # DKL 預測
            dkl_mu, dkl_var = self.model(X1, y1, X2)
            
            # 損失
            loss_ell = safe_ell(y2, dkl_mu, dkl_var, self.model.noise)
            loss_kl = safe_kl(dkl_mu, dkl_var, nngp_mu.detach(), nngp_var.detach())
            
            gdkl = (loss_ell * w2).mean() + current_beta * (loss_kl * w2).mean()
            
            # MLL 正則
            mll_loss = -self.model.mll(self.X_tr, self.y_tr) / n
            
            total = gdkl + cfg['mll_weight'] * mll_loss
            
            if torch.isnan(total) or torch.isinf(total):
                continue
            
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            
            lv = total.item()
            
            if verbose and (ep + 1) % 100 == 0:
                print(f"    Epoch {ep+1}: ELL={loss_ell.mean().item():.3f}, "
                      f"KL={loss_kl.mean().item():.3f}, β={current_beta:.2f}, Total={lv:.4f}")
            
            if lv < best_loss:
                best_loss = lv
                patience = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience += 1
            
            if patience >= cfg['patience']:
                if verbose:
                    print(f"    早停 @ {ep+1}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        if verbose:
            print(f"\n完成 (Best: {best_loss:.4f})")
    
    def predict(self, X):
        self.model.eval()
        X_s = torch.from_numpy(self.scaler_x.transform(X)).to(device)
        
        with torch.no_grad():
            mu, var = self.model(self.X_tr, self.y_tr, X_s)
        
        pred = self.scaler_y.inverse_transform(mu.cpu().numpy().reshape(-1, 1)).flatten()
        std = torch.sqrt(var).cpu().numpy() * self.scaler_y.scale_[0]
        
        return pred, std


# ==========================================
# Evaluate
# ==========================================

def evaluate(trainer, X, y, verbose=True):
    pred, std = trainer.predict(X)
    err = np.abs((y - pred) / y) * 100
    
    out20 = (err > 20).sum()
    out15 = (err > 15).sum()
    t3 = X[:, 0] == 3
    t3_out = ((err > 20) & t3).sum()
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"結果: MAPE={err.mean():.2f}%")
        print(f"異常點 >20%: {out20}/{len(y)} ({out20/len(y)*100:.1f}%)")
        print(f"異常點 >15%: {out15}/{len(y)}")
        print(f"Type3 異常: {t3_out}/{t3.sum()}")
        print(f"\nTop 5 誤差:")
        for i in np.argsort(err)[-5:][::-1]:
            print(f"  T{X[i,0]:.0f} Th{X[i,1]:.0f} C{X[i,2]:.1f}: "
                  f"True={y[i]:.4f} Pred={pred[i]:.4f} Err={err[i]:.1f}%")
        print(f"{'='*50}")
    
    return {'mape': err.mean(), 'outliers_20': out20, 'outliers_15': out15,
            'type3_outliers': t3_out, 'predictions': pred, 'errors': err, 'std': std}


# ==========================================
# Main
# ==========================================

def main(seed=2024, beta=1.0):
    set_seed(seed)
    print(f"\n裝置: {device}")
    print("="*50)
    print("GDKL V4")
    print("="*50)
    
    cfg = {
        'hidden': [64, 32],
        'feat_dim': 8,
        'dropout': 0.1,
        'lr': 0.005,
        'pretrain_epochs': 150,  # DKL 預訓練
        'epochs': 500,
        'patience': 80,
        'beta': beta,
        'mll_weight': 0.3,
        'nngp_sw': 1.0,
        'nngp_sb': 0.05,
        'nngp_layers': 2,
        'nngp_noise': 0.1,
        'weight_factor': 3.0,
    }
    
    print(f"\nβ={beta}")
    
    # 資料
    train_df = pd.read_excel('data/train/Above.xlsx')
    test_df = pd.read_excel('data/test/Above.xlsx')
    
    cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    train_df = train_df.groupby(cols, as_index=False).agg({'Theta.JC': 'mean'})
    
    X_tr, y_tr = train_df[cols].values, train_df['Theta.JC'].values
    X_te, y_te = test_df[cols].values, test_df['Theta.JC'].values
    
    print(f"訓練: {len(X_tr)}, 測試: {len(X_te)}")
    
    # 訓練
    trainer = GDKLTrainer(cfg)
    trainer.train(X_tr, y_tr)
    
    # 評估
    res = evaluate(trainer, X_te, y_te)
    
    # 保存
    df = pd.DataFrame({
        'TIM_TYPE': X_te[:, 0], 'TIM_THICKNESS': X_te[:, 1], 'TIM_COVERAGE': X_te[:, 2],
        'TRUE': y_te, 'Predicted': res['predictions'], 'Error%': res['errors']
    })
    df.to_csv(f'gdkl_v4_seed{seed}_beta{beta}.csv', index=False)
    print(f"\n✓ 保存: gdkl_v4_seed{seed}_beta{beta}.csv")
    
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    
    main(args.seed, args.beta)
