"""
GDKL (Guided Deep Kernel Learning) for Thermal Resistance Prediction - V2
基於論文: "Guided Deep Kernel Learning" (UAI 2023)

修正版本: 解決 GPyTorch ExactGP 的訓練模式限制

需要安裝的套件:
    pip install torch gpytorch pandas numpy scikit-learn openpyxl

使用方法:
    python gdkl_thermal_v2.py --seed 2024
    python gdkl_thermal_v2.py --seed 2024 --beta 1.0 -v
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from sklearn.preprocessing import StandardScaler
import warnings
import random
import os
import argparse
from typing import Tuple, Dict, Optional

warnings.filterwarnings('ignore')

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 工具函數
# ==========================================

def set_seed(seed: int):
    """設置隨機種子以確保可重現性"""
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
# NNGP Kernel 實作
# ==========================================

class NNGPKernel:
    """
    Neural Network Gaussian Process Kernel
    
    對於 ReLU 激活函數，有解析解 (Cho & Saul, 2009)
    """
    
    def __init__(self, sigma_w: float = 1.5, sigma_b: float = 0.1, num_layers: int = 3):
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.num_layers = num_layers
    
    def _relu_kernel_expectation(self, k11: torch.Tensor, k12: torch.Tensor, 
                                  k22: torch.Tensor) -> torch.Tensor:
        """計算 ReLU 激活函數的 kernel expectation"""
        eps = 1e-8
        denom = torch.sqrt(k11 * k22 + eps)
        rho = torch.clamp(k12 / denom, -1 + eps, 1 - eps)
        theta = torch.acos(rho)
        result = (1 / (2 * np.pi)) * denom * (torch.sin(theta) + (np.pi - theta) * rho)
        return result
    
    def compute_kernel(self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """計算 NNGP kernel matrix"""
        if X2 is None:
            X2 = X1
        
        n1, d = X1.shape
        n2 = X2.shape[0]
        
        # 第一層
        K12 = self.sigma_b ** 2 + self.sigma_w ** 2 * (X1 @ X2.T) / d
        K11_diag = self.sigma_b ** 2 + self.sigma_w ** 2 * (X1 ** 2).sum(dim=1) / d
        K22_diag = self.sigma_b ** 2 + self.sigma_w ** 2 * (X2 ** 2).sum(dim=1) / d
        
        # 遞迴計算後續層
        for layer in range(1, self.num_layers):
            K11 = K11_diag.unsqueeze(1).expand(n1, n2)
            K22 = K22_diag.unsqueeze(0).expand(n1, n2)
            K12_new = self._relu_kernel_expectation(K11, K12, K22)
            K12 = self.sigma_b ** 2 + self.sigma_w ** 2 * K12_new
            K11_diag_new = self._relu_kernel_expectation(K11_diag, K11_diag, K11_diag)
            K22_diag_new = self._relu_kernel_expectation(K22_diag, K22_diag, K22_diag)
            K11_diag = self.sigma_b ** 2 + self.sigma_w ** 2 * K11_diag_new
            K22_diag = self.sigma_b ** 2 + self.sigma_w ** 2 * K22_diag_new
        
        return K12


class NNGPModel:
    """NNGP 模型 - 用於計算 p(f*|x*, D1) 作為 GDKL 的 guide"""
    
    def __init__(self, sigma_w: float = 1.5, sigma_b: float = 0.1, 
                 num_layers: int = 3, noise_var: float = 0.1):
        self.kernel = NNGPKernel(sigma_w, sigma_b, num_layers)
        self.noise_var = noise_var
        self.K_train = None
        self.K_train_inv = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """擬合 NNGP 模型"""
        self.X_train = X
        self.y_train = y
        self.K_train = self.kernel.compute_kernel(X)
        n = X.shape[0]
        K_noisy = self.K_train + self.noise_var * torch.eye(n, device=X.device, dtype=X.dtype)
        
        # 添加 jitter 確保數值穩定
        jitter = 1e-6
        try:
            L = torch.linalg.cholesky(K_noisy + jitter * torch.eye(n, device=X.device, dtype=X.dtype))
            self.K_train_inv = torch.cholesky_inverse(L)
        except:
            self.K_train_inv = torch.linalg.inv(K_noisy + jitter * torch.eye(n, device=X.device, dtype=X.dtype))
    
    def predict(self, X_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """預測 p(f*|x*, D_train)"""
        K_star = self.kernel.compute_kernel(X_test, self.X_train)
        K_star_star_diag = torch.diag(self.kernel.compute_kernel(X_test))
        mean = K_star @ self.K_train_inv @ self.y_train
        var = K_star_star_diag - torch.sum(K_star @ self.K_train_inv * K_star, dim=1)
        var = torch.clamp(var, min=1e-6)
        return mean, var


# ==========================================
# DKL 模型定義 (直接計算 GP 預測，不依賴 GPyTorch 的 ExactGP)
# ==========================================

class DnnFeatureExtractor(nn.Module):
    """深度神經網路特徵提取器"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16], 
                 output_dim: int = 8, dropout: float = 0.1):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepKernelGP(nn.Module):
    """
    Deep Kernel GP - 直接實作 GP 預測
    避開 GPyTorch ExactGP 的訓練限制
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, feature_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_extractor = DnnFeatureExtractor(
            input_dim, hidden_dims, feature_dim, dropout
        )
        
        # RBF kernel 參數 (可學習)
        self.log_lengthscale = nn.Parameter(torch.zeros(feature_dim))
        self.log_outputscale = nn.Parameter(torch.zeros(1))
        self.log_noise = nn.Parameter(torch.tensor(-2.0))  # 初始 noise ~ 0.135
        
        # 訓練數據緩存
        self.train_x = None
        self.train_y = None
        self.K_inv = None
    
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)
    
    @property
    def outputscale(self):
        return torch.exp(self.log_outputscale)
    
    @property
    def noise(self):
        return torch.exp(self.log_noise)
    
    def _compute_kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """計算 RBF kernel with ARD"""
        # 特徵提取
        Z1 = self.feature_extractor(X1)
        Z2 = self.feature_extractor(X2)
        
        # Scaled by lengthscale
        Z1_scaled = Z1 / self.lengthscale
        Z2_scaled = Z2 / self.lengthscale
        
        # 計算距離
        dist_sq = torch.cdist(Z1_scaled, Z2_scaled, p=2) ** 2
        
        # RBF kernel
        K = self.outputscale * torch.exp(-0.5 * dist_sq)
        
        return K
    
    def set_train_data(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """設置訓練數據"""
        self.train_x = train_x
        self.train_y = train_y
    
    def compute_loss_and_prediction(self, X_test: torch.Tensor, 
                                     y_test: Optional[torch.Tensor] = None,
                                     X_train: Optional[torch.Tensor] = None,
                                     y_train: Optional[torch.Tensor] = None) -> Dict:
        """
        計算預測分布和損失
        
        Returns:
            dict with: mean, var, nll (if y_test provided)
        """
        if X_train is None:
            X_train = self.train_x
            y_train = self.train_y
        
        n_train = X_train.shape[0]
        
        # 計算 kernel matrices
        K_train = self._compute_kernel(X_train, X_train)
        K_train_noisy = K_train + self.noise * torch.eye(n_train, device=X_train.device, dtype=X_train.dtype)
        
        # 添加 jitter
        jitter = 1e-5 * torch.eye(n_train, device=X_train.device, dtype=X_train.dtype)
        K_train_noisy = K_train_noisy + jitter
        
        # Cholesky 分解
        try:
            L = torch.linalg.cholesky(K_train_noisy)
        except:
            # Fallback: 加更多 jitter
            K_train_noisy = K_train_noisy + 1e-4 * torch.eye(n_train, device=X_train.device, dtype=X_train.dtype)
            L = torch.linalg.cholesky(K_train_noisy)
        
        # K_test_train
        K_test_train = self._compute_kernel(X_test, X_train)
        
        # K_test_test diagonal
        K_test_test_diag = self.outputscale * torch.ones(X_test.shape[0], device=X_test.device, dtype=X_test.dtype)
        
        # 預測均值: K_*  K^{-1} y = K_* @ (L^{-T} @ L^{-1} @ y)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)
        mean = K_test_train @ alpha
        
        # 預測方差: K_** - K_* @ K^{-1} @ K_*^T
        v = torch.linalg.solve_triangular(L, K_test_train.T, upper=False)
        var = K_test_test_diag - torch.sum(v ** 2, dim=0)
        var = torch.clamp(var, min=1e-6)
        
        result = {'mean': mean, 'var': var}
        
        # 計算 NLL (用於標準 GP 訓練)
        if y_test is not None:
            # Predictive NLL
            pred_nll = 0.5 * torch.log(2 * np.pi * (var + self.noise)) + \
                       0.5 * (y_test - mean) ** 2 / (var + self.noise)
            result['nll'] = pred_nll.mean()
        
        return result
    
    def compute_mll(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """計算 marginal log-likelihood"""
        n = X.shape[0]
        K = self._compute_kernel(X, X)
        K_noisy = K + self.noise * torch.eye(n, device=X.device, dtype=X.dtype)
        K_noisy = K_noisy + 1e-5 * torch.eye(n, device=X.device, dtype=X.dtype)
        
        try:
            L = torch.linalg.cholesky(K_noisy)
        except:
            K_noisy = K_noisy + 1e-4 * torch.eye(n, device=X.device, dtype=X.dtype)
            L = torch.linalg.cholesky(K_noisy)
        
        # log|K| = 2 * sum(log(diag(L)))
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        
        # y^T K^{-1} y
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        quad_form = torch.dot(y, alpha)
        
        # MLL = -0.5 * (y^T K^{-1} y + log|K| + n*log(2π))
        mll = -0.5 * (quad_form + log_det + n * np.log(2 * np.pi))
        
        return mll


# ==========================================
# GDKL 損失函數
# ==========================================

def gaussian_kl_divergence(mu_q: torch.Tensor, var_q: torch.Tensor,
                           mu_p: torch.Tensor, var_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) for Gaussians - 論文 Eq. 14"""
    eps = 1e-8
    std_q = torch.sqrt(var_q + eps)
    std_p = torch.sqrt(var_p + eps)
    kl = torch.log(std_p / std_q + eps) + (var_q + (mu_q - mu_p) ** 2) / (2 * var_p + eps) - 0.5
    return kl


def expected_log_likelihood(y: torch.Tensor, mu_q: torch.Tensor, 
                            var_q: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    """E_q[log p(y|f)] - 論文 Eq. 15"""
    ell = 0.5 * (np.log(2 * np.pi) + torch.log(noise_var) + 
                 ((y - mu_q) ** 2 + var_q) / noise_var)
    return ell


def compute_sample_weights(X: np.ndarray, weight_factor: float = 3.0) -> np.ndarray:
    """計算樣本權重"""
    weights = np.ones(len(X))
    difficult_mask = (
        (X[:, 0] == 3) &      
        (X[:, 2] >= 0.75) &   
        (X[:, 1] >= 200)      
    )
    weights[difficult_mask] *= weight_factor
    return weights


# ==========================================
# GDKL 訓練器
# ==========================================

class GDKLTrainer:
    """Guided Deep Kernel Learning 訓練器 - V2"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.nngp_model = None
        self.dkl_model = None
        self.scaler_x = None
        self.scaler_y = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              verbose: bool = True) -> None:
        """訓練 GDKL 模型"""
        config = self.config
        
        # 計算樣本權重
        sample_weights_np = compute_sample_weights(X_train, config['sample_weight_factor'])
        
        if verbose:
            difficult_count = np.sum(sample_weights_np > 1.0)
            print(f"\n樣本權重:")
            print(f"  困難樣本數: {difficult_count} ({difficult_count/len(X_train)*100:.2f}%)")
        
        # 標準化
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        train_x = torch.from_numpy(X_train_scaled).to(device)
        train_y = torch.from_numpy(y_train_scaled).to(device)
        sample_weights = torch.from_numpy(sample_weights_np).to(device)
        
        # ==========================================
        # Step 1: 初始化 NNGP Guide (預計算全部 kernel)
        # ==========================================
        if verbose:
            print(f"\n[Step 1] 初始化 NNGP Guide...")
        
        self.nngp_model = NNGPModel(
            sigma_w=config['nngp_sigma_w'],
            sigma_b=config['nngp_sigma_b'],
            num_layers=config['nngp_layers'],
            noise_var=config['nngp_noise_var']
        )
        
        # 預計算完整的 NNGP kernel (用於後續快速提取子矩陣)
        self.nngp_full_kernel = self.nngp_model.kernel.compute_kernel(train_x)
        
        if verbose:
            print(f"  NNGP 參數: σ_w={config['nngp_sigma_w']}, σ_b={config['nngp_sigma_b']}, "
                  f"layers={config['nngp_layers']}")
        
        # ==========================================
        # Step 2: 初始化 DKL 模型
        # ==========================================
        if verbose:
            print(f"\n[Step 2] 初始化 DKL 模型...")
        
        self.dkl_model = DeepKernelGP(
            input_dim=train_x.shape[1],
            hidden_dims=config['hidden_dims'],
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        ).to(device)
        
        self.dkl_model.set_train_data(train_x, train_y)
        
        # ==========================================
        # Step 3: GDKL 訓練
        # ==========================================
        if verbose:
            print(f"\n[Step 3] GDKL 訓練 (β={config['beta']})...")
        
        optimizer = optim.Adam(self.dkl_model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        n_samples = train_x.shape[0]
        
        for epoch in range(config['epochs']):
            self.dkl_model.train()
            optimizer.zero_grad()
            
            # 隨機分割數據 D1, D2 (論文核心)
            perm = torch.randperm(n_samples)
            split_idx = int(n_samples * 0.7)
            idx1 = perm[:split_idx]
            idx2 = perm[split_idx:]
            
            X1, y1 = train_x[idx1], train_y[idx1]
            X2, y2 = train_x[idx2], train_y[idx2]
            w2 = sample_weights[idx2]
            
            # ==========================================
            # 計算 NNGP 在 D2 上的預測 (使用 D1 作為訓練)
            # ==========================================
            
            # 從預計算的 kernel 中提取子矩陣
            K_D1 = self.nngp_full_kernel[idx1][:, idx1]
            K_D2_D1 = self.nngp_full_kernel[idx2][:, idx1]
            K_D2_diag = torch.diag(self.nngp_full_kernel)[idx2]
            
            # NNGP 預測
            n1 = len(idx1)
            K_D1_noisy = K_D1 + config['nngp_noise_var'] * torch.eye(n1, device=device, dtype=train_x.dtype)
            K_D1_noisy = K_D1_noisy + 1e-6 * torch.eye(n1, device=device, dtype=train_x.dtype)
            
            try:
                L_nngp = torch.linalg.cholesky(K_D1_noisy)
                alpha_nngp = torch.cholesky_solve(y1.unsqueeze(-1), L_nngp).squeeze(-1)
                v_nngp = torch.linalg.solve_triangular(L_nngp, K_D2_D1.T, upper=False)
            except Exception as e:
                # Fallback
                K_D1_inv = torch.linalg.inv(K_D1_noisy + 1e-4 * torch.eye(n1, device=device, dtype=train_x.dtype))
                alpha_nngp = K_D1_inv @ y1
                v_nngp = K_D1_inv @ K_D2_D1.T
            
            nngp_mean = K_D2_D1 @ alpha_nngp
            nngp_var = K_D2_diag - torch.sum(K_D2_D1 @ torch.linalg.inv(K_D1_noisy) * K_D2_D1, dim=1)
            nngp_var = torch.clamp(nngp_var, min=1e-6)
            
            # ==========================================
            # 計算 DKL 在 D2 上的預測 (使用 D1 作為訓練)
            # ==========================================
            
            dkl_result = self.dkl_model.compute_loss_and_prediction(
                X_test=X2, X_train=X1, y_train=y1
            )
            dkl_mean = dkl_result['mean']
            dkl_var = dkl_result['var']
            
            # ==========================================
            # GDKL Loss (論文 Eq. 7)
            # ==========================================
            
            # Expected Log-Likelihood
            ell_loss = expected_log_likelihood(y2, dkl_mean, dkl_var, self.dkl_model.noise)
            weighted_ell = (ell_loss * w2).sum() / w2.sum()
            
            # KL Divergence
            kl_loss = gaussian_kl_divergence(dkl_mean, dkl_var, nngp_mean, nngp_var)
            weighted_kl = (kl_loss * w2).sum() / w2.sum()
            
            # 標準 MLL (輔助損失，穩定訓練)
            mll_loss = -self.dkl_model.compute_mll(train_x, train_y) / n_samples
            
            # 總損失
            gdkl_loss = weighted_ell + config['beta'] * weighted_kl
            total_loss = gdkl_loss + config['mll_weight'] * mll_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dkl_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            current_loss = total_loss.item()
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: ELL={weighted_ell.item():.4f}, "
                      f"KL={weighted_kl.item():.4f}, MLL={mll_loss.item():.4f}, "
                      f"Total={current_loss:.4f}")
            
            # Early stopping
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                best_state = self.dkl_model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                if verbose:
                    print(f"早停 at Epoch {epoch+1}")
                break
        
        # 載入最佳模型
        if best_state is not None:
            self.dkl_model.load_state_dict(best_state)
        
        if verbose:
            print(f"訓練完成 (Final Loss: {best_loss:.4f})")
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """預測"""
        self.dkl_model.eval()
        
        X_test_scaled = self.scaler_x.transform(X_test)
        test_x = torch.from_numpy(X_test_scaled).to(device)
        
        with torch.no_grad():
            result = self.dkl_model.compute_loss_and_prediction(
                X_test=test_x, 
                X_train=self.dkl_model.train_x,
                y_train=self.dkl_model.train_y
            )
            y_pred_scaled = result['mean'].cpu().numpy()
            y_var_scaled = result['var'].cpu().numpy()
        
        # 反標準化
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_std = np.sqrt(y_var_scaled) * self.scaler_y.scale_[0]
        
        return y_pred, y_std


# ==========================================
# 評估函數
# ==========================================

def evaluate_model(trainer: GDKLTrainer, X_test: np.ndarray, y_test: np.ndarray,
                   verbose: bool = True) -> Dict:
    """評估模型"""
    y_pred, y_std = trainer.predict(X_test)
    
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((relative_errors > 20) & type3_mask)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GDKL 評估結果")
        print(f"{'='*60}")
        print(f"樣本數: {len(y_test)}")
        print(f"\n準確度:")
        print(f"  MAPE:      {mape:.2f}%")
        print(f"  MAE:       {mae:.4f}")
        print(f"  Max Error: {max_error:.2f}%")
        print(f"\n異常點 (Error > 20%):")
        print(f"  總數: {outliers_20}/{len(y_test)} ({outliers_20/len(y_test)*100:.2f}%)")
        print(f"  >15%: {outliers_15}/{len(y_test)}")
        print(f"  >10%: {outliers_10}/{len(y_test)}")
        
        if np.sum(type3_mask) > 0:
            type3_mape = np.mean(relative_errors[type3_mask])
            print(f"\nType 3 分析:")
            print(f"  樣本數: {np.sum(type3_mask)}")
            print(f"  MAPE: {type3_mape:.2f}%")
            print(f"  異常點: {type3_outliers}/{np.sum(type3_mask)}")
        
        print(f"\n最大誤差樣本 (Top 5):")
        worst_idx = np.argsort(relative_errors)[-5:][::-1]
        for idx in worst_idx:
            print(f"  Type={X_test[idx,0]:.0f}, Thick={X_test[idx,1]:.0f}, "
                  f"Cov={X_test[idx,2]:.2f}: True={y_test[idx]:.4f}, "
                  f"Pred={y_pred[idx]:.4f}, Error={relative_errors[idx]:.2f}%")
        
        print(f"{'='*60}\n")
    
    return {
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


def save_predictions(X_test: np.ndarray, y_test: np.ndarray, 
                     results: Dict, filename: str):
    """保存預測結果"""
    df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'TRUE': y_test,
        'Predicted': results['predictions'],
        'Error%': results['errors'],
        'Std': results['std']
    })
    df.to_csv(filename, index=False)
    print(f"✓ 預測結果已保存到: {filename}")


# ==========================================
# 主函數
# ==========================================

def main(seed: int = 2024, beta: float = 1.0, verbose: bool = True):
    """主訓練流程"""
    clear_gpu_cache()
    set_seed(seed)
    
    print(f"\n使用裝置: {device}\n")
    print("="*60)
    print("GDKL (Guided Deep Kernel Learning) V2 - 熱阻預測")
    print("="*60)
    
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # GDKL 配置
    config = {
        # DKL 架構
        'hidden_dims': [64, 32, 16],
        'feature_dim': 8,
        'dropout': 0.1,
        
        # 訓練參數
        'lr': 0.005,
        'epochs': 600,
        'patience': 80,
        
        # GDKL 參數
        'beta': beta,
        'mll_weight': 0.3,
        
        # NNGP 參數
        'nngp_sigma_w': 1.5,
        'nngp_sigma_b': 0.1,
        'nngp_layers': 3,
        'nngp_noise_var': 0.05,
        
        # 樣本加權
        'sample_weight_factor': 3.0,
    }
    
    if verbose:
        print(f"\nGDKL 配置:")
        print(f"  β (KL weight): {config['beta']}")
        print(f"  MLL weight: {config['mll_weight']}")
        print(f"  Learning rate: {config['lr']}")
    
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
    
    X_train = train_above_clean[feature_cols].values
    y_train = train_above_clean[target_col].values
    X_test = test_above[feature_cols].values
    y_test = test_above[target_col].values
    
    # 訓練
    trainer = GDKLTrainer(config)
    trainer.train(X_train, y_train, verbose=verbose)
    
    # 評估
    results = evaluate_model(trainer, X_test, y_test, verbose=verbose)
    
    # 保存
    save_predictions(X_test, y_test, results, 
                     f'gdkl_v2_above_seed{seed}_beta{beta}_predictions.csv')
    
    # ==========================================
    # 總結
    # ==========================================
    
    print("\n" + "="*60)
    print("GDKL V2 結果總結")
    print("="*60)
    print(f"隨機種子: {seed}")
    print(f"β: {beta}")
    print(f"\nAbove資料集:")
    print(f"  異常點 (>20%): {results['outliers_20']}/{len(y_test)} "
          f"({results['outliers_20']/len(y_test)*100:.2f}%)")
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"  Type 3異常點: {results['type3_outliers']}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GDKL V2 熱阻預測')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('--beta', type=float, default=1.0, 
                        help='KL divergence 權重 (0.1-2.0)')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, beta=args.beta, verbose=True)
    
    print("\n💡 調參建議:")
    print(f"   試試不同的 β 值:")
    print(f"   python gdkl_thermal_v2.py --seed 2024 --beta 0.5")
    print(f"   python gdkl_thermal_v2.py --seed 2024 --beta 1.5")
    print(f"   python gdkl_thermal_v2.py --seed 2024 --beta 2.0")