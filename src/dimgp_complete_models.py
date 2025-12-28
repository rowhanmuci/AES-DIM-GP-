"""
完整的DIM-GP變體實作
包含：
1. Deep Kernel Learning (DKL) - Wilson 2016風格
2. Deep Mixture of GP Experts (MoE) - Ultra-fast風格
3. 所有Baseline模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# 1. DEEP KERNEL LEARNING (DKL)
# =====================================================================

class FeatureExtractor(nn.Module):
    """DNN特徵提取器"""
    def __init__(self, input_dim=3, hidden_dims=[64, 32, 16], output_dim=8):
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
        
        # 最後投影到output_dim維度
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x):
        return self.network(x)


class GPRegressionModel(gpytorch.models.ExactGP):
    """GP回歸模型"""
    def __init__(self, train_x, train_y, likelihood, feature_dim=8):
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepKernelLearning:
    """
    Deep Kernel Learning模型
    結合DNN特徵提取和GP回歸
    """
    def __init__(self, input_dim=3, hidden_dims=[64, 32, 16], feature_dim=8, 
                 lr=0.01, epochs=100, batch_size=256):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特徵提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim
        ).to(self.device)
        
        self.likelihood = None
        self.gp_model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.train_losses = []
        
    def fit(self, X, y):
        """訓練模型"""
        print(f"Training Deep Kernel Learning on {len(X)} samples...")
        
        # 標準化
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # 轉換為tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # 第一階段：預訓練特徵提取器
        print("Stage 1: Pre-training feature extractor...")
        self._pretrain_features(X_tensor, y_tensor)
        
        # 提取特徵
        self.feature_extractor.eval()
        with torch.no_grad():
            train_features = self.feature_extractor(X_tensor)
        
        # 第二階段：訓練GP
        print("Stage 2: Training GP layer...")
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = GPRegressionModel(
            train_features, y_tensor, self.likelihood, self.feature_dim
        ).to(self.device)
        
        self._train_gp(train_features, y_tensor)
        
        # 第三階段：聯合微調
        print("Stage 3: Joint fine-tuning...")
        self._joint_finetune(X_tensor, y_tensor)
        
        print("DKL training completed!")
        
    def _pretrain_features(self, X, y, epochs=50):
        """預訓練特徵提取器"""
        # 簡單的MLP預訓練
        predictor = nn.Linear(self.feature_dim, 1).to(self.device)
        
        optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) + list(predictor.parameters()),
            lr=self.lr
        )
        criterion = nn.MSELoss()
        
        self.feature_extractor.train()
        predictor.train()
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                features = self.feature_extractor(batch_x)
                pred = predictor(features).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def _train_gp(self, train_x, train_y, epochs=50):
        """訓練GP層"""
        self.gp_model.train()
        self.likelihood.train()
        
        optimizer = optim.Adam(self.gp_model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    def _joint_finetune(self, X, y, epochs=30):
        """聯合微調特徵提取器和GP"""
        print("  Initializing joint training...")
        
        self.feature_extractor.train()
        
        for epoch in range(epochs):
            # Step 1: 提取當前特徵
            self.feature_extractor.eval()
            with torch.no_grad():
                current_features = self.feature_extractor(X)
            
            # Step 2: 用當前特徵重新訓練GP
            self.gp_model = GPRegressionModel(
                current_features, y, self.likelihood, self.feature_dim
            ).to(self.device)
            self.gp_model.train()
            self.likelihood.train()
            
            gp_optimizer = optim.Adam(self.gp_model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
            
            # 快速訓練GP幾步
            for _ in range(5):
                gp_optimizer.zero_grad()
                output = self.gp_model(current_features)
                gp_loss = -mll(output, y)
                gp_loss.backward()
                gp_optimizer.step()
            
            # Step 3: 更新特徵提取器
            self.feature_extractor.train()
            self.gp_model.eval()
            
            feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=self.lr * 0.1)
            
            feature_optimizer.zero_grad()
            features = self.feature_extractor(X)
            
            # 用detach來避免梯度回傳問題
            with gpytorch.settings.fast_pred_var():
                output = self.gp_model(features)
                loss = -mll(output, y)
            
            loss.backward()
            feature_optimizer.step()
            
            self.train_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    def predict(self, X, return_std=False):
        """預測"""
        self.feature_extractor.eval()
        self.gp_model.eval()
        self.likelihood.eval()
        
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            features = self.feature_extractor(X_tensor)
            pred_dist = self.likelihood(self.gp_model(features))
            
            mean = pred_dist.mean.cpu().numpy()
            variance = pred_dist.variance.cpu().numpy()
        
        # 反標準化
        mean = self.scaler_y.inverse_transform(mean.reshape(-1, 1)).ravel()
        std = np.sqrt(variance) * self.scaler_y.scale_[0]
        
        if return_std:
            return mean, std
        return mean


# =====================================================================
# 2. DEEP MIXTURE OF GP EXPERTS (MoE)
# =====================================================================

class DeepMixtureGPExperts:
    """
    Deep Mixture of GP Experts
    使用DNN做gating network，Sparse GP做experts
    """
    def __init__(self, n_experts=3, n_inducing=100, hidden_dims=(32, 16)):
        self.n_experts = n_experts
        self.n_inducing = n_inducing
        self.hidden_dims = hidden_dims
        
        # DNN Gating Network
        self.gating_network = MLPClassifier(
            hidden_layer_sizes=hidden_dims,
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Sparse GP Experts
        self.experts = []
        self.expert_scalers = []
        
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        """訓練模型"""
        print(f"Training Deep Mixture of GP Experts ({self.n_experts} experts)...")
        
        # 標準化
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Step 1: Cluster (K-means)
        print("Step 1: Clustering data...")
        kmeans = KMeans(n_clusters=self.n_experts, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        print(f"Cluster sizes: {[np.sum(clusters == i) for i in range(self.n_experts)]}")
        
        # Step 2: Train DNN Gating Network
        print("Step 2: Training DNN gating network...")
        self.gating_network.fit(X_scaled, clusters)
        gating_acc = self.gating_network.score(X_scaled, clusters)
        print(f"  Gating network accuracy: {gating_acc:.4f}")
        
        # Step 3: Train GP Experts
        print("Step 3: Training GP experts...")
        for expert_id in range(self.n_experts):
            mask = clusters == expert_id
            n_samples = np.sum(mask)
            
            if n_samples < 10:
                print(f"  Expert {expert_id}: Too few samples ({n_samples}), skipping...")
                self.experts.append(None)
                self.expert_scalers.append(None)
                continue
            
            X_expert = X_scaled[mask]
            y_expert = y_scaled[mask]
            
            # Subsample if too many points
            if len(X_expert) > self.n_inducing:
                indices = np.random.choice(len(X_expert), self.n_inducing, replace=False)
                X_expert = X_expert[indices]
                y_expert = y_expert[indices]
            
            # Train GP expert
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=3,
                random_state=42,
                alpha=1e-6
            )
            
            try:
                gp.fit(X_expert, y_expert)
                self.experts.append(gp)
                print(f"  Expert {expert_id}: Trained on {len(X_expert)} samples")
            except Exception as e:
                print(f"  Expert {expert_id}: Training failed - {e}")
                self.experts.append(None)
        
        print("Deep Mixture of GP Experts training completed!")
    
    def predict(self, X, return_std=False):
        """預測 - 使用soft gating"""
        X_scaled = self.scaler_x.transform(X)
        
        # Get gating probabilities
        gate_probs = self.gating_network.predict_proba(X_scaled)
        
        # Weighted predictions from all experts
        predictions = np.zeros(len(X))
        variances = np.zeros(len(X))
        
        for expert_id, expert in enumerate(self.experts):
            if expert is None:
                continue
            
            try:
                pred, std = expert.predict(X_scaled, return_std=True)
                weights = gate_probs[:, expert_id]
                
                # Weighted mean and variance (law of total variance)
                predictions += weights * pred
                variances += weights * (std**2 + pred**2)
            except Exception as e:
                print(f"Warning: Expert {expert_id} prediction failed - {e}")
                continue
        
        # Correct variance
        variances -= predictions**2
        variances = np.maximum(variances, 1e-10)  # 避免負值
        
        # 反標準化
        predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()
        std = np.sqrt(variances) * self.scaler_y.scale_[0]
        
        if return_std:
            return predictions, std
        return predictions


# =====================================================================
# 3. BASELINE MODELS
# =====================================================================

class MLPModel:
    """MLP Baseline"""
    def __init__(self, hidden_dims=(64, 32, 16)):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            activation='relu',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            batch_size=256,
            learning_rate_init=0.001
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.model.fit(X_scaled, y_scaled)
    
    def predict(self, X, return_std=False):
        X_scaled = self.scaler_x.transform(X)
        pred = self.model.predict(X_scaled)
        pred = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
        
        if return_std:
            # MLP沒有不確定性，返回0
            return pred, np.zeros_like(pred)
        return pred


class XGBoostModel:
    """XGBoost Baseline"""
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X, return_std=False):
        pred = self.model.predict(X)
        
        if return_std:
            # XGBoost沒有不確定性，返回0
            return pred, np.zeros_like(pred)
        return pred


class StandardGP:
    """標準GP Baseline"""
    def __init__(self, subsample=1000):
        self.subsample = subsample
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            alpha=1e-6
        )
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Subsample if needed
        if len(X_scaled) > self.subsample:
            indices = np.random.choice(len(X_scaled), self.subsample, replace=False)
            X_scaled = X_scaled[indices]
            y_scaled = y_scaled[indices]
        
        self.model.fit(X_scaled, y_scaled)
    
    def predict(self, X, return_std=False):
        X_scaled = self.scaler_x.transform(X)
        pred, std = self.model.predict(X_scaled, return_std=True)
        
        pred = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
        std = std * self.scaler_y.scale_[0]
        
        if return_std:
            return pred, std
        return pred


# =====================================================================
# 4. ENSEMBLE MODEL
# =====================================================================

class EnsembleModel:
    """Ensemble: MLP + XGBoost + GP for uncertainty"""
    def __init__(self, mlp_weight=0.5, xgb_weight=0.5):
        self.mlp = MLPModel()
        self.xgb = XGBoostModel()
        self.gp = StandardGP(subsample=500)
        self.mlp_weight = mlp_weight
        self.xgb_weight = xgb_weight
    
    def fit(self, X, y):
        print("Training Ensemble Model...")
        print("  Training MLP...")
        self.mlp.fit(X, y)
        print("  Training XGBoost...")
        self.xgb.fit(X, y)
        print("  Training GP for uncertainty...")
        self.gp.fit(X, y)
        print("Ensemble training completed!")
    
    def predict(self, X, return_std=False):
        pred_mlp = self.mlp.predict(X)
        pred_xgb = self.xgb.predict(X)
        
        # Ensemble prediction
        pred = self.mlp_weight * pred_mlp + self.xgb_weight * pred_xgb
        
        if return_std:
            # Use GP for uncertainty
            _, std = self.gp.predict(X, return_std=True)
            return pred, std
        return pred


# =====================================================================
# 5. MODEL FACTORY
# =====================================================================

def get_model(model_name, **kwargs):
    """模型工廠函數"""
    models = {
        'MLP': MLPModel,
        'XGBoost': XGBoostModel,
        'GP': StandardGP,
        'DKL': DeepKernelLearning,
        'MoE': DeepMixtureGPExperts,
        'Ensemble': EnsembleModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    print("DIM-GP Complete Models Module")
    print("Available models:", ['MLP', 'XGBoost', 'GP', 'DKL', 'MoE', 'Ensemble'])