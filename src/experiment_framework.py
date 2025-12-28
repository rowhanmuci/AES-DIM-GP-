"""
完整實驗框架 - 統一評估所有DIM-GP變體
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from dimgp_complete_models import (
    MLPModel, XGBoostModel, StandardGP,
    DeepKernelLearning, DeepMixtureGPExperts, EnsembleModel,
    get_model
)


class ExperimentFramework:
    """統一的實驗框架"""
    
    def __init__(self, dataset_name='Above'):
        self.dataset_name = dataset_name
        self.results = {}
        self.models = {}
        
    def load_data(self, X_train, y_train, X_test, y_test):
        """載入資料"""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\n{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        print(f"{'='*60}\n")
    
    def run_model(self, model_name, model_params=None):
        """訓練和評估單個模型"""
        print(f"\n{'='*60}")
        print(f"Running: {model_name}")
        print(f"{'='*60}")
        
        if model_params is None:
            model_params = {}
        
        # 建立模型
        try:
            model = get_model(model_name, **model_params)
        except Exception as e:
            print(f"Error creating model: {e}")
            return None
        
        # 訓練
        start_time = time.time()
        try:
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            print(f"✓ Training completed in {train_time:.2f}s")
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return None
        
        # 預測
        try:
            start_time = time.time()
            
            # 檢查模型是否支援不確定性估計
            if hasattr(model, 'predict'):
                import inspect
                sig = inspect.signature(model.predict)
                if 'return_std' in sig.parameters:
                    y_pred, y_std = model.predict(self.X_test, return_std=True)
                    has_uncertainty = True
                else:
                    y_pred = model.predict(self.X_test)
                    y_std = np.zeros_like(y_pred)
                    has_uncertainty = False
            else:
                y_pred = model.predict(self.X_test)
                y_std = np.zeros_like(y_pred)
                has_uncertainty = False
            
            pred_time = time.time() - start_time
            print(f"✓ Prediction completed in {pred_time:.2f}s")
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            return None
        
        # 評估指標
        metrics = self._compute_metrics(y_pred, y_std, has_uncertainty)
        
        # 儲存結果
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'std': y_std,
            'train_time': train_time,
            'pred_time': pred_time,
            'has_uncertainty': has_uncertainty
        }
        
        self.models[model_name] = model
        
        # 顯示結果
        self._print_metrics(model_name, metrics, train_time)
        
        return metrics
    
    def _compute_metrics(self, y_pred, y_std, has_uncertainty):
        """計算評估指標"""
        metrics = {}
        
        # 基本指標
        metrics['RMSE'] = np.sqrt(mean_squared_error(self.y_test, y_pred))
        metrics['MAE'] = mean_absolute_error(self.y_test, y_pred)
        metrics['R2'] = r2_score(self.y_test, y_pred)
        metrics['MAPE'] = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # 不確定性指標
        if has_uncertainty and np.any(y_std > 0):
            # 95% 信賴區間
            ci_lower = y_pred - 1.96 * y_std
            ci_upper = y_pred + 1.96 * y_std
            
            # 覆蓋率 (Coverage)
            coverage = np.mean((self.y_test >= ci_lower) & (self.y_test <= ci_upper))
            metrics['CI_Coverage'] = coverage * 100
            
            # 平均CI寬度
            metrics['CI_Width'] = np.mean(ci_upper - ci_lower)
            
            # Calibration: 檢查預測誤差是否與不確定性相符
            errors = np.abs(self.y_test - y_pred)
            metrics['Mean_Error'] = np.mean(errors)
            metrics['Mean_Std'] = np.mean(y_std)
            
            # Negative Log Predictive Density (NLPD)
            # 假設Gaussian分佈
            nlpd = 0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * ((self.y_test - y_pred)**2 / y_std**2)
            metrics['NLPD'] = np.mean(nlpd)
        else:
            metrics['CI_Coverage'] = None
            metrics['CI_Width'] = None
            metrics['Mean_Error'] = None
            metrics['Mean_Std'] = None
            metrics['NLPD'] = None
        
        return metrics
    
    def _print_metrics(self, model_name, metrics, train_time):
        """顯示指標"""
        print(f"\n{'─'*60}")
        print(f"Results for {model_name}:")
        print(f"{'─'*60}")
        print(f"  RMSE:      {metrics['RMSE']:.6f}")
        print(f"  MAE:       {metrics['MAE']:.6f}")
        print(f"  R²:        {metrics['R2']:.6f}")
        print(f"  MAPE:      {metrics['MAPE']:.2f}%")
        print(f"  Time:      {train_time:.2f}s")
        
        if metrics['CI_Coverage'] is not None:
            print(f"\n  Uncertainty Quantification:")
            print(f"  CI Coverage:  {metrics['CI_Coverage']:.2f}%")
            print(f"  CI Width:     {metrics['CI_Width']:.6f}")
            print(f"  Mean Error:   {metrics['Mean_Error']:.6f}")
            print(f"  Mean Std:     {metrics['Mean_Std']:.6f}")
            print(f"  NLPD:         {metrics['NLPD']:.4f}")
        
        print(f"{'─'*60}\n")
    
    def run_all_models(self):
        """執行所有模型"""
        print(f"\n{'#'*60}")
        print(f"# Running Complete Experiment: {self.dataset_name}")
        print(f"{'#'*60}\n")
        
        # 模型配置
        model_configs = {
            'MLP': {},
            'XGBoost': {},
            'GP': {'subsample': 1000},
            'DKL': {
                'input_dim': self.X_train.shape[1],
                'hidden_dims': [64, 32, 16],
                'feature_dim': 8,
                'epochs': 100
            },
            'MoE': {
                'n_experts': 3,
                'n_inducing': 100,
                'hidden_dims': (32, 16)
            },
            'Ensemble': {
                'mlp_weight': 0.5,
                'xgb_weight': 0.5
            }
        }
        
        # 執行所有模型
        for model_name, params in model_configs.items():
            try:
                self.run_model(model_name, params)
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                continue
        
        print(f"\n{'#'*60}")
        print(f"# All Models Completed!")
        print(f"{'#'*60}\n")
    
    def get_summary_table(self):
        """生成結果摘要表"""
        data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R2'],
                'MAPE (%)': metrics['MAPE'],
                'Train Time (s)': result['train_time'],
                'Has UQ': '✓' if result['has_uncertainty'] else '✗'
            }
            
            if metrics['CI_Coverage'] is not None:
                row['CI Coverage (%)'] = metrics['CI_Coverage']
                row['CI Width'] = metrics['CI_Width']
                row['NLPD'] = metrics['NLPD']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 排序：先按R²降序，再按RMSE升序
        if len(df) > 0:
            df = df.sort_values(['R²', 'RMSE'], ascending=[False, True])
        
        return df
    
    def print_summary(self):
        """顯示摘要"""
        df = self.get_summary_table()
        
        print(f"\n{'='*80}")
        print(f"SUMMARY: {self.dataset_name} Dataset")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*80}\n")
        
        # 最佳模型
        if len(df) > 0:
            best_acc = df.iloc[0]['Model']
            print(f"🏆 Best Accuracy: {best_acc} (R²={df.iloc[0]['R²']:.6f})")
            
            if 'CI Coverage (%)' in df.columns:
                uq_models = df[df['Has UQ'] == '✓']
                if len(uq_models) > 0:
                    best_uq = uq_models.iloc[0]['Model']
                    print(f"🎯 Best with UQ: {best_uq}")
        
        return df
    
    def save_results(self, filename):
        """儲存結果"""
        df = self.get_summary_table()
        df.to_csv(filename, index=False)
        print(f"✓ Results saved to {filename}")


def run_complete_experiment(X_train, y_train, X_test, y_test, dataset_name='Dataset'):
    """執行完整實驗的便捷函數"""
    
    exp = ExperimentFramework(dataset_name=dataset_name)
    exp.load_data(X_train, y_train, X_test, y_test)
    exp.run_all_models()
    summary = exp.print_summary()
    
    return exp, summary


if __name__ == "__main__":
    print("Experiment Framework Ready!")
    print("Use: run_complete_experiment(X_train, y_train, X_test, y_test)")
