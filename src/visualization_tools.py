"""
完整視覺化腳本 - 生成所有比較圖表
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體和樣式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ExperimentVisualizer:
    """實驗視覺化工具"""
    
    def __init__(self, experiment_framework):
        self.exp = experiment_framework
        self.dataset_name = experiment_framework.dataset_name
        
        # 顏色配置
        self.colors = {
            'MLP': '#FF6B6B',
            'XGBoost': '#4ECDC4',
            'GP': '#45B7D1',
            'DKL': '#FFA07A',
            'MoE': '#98D8C8',
            'Ensemble': '#F7DC6F'
        }
    
    def plot_all(self, save_path=None):
        """生成所有圖表"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 預測 vs 真實值散點圖 (2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._plot_predictions(ax1)
        
        # 2. 誤差分布箱型圖
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_error_distribution(ax2)
        
        # 3. 性能指標雷達圖
        ax3 = fig.add_subplot(gs[1, 2], projection='polar')
        self._plot_radar(ax3)
        
        # 4. 不確定性比較
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_uncertainty_comparison(ax4)
        
        # 5. CI覆蓋率 vs CI寬度
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_ci_quality(ax5)
        
        # 6. 訓練時間比較
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_training_time(ax6)
        
        plt.suptitle(f'Complete DIM-GP Comparison: {self.dataset_name} Dataset', 
                     fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comprehensive plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def _plot_predictions(self, ax):
        """預測 vs 真實值散點圖"""
        y_true = self.exp.y_test
        
        n_models = len(self.exp.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig_pred, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, result) in enumerate(self.exp.results.items()):
            ax_sub = axes[idx]
            y_pred = result['predictions']
            y_std = result['std']
            
            # 散點圖
            ax_sub.scatter(y_true, y_pred, alpha=0.5, 
                          c=self.colors.get(model_name, 'gray'),
                          label='Predictions', s=30)
            
            # 對角線 (perfect prediction)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax_sub.plot([min_val, max_val], [min_val, max_val], 
                       'r--', lw=2, label='Perfect')
            
            # 如果有不確定性，顯示誤差棒
            if result['has_uncertainty'] and np.any(y_std > 0):
                # 只顯示部分點的誤差棒，避免過於擁擠
                sample_indices = np.random.choice(len(y_true), 
                                                 min(50, len(y_true)), 
                                                 replace=False)
                ax_sub.errorbar(y_true[sample_indices], y_pred[sample_indices],
                              yerr=1.96*y_std[sample_indices],
                              fmt='none', ecolor='gray', alpha=0.3,
                              label='95% CI')
            
            # 指標標註
            r2 = result['metrics']['R2']
            rmse = result['metrics']['RMSE']
            ax_sub.text(0.05, 0.95, 
                       f"R²={r2:.4f}\nRMSE={rmse:.6f}",
                       transform=ax_sub.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax_sub.set_xlabel('True Values')
            ax_sub.set_ylabel('Predictions')
            ax_sub.set_title(f'{model_name}')
            ax_sub.legend()
            ax_sub.grid(True, alpha=0.3)
        
        # 隱藏多餘的子圖
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_predictions_grid.png', dpi=300, bbox_inches='tight')
        print(f"✓ Predictions grid saved")
        plt.close()
    
    def _plot_error_distribution(self, ax):
        """誤差分布箱型圖"""
        errors_dict = {}
        for model_name, result in self.exp.results.items():
            errors = result['predictions'] - self.exp.y_test
            errors_dict[model_name] = errors
        
        positions = range(len(errors_dict))
        bp = ax.boxplot(errors_dict.values(), positions=positions, 
                        labels=errors_dict.keys(), patch_artist=True)
        
        # 顏色設定
        for patch, model_name in zip(bp['boxes'], errors_dict.keys()):
            patch.set_facecolor(self.colors.get(model_name, 'gray'))
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_radar(self, ax):
        """性能指標雷達圖"""
        # 收集指標
        metrics_names = ['R²', 'RMSE', 'MAE', 'Speed', 'UQ']
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]  # 閉合
        
        for model_name, result in self.exp.results.items():
            metrics = result['metrics']
            
            # 正規化指標 (0-1)
            values = []
            
            # R² (已經是0-1)
            values.append(max(0, metrics['R2']))
            
            # RMSE (反向，越小越好)
            all_rmse = [r['metrics']['RMSE'] for r in self.exp.results.values()]
            rmse_norm = 1 - (metrics['RMSE'] - min(all_rmse)) / (max(all_rmse) - min(all_rmse) + 1e-10)
            values.append(rmse_norm)
            
            # MAE (反向)
            all_mae = [r['metrics']['MAE'] for r in self.exp.results.values()]
            mae_norm = 1 - (metrics['MAE'] - min(all_mae)) / (max(all_mae) - min(all_mae) + 1e-10)
            values.append(mae_norm)
            
            # Speed (反向時間)
            all_time = [r['train_time'] for r in self.exp.results.values()]
            time_norm = 1 - (result['train_time'] - min(all_time)) / (max(all_time) - min(all_time) + 1e-10)
            values.append(time_norm)
            
            # UQ (有不確定性=1，沒有=0)
            values.append(1.0 if result['has_uncertainty'] else 0.0)
            
            values += values[:1]  # 閉合
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model_name, 
                   color=self.colors.get(model_name, 'gray'))
            ax.fill(angles, values, alpha=0.15, 
                   color=self.colors.get(model_name, 'gray'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    def _plot_uncertainty_comparison(self, ax):
        """不確定性比較 - 顯示預測±2σ"""
        # 只顯示有UQ的模型
        uq_models = {name: result for name, result in self.exp.results.items() 
                     if result['has_uncertainty'] and np.any(result['std'] > 0)}
        
        if not uq_models:
            ax.text(0.5, 0.5, 'No models with\nuncertainty quantification',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Uncertainty Quantification')
            return
        
        # 排序測試樣本
        sorted_indices = np.argsort(self.exp.y_test)
        y_true_sorted = self.exp.y_test[sorted_indices]
        x_axis = np.arange(len(y_true_sorted))
        
        # 只顯示部分樣本，避免過於擁擠
        sample_size = min(100, len(y_true_sorted))
        step = len(y_true_sorted) // sample_size
        sample_indices = sorted_indices[::step][:sample_size]
        
        for model_name, result in uq_models.items():
            y_pred = result['predictions'][sample_indices]
            y_std = result['std'][sample_indices]
            y_true = self.exp.y_test[sample_indices]
            
            # 排序
            sort_idx = np.argsort(y_true)
            y_true = y_true[sort_idx]
            y_pred = y_pred[sort_idx]
            y_std = y_std[sort_idx]
            
            x = np.arange(len(y_true))
            
            color = self.colors.get(model_name, 'gray')
            ax.plot(x, y_pred, '-', label=model_name, color=color, linewidth=2)
            ax.fill_between(x, y_pred - 2*y_std, y_pred + 2*y_std, 
                           alpha=0.2, color=color)
        
        # 真實值
        ax.scatter(x, y_true, c='black', s=20, alpha=0.5, label='True', zorder=10)
        
        ax.set_xlabel('Samples (sorted)')
        ax.set_ylabel('Value')
        ax.set_title('Uncertainty Quantification (±2σ)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_ci_quality(self, ax):
        """CI覆蓋率 vs CI寬度散點圖"""
        uq_data = []
        
        for model_name, result in self.exp.results.items():
            metrics = result['metrics']
            if metrics['CI_Coverage'] is not None:
                uq_data.append({
                    'model': model_name,
                    'coverage': metrics['CI_Coverage'],
                    'width': metrics['CI_Width']
                })
        
        if not uq_data:
            ax.text(0.5, 0.5, 'No uncertainty\nquantification data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('CI Quality')
            return
        
        for data in uq_data:
            ax.scatter(data['width'], data['coverage'], 
                      s=200, alpha=0.7,
                      c=self.colors.get(data['model'], 'gray'),
                      label=data['model'])
            ax.text(data['width'], data['coverage'], data['model'],
                   ha='center', va='center', fontsize=8)
        
        # 理想區域：95%覆蓋率，窄CI
        ax.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='Target (95%)')
        ax.fill_between([0, ax.get_xlim()[1]], 93, 97, alpha=0.1, color='green',
                       label='Good Coverage')
        
        ax.set_xlabel('CI Width (narrower is better)')
        ax.set_ylabel('CI Coverage (%)')
        ax.set_title('CI Quality: Coverage vs Width')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_time(self, ax):
        """訓練時間比較"""
        models = list(self.exp.results.keys())
        times = [self.exp.results[m]['train_time'] for m in models]
        colors_list = [self.colors.get(m, 'gray') for m in models]
        
        bars = ax.barh(models, times, color=colors_list, alpha=0.7)
        
        # 標註數值
        for bar, time in zip(bars, times):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{time:.2f}s',
                   ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.grid(True, alpha=0.3, axis='x')
    
    def plot_calibration(self, save_path=None):
        """Calibration plot - 預測不確定性 vs 實際誤差"""
        uq_models = {name: result for name, result in self.exp.results.items() 
                     if result['has_uncertainty'] and np.any(result['std'] > 0)}
        
        if not uq_models:
            print("No models with uncertainty quantification")
            return
        
        fig, axes = plt.subplots(1, len(uq_models), figsize=(5*len(uq_models), 5))
        if len(uq_models) == 1:
            axes = [axes]
        
        for ax, (model_name, result) in zip(axes, uq_models.items()):
            y_pred = result['predictions']
            y_std = result['std']
            y_true = self.exp.y_test
            
            # 實際誤差
            errors = np.abs(y_true - y_pred)
            
            # 分箱
            n_bins = 10
            std_bins = np.percentile(y_std, np.linspace(0, 100, n_bins+1))
            
            bin_centers = []
            mean_stds = []
            mean_errors = []
            
            for i in range(n_bins):
                mask = (y_std >= std_bins[i]) & (y_std < std_bins[i+1])
                if np.sum(mask) > 0:
                    bin_centers.append((std_bins[i] + std_bins[i+1]) / 2)
                    mean_stds.append(np.mean(y_std[mask]))
                    mean_errors.append(np.mean(errors[mask]))
            
            # 繪圖
            ax.scatter(mean_stds, mean_errors, s=100, alpha=0.6,
                      c=self.colors.get(model_name, 'gray'))
            
            # 理想線 (y=x)
            max_val = max(max(mean_stds), max(mean_errors))
            ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Calibration')
            
            ax.set_xlabel('Predicted Std')
            ax.set_ylabel('Actual Error')
            ax.set_title(f'{model_name} Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Calibration plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_feature_importance(self, save_path=None):
        """特徵重要性比較 (針對XGBoost和Ensemble)"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # XGBoost
        if 'XGBoost' in self.exp.models:
            xgb_model = self.exp.models['XGBoost'].model
            importance = xgb_model.feature_importances_
            
            feature_names = [f'Feature {i}' for i in range(len(importance))]
            if hasattr(self.exp, 'feature_names'):
                feature_names = self.exp.feature_names
            
            axes[0].barh(feature_names, importance, color='#4ECDC4', alpha=0.7)
            axes[0].set_xlabel('Importance')
            axes[0].set_title('XGBoost Feature Importance')
            axes[0].grid(True, alpha=0.3, axis='x')
        
        # Ensemble
        if 'Ensemble' in self.exp.models:
            ens_model = self.exp.models['Ensemble']
            importance = ens_model.xgb.model.feature_importances_
            
            feature_names = [f'Feature {i}' for i in range(len(importance))]
            if hasattr(self.exp, 'feature_names'):
                feature_names = self.exp.feature_names
            
            axes[1].barh(feature_names, importance, color='#F7DC6F', alpha=0.7)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Ensemble Feature Importance')
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        plt.show()
        return fig


def create_all_visualizations(experiment_framework, output_prefix='results'):
    """生成所有視覺化"""
    visualizer = ExperimentVisualizer(experiment_framework)
    
    print("\n" + "="*60)
    print("Generating All Visualizations...")
    print("="*60 + "\n")
    
    # 1. 主要綜合圖
    visualizer.plot_all(f'{output_prefix}_comprehensive.png')
    
    # 2. Calibration圖
    visualizer.plot_calibration(f'{output_prefix}_calibration.png')
    
    # 3. Feature importance
    visualizer.plot_feature_importance(f'{output_prefix}_feature_importance.png')
    
    print("\n" + "="*60)
    print("✓ All Visualizations Generated!")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Visualization Tools Ready!")
    print("Use: create_all_visualizations(experiment_framework)")
