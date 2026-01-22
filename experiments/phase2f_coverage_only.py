"""
Phase 2F: Coverage-Only 模型 (Type 3)
最激進的策略：完全忽略 Type 3 的 Thickness

核心思想：
- Type 3: 只用 Coverage 預測（Thickness 相關性 = 0.04）
- 使用高階多項式回歸 + 局部加權回歸 (LOWESS)

使用方法:
    python phase2f_coverage_only.py --seed 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
import argparse

warnings.filterwarnings('ignore')


def set_seed(seed):
    """設置隨機種子"""
    np.random.seed(seed)
    print(f"✓ 隨機種子設定為: {seed}")


# ==========================================
# Type 3 Coverage-Only 模型
# ==========================================

class Type3CoverageModel:
    """
    Type 3 專用模型：只使用 Coverage
    
    組合多種回歸方法：
    1. 高階多項式回歸
    2. LOWESS (局部加權回歸)
    3. Spline 插值
    4. Isotonic 回歸（保證單調性）
    """
    
    def __init__(self, X_train, y_train, degree=5):
        """
        Args:
            X_train: [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
            y_train: Theta.JC
            degree: 多項式階數
        """
        # 只用 Type 3 資料
        type3_mask = X_train[:, 0] == 3
        coverage = X_train[type3_mask, 2].reshape(-1, 1)
        theta = y_train[type3_mask]
        
        self.degree = degree
        
        print(f"\n{'='*60}")
        print(f"訓練 Type 3 Coverage-Only 模型")
        print(f"{'='*60}")
        print(f"訓練樣本數: {len(coverage)}")
        print(f"Coverage 範圍: [{coverage.min():.2f}, {coverage.max():.2f}]")
        print(f"Theta.JC 範圍: [{theta.min():.4f}, {theta.max():.4f}]")
        
        # ===== 方法 1: 高階多項式回歸 =====
        self.poly_features = PolynomialFeatures(degree=degree)
        coverage_poly = self.poly_features.fit_transform(coverage)
        
        # 使用 Huber Regressor（對異常值更穩健）
        self.poly_model = HuberRegressor(epsilon=1.5, alpha=0.001)
        self.poly_model.fit(coverage_poly, theta)
        
        poly_pred = self.poly_model.predict(coverage_poly)
        poly_mape = np.mean(np.abs((theta - poly_pred) / theta)) * 100
        print(f"\n多項式回歸 (degree={degree}):")
        print(f"  訓練 MAPE: {poly_mape:.2f}%")
        
        # ===== 方法 2: LOWESS (局部加權回歸) =====
        # 按 Coverage 排序
        sort_idx = np.argsort(coverage.flatten())
        coverage_sorted = coverage.flatten()[sort_idx]
        theta_sorted = theta[sort_idx]
        
        # LOWESS 擬合
        lowess_result = lowess(theta_sorted, coverage_sorted, 
                               frac=0.3, it=3, return_sorted=True)
        self.lowess_coverage = lowess_result[:, 0]
        self.lowess_theta = lowess_result[:, 1]
        
        # 用於插值
        from scipy.interpolate import interp1d
        self.lowess_interp = interp1d(
            self.lowess_coverage, self.lowess_theta,
            kind='cubic', fill_value='extrapolate'
        )
        
        lowess_pred = self.lowess_interp(coverage_sorted)
        lowess_mape = np.mean(np.abs((theta_sorted - lowess_pred) / theta_sorted)) * 100
        print(f"\nLOWESS:")
        print(f"  訓練 MAPE: {lowess_mape:.2f}%")
        
        # ===== 方法 3: Cubic Spline =====
        # 選擇結點（在數據密集的地方放更多結點）
        knots = np.percentile(coverage_sorted, [10, 25, 40, 50, 60, 75, 90])
        
        self.spline_model = LSQUnivariateSpline(
            coverage_sorted, theta_sorted, 
            t=knots[1:-1],  # 去掉邊界結點
            k=3  # 三次 spline
        )
        
        spline_pred = self.spline_model(coverage_sorted)
        spline_mape = np.mean(np.abs((theta_sorted - spline_pred) / theta_sorted)) * 100
        print(f"\nCubic Spline:")
        print(f"  訓練 MAPE: {spline_mape:.2f}%")
        
        # ===== 方法 4: Isotonic 回歸（保證單調遞減）=====
        self.isotonic_model = IsotonicRegression(increasing=False, out_of_bounds='clip')
        self.isotonic_model.fit(coverage_sorted, theta_sorted)
        
        isotonic_pred = self.isotonic_model.predict(coverage_sorted)
        isotonic_mape = np.mean(np.abs((theta_sorted - isotonic_pred) / theta_sorted)) * 100
        print(f"\nIsotonic 回歸 (單調遞減):")
        print(f"  訓練 MAPE: {isotonic_mape:.2f}%")
        
        # ===== 保存訓練資料（用於分位數預測）=====
        self.train_coverage = coverage.flatten()
        self.train_theta = theta
        
        # 按 Coverage 分組統計
        self.coverage_stats = {}
        for cov in np.unique(coverage.flatten()):
            mask = coverage.flatten() == cov
            self.coverage_stats[cov] = {
                'mean': np.mean(theta[mask]),
                'median': np.median(theta[mask]),
                'std': np.std(theta[mask]),
                'q25': np.percentile(theta[mask], 25),
                'q75': np.percentile(theta[mask], 75),
                'count': np.sum(mask)
            }
        
        print(f"\n✓ Type 3 模型訓練完成")
        print(f"{'='*60}")
        
        # 選擇最佳方法（目前使用加權組合）
        self.best_method = 'ensemble'
    
    def predict(self, X_test, method='ensemble'):
        """
        預測
        
        Args:
            X_test: [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
            method: 'poly', 'lowess', 'spline', 'isotonic', 'ensemble'
        
        Returns:
            predictions, std
        """
        type3_mask = X_test[:, 0] == 3
        n_type3 = np.sum(type3_mask)
        
        if n_type3 == 0:
            return np.zeros(len(X_test)), np.zeros(len(X_test))
        
        coverage = X_test[type3_mask, 2].reshape(-1, 1)
        
        predictions = np.zeros(len(X_test))
        stds = np.zeros(len(X_test))
        
        if method == 'poly':
            # 多項式預測
            coverage_poly = self.poly_features.transform(coverage)
            pred = self.poly_model.predict(coverage_poly)
        
        elif method == 'lowess':
            # LOWESS 預測
            pred = self.lowess_interp(coverage.flatten())
        
        elif method == 'spline':
            # Spline 預測
            pred = self.spline_model(coverage.flatten())
        
        elif method == 'isotonic':
            # Isotonic 預測
            pred = self.isotonic_model.predict(coverage.flatten())
        
        elif method == 'ensemble':
            # 加權組合（給表現好的方法更高權重）
            coverage_flat = coverage.flatten()
            coverage_poly = self.poly_features.transform(coverage)
            
            pred_poly = self.poly_model.predict(coverage_poly)
            pred_lowess = self.lowess_interp(coverage_flat)
            pred_spline = self.spline_model(coverage_flat)
            pred_isotonic = self.isotonic_model.predict(coverage_flat)
            
            # 動態權重（根據 Coverage 位置）
            weights = np.ones((n_type3, 4))
            
            for i, cov in enumerate(coverage_flat):
                # 如果 Coverage 在訓練集中出現過，增加 isotonic 權重
                if cov in self.coverage_stats:
                    weights[i, 3] *= 2.0  # isotonic
                
                # 高 Coverage (>0.7) 增加 lowess 權重
                if cov > 0.7:
                    weights[i, 1] *= 1.5  # lowess
            
            # 歸一化
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            # 加權平均
            pred = (
                weights[:, 0] * pred_poly +
                weights[:, 1] * pred_lowess +
                weights[:, 2] * pred_spline +
                weights[:, 3] * pred_isotonic
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        predictions[type3_mask] = pred
        
        # 估計標準差（用最近的訓練樣本）
        for i, cov in enumerate(coverage.flatten()):
            if cov in self.coverage_stats:
                stds[type3_mask][i] = self.coverage_stats[cov]['std']
            else:
                # 找最近的 Coverage
                distances = np.abs(self.train_coverage - cov)
                nearest_idx = np.argmin(distances)
                nearest_cov = self.train_coverage[nearest_idx]
                if nearest_cov in self.coverage_stats:
                    stds[type3_mask][i] = self.coverage_stats[nearest_cov]['std']
                else:
                    stds[type3_mask][i] = 0.005  # 預設值
        
        return predictions, stds


# ==========================================
# Type 1, 2 簡單模型
# ==========================================

class SimpleLinearModel:
    """Type 1, 2 用簡單線性模型"""
    
    def __init__(self, X_train, y_train):
        """
        Args:
            X_train: [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
            y_train: Theta.JC
        """
        # 只用 Type 1, 2 資料
        others_mask = X_train[:, 0] != 3
        X_others = X_train[others_mask]
        y_others = y_train[others_mask]
        
        print(f"\n訓練 Type 1, 2 模型 ({len(X_others)} 筆)...")
        
        # 多項式特徵
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=False)
        X_poly = self.poly_features.fit_transform(X_others)
        
        # Ridge 回歸
        self.model = Ridge(alpha=0.1)
        self.model.fit(X_poly, y_others)
        
        # 訓練誤差
        pred = self.model.predict(X_poly)
        mape = np.mean(np.abs((y_others - pred) / y_others)) * 100
        print(f"  訓練 MAPE: {mape:.2f}%")
    
    def predict(self, X_test):
        """預測"""
        others_mask = X_test[:, 0] != 3
        
        predictions = np.zeros(len(X_test))
        stds = np.zeros(len(X_test))
        
        if np.sum(others_mask) > 0:
            X_others = X_test[others_mask]
            X_poly = self.poly_features.transform(X_others)
            predictions[others_mask] = self.model.predict(X_poly)
            stds[others_mask] = 0.01  # 預設值
        
        return predictions, stds


# ==========================================
# 混合模型
# ==========================================

class HybridCoverageModel:
    """混合模型：Type 3 用 Coverage-Only，Type 1/2 用標準模型"""
    
    def __init__(self, X_train, y_train, config):
        """初始化"""
        # Type 3 Coverage-Only 模型
        self.type3_model = Type3CoverageModel(X_train, y_train, degree=config['degree'])
        
        # Type 1, 2 模型
        self.standard_model = SimpleLinearModel(X_train, y_train)
    
    def predict(self, X_test):
        """預測"""
        # Type 3 預測
        pred_type3, std_type3 = self.type3_model.predict(X_test, method='ensemble')
        
        # Type 1, 2 預測
        pred_others, std_others = self.standard_model.predict(X_test)
        
        # 組合
        predictions = pred_type3 + pred_others
        stds = std_type3 + std_others
        
        return predictions, stds


# ==========================================
# 評估函數
# ==========================================

def evaluate_model(model, X_test, y_test, verbose=True):
    """評估模型"""
    y_pred, y_std = model.predict(X_test)
    
    # 計算指標
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    # Type 3 分析
    type3_mask = X_test[:, 0] == 3
    if np.sum(type3_mask) > 0:
        type3_errors = relative_errors[type3_mask]
        type3_mape = np.mean(type3_errors)
        type3_outliers = np.sum(type3_errors > 20)
        
        # Coverage 0.8 分析
        cov08_mask = type3_mask & (X_test[:, 2] == 0.8)
        if np.sum(cov08_mask) > 0:
            cov08_errors = relative_errors[cov08_mask]
            cov08_mape = np.mean(cov08_errors)
            cov08_outliers = np.sum(cov08_errors > 20)
            
            # 詳細顯示 Coverage 0.8 預測
            if verbose:
                print(f"\n{'='*60}")
                print("Coverage 0.8 詳細預測")
                print(f"{'='*60}")
                cov08_data = X_test[cov08_mask]
                cov08_true = y_test[cov08_mask]
                cov08_pred = y_pred[cov08_mask]
                cov08_err = relative_errors[cov08_mask]
                
                for i in range(len(cov08_true)):
                    marker = "❌" if cov08_err[i] > 20 else "✓"
                    print(f"{marker} Thick={cov08_data[i, 1]:.0f}, "
                          f"True={cov08_true[i]:.3f}, Pred={cov08_pred[i]:.3f}, "
                          f"Error={cov08_err[i]:.1f}%")
        else:
            cov08_mape = 0
            cov08_outliers = 0
    else:
        type3_mape = 0
        type3_outliers = 0
        cov08_mape = 0
        cov08_outliers = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"評估結果 (Coverage-Only 模型)")
        print(f"{'='*60}")
        print(f"樣本數: {len(y_test)}")
        print(f"\n準確度:")
        print(f"  MAPE:      {mape:.2f}%")
        print(f"  MAE:       {mae:.4f}")
        print(f"  Max Error: {max_error:.2f}%")
        print(f"\n異常點:")
        print(f"  >20%: {outliers_20}/{len(y_test)} ({outliers_20/len(y_test)*100:.2f}%)")
        print(f"  >15%: {outliers_15}/{len(y_test)} ({outliers_15/len(y_test)*100:.2f}%)")
        print(f"  >10%: {outliers_10}/{len(y_test)} ({outliers_10/len(y_test)*100:.2f}%)")
        
        if np.sum(type3_mask) > 0:
            print(f"\nType 3 詳細:")
            print(f"  樣本數: {np.sum(type3_mask)}")
            print(f"  MAPE: {type3_mape:.2f}%")
            print(f"  異常點: {type3_outliers}/{np.sum(type3_mask)}")
            if np.sum(cov08_mask) > 0:
                print(f"\n  Coverage 0.8 子集:")
                print(f"    MAPE: {cov08_mape:.2f}%")
                print(f"    異常點: {cov08_outliers}/{np.sum(cov08_mask)}")
        print(f"{'='*60}\n")
    
    results = {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'type3_mape': type3_mape,
        'type3_outliers': type3_outliers,
        'cov08_mape': cov08_mape,
        'cov08_outliers': cov08_outliers,
        'predictions': y_pred,
        'std': y_std,
        'errors': relative_errors
    }
    
    return results


def save_predictions(X_test, y_test, results, filename):
    """保存預測結果"""
    df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Predicted': results['predictions'],
        'Error%': results['errors'],
        'Std': results['std']
    })
    
    df.to_csv(filename, index=False)
    print(f"✓ 預測結果已保存到: {filename}")


# ==========================================
# 主函數
# ==========================================

def main(seed=2024, verbose=True):
    """主訓練流程"""
    set_seed(seed)
    
    print("\n" + "="*60)
    print("Phase 2F: Coverage-Only 模型 (Type 3)")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置
    config = {
        'degree': 6,  # 多項式階數
        'seed': seed,
    }
    
    print(f"\n配置: 多項式階數 = {config['degree']}")
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    # 訓練集清理
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"\n訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆")
    
    X_train = train_above_clean[feature_cols].values
    y_train = train_above_clean[target_col].values
    
    X_test = test_above[feature_cols].values
    y_test = test_above[target_col].values
    
    # 訓練混合模型
    model = HybridCoverageModel(X_train, y_train, config)
    
    # 評估
    results = evaluate_model(model, X_test, y_test, verbose=verbose)
    
    # 保存預測結果
    save_predictions(X_test, y_test, results,
                     f'phase2f_coverage_only_seed{seed}_predictions.csv')
    
    # 總結
    print("\n" + "="*60)
    print("最終結果總結 (Phase 2F)")
    print("="*60)
    print(f"策略:")
    print(f"  ✓ Type 3: 完全忽略 Thickness (相關性 = 0.04)")
    print(f"  ✓ Type 3: 組合 4 種回歸方法")
    print(f"    - 高階多項式回歸")
    print(f"    - LOWESS (局部加權)")
    print(f"    - Cubic Spline")
    print(f"    - Isotonic 回歸 (單調性)")
    print(f"  ✓ Type 1, 2: 多項式回歸")
    print(f"\n結果:")
    print(f"  總體 MAPE: {results['mape']:.2f}%")
    print(f"  Type 3 MAPE: {results['type3_mape']:.2f}%")
    print(f"  Coverage 0.8 MAPE: {results['cov08_mape']:.2f}%")
    print(f"  異常點: {results['outliers_20']}/{len(y_test)}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2F Coverage-Only 模型')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細模式')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
    
    print("\n💡 說明:")
    print("  此版本對 Type 3 完全忽略 Thickness")
    print("  只用 Coverage 建立 Theta.JC 的映射關係")
    print("  預期: 減少過度平滑問題\n")
