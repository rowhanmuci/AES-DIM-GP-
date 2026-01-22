"""
Phase 2G: 簡化穩健版 - 分位數回歸
針對 Type 3 過度平滑問題的最簡單解決方案

核心思想：
- Type 3: 只用 Coverage，用分位數回歸（更穩健）
- 預測 median + 不確定性區間

使用方法:
    python phase2g_simple_robust.py --seed 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
import warnings
import argparse

warnings.filterwarnings('ignore')


def set_seed(seed):
    """設置隨機種子"""
    np.random.seed(seed)
    print(f"✓ 隨機種子設定為: {seed}")


# ==========================================
# Type 3 分位數回歸模型
# ==========================================

class Type3QuantileModel:
    """
    Type 3 專用：分位數回歸
    
    策略：
    1. 只用 Coverage（忽略 Thickness）
    2. 同時預測 p10, p50, p90 分位數
    3. 最終預測用 median (p50)
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
        print(f"訓練 Type 3 分位數回歸模型")
        print(f"{'='*60}")
        print(f"訓練樣本數: {len(coverage)}")
        print(f"Coverage 範圍: [{coverage.min():.2f}, {coverage.max():.2f}]")
        print(f"Theta.JC 範圍: [{theta.min():.4f}, {theta.max():.4f}]")
        
        # 多項式特徵
        self.poly_features = PolynomialFeatures(degree=degree)
        coverage_poly = self.poly_features.fit_transform(coverage)
        
        # 訓練 3 個分位數模型
        self.quantile_models = {}
        for quantile in [0.1, 0.5, 0.9]:
            model = QuantileRegressor(
                quantile=quantile,
                alpha=0.01,
                solver='highs'
            )
            model.fit(coverage_poly, theta)
            self.quantile_models[quantile] = model
            
            # 訓練誤差
            pred = model.predict(coverage_poly)
            mape = np.mean(np.abs((theta - pred) / theta)) * 100
            print(f"  Q{int(quantile*100)} MAPE: {mape:.2f}%")
        
        # 保存訓練資料統計
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
                'count': np.sum(mask)
            }
        
        # 建立簡單的查表插值（作為後備）
        coverage_unique = sorted(self.coverage_stats.keys())
        medians = [self.coverage_stats[c]['median'] for c in coverage_unique]
        
        self.lookup_interp = interp1d(
            coverage_unique, medians,
            kind='linear', fill_value='extrapolate'
        )
        
        print(f"\n✓ 模型訓練完成")
        print(f"{'='*60}")
    
    def predict(self, X_test, use_median=True):
        """
        預測
        
        Args:
            X_test: [TIM_TYPE, TIM_THICKNESS, TIM_COVERAGE]
            use_median: True=用 p50, False=用加權組合
        
        Returns:
            predictions, std
        """
        type3_mask = X_test[:, 0] == 3
        n_type3 = np.sum(type3_mask)
        
        if n_type3 == 0:
            return np.zeros(len(X_test)), np.zeros(len(X_test))
        
        coverage = X_test[type3_mask, 2].reshape(-1, 1)
        coverage_poly = self.poly_features.transform(coverage)
        
        predictions = np.zeros(len(X_test))
        stds = np.zeros(len(X_test))
        
        # 預測 3 個分位數
        pred_p10 = self.quantile_models[0.1].predict(coverage_poly)
        pred_p50 = self.quantile_models[0.5].predict(coverage_poly)
        pred_p90 = self.quantile_models[0.9].predict(coverage_poly)
        
        if use_median:
            # 使用 median
            pred = pred_p50
        else:
            # 加權組合（給接近訓練集的 coverage 更多 p50 權重）
            pred = np.zeros(n_type3)
            for i, cov in enumerate(coverage.flatten()):
                if cov in self.coverage_stats:
                    # 訓練集中見過，直接用 median
                    pred[i] = pred_p50[i]
                else:
                    # 未見過，用分位數加權
                    pred[i] = 0.2 * pred_p10[i] + 0.6 * pred_p50[i] + 0.2 * pred_p90[i]
        
        predictions[type3_mask] = pred
        
        # 不確定性估計 (IQR / 1.35)
        iqr = pred_p90 - pred_p10
        stds[type3_mask] = iqr / 1.35
        
        return predictions, stds


# ==========================================
# Type 1, 2 簡單模型
# ==========================================

class SimplePolyModel:
    """Type 1, 2 用多項式回歸"""
    
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
        
        # Quantile 回歸 (median)
        self.model = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
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

class HybridQuantileModel:
    """混合模型：Type 3 用分位數回歸，Type 1/2 用標準模型"""
    
    def __init__(self, X_train, y_train, config):
        """初始化"""
        # Type 3 分位數模型
        self.type3_model = Type3QuantileModel(X_train, y_train, degree=config['degree'])
        
        # Type 1, 2 模型
        self.standard_model = SimplePolyModel(X_train, y_train)
    
    def predict(self, X_test):
        """預測"""
        # Type 3 預測
        pred_type3, std_type3 = self.type3_model.predict(X_test, use_median=True)
        
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
        print(f"評估結果 (分位數回歸)")
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
    print("Phase 2G: 簡化穩健版 - 分位數回歸")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # 配置
    config = {
        'degree': 5,  # 多項式階數
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
    model = HybridQuantileModel(X_train, y_train, config)
    
    # 評估
    results = evaluate_model(model, X_test, y_test, verbose=verbose)
    
    # 保存預測結果
    save_predictions(X_test, y_test, results,
                     f'phase2g_quantile_seed{seed}_predictions.csv')
    
    # 總結
    print("\n" + "="*60)
    print("最終結果總結 (Phase 2G)")
    print("="*60)
    print(f"策略:")
    print(f"  ✓ Type 3: 完全忽略 Thickness")
    print(f"  ✓ Type 3: 分位數回歸 (p10, p50, p90)")
    print(f"  ✓ Type 1, 2: Median 回歸")
    print(f"\n結果:")
    print(f"  總體 MAPE: {results['mape']:.2f}%")
    print(f"  Type 3 MAPE: {results['type3_mape']:.2f}%")
    print(f"  Coverage 0.8 MAPE: {results['cov08_mape']:.2f}%")
    print(f"  異常點: {results['outliers_20']}/{len(y_test)}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2G 分位數回歸')
    parser.add_argument('--seed', type=int, default=2024, help='隨機種子')
    parser.add_argument('-v', '--verbose', action='store_true', help='詳細模式')
    
    args = parser.parse_args()
    
    results = main(seed=args.seed, verbose=args.verbose)
    
    print("\n💡 說明:")
    print("  此版本使用分位數回歸（最簡單穩健）")
    print("  預測 median 而非 mean（對異常值更穩健）\n")
