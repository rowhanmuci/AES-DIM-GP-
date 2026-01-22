"""
XGBoost Baseline - 與 Phase 2B DKL 對照實驗
使用相同的評估指標和資料集
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
import argparse

warnings.filterwarnings('ignore')


def set_seed(seed):
    """設置隨機種子"""
    np.random.seed(seed)
    print(f"✓ 隨機種子設定為: {seed}")


def compute_sample_weights(X, weight_factor=3.0):
    """
    計算樣本權重（與 Phase 2B 相同）
    困難樣本定義: TIM_TYPE=3 AND Coverage=0.8 AND THICKNESS>=220
    """
    weights = np.ones(len(X))
    
    difficult_mask = (
        (X[:, 0] == 3) &      # TIM_TYPE = 3
        (X[:, 2] == 0.8) &    # TIM_COVERAGE = 0.8
        (X[:, 1] >= 220)      # TIM_THICKNESS >= 220
    )
    
    weights[difficult_mask] *= weight_factor
    
    return weights


def train_xgboost(X_train, y_train, use_weights=True, weight_factor=3.0, 
                  tune_hyperparams=False, seed=2024, verbose=True):
    """
    訓練 XGBoost 模型
    
    Args:
        X_train: 訓練特徵
        y_train: 訓練標籤
        use_weights: 是否使用樣本權重
        weight_factor: 權重倍數
        tune_hyperparams: 是否進行超參數搜索
        seed: 隨機種子
        verbose: 是否顯示訓練過程
        
    Returns:
        model: 訓練好的 XGBoost 模型
    """
    
    # 計算樣本權重
    if use_weights:
        sample_weights = compute_sample_weights(X_train, weight_factor)
        if verbose:
            difficult_count = np.sum(sample_weights > 1.0)
            print(f"\n計算樣本權重:")
            print(f"  困難樣本數: {difficult_count} ({difficult_count/len(X_train)*100:.2f}%)")
            print(f"  權重倍數: {weight_factor}x")
    else:
        sample_weights = None
        if verbose:
            print(f"\n不使用樣本權重")
    
    if tune_hyperparams:
        # 超參數搜索
        if verbose:
            print(f"\n進行超參數搜索...")
        
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 500],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=seed,
            tree_method='hist'
        )
        
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        if verbose:
            print(f"\n最佳參數:")
            for key, value in grid_search.best_params_.items():
                print(f"  {key}: {value}")
        
        model = grid_search.best_estimator_
    
    else:
        # 使用預設良好參數
        params = {
            'max_depth': 7,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': seed,
            'tree_method': 'hist',
            'n_jobs': -1,
        }
        
        if verbose:
            print(f"\n使用參數:")
            for key, value in params.items():
                if key not in ['random_state', 'tree_method', 'n_jobs']:
                    print(f"  {key}: {value}")
        
        model = xgb.XGBRegressor(**params)
        
        # 訓練
        if verbose:
            print(f"\n開始訓練...")
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train)],
            verbose=50 if verbose else 0
        )
    
    if verbose:
        print(f"訓練完成")
    
    return model


def evaluate_model(model, X_test, y_test, verbose=True):
    """
    評估模型（與 Phase 2B 相同的評估邏輯）
    
    Returns:
        results: 包含 MAPE, outliers 等指標的字典
    """
    # 預測
    y_pred = model.predict(X_test)
    
    # 計算指標（在原始空間）
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    mape = np.mean(relative_errors)
    mae = np.mean(np.abs(y_test - y_pred))
    max_error = np.max(relative_errors)
    
    outliers_20 = np.sum(relative_errors > 20)
    outliers_15 = np.sum(relative_errors > 15)
    outliers_10 = np.sum(relative_errors > 10)
    
    # Type 3 分析
    type3_mask = X_test[:, 0] == 3
    type3_outliers = np.sum((relative_errors > 20) & type3_mask)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"評估結果")
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
            print(f"\nType 3異常點: {type3_outliers}/{np.sum(type3_mask)}")
        print(f"{'='*60}\n")
    
    results = {
        'mape': mape,
        'mae': mae,
        'max_error': max_error,
        'outliers_20': outliers_20,
        'outliers_15': outliers_15,
        'outliers_10': outliers_10,
        'type3_outliers': type3_outliers,
        'predictions': y_pred,
        'errors': relative_errors
    }
    
    return results


def save_predictions(X_test, y_test, results, filename):
    """保存預測結果到 CSV"""
    df = pd.DataFrame({
        'TIM_TYPE': X_test[:, 0],
        'TIM_THICKNESS': X_test[:, 1],
        'TIM_COVERAGE': X_test[:, 2],
        'True': y_test,
        'Predicted': results['predictions'],
        'Error%': results['errors']
    })
    
    df.to_csv(filename, index=False)
    print(f"✓ 預測結果已保存到: {filename}")


def main(seed=2024, use_weights=True, weight_factor=3.0, 
         tune_hyperparams=False, verbose=True):
    """
    主訓練流程
    
    Args:
        seed: 隨機種子
        use_weights: 是否使用樣本權重
        weight_factor: 權重倍數
        tune_hyperparams: 是否進行超參數搜索
        verbose: 是否顯示詳細信息
    """
    set_seed(seed)
    
    print("="*60)
    print("XGBoost Baseline - 與 DKL 對照實驗")
    print("="*60)
    
    # 特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    # ==========================================
    # Above Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Above 50% Coverage'}\n")
    
    # 載入資料
    train_above = pd.read_excel('data/train/Above.xlsx')
    test_above = pd.read_excel('data/test/Above.xlsx')
    
    # 訓練集清理（去除重複，取平均）
    train_above_clean = train_above.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_above_clean)} 筆")
    print(f"測試集: {len(test_above)} 筆")
    
    X_train_above = train_above_clean[feature_cols].values
    y_train_above = train_above_clean[target_col].values
    
    X_test_above = test_above[feature_cols].values
    y_test_above = test_above[target_col].values
    
    # 訓練
    model_above = train_xgboost(
        X_train_above, y_train_above,
        use_weights=use_weights,
        weight_factor=weight_factor,
        tune_hyperparams=tune_hyperparams,
        seed=seed,
        verbose=verbose
    )
    
    # 評估
    results_above = evaluate_model(
        model_above,
        X_test_above, y_test_above,
        verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_above, y_test_above, results_above,
                     f'xgboost_above_seed{seed}_predictions.csv')
    
    # ==========================================
    # Below Dataset
    # ==========================================
    
    print(f"\n\n{'🔵 Below 50% Coverage'}\n")
    
    # 載入資料
    train_below = pd.read_excel('data/train/Below.xlsx')
    test_below = pd.read_excel('data/test/Below.xlsx')
    
    # 訓練集清理
    train_below_clean = train_below.groupby(feature_cols, as_index=False).agg({
        target_col: 'mean'
    })
    
    print(f"訓練集: {len(train_below_clean)} 筆")
    print(f"測試集: {len(test_below)} 筆")
    
    X_train_below = train_below_clean[feature_cols].values
    y_train_below = train_below_clean[target_col].values
    
    X_test_below = test_below[feature_cols].values
    y_test_below = test_below[target_col].values
    
    # 訓練
    model_below = train_xgboost(
        X_train_below, y_train_below,
        use_weights=use_weights,
        weight_factor=weight_factor,
        tune_hyperparams=tune_hyperparams,
        seed=seed,
        verbose=verbose
    )
    
    # 評估
    results_below = evaluate_model(
        model_below,
        X_test_below, y_test_below,
        verbose=verbose
    )
    
    # 保存預測結果
    save_predictions(X_test_below, y_test_below, results_below,
                     f'xgboost_below_seed{seed}_predictions.csv')
    
    # ==========================================
    # 總結
    # ==========================================
    
    print("\n" + "="*60)
    print("XGBoost 最終結果總結")
    print("="*60)
    print(f"隨機種子: {seed}")
    print(f"樣本權重: {'啟用' if use_weights else '停用'} (factor={weight_factor})")
    
    print(f"\nAbove資料集:")
    print(f"  異常點 (>20%): {results_above['outliers_20']}/{len(y_test_above)} ({results_above['outliers_20']/len(y_test_above)*100:.2f}%)")
    print(f"  MAPE: {results_above['mape']:.2f}%")
    print(f"  Type 3異常點: {results_above['type3_outliers']}")
    
    print(f"\nBelow資料集:")
    print(f"  異常點 (>20%): {results_below['outliers_20']}/{len(y_test_below)} ({results_below['outliers_20']/len(y_test_below)*100:.2f}%)")
    print(f"  MAPE: {results_below['mape']:.2f}%")
    
    print("\n" + "="*60)
    print("✓ XGBoost 訓練完成！")
    print("="*60 + "\n")
    
    return {
        'above': results_above,
        'below': results_below,
        'seed': seed
    }


if __name__ == "__main__":
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='XGBoost Baseline')
    parser.add_argument('--seed', type=int, default=2024,
                        help='隨機種子 (預設: 2024)')
    parser.add_argument('--no-weights', action='store_true',
                        help='停用樣本權重')
    parser.add_argument('--weight-factor', type=float, default=3.0,
                        help='樣本權重倍數 (預設: 3.0)')
    parser.add_argument('--tune', action='store_true',
                        help='進行超參數搜索')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='顯示詳細訓練過程')
    
    args = parser.parse_args()
    
    # 運行訓練
    results = main(
        seed=args.seed,
        use_weights=not args.no_weights,
        weight_factor=args.weight_factor,
        tune_hyperparams=args.tune,
        verbose=args.verbose
    )
    
    print("\n💡 使用範例:")
    print("  python xgboost_baseline.py                    # 基本版本")
    print("  python xgboost_baseline.py --seed 42 -v       # 指定種子，詳細模式")
    print("  python xgboost_baseline.py --no-weights       # 不使用樣本權重")
    print("  python xgboost_baseline.py --tune             # 超參數搜索（較慢）\n")