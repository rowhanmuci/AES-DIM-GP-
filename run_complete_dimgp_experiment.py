"""
主執行腳本 - 完整的DIM-GP變體比較實驗
一鍵執行所有模型訓練、評估和視覺化
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from experiment_framework import ExperimentFramework
from visualization_tools import create_all_visualizations

def load_ase_data():
    """載入日月光資料"""
    print("Loading ASE FOCoS Data...")
    
    # Above 50% TIM Coverage
    train_above = pd.read_excel('D:/NSYSU/Aes/data1/FOCoS_PKG_Type4_Cavity_TIM_50%_Above_Training_Data.xlsx')
    test_above = pd.read_excel('D:/NSYSU/Aes/data1/FOCoS_PKG_Type4_Cavity_TIM_50%_Above_Test_Data.xlsx')
    
    # Below 50% TIM Coverage
    train_below = pd.read_excel('D:/NSYSU/Aes/data2/FOCoS_PKG_Type4_Cavity_TIM_50%_Below_Training_Data.xlsx')
    test_below = pd.read_excel('D:/NSYSU/Aes/data2/FOCoS_PKG_Type4_Cavity_TIM_50%_Below_Test_Data.xlsx')
    
    # 準備特徵和目標
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    target_col = 'Theta.JC'
    
    data = {
        'above': {
            'X_train': train_above[feature_cols].values,
            'y_train': train_above[target_col].values,
            'X_test': test_above[feature_cols].values,
            'y_test': test_above[target_col].values
        },
        'below': {
            'X_train': train_below[feature_cols].values,
            'y_train': train_below[target_col].values,
            'X_test': test_below[feature_cols].values,
            'y_test': test_below[target_col].values
        }
    }
    
    print(f"✓ Data loaded successfully!")
    print(f"  Above: Train={len(data['above']['X_train'])}, Test={len(data['above']['X_test'])}")
    print(f"  Below: Train={len(data['below']['X_train'])}, Test={len(data['below']['X_test'])}")
    
    return data, feature_cols


def run_single_dataset_experiment(X_train, y_train, X_test, y_test, 
                                  dataset_name, output_prefix):
    """執行單一資料集的完整實驗"""
    
    print("\n" + "="*80)
    print(f"STARTING EXPERIMENT: {dataset_name}")
    print("="*80 + "\n")
    
    # 建立實驗框架
    exp = ExperimentFramework(dataset_name=dataset_name)
    exp.load_data(X_train, y_train, X_test, y_test)
    
    # 執行所有模型
    exp.run_all_models()
    
    # 顯示摘要
    summary = exp.print_summary()
    
    # 儲存結果
    exp.save_results(f'{output_prefix}_results.csv')
    
    # 生成視覺化
    create_all_visualizations(exp, output_prefix=output_prefix)
    
    print("\n" + "="*80)
    print(f"✓ EXPERIMENT COMPLETED: {dataset_name}")
    print("="*80 + "\n")
    
    return exp, summary


def run_complete_ase_experiment():
    """執行完整的日月光實驗"""
    
    print("\n" + "#"*80)
    print("#" + " "*20 + "ASE FOCOS THERMAL PREDICTION" + " "*20 + "#")
    print("#" + " "*15 + "Complete DIM-GP Variants Comparison" + " "*15 + "#")
    print("#"*80 + "\n")
    
    # 載入資料
    data, feature_cols = load_ase_data()
    
    # Above 50% 實驗
    exp_above, summary_above = run_single_dataset_experiment(
        data['above']['X_train'],
        data['above']['y_train'],
        data['above']['X_test'],
        data['above']['y_test'],
        dataset_name='Above 50% Coverage',
        output_prefix='above'
    )
    
    # Below 50% 實驗
    exp_below, summary_below = run_single_dataset_experiment(
        data['below']['X_train'],
        data['below']['y_train'],
        data['below']['X_test'],
        data['below']['y_test'],
        dataset_name='Below 50% Coverage',
        output_prefix='below'
    )
    
    # 生成比較報告
    generate_comparison_report(exp_above, exp_below, summary_above, summary_below)
    
    print("\n" + "#"*80)
    print("#" + " "*25 + "ALL EXPERIMENTS COMPLETED!" + " "*25 + "#")
    print("#"*80 + "\n")
    
    return exp_above, exp_below


def generate_comparison_report(exp_above, exp_below, summary_above, summary_below):
    """生成兩個資料集的比較報告"""
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: Above vs Below 50% Coverage")
    print("="*80 + "\n")
    
    # 合併摘要
    summary_above['Dataset'] = 'Above 50%'
    summary_below['Dataset'] = 'Below 50%'
    
    combined = pd.concat([summary_above, summary_below], ignore_index=True)
    
    # 按模型分組比較
    print("\n📊 Model Performance Comparison:\n")
    
    for model_name in summary_above['Model'].unique():
        if model_name in summary_below['Model'].values:
            above_metrics = summary_above[summary_above['Model'] == model_name].iloc[0]
            below_metrics = summary_below[summary_below['Model'] == model_name].iloc[0]
            
            print(f"─"*80)
            print(f"{model_name}:")
            print(f"  Above 50%: R²={above_metrics['R²']:.6f}, RMSE={above_metrics['RMSE']:.6f}")
            print(f"  Below 50%: R²={below_metrics['R²']:.6f}, RMSE={below_metrics['RMSE']:.6f}")
            
            # 比較UQ
            if above_metrics['Has UQ'] == '✓' and 'CI Coverage (%)' in above_metrics:
                print(f"  UQ Above: Coverage={above_metrics['CI Coverage (%)']:.2f}%, Width={above_metrics['CI Width']:.6f}")
                print(f"  UQ Below: Coverage={below_metrics['CI Coverage (%)']:.2f}%, Width={below_metrics['CI Width']:.6f}")
    
    print(f"─"*80 + "\n")
    
    # 最佳模型
    print("\n🏆 Best Models:\n")
    print(f"  Above 50%: {summary_above.iloc[0]['Model']} (R²={summary_above.iloc[0]['R²']:.6f})")
    print(f"  Below 50%: {summary_below.iloc[0]['Model']} (R²={summary_below.iloc[0]['R²']:.6f})")
    
    # 最佳UQ模型
    uq_above = summary_above[summary_above['Has UQ'] == '✓']
    uq_below = summary_below[summary_below['Has UQ'] == '✓']
    
    if len(uq_above) > 0 and len(uq_below) > 0:
        print(f"\n🎯 Best with Uncertainty Quantification:\n")
        print(f"  Above 50%: {uq_above.iloc[0]['Model']} (R²={uq_above.iloc[0]['R²']:.6f})")
        print(f"  Below 50%: {uq_below.iloc[0]['Model']} (R²={uq_below.iloc[0]['R²']:.6f})")
    
    print("\n" + "="*80 + "\n")
    
    # 儲存比較報告
    combined.to_csv('comparison_report.csv', index=False)
    print("✓ Comparison report saved to comparison_report.csv\n")


def quick_test():
    """快速測試 - 只跑Above資料集的部分模型"""
    print("\n" + "="*80)
    print("QUICK TEST MODE")
    print("="*80 + "\n")
    
    data, _ = load_ase_data()
    
    exp = ExperimentFramework(dataset_name='Above 50% (Quick Test)')
    exp.load_data(
        data['above']['X_train'],
        data['above']['y_train'],
        data['above']['X_test'],
        data['above']['y_test']
    )
    
    # 只跑幾個快速模型
    print("Running: MLP, XGBoost, GP\n")
    exp.run_model('MLP')
    exp.run_model('XGBoost')
    exp.run_model('GP', {'subsample': 500})
    
    exp.print_summary()
    
    print("\n✓ Quick test completed!\n")
    return exp


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("ASE FOCOS THERMAL PREDICTION - DIM-GP COMPLETE COMPARISON")
    print("="*80 + "\n")
    
    print("Available modes:")
    print("  1. Full experiment (all models, both datasets)")
    print("  2. Quick test (few models, one dataset)")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # 快速測試模式
        exp = quick_test()
    else:
        # 完整實驗
        print("Starting FULL EXPERIMENT...")
        print("This will take approximately 15-30 minutes.\n")
        
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            exp_above, exp_below = run_complete_ase_experiment()
            
            print("\n📁 Generated Files:")
            print("  - above_results.csv, below_results.csv")
            print("  - above_comprehensive.png, below_comprehensive.png")
            print("  - above_predictions_grid.png, below_predictions_grid.png")
            print("  - above_calibration.png, below_calibration.png")
            print("  - above_feature_importance.png, below_feature_importance.png")
            print("  - comparison_report.csv")
            print()
        else:
            print("Experiment cancelled.")
