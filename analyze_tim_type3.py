"""
異常點深度分析 - 針對TIM_TYPE=3
發現: 10個異常點中7個都是TIM_TYPE=3，且THICKNESS都很大
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_tim_type3_outliers():
    """分析TIM_TYPE=3為何異常"""
    
    print("="*60)
    print("TIM_TYPE=3 異常點深度分析")
    print("="*60 + "\n")
    
    # 載入資料
    train_df = pd.read_excel('data/train/Above.xlsx')
    test_df = pd.read_csv('phase1_above_predictions.csv')
    
    # 分離異常點
    outliers = test_df[test_df['Error%'] > 20]
    normals = test_df[test_df['Error%'] <= 20]
    
    print(f"異常點總數: {len(outliers)}")
    print(f"TIM_TYPE=3的異常點: {len(outliers[outliers['TIM_TYPE'] == 3])}\n")
    
    # 分析1: TIM_TYPE=3在訓練集中的分布
    print("="*60)
    print("訓練集中TIM_TYPE分布")
    print("="*60 + "\n")
    
    type_counts_train = train_df['TIM_TYPE'].value_counts().sort_index()
    print("訓練集:")
    for tim_type, count in type_counts_train.items():
        pct = count / len(train_df) * 100
        print(f"  Type {tim_type}: {count} 筆 ({pct:.2f}%)")
    
    print("\n測試集:")
    type_counts_test = test_df['TIM_TYPE'].value_counts().sort_index()
    for tim_type, count in type_counts_test.items():
        pct = count / len(test_df) * 100
        print(f"  Type {tim_type}: {count} 筆 ({pct:.2f}%)")
    
    # 分析2: TIM_TYPE=3的THICKNESS分布
    print("\n" + "="*60)
    print("TIM_TYPE=3的THICKNESS分析")
    print("="*60 + "\n")
    
    train_type3 = train_df[train_df['TIM_TYPE'] == 3]
    test_type3 = test_df[test_df['TIM_TYPE'] == 3]
    outlier_type3 = outliers[outliers['TIM_TYPE'] == 3]
    
    print("THICKNESS統計:")
    print(f"\n訓練集 (Type 3):")
    print(f"  範圍: [{train_type3['TIM_THICKNESS'].min():.1f}, {train_type3['TIM_THICKNESS'].max():.1f}]")
    print(f"  平均: {train_type3['TIM_THICKNESS'].mean():.1f}")
    print(f"  中位數: {train_type3['TIM_THICKNESS'].median():.1f}")
    
    print(f"\n測試集 (Type 3):")
    print(f"  範圍: [{test_type3['TIM_THICKNESS'].min():.1f}, {test_type3['TIM_THICKNESS'].max():.1f}]")
    print(f"  平均: {test_type3['TIM_THICKNESS'].mean():.1f}")
    
    print(f"\n異常點 (Type 3):")
    print(f"  範圍: [{outlier_type3['TIM_THICKNESS'].min():.1f}, {outlier_type3['TIM_THICKNESS'].max():.1f}]")
    print(f"  平均: {outlier_type3['TIM_THICKNESS'].mean():.1f}")
    
    # 檢查外推
    train_max_thick = train_type3['TIM_THICKNESS'].max()
    outlier_thick = outlier_type3['TIM_THICKNESS'].values
    
    print(f"\n⚠️  外推問題:")
    out_of_range = outlier_thick[outlier_thick > train_max_thick]
    if len(out_of_range) > 0:
        print(f"  {len(out_of_range)} 個異常點的THICKNESS超出訓練範圍")
        print(f"  訓練最大值: {train_max_thick:.1f}")
        print(f"  異常點超出值: {out_of_range}")
    else:
        print(f"  所有異常點都在訓練範圍內")
    
    # 分析3: 不同COVERAGE的表現
    print("\n" + "="*60)
    print("TIM_TYPE=3的COVERAGE分析")
    print("="*60 + "\n")
    
    for coverage in sorted(outlier_type3['TIM_COVERAGE'].unique()):
        subset = outlier_type3[outlier_type3['TIM_COVERAGE'] == coverage]
        print(f"COVERAGE={coverage}:")
        print(f"  異常點數: {len(subset)}")
        print(f"  平均誤差: {subset['Error%'].mean():.2f}%")
        print(f"  THICKNESS範圍: [{subset['TIM_THICKNESS'].min():.1f}, {subset['TIM_THICKNESS'].max():.1f}]")
        print()
    
    # 分析4: 訓練集中Type3的Theta.JC分布
    print("="*60)
    print("Theta.JC分布比較")
    print("="*60 + "\n")
    
    print(f"訓練集 (Type 3):")
    print(f"  範圍: [{train_type3['Theta.JC'].min():.4f}, {train_type3['Theta.JC'].max():.4f}]")
    print(f"  平均: {train_type3['Theta.JC'].mean():.4f}")
    
    print(f"\n異常點 (Type 3):")
    print(f"  真實值範圍: [{outlier_type3['Theta.JC'].min():.4f}, {outlier_type3['Theta.JC'].max():.4f}]")
    print(f"  預測值範圍: [{outlier_type3['Prediction'].min():.4f}, {outlier_type3['Prediction'].max():.4f}]")
    
    # 視覺化
    create_type3_visualization(train_df, test_df, outliers)
    
    # 改進建議
    print("\n" + "="*60)
    print("💡 改進建議")
    print("="*60 + "\n")
    
    suggestions = [
        "1. TIM_TYPE=3的樣本可能需要特殊處理",
        "2. 大THICKNESS值的外推問題 → 考慮增加大THICKNESS的訓練樣本",
        "3. TIM_TYPE可能需要更好的特徵表示 (Entity Embedding)",
        "4. 考慮對TIM_TYPE=3使用不同的模型或參數",
        "5. 對大THICKNESS區域使用樣本加權訓練",
    ]
    
    for s in suggestions:
        print(f"  {s}")
    
    print("\n" + "="*60 + "\n")


def create_type3_visualization(train_df, test_df, outliers):
    """創建TIM_TYPE=3的視覺化"""
    
    print("生成視覺化圖表...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TIM_TYPE=3 異常點分析', fontsize=16, fontweight='bold')
    
    # 1. THICKNESS分布對比
    ax1 = axes[0, 0]
    
    train_type3 = train_df[train_df['TIM_TYPE'] == 3]
    test_type3 = test_df[test_df['TIM_TYPE'] == 3]
    outlier_type3 = outliers[outliers['TIM_TYPE'] == 3]
    normal_type3 = test_type3[test_type3['Error%'] <= 20]
    
    ax1.hist(train_type3['TIM_THICKNESS'], bins=20, alpha=0.5, label='Training', color='blue')
    ax1.hist(normal_type3['TIM_THICKNESS'], bins=20, alpha=0.5, label='Test (Normal)', color='green')
    ax1.hist(outlier_type3['TIM_THICKNESS'], bins=10, alpha=0.7, label='Test (Outliers)', color='red')
    
    ax1.set_xlabel('TIM_THICKNESS')
    ax1.set_ylabel('Count')
    ax1.set_title('TIM_TYPE=3: THICKNESS分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Theta.JC vs THICKNESS (Type 3 only)
    ax2 = axes[0, 1]
    
    ax2.scatter(train_type3['TIM_THICKNESS'], train_type3['Theta.JC'], 
               alpha=0.3, s=20, label='Training', color='blue')
    ax2.scatter(normal_type3['TIM_THICKNESS'], normal_type3['Theta.JC'], 
               alpha=0.7, s=50, label='Test (Normal)', color='green')
    ax2.scatter(outlier_type3['TIM_THICKNESS'], outlier_type3['Theta.JC'], 
               alpha=0.9, s=100, marker='X', label='Test (Outliers)', color='red')
    
    ax2.set_xlabel('TIM_THICKNESS')
    ax2.set_ylabel('Theta.JC')
    ax2.set_title('TIM_TYPE=3: Theta.JC vs THICKNESS')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 誤差 vs THICKNESS
    ax3 = axes[1, 0]
    
    ax3.scatter(test_type3['TIM_THICKNESS'], test_type3['Error%'], 
               c=test_type3['Error%'], cmap='RdYlGn_r', s=100, alpha=0.7)
    ax3.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Threshold (20%)')
    
    ax3.set_xlabel('TIM_THICKNESS')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('TIM_TYPE=3: 誤差 vs THICKNESS')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 不同COVERAGE的誤差分布
    ax4 = axes[1, 1]
    
    coverage_values = sorted(test_type3['TIM_COVERAGE'].unique())
    errors_by_coverage = [test_type3[test_type3['TIM_COVERAGE'] == c]['Error%'].values 
                         for c in coverage_values]
    
    bp = ax4.boxplot(errors_by_coverage, labels=[f'{c}' for c in coverage_values])
    ax4.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Threshold')
    
    ax4.set_xlabel('TIM_COVERAGE')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('TIM_TYPE=3: 不同COVERAGE的誤差分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tim_type3_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 視覺化已保存: tim_type3_analysis.png\n")
    plt.close()


if __name__ == "__main__":
    analyze_tim_type3_outliers()
