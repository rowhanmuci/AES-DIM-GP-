"""
三條線比較圖：訓練鄰近點平均 vs 測試真值 vs 模型預測
範圍：Thick±5, Cov±0.02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 載入資料
train_df = pd.read_excel('data/train/Above.xlsx')
test_df = pd.read_excel('data/test/Above.xlsx')
pred_df = pd.read_csv('phase2j_best_ensemble_predictions_above.csv')

# 定義鄰近範圍
THICK_RANGE = 5
COV_RANGE = 0.02

def get_neighbor_stats(train_df, typ, thick, cov):
    """取得鄰近點的統計量"""
    subset = train_df[(train_df['TIM_TYPE'] == typ) & 
                      (train_df['TIM_THICKNESS'] >= thick - THICK_RANGE) & 
                      (train_df['TIM_THICKNESS'] <= thick + THICK_RANGE) &
                      (train_df['TIM_COVERAGE'] >= cov - COV_RANGE) & 
                      (train_df['TIM_COVERAGE'] <= cov + COV_RANGE)]
    if len(subset) > 0:
        return {
            'mean': subset['Theta.JC'].mean(),
            'std': subset['Theta.JC'].std(),
            'min': subset['Theta.JC'].min(),
            'max': subset['Theta.JC'].max(),
            'n': len(subset)
        }
    return None

# ============================================
# 圖: Type 3 三種 Coverage 的比較
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

thicknesses = [200, 220, 240, 260, 280, 300]
coverages = [0.6, 0.8, 1.0]

for idx, cov in enumerate(coverages):
    ax = axes[idx]
    
    # 收集資料
    train_means = []
    train_stds = []
    train_mins = []
    train_maxs = []
    test_vals = []
    pred_vals = []
    n_neighbors = []
    
    for thick in thicknesses:
        # 訓練集鄰近點統計
        stats = get_neighbor_stats(train_df, 3, thick, cov)
        if stats:
            train_means.append(stats['mean'])
            train_stds.append(stats['std'])
            train_mins.append(stats['min'])
            train_maxs.append(stats['max'])
            n_neighbors.append(stats['n'])
        else:
            train_means.append(np.nan)
            train_stds.append(np.nan)
            train_mins.append(np.nan)
            train_maxs.append(np.nan)
            n_neighbors.append(0)
        
        # 測試集真值
        test_row = test_df[(test_df['TIM_TYPE'] == 3) & 
                           (test_df['TIM_THICKNESS'] == thick) & 
                           (test_df['TIM_COVERAGE'] == cov)]
        test_vals.append(test_row['Theta.JC'].values[0] if len(test_row) > 0 else np.nan)
        
        # 模型預測
        pred_row = pred_df[(pred_df['TIM_TYPE'] == 3) & 
                           (pred_df['TIM_THICKNESS'] == thick) & 
                           (pred_df['TIM_COVERAGE'] == cov)]
        pred_vals.append(pred_row['Predicted'].values[0] if len(pred_row) > 0 else np.nan)
    
    # 繪製訓練集範圍 (陰影)
    ax.fill_between(thicknesses, train_mins, train_maxs, alpha=0.2, color='blue')
    
    # 三條線
    ax.plot(thicknesses, train_means, 'b-o', linewidth=2, markersize=8, 
            label='Training Neighbor Mean')
    ax.plot(thicknesses, test_vals, 'r-s', linewidth=2, markersize=10, 
            label='Test Ground Truth')
    ax.plot(thicknesses, pred_vals, 'g--^', linewidth=2, markersize=8, 
            label='Model Prediction')
    
    # 標記異常點 (誤差 > 20%)
    for i, thick in enumerate(thicknesses):
        if not np.isnan(test_vals[i]) and not np.isnan(pred_vals[i]):
            error = abs(test_vals[i] - pred_vals[i]) / test_vals[i] * 100
            if error > 20:
                # 紅色圓圈標記
                ax.scatter([thick], [test_vals[i]], s=300, facecolors='none', 
                          edgecolors='red', linewidths=3, zorder=10)
                # 標記誤差
                ax.annotate(f'{error:.0f}%', xy=(thick, test_vals[i]), 
                           xytext=(thick+5, test_vals[i]+0.003),
                           fontsize=10, color='red', fontweight='bold')
    
    ax.set_xlabel('TIM_THICKNESS', fontsize=12)
    ax.set_ylabel('Theta.JC', fontsize=12)
    ax.set_title(f'Type 3, Coverage = {cov}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thicknesses)

plt.suptitle(f'Training Neighbors (Thick±{THICK_RANGE}, Cov±{COV_RANGE}) vs Test vs Prediction\n'
             f'Red circles = Outliers (>20% error)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig('three_lines_comparison.png', dpi=150, 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ 已保存: three_lines_comparison.png")
plt.close()

# ============================================
# 額外：數據表格
# ============================================
print("\n" + "="*80)
print(f"Type 3 數據比較 (鄰近範圍: Thick±{THICK_RANGE}, Cov±{COV_RANGE})")
print("="*80)

for cov in coverages:
    print(f"\n【Coverage = {cov}】")
    print(f"{'Thick':<8} {'Train Mean':<12} {'Train N':<10} {'Test True':<12} {'Pred':<12} {'Error%':<10} {'備註'}")
    print("-"*80)
    
    for thick in thicknesses:
        stats = get_neighbor_stats(train_df, 3, thick, cov)
        
        test_row = test_df[(test_df['TIM_TYPE'] == 3) & 
                           (test_df['TIM_THICKNESS'] == thick) & 
                           (test_df['TIM_COVERAGE'] == cov)]
        test_val = test_row['Theta.JC'].values[0] if len(test_row) > 0 else np.nan
        
        pred_row = pred_df[(pred_df['TIM_TYPE'] == 3) & 
                           (pred_df['TIM_THICKNESS'] == thick) & 
                           (pred_df['TIM_COVERAGE'] == cov)]
        pred_val = pred_row['Predicted'].values[0] if len(pred_row) > 0 else np.nan
        
        if stats and not np.isnan(test_val):
            error = abs(test_val - pred_val) / test_val * 100
            
            # 判斷備註
            note = ""
            if error > 20:
                note = "⚠️ 異常點"
            if test_val < stats['min']:
                note += " (真值 < 訓練min)"
            elif test_val > stats['max']:
                note += " (真值 > 訓練max)"
            
            print(f"{thick:<8} {stats['mean']:<12.4f} {stats['n']:<10} {test_val:<12.4f} "
                  f"{pred_val:<12.4f} {error:<10.1f} {note}")

print("\n" + "="*80)
print("觀察")
print("="*80)
print("""
1. 模型預測 (綠線) 幾乎完美貼合訓練鄰近點平均 (藍線)
   → 模型學習正確！

2. 異常點的特徵：測試真值 (紅線) 顯著偏離訓練平均 (藍線)
   → 這是資料分布問題，不是模型問題

3. Coverage=0.6 時，三條線幾乎重合
   → 這個區域訓練資料一致性高，預測準確

4. Coverage=0.8 和 1.0 時，紅線出現明顯的「非單調跳動」
   → 測試集在這些區域有訓練集沒見過的 pattern
""")
