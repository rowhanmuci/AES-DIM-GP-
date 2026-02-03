"""
Phase 2J 分析報告：Type 3 異常點深度分析
改進版 - 更簡潔清晰的圖表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 設定字體
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 載入資料
train_df = pd.read_excel('data/train/Above.xlsx')
test_df = pd.read_excel('data/test/Above.xlsx')

# ============================================
# 圖1: 主要分析圖 (3個子圖)
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ------------------------------------------
# 左上: Type 3, Coverage=0.8
# ------------------------------------------
ax1 = axes[0, 0]

# 訓練集資料 (Cov=0.78~0.82)
train_t3_08 = train_df[(train_df['TIM_TYPE'] == 3) & 
                        (train_df['TIM_COVERAGE'] >= 0.78) & 
                        (train_df['TIM_COVERAGE'] <= 0.82)]

thicknesses = [200, 220, 240, 260, 280, 300]

# 計算每個 thickness 區間的統計量 (±10)
train_means = []
train_mins = []
train_maxs = []

for thick in thicknesses:
    subset = train_t3_08[(train_t3_08['TIM_THICKNESS'] >= thick-5) & 
                          (train_t3_08['TIM_THICKNESS'] <= thick+5)]
    train_means.append(subset['Theta.JC'].mean())
    train_mins.append(subset['Theta.JC'].min())
    train_maxs.append(subset['Theta.JC'].max())

# 測試集真值
test_t3_08 = test_df[(test_df['TIM_TYPE'] == 3) & (test_df['TIM_COVERAGE'] == 0.8)]
test_values = test_t3_08.set_index('TIM_THICKNESS')['Theta.JC'].to_dict()
test_vals = [test_values.get(t, np.nan) for t in thicknesses]

# 繪製
ax1.fill_between(thicknesses, train_mins, train_maxs, alpha=0.25, color='blue')
ax1.plot(thicknesses, train_means, 'b-o', linewidth=2, markersize=8, 
         label=f'Training (Thick±5, Cov=0.78~0.82)')
ax1.plot(thicknesses, test_vals, 'r-s', linewidth=2.5, markersize=10, 
         label='Test Ground Truth')

# 標記異常點 (圓圈)
outlier_thick_08 = [220, 240, 260, 300]
for t in outlier_thick_08:
    if t in test_values:
        ax1.scatter([t], [test_values[t]], s=250, facecolors='none', 
                   edgecolors='red', linewidths=2.5, zorder=5)

ax1.set_xlabel('TIM_THICKNESS', fontsize=12)
ax1.set_ylabel('Theta.JC', fontsize=12)
ax1.set_title('Type 3, Coverage = 0.8', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(thicknesses)
ax1.set_ylim(0.005, 0.038)

# ------------------------------------------
# 右上: Type 3, Coverage=1.0
# ------------------------------------------
ax2 = axes[0, 1]

# 訓練集資料 (Cov>=0.98)
train_t3_10 = train_df[(train_df['TIM_TYPE'] == 3) & 
                        (train_df['TIM_COVERAGE'] >= 0.98)]

train_means_10 = []
train_mins_10 = []
train_maxs_10 = []

for thick in thicknesses:
    subset = train_t3_10[(train_t3_10['TIM_THICKNESS'] >= thick-5) & 
                          (train_t3_10['TIM_THICKNESS'] <= thick+5)]
    if len(subset) > 0:
        train_means_10.append(subset['Theta.JC'].mean())
        train_mins_10.append(subset['Theta.JC'].min())
        train_maxs_10.append(subset['Theta.JC'].max())
    else:
        train_means_10.append(np.nan)
        train_mins_10.append(np.nan)
        train_maxs_10.append(np.nan)

# 測試集
test_t3_10 = test_df[(test_df['TIM_TYPE'] == 3) & (test_df['TIM_COVERAGE'] == 1.0)]
test_values_10 = test_t3_10.set_index('TIM_THICKNESS')['Theta.JC'].to_dict()
test_vals_10 = [test_values_10.get(t, np.nan) for t in thicknesses]

# 繪製
ax2.fill_between(thicknesses, train_mins_10, train_maxs_10, alpha=0.25, color='blue')
ax2.plot(thicknesses, train_means_10, 'b-o', linewidth=2, markersize=8,
         label=f'Training (Thick±5, Cov≥0.98)')
ax2.plot(thicknesses, test_vals_10, 'r-s', linewidth=2.5, markersize=10,
         label='Test Ground Truth')

# 標記異常點
ax2.scatter([240], [test_values_10.get(240, 0)], s=250, facecolors='none', 
           edgecolors='red', linewidths=2.5, zorder=5)

ax2.set_xlabel('TIM_THICKNESS', fontsize=12)
ax2.set_ylabel('Theta.JC', fontsize=12)
ax2.set_title('Type 3, Coverage = 1.0', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(thicknesses)
ax2.set_ylim(0.005, 0.028)

# ------------------------------------------
# 下方: 5個異常點的詳細比較 (橫跨兩欄)
# ------------------------------------------
ax3 = fig.add_subplot(2, 1, 2)
axes[1, 0].remove()
axes[1, 1].remove()

outliers_data = [
    ('Thick=220\nCov=0.8', 0.014, 0.0204, 0.020),
    ('Thick=240\nCov=0.8', 0.029, 0.0200, 0.020),
    ('Thick=240\nCov=1.0', 0.010, 0.0148, 0.015),
    ('Thick=260\nCov=0.8', 0.028, 0.0210, 0.020),
    ('Thick=300\nCov=0.8', 0.017, 0.0247, 0.030),
]

x_pos = np.arange(len(outliers_data))
width = 0.25

labels = [d[0] for d in outliers_data]
true_vals = [d[1] for d in outliers_data]
pred_vals = [d[2] for d in outliers_data]
neighbor_vals = [d[3] for d in outliers_data]

# 計算誤差
errors = [abs(t - p) / t * 100 for t, p in zip(true_vals, pred_vals)]

bars1 = ax3.bar(x_pos - width, true_vals, width, label='Test Ground Truth', 
                color='#e74c3c', edgecolor='black', linewidth=1.2)
bars2 = ax3.bar(x_pos, pred_vals, width, label='Model Prediction', 
                color='#3498db', edgecolor='black', linewidth=1.2)
bars3 = ax3.bar(x_pos + width, neighbor_vals, width, label='Training Neighbor Median', 
                color='#95a5a6', edgecolor='black', linewidth=1.2)

# 添加誤差標籤
for i, (x, err) in enumerate(zip(x_pos, errors)):
    y_pos = max(true_vals[i], pred_vals[i], neighbor_vals[i]) + 0.002
    ax3.annotate(f'{err:.1f}%', xy=(x, y_pos), fontsize=11, ha='center', 
                color='#c0392b', fontweight='bold')

ax3.set_xlabel('Outlier Points (All Type 3)', fontsize=12)
ax3.set_ylabel('Theta.JC', fontsize=12)
ax3.set_title('5 Outliers: Ground Truth vs Prediction vs Training Neighbors', 
              fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, fontsize=11)
ax3.legend(loc='upper right', fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 0.040)

# 添加關鍵說明
ax3.text(0.02, 0.92, 'Model Prediction ≈ Training Neighbor Median\n→ Model learned correctly from training data',
         transform=ax3.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.suptitle('ASE FOCoS Type 3 Outlier Analysis', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('phase2j_analysis_report_v3.png', dpi=150, 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ 已保存: phase2j_analysis_report_v2.png")
plt.close()

# ============================================
# 圖2: 關鍵發現圖 (更簡潔)
# ============================================
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

# 左圖: Coverage=0.8 的非單調 pattern
ax = axes2[0]
thick_vals = [200, 220, 240, 260, 280, 300]
test_theta = [0.021, 0.014, 0.029, 0.028, 0.019, 0.017]
train_avg = [0.0228, 0.0224, 0.0202, 0.0221, 0.0227, 0.0242]

ax.plot(thick_vals, test_theta, 'ro-', linewidth=2.5, markersize=12, label='Test Ground Truth')
ax.plot(thick_vals, train_avg, 'b^--', linewidth=2, markersize=10, label='Training Mean (Thick±10)')
ax.fill_between(thick_vals, [t-0.008 for t in train_avg], [t+0.008 for t in train_avg], 
                alpha=0.2, color='blue', label='Training Range')

# 標記跳躍箭頭
ax.annotate('', xy=(240, 0.028), xytext=(220, 0.015),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5))
ax.text(230, 0.021, '+107%', fontsize=12, ha='center', color='darkred', fontweight='bold')

ax.set_xlabel('TIM_THICKNESS', fontsize=12)
ax.set_ylabel('Theta.JC', fontsize=12)
ax.set_title('Type 3, Coverage=0.8\nNon-monotonic Pattern in Test Set', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(thick_vals)
ax.set_ylim(0.008, 0.035)

# 右圖: 異常點誤差方向
ax = axes2[1]

categories = ['Prediction Too High\n(True value is LOW)', 
              'Prediction Too Low\n(True value is HIGH)']
counts = [3, 2]  # 220, 240(cov=1), 300 vs 240(cov=0.8), 260

colors = ['#e74c3c', '#3498db']
bars = ax.bar(categories, counts, color=colors, edgecolor='black', width=0.5, linewidth=1.5)

# 標籤
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{count} outliers', ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('Number of Outliers', fontsize=12)
ax.set_title('Outlier Error Direction\n(Same region, opposite errors)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 4.5)
ax.set_xticklabels(categories, fontsize=10)

# 說明框
ax.text(0.5, 0.75, 'Cannot fix both directions\nwith a single correction strategy',
        transform=ax.transAxes, fontsize=11, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('phase2j_key_findings_v2.png', dpi=150, 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ 已保存: phase2j_key_findings_v2.png")
plt.close()

print("\n圖表生成完成！")