"""
Above 資料集中 100% 誤差的點分布 - 修正版 (上下圖)
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

# 找出 100% 誤差的重複組合
feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
grouped = train_df.groupby(feature_cols)

dup_100 = []
for name, group in grouped:
    if len(group) > 1:
        vals = group['Theta.JC'].values
        if vals.min() > 0:
            error_pct = (vals.max() - vals.min()) / vals.min() * 100
            if error_pct >= 100:
                typ, thick, cov = name
                dup_100.append({
                    'type': typ, 'thick': thick, 'cov': cov,
                    'min': vals.min(), 'max': vals.max(),
                    'values': sorted(vals.tolist())
                })

# 上下兩圖
fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1.5]})

# ============================================
# 上圖: Thickness vs Coverage 散點圖
# ============================================
ax = axes[0]

# 所有 Type 3 訓練資料 (灰色背景)
train_t3 = train_df[train_df['TIM_TYPE'] == 3]
ax.scatter(train_t3['TIM_THICKNESS'], train_t3['TIM_COVERAGE'], 
           alpha=0.2, s=20, c='gray', label='All Type 3 data')

# 100% 誤差的點 (紅色)
dup_100_t3 = [d for d in dup_100 if d['type'] == 3]
if dup_100_t3:
    thick_100 = [d['thick'] for d in dup_100_t3]
    cov_100 = [d['cov'] for d in dup_100_t3]
    ax.scatter(thick_100, cov_100, s=120, c='red', marker='X', 
               edgecolors='darkred', linewidths=1.5, zorder=5,
               label=f'100% error groups (n={len(dup_100_t3)})')

ax.set_xlabel('TIM_THICKNESS', fontsize=12)
ax.set_ylabel('TIM_COVERAGE', fontsize=12)
ax.set_title('Type 3: Location of 100% Error Groups\n(Same features but Theta.JC differs by 100%+)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)

# 標註區域
ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(305, 0.81, 'Coverage ≥ 0.8', fontsize=11, color='red', fontweight='bold')

# ============================================
# 下圖: 100% 誤差點的詳細列表
# ============================================
ax = axes[1]
ax.axis('off')

# 準備表格資料
table_data = []
for d in sorted(dup_100, key=lambda x: (x['type'], x['thick'], x['cov'])):
    vals_str = ', '.join([f"{v:.2f}" for v in d['values']])
    table_data.append([f"{d['type']}", f"{d['thick']:.0f}", f"{d['cov']:.2f}", 
                       f"{d['min']:.2f}", f"{d['max']:.2f}", vals_str])

columns = ['Type', 'Thick', 'Cov', 'Min', 'Max', 'All Values']

# 建立表格
table = ax.table(cellText=table_data, colLabels=columns, loc='upper center', cellLoc='center',
                 colColours=['#d5dbdb']*6,
                 colWidths=[0.08, 0.10, 0.10, 0.10, 0.10, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# 標題列樣式
for i in range(6):
    table[(0, i)].set_text_props(fontweight='bold')

ax.set_title(f'All 100% Error Groups in Training Data (n={len(dup_100)})\nSame (Type, Thick, Cov) but Theta.JC = 0.01 and 0.02', 
             fontsize=13, fontweight='bold', y=1.0, pad=20)

plt.suptitle('Above Dataset: Training Data with 100% Internal Inconsistency', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('above_100pct_error_points.png', dpi=150, 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ 已保存: above_100pct_error_points.png")
plt.close()

print("\n圖表生成完成！")