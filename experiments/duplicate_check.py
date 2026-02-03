import pandas as pd
import numpy as np

train_df = pd.read_excel('data/train/Above.xlsx')
#train_df = pd.read_excel('data/train/Below.xlsx')
print("="*80)
print("分析 2: 訓練集重複資料的 Theta.JC 差異")
print("="*80)

# 找出重複的組合
feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
grouped = train_df.groupby(feature_cols)

# 計算每個組合的統計量
dup_analysis = []
for name, group in grouped:
    if len(group) > 1:  # 只看有重複的
        typ, thick, cov = name
        vals = group['Theta.JC'].values
        min_val = vals.min()
        max_val = vals.max()
        
        if min_val > 0:
            # 用 MAPE 的方式計算：相對於平均值的變異
            mean_val = vals.mean()
            max_diff_pct = (max_val - min_val) / mean_val * 100
            
            # 也計算最大值相對最小值的誤差 (你說的方式)
            error_pct = (max_val - min_val) / min_val * 100
            
            dup_analysis.append({
                'type': typ, 'thick': thick, 'cov': cov,
                'n': len(group), 'min': min_val, 'max': max_val,
                'mean': mean_val, 'std': vals.std(),
                'diff': max_val - min_val,
                'error_pct': error_pct,
                'max_diff_pct': max_diff_pct,
                'values': sorted(vals.tolist())
            })

print(f"\n總共有 {len(dup_analysis)} 個重複的組合")

# 按誤差排序
dup_sorted = sorted(dup_analysis, key=lambda x: x['error_pct'], reverse=True)

print(f"\n誤差 >= 50% 的重複組合 (共 {sum(1 for d in dup_sorted if d['error_pct'] >= 50)} 個):")
print("-"*100)
print(f"{'Type':<6} {'Thick':<8} {'Cov':<8} {'N':<4} {'Min':<8} {'Max':<8} {'誤差%':<10} {'所有值'}")
print("-"*100)

count = 0
for d in dup_sorted:
    if d['error_pct'] >= 50:
        vals_str = ', '.join([f"{v:.3f}" for v in d['values']])
        print(f"{d['type']:<6} {d['thick']:<8.0f} {d['cov']:<8.2f} {d['n']:<4} "
              f"{d['min']:<8.3f} {d['max']:<8.3f} {d['error_pct']:<10.1f} [{vals_str}]")
        count += 1
        if count >= 30:
            print(f"... 還有 {sum(1 for d in dup_sorted if d['error_pct'] >= 50) - 30} 個")
            break

# 統計
print(f"\n" + "="*80)
print("重複組合的誤差分布統計")
print("="*80)

errors = [d['error_pct'] for d in dup_analysis]
print(f"總共 {len(errors)} 個重複組合")
print(f"誤差 >= 100% (翻倍): {sum(1 for e in errors if e >= 100)} 個 ({sum(1 for e in errors if e >= 100)/len(errors)*100:.1f}%)")
print(f"誤差 >= 50%: {sum(1 for e in errors if e >= 50)} 個 ({sum(1 for e in errors if e >= 50)/len(errors)*100:.1f}%)")
print(f"誤差 >= 20%: {sum(1 for e in errors if e >= 20)} 個 ({sum(1 for e in errors if e >= 20)/len(errors)*100:.1f}%)")
print(f"誤差 < 20%: {sum(1 for e in errors if e < 20)} 個 ({sum(1 for e in errors if e < 20)/len(errors)*100:.1f}%)")
print(f"誤差 < 5%: {sum(1 for e in errors if e < 5)} 個 ({sum(1 for e in errors if e < 5)/len(errors)*100:.1f}%)")

print(f"\n誤差統計: min={min(errors):.1f}%, max={max(errors):.1f}%, "
      f"mean={np.mean(errors):.1f}%, median={np.median(errors):.1f}%")