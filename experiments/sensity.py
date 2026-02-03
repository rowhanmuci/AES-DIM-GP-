import pandas as pd
import numpy as np

train_df = pd.read_excel('data/train/Above.xlsx')
print("="*80)
print("對其他 4 個異常點也做同樣的敏感性分析")
print("="*80)

outliers = [
    (3, 220, 0.8, 0.014),
    (3, 240, 1.0, 0.010),
    (3, 260, 0.8, 0.028),
    (3, 300, 0.8, 0.017),
]

thick_ranges = [5, 10, 20]
cov_ranges = [0.02, 0.05, 0.10, 0.15]

for test_type, test_thick, test_cov, test_true in outliers:
    print(f"\n{'='*80}")
    print(f"Type={test_type}, Thick={test_thick}, Cov={test_cov}, 真值={test_true}")
    print(f"{'='*80}")
    print(f"{'Thick±':<8} {'Cov±':<8} {'N':<6} {'Mean':<10} {'Min~Max':<15} {'誤差%'}")
    print("-"*60)
    
    best_error = 999
    best_config = None
    
    for thick_r in thick_ranges:
        for cov_r in cov_ranges:
            subset = train_df[(train_df['TIM_TYPE'] == test_type) & 
                              (train_df['TIM_THICKNESS'] >= test_thick - thick_r) & 
                              (train_df['TIM_THICKNESS'] <= test_thick + thick_r) &
                              (train_df['TIM_COVERAGE'] >= test_cov - cov_r) & 
                              (train_df['TIM_COVERAGE'] <= test_cov + cov_r)]
            
            if len(subset) > 0:
                mean_val = subset['Theta.JC'].mean()
                min_val = subset['Theta.JC'].min()
                max_val = subset['Theta.JC'].max()
                error = abs(mean_val - test_true) / test_true * 100
                
                print(f"{thick_r:<8} {cov_r:<8.2f} {len(subset):<6} {mean_val:<10.4f} "
                      f"{min_val:.3f}~{max_val:.3f}   {error:.1f}%")
                
                if error < best_error:
                    best_error = error
                    best_config = (thick_r, cov_r, len(subset), mean_val)
    
    print(f"\n→ 最佳: Thick±{best_config[0]}, Cov±{best_config[1]:.2f}, "
          f"n={best_config[2]}, mean={best_config[3]:.4f}, 誤差={best_error:.1f}%")
    
    # 檢查真值是否在訓練集範圍內
    subset_wide = train_df[(train_df['TIM_TYPE'] == test_type) & 
                           (train_df['TIM_THICKNESS'] >= test_thick - 20) & 
                           (train_df['TIM_THICKNESS'] <= test_thick + 20) &
                           (train_df['TIM_COVERAGE'] >= test_cov - 0.15) & 
                           (train_df['TIM_COVERAGE'] <= test_cov + 0.15)]
    if len(subset_wide) > 0:
        if test_true < subset_wide['Theta.JC'].min():
            print(f"⚠️ 真值 {test_true} < 訓練集最小值 {subset_wide['Theta.JC'].min():.3f}")
        elif test_true > subset_wide['Theta.JC'].max():
            print(f"⚠️ 真值 {test_true} > 訓練集最大值 {subset_wide['Theta.JC'].max():.3f}")
        else:
            print(f"✓ 真值 {test_true} 在訓練集範圍內 [{subset_wide['Theta.JC'].min():.3f}, {subset_wide['Theta.JC'].max():.3f}]")