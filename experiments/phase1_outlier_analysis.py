"""
Phase 1: 異常點深度分析
目標: 找出Above資料集中16個異常點的共同特徵
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """載入Above資料集"""
    print("="*60)
    print("載入Above資料集...")
    print("="*60 + "\n")
    
    train_above = pd.read_excel('D:/NSYSU/Aes/data1/FOCoS_PKG_Type4_Cavity_TIM_50%_Above_Training_Data.xlsx')
    test_above = pd.read_excel('D:/NSYSU/Aes/data1/FOCoS_PKG_Type4_Cavity_TIM_50%_Above_Test_Data.xlsx')

    print(f"訓練集: {len(train_above)} 筆")
    print(f"測試集: {len(test_above)} 筆\n")
    
    return train_above, test_above


def analyze_outlier_features(outlier_df, normal_df, feature_cols):
    """比較異常點和正常點的特徵分布"""
    
    print("\n" + "="*60)
    print("特徵分布對比: 異常點 vs 正常點")
    print("="*60 + "\n")
    
    analysis_results = {}
    
    for feat in feature_cols:
        print(f"📊 {feat}:")
        print("-" * 40)
        
        outlier_vals = outlier_df[feat]
        normal_vals = normal_df[feat]
        
        stats = {
            'outlier': {
                'mean': outlier_vals.mean(),
                'std': outlier_vals.std(),
                'min': outlier_vals.min(),
                'max': outlier_vals.max(),
                'median': outlier_vals.median(),
            },
            'normal': {
                'mean': normal_vals.mean(),
                'std': normal_vals.std(),
                'min': normal_vals.min(),
                'max': normal_vals.max(),
                'median': normal_vals.median(),
            }
        }
        
        print(f"  異常點 - 均值: {stats['outlier']['mean']:.4f}, "
              f"標準差: {stats['outlier']['std']:.4f}, "
              f"範圍: [{stats['outlier']['min']:.4f}, {stats['outlier']['max']:.4f}]")
        
        print(f"  正常點 - 均值: {stats['normal']['mean']:.4f}, "
              f"標準差: {stats['normal']['std']:.4f}, "
              f"範圍: [{stats['normal']['min']:.4f}, {stats['normal']['max']:.4f}]")
        
        # 差異分析
        mean_diff = abs(stats['outlier']['mean'] - stats['normal']['mean'])
        mean_diff_pct = mean_diff / stats['normal']['mean'] * 100
        
        print(f"  ⚡ 均值差異: {mean_diff:.4f} ({mean_diff_pct:.2f}%)")
        
        analysis_results[feat] = stats
        print()
    
    return analysis_results


def check_outlier_patterns(outlier_df):
    """檢查異常點的模式"""
    
    print("\n" + "="*60)
    print("異常點模式分析")
    print("="*60 + "\n")
    
    # TIM_TYPE分布
    print("📌 TIM_TYPE分布:")
    print("-" * 40)
    type_dist = outlier_df['TIM_TYPE'].value_counts().sort_index()
    for tim_type, count in type_dist.items():
        percentage = count / len(outlier_df) * 100
        print(f"  Type {tim_type}: {count} 筆 ({percentage:.1f}%)")
    
    # THICKNESS範圍
    print("\n📌 TIM_THICKNESS分布:")
    print("-" * 40)
    thickness = outlier_df['TIM_THICKNESS']
    
    bins = [(0, 0.1, 'Low (<0.1)'),
            (0.1, 0.2, 'Mid (0.1-0.2)'),
            (0.2, float('inf'), 'High (>0.2)')]
    
    for low, high, label in bins:
        if high == float('inf'):
            count = len(thickness[thickness >= low])
        else:
            count = len(thickness[(thickness >= low) & (thickness < high)])
        percentage = count / len(thickness) * 100
        print(f"  {label}: {count} 筆 ({percentage:.1f}%)")
    
    # COVERAGE範圍
    print("\n📌 TIM_COVERAGE分布:")
    print("-" * 40)
    coverage = outlier_df['TIM_COVERAGE']
    
    bins = [(0, 30, 'Low (<30%)'),
            (30, 70, 'Mid (30-70%)'),
            (70, 100, 'High (>70%)')]
    
    for low, high, label in bins:
        count = len(coverage[(coverage >= low) & (coverage < high)])
        percentage = count / len(coverage) * 100
        print(f"  {label}: {count} 筆 ({percentage:.1f}%)")
    
    # Theta.JC分布
    print("\n📌 Theta.JC (真實值)分布:")
    print("-" * 40)
    theta = outlier_df['Theta.JC']
    print(f"  均值: {theta.mean():.4f}")
    print(f"  標準差: {theta.std():.4f}")
    print(f"  範圍: [{theta.min():.4f}, {theta.max():.4f}]")
    
    return {
        'type_dist': type_dist,
        'thickness_stats': thickness.describe(),
        'coverage_stats': coverage.describe(),
        'theta_stats': theta.describe()
    }


def check_training_data_coverage(train_df, outlier_df, feature_cols):
    """檢查訓練集是否覆蓋異常點的特徵空間"""
    
    print("\n" + "="*60)
    print("訓練集覆蓋度分析")
    print("="*60 + "\n")
    
    for feat in feature_cols:
        outlier_min = outlier_df[feat].min()
        outlier_max = outlier_df[feat].max()
        
        train_min = train_df[feat].min()
        train_max = train_df[feat].max()
        
        print(f"📊 {feat}:")
        print(f"  訓練集範圍: [{train_min:.4f}, {train_max:.4f}]")
        print(f"  異常點範圍: [{outlier_min:.4f}, {outlier_max:.4f}]")
        
        # 檢查是否超出訓練範圍
        if outlier_min < train_min or outlier_max > train_max:
            print(f"  ⚠️  異常點超出訓練範圍！")
            if outlier_min < train_min:
                print(f"     - 最小值超出: {outlier_min:.4f} < {train_min:.4f}")
            if outlier_max > train_max:
                print(f"     - 最大值超出: {outlier_max:.4f} > {train_max:.4f}")
        else:
            print(f"  ✅ 異常點在訓練範圍內")
        print()


def find_similar_training_samples(train_df, outlier_sample, feature_cols, top_k=5):
    """找出訓練集中與異常點最相似的樣本"""
    
    # 標準化
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols])
    outlier_features = scaler.transform(outlier_sample[feature_cols].values.reshape(1, -1))
    
    # 計算歐式距離
    distances = np.linalg.norm(train_features - outlier_features, axis=1)
    
    # 找最近的k個
    nearest_indices = np.argsort(distances)[:top_k]
    
    similar_samples = train_df.iloc[nearest_indices].copy()
    similar_samples['distance'] = distances[nearest_indices]
    
    return similar_samples


def analyze_each_outlier(train_df, test_df_with_predictions, feature_cols, threshold=20):
    """逐一分析每個異常點"""
    
    print("\n" + "="*60)
    print("逐一分析異常點 (前10筆詳細)")
    print("="*60 + "\n")
    
    # 篩選異常點
    outliers = test_df_with_predictions[test_df_with_predictions['Error%'] > threshold].copy()
    outliers = outliers.sort_values('Error%', ascending=False)
    
    for idx, (i, row) in enumerate(outliers.head(10).iterrows()):
        print(f"\n{'─'*60}")
        print(f"異常點 #{idx+1} (測試集第 {i} 筆)")
        print(f"{'─'*60}")
        
        print(f"特徵值:")
        for feat in feature_cols:
            print(f"  {feat}: {row[feat]:.4f}")
        
        print(f"\n預測結果:")
        print(f"  真實值 (Theta.JC): {row['Theta.JC']:.4f}")
        print(f"  預測值: {row['Prediction']:.4f}")
        print(f"  誤差: {row['Error%']:.2f}%")
        
        # 找相似的訓練樣本
        print(f"\n訓練集中最相似的5個樣本:")
        similar = find_similar_training_samples(train_df, row, feature_cols, top_k=5)
        
        for j, (_, sim_row) in enumerate(similar.iterrows(), 1):
            print(f"\n  相似樣本 #{j} (距離={sim_row['distance']:.4f}):")
            for feat in feature_cols:
                print(f"    {feat}: {sim_row[feat]:.4f}")
            print(f"    Theta.JC: {sim_row['Theta.JC']:.4f}")
    
    print(f"\n{'─'*60}")
    print(f"剩餘 {len(outliers) - 10} 個異常點未詳細顯示")
    print(f"{'─'*60}\n")


def visualize_outliers(train_df, test_df_with_predictions, feature_cols, threshold=20):
    """視覺化異常點"""
    
    print("\n生成視覺化圖表...")
    
    outliers = test_df_with_predictions[test_df_with_predictions['Error%'] > threshold]
    normals = test_df_with_predictions[test_df_with_predictions['Error%'] <= threshold]
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Above資料集 - 異常點分析', fontsize=16, fontweight='bold')
    
    # 1. Feature分布對比
    ax1 = axes[0, 0]
    x = np.arange(len(feature_cols))
    width = 0.35
    
    outlier_means = [outliers[f].mean() for f in feature_cols]
    normal_means = [normals[f].mean() for f in feature_cols]
    train_means = [train_df[f].mean() for f in feature_cols]
    
    ax1.bar(x - width, outlier_means, width, label='Outliers', color='red', alpha=0.7)
    ax1.bar(x, normal_means, width, label='Normals', color='green', alpha=0.7)
    ax1.bar(x + width, train_means, width, label='Training', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('特徵均值對比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. TIM_TYPE分布
    ax2 = axes[0, 1]
    type_counts_outlier = outliers['TIM_TYPE'].value_counts()
    type_counts_normal = normals['TIM_TYPE'].value_counts()
    
    types = sorted(set(list(type_counts_outlier.index) + list(type_counts_normal.index)))
    outlier_vals = [type_counts_outlier.get(t, 0) for t in types]
    normal_vals = [type_counts_normal.get(t, 0) for t in types]
    
    x = np.arange(len(types))
    ax2.bar(x - width/2, outlier_vals, width, label='Outliers', color='red', alpha=0.7)
    ax2.bar(x + width/2, normal_vals, width, label='Normals', color='green', alpha=0.7)
    
    ax2.set_xlabel('TIM_TYPE')
    ax2.set_ylabel('Count')
    ax2.set_title('TIM_TYPE分布對比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. THICKNESS vs COVERAGE散點圖
    ax3 = axes[1, 0]
    ax3.scatter(normals['TIM_THICKNESS'], normals['TIM_COVERAGE'], 
               c='green', alpha=0.5, s=50, label='Normal')
    ax3.scatter(outliers['TIM_THICKNESS'], outliers['TIM_COVERAGE'], 
               c='red', alpha=0.7, s=100, marker='X', label='Outliers')
    ax3.scatter(train_df['TIM_THICKNESS'], train_df['TIM_COVERAGE'], 
               c='blue', alpha=0.1, s=10, label='Training')
    
    ax3.set_xlabel('TIM_THICKNESS')
    ax3.set_ylabel('TIM_COVERAGE')
    ax3.set_title('THICKNESS vs COVERAGE分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 誤差分布直方圖
    ax4 = axes[1, 1]
    errors_all = test_df_with_predictions['Error%']
    
    ax4.hist(errors_all, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
    ax4.axvline(x=errors_all.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({errors_all.mean():.2f}%)')
    
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_ylabel('Count')
    ax4.set_title('誤差分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 圖表已儲存: outlier_analysis.png\n")
    plt.close()


def generate_summary_report(train_df, test_df_with_predictions, analysis_results, threshold=20):
    """生成總結報告"""
    
    outliers = test_df_with_predictions[test_df_with_predictions['Error%'] > threshold]
    
    print("\n" + "="*60)
    print("🎯 總結報告")
    print("="*60 + "\n")
    
    print(f"📊 基本統計:")
    print(f"  測試集總數: {len(test_df_with_predictions)} 筆")
    print(f"  異常點數量: {len(outliers)} 筆 ({len(outliers)/len(test_df_with_predictions)*100:.2f}%)")
    print(f"  正常點數量: {len(test_df_with_predictions) - len(outliers)} 筆")
    
    print(f"\n🔍 異常點特徵摘要:")
    
    # 檢查是否集中在某些TIM_TYPE
    type_dist = outliers['TIM_TYPE'].value_counts()
    dominant_type = type_dist.idxmax() if len(type_dist) > 0 else None
    
    if dominant_type is not None:
        dominant_pct = type_dist[dominant_type] / len(outliers) * 100
        print(f"  主要TIM_TYPE: Type {dominant_type} ({type_dist[dominant_type]} 筆, {dominant_pct:.1f}%)")
    
    # 檢查特徵範圍
    print(f"\n  特徵範圍:")
    for feat in ['TIM_THICKNESS', 'TIM_COVERAGE']:
        outlier_range = (outliers[feat].min(), outliers[feat].max())
        train_range = (train_df[feat].min(), train_df[feat].max())
        
        print(f"    {feat}:")
        print(f"      異常點: [{outlier_range[0]:.4f}, {outlier_range[1]:.4f}]")
        print(f"      訓練集: [{train_range[0]:.4f}, {train_range[1]:.4f}]")
        
        # 超出訓練範圍的異常點
        out_of_range = outliers[(outliers[feat] < train_range[0]) | (outliers[feat] > train_range[1])]
        if len(out_of_range) > 0:
            print(f"      ⚠️  {len(out_of_range)} 個異常點超出訓練範圍")
    
    print(f"\n💡 改進建議:")
    
    # 根據分析結果提供建議
    suggestions = []
    
    # 建議1: 超出訓練範圍
    for feat in ['TIM_THICKNESS', 'TIM_COVERAGE']:
        outlier_range = (outliers[feat].min(), outliers[feat].max())
        train_range = (train_df[feat].min(), train_df[feat].max())
        out_of_range = outliers[(outliers[feat] < train_range[0]) | (outliers[feat] > train_range[1])]
        
        if len(out_of_range) > 0:
            suggestions.append(f"增加訓練集在 {feat} 極端值區域的樣本")
    
    # 建議2: TIM_TYPE不平衡
    if dominant_type is not None and dominant_pct > 50:
        suggestions.append(f"特別關注 TIM_TYPE={dominant_type} 的預測")
    
    # 建議3: 超參數
    suggestions.append("使用超參數搜尋優化模型")
    suggestions.append("嘗試不同的kernel組合")
    suggestions.append("調整feature_dim (潛在空間維度)")
    
    # 建議4: 損失函數
    suggestions.append("使用MAPE loss直接優化相對誤差")
    suggestions.append("對異常區域樣本加權")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\n{'='*60}\n")


def main():
    """主函數"""
    
    # 載入資料
    train_df, test_df = load_data()
    
    # 特徵欄位
    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    
    # ⚠️ 這裡需要你提供組員模型的預測結果
    # 我先創建一個模擬的預測結果示範
    # 實際使用時，請替換成真實的預測
    
    # 載入預測結果
    test_df = pd.read_csv('phase1_predictions.csv')
    train_df = pd.read_excel('D:/NSYSU/Aes/data1/FOCoS_PKG_Type4_Cavity_TIM_50%_Above_Training_Data.xlsx')

    feature_cols = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']

    # 分離異常點和正常點
    outliers = test_df[test_df['Error%'] > 20]
    normals = test_df[test_df['Error%'] <= 20]

    # 執行分析
    print(f"\n找到 {len(outliers)} 個異常點\n")

    # 1. 特徵分布分析
    results = analyze_outlier_features(outliers, normals, feature_cols)

    # 2. 模式分析
    patterns = check_outlier_patterns(outliers)

    # 3. 訓練集覆蓋度
    check_training_data_coverage(train_df, outliers, feature_cols)

    # 4. 逐一分析
    analyze_each_outlier(train_df, test_df, feature_cols)

    # 5. 視覺化
    visualize_outliers(train_df, test_df, feature_cols)

    # 6. 總結報告
    generate_summary_report(train_df, test_df, results)


if __name__ == "__main__":
    main()
