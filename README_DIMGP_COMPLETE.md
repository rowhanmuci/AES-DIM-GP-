# Complete DIM-GP Variants Comparison

完整的Deep Infinite Mixture of Gaussian Process變體比較實驗框架

## 📦 安裝依賴

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn torch gpytorch openpyxl
```

## 🚀 快速開始

### 方法1: 執行完整實驗 (推薦)

執行所有模型在兩個資料集上：

```bash
python run_complete_dimgp_experiment.py
```

這會：
- 訓練6個模型 (MLP, XGBoost, GP, DKL, MoE, Ensemble)
- 在Above和Below兩個資料集上評估
- 生成所有視覺化圖表
- 產生比較報告

⏱️ 預計時間: 15-30分鐘

### 方法2: 快速測試

只跑Above資料集的3個基礎模型：

```bash
python run_complete_dimgp_experiment.py quick
```

⏱️ 預計時間: 2-3分鐘

### 方法3: 自定義實驗

```python
from experiment_framework import ExperimentFramework
from visualization_tools import create_all_visualizations
import pandas as pd

# 載入資料
train_data = pd.read_excel('your_train_data.xlsx')
test_data = pd.read_excel('your_test_data.xlsx')

X_train = train_data[['feature1', 'feature2', 'feature3']].values
y_train = train_data['target'].values
X_test = test_data[['feature1', 'feature2', 'feature3']].values
y_test = test_data['target'].values

# 建立實驗
exp = ExperimentFramework(dataset_name='My Dataset')
exp.load_data(X_train, y_train, X_test, y_test)

# 執行所有模型
exp.run_all_models()

# 顯示結果
summary = exp.print_summary()

# 生成視覺化
create_all_visualizations(exp, output_prefix='my_results')
```

## 📊 包含的模型

### Baseline模型
1. **MLP** - Multi-Layer Perceptron
   - 3層隱藏層 (64-32-16)
   - ReLU激活函數
   - Early stopping

2. **XGBoost** - Gradient Boosting
   - 200棵樹
   - Depth=5
   - Learning rate=0.05

3. **GP** - 標準Gaussian Process
   - RBF kernel
   - 最多1000個樣本

### DIM-GP變體
4. **DKL** - Deep Kernel Learning
   - DNN特徵提取 (64-32-16-8)
   - GP回歸層
   - 聯合訓練
   - 📄 論文: Wilson et al., 2016

5. **MoE** - Deep Mixture of GP Experts
   - DNN gating network
   - 3個Sparse GP experts
   - 軟權重預測
   - 📄 論文: Ultra-fast Deep Mixtures, 2022

6. **Ensemble** - 混合模型
   - 0.5*MLP + 0.5*XGBoost
   - GP提供不確定性

## 📈 評估指標

### 準確度指標
- **RMSE** - Root Mean Squared Error (越小越好)
- **MAE** - Mean Absolute Error (越小越好)
- **R²** - R-squared (越大越好，最大1.0)
- **MAPE** - Mean Absolute Percentage Error

### 不確定性指標 (針對GP相關模型)
- **CI Coverage** - 95%信賴區間覆蓋率 (應該≈95%)
- **CI Width** - 信賴區間平均寬度 (越窄越好)
- **NLPD** - Negative Log Predictive Density (越小越好)
- **Calibration** - 預測不確定性 vs 實際誤差

### 效率指標
- **Train Time** - 訓練時間 (秒)
- **Pred Time** - 預測時間 (秒)

## 🎨 生成的視覺化

### 1. Comprehensive Plot (`*_comprehensive.png`)
包含6個子圖的綜合比較：
- 預測 vs 真實值散點圖 (2x2大圖)
- 誤差分布箱型圖
- 性能雷達圖
- 不確定性比較
- CI質量散點圖
- 訓練時間比較

### 2. Predictions Grid (`*_predictions_grid.png`)
每個模型的預測散點圖，包含：
- 真實值 vs 預測值
- 理想線 (y=x)
- 誤差棒 (如果有UQ)
- R²和RMSE標註

### 3. Calibration Plot (`*_calibration.png`)
檢查不確定性估計是否準確：
- 預測標準差 vs 實際誤差
- 應該在y=x線附近

### 4. Feature Importance (`*_feature_importance.png`)
特徵重要性分析 (XGBoost和Ensemble)

## 📁 輸出文件

### Above 50% Coverage
- `above_results.csv` - 數值結果表格
- `above_comprehensive.png` - 綜合比較圖
- `above_predictions_grid.png` - 預測網格
- `above_calibration.png` - 校準圖
- `above_feature_importance.png` - 特徵重要性

### Below 50% Coverage
- `below_results.csv`
- `below_comprehensive.png`
- `below_predictions_grid.png`
- `below_calibration.png`
- `below_feature_importance.png`

### 比較報告
- `comparison_report.csv` - Above vs Below完整比較

## 🔧 模型配置

如果要調整模型參數：

```python
# DKL參數
exp.run_model('DKL', {
    'input_dim': 3,
    'hidden_dims': [64, 32, 16],
    'feature_dim': 8,
    'epochs': 100,
    'lr': 0.01,
    'batch_size': 256
})

# MoE參數
exp.run_model('MoE', {
    'n_experts': 3,
    'n_inducing': 100,
    'hidden_dims': (32, 16)
})
```

## 💡 使用建議

### 1. 快速測試先
```bash
python run_complete_dimgp_experiment.py quick
```
確保環境正常後再跑完整實驗

### 2. 記憶體不足？
如果遇到記憶體問題：
- 減少GP的subsample: `{'subsample': 500}`
- 減少DKL的epochs: `{'epochs': 50}`
- 減少MoE的experts: `{'n_experts': 2}`

### 3. 速度優化
如果想要更快：
- 只跑需要的模型: `exp.run_model('DKL')`
- 減少DKL的epochs和hidden_dims
- 使用GPU (如果有): DKL會自動使用

### 4. 只要不確定性？
跑這些模型: GP, DKL, MoE
```python
exp.run_model('GP')
exp.run_model('DKL')
exp.run_model('MoE')
```

## 📚 論文參考

1. **Deep Kernel Learning**
   - Wilson et al., "Deep Kernel Learning", AISTATS 2016
   - 結合DNN特徵提取和GP

2. **Deep Mixture of GP Experts**
   - Etienam et al., "Ultra-fast Deep Mixtures of Gaussian Process Experts", 2022
   - DNN gating + Sparse GP experts

3. **Mixture of Experts**
   - Rasmussen & Ghahramani, "Infinite Mixtures of Gaussian Process Experts", NIPS 2002
   - 經典MoE with GP

## ❓ 常見問題

### Q: DKL訓練很慢？
A: 正常，因為要聯合訓練DNN+GP。可以減少epochs或使用GPU。

### Q: MoE表現不好？
A: 可能是資料特性不適合分群。試試調整n_experts或n_inducing。

### Q: 為什麼MLP/XGBoost比DIM-GP好？
A: 很正常！對於簡單、低維度資料，簡單模型often表現更好。DIM-GP的優勢在於：
   - 不確定性量化
   - 複雜、非平穩資料
   - 小樣本情況

### Q: CI Coverage不到95%？
A: 可能的原因：
   - 模型欠擬合/過擬合
   - 不確定性估計偏差
   - 資料分布問題
   
試試調整模型參數或檢查資料品質。

## 🎯 實驗建議

### 對於報告
1. 執行完整實驗
2. 重點討論不確定性量化
3. 比較Above vs Below的差異
4. 分析為何某些模型表現好/差

### 重點指標
- **準確度**: R², RMSE
- **UQ品質**: CI Coverage (應≈95%), CI Width
- **效率**: 訓練時間

### 結論建議
- 承認MLP/XGBoost準確度高
- 強調DKL/MoE的UQ優勢
- 討論商業版DIM-GP可能的實作方向
- 說明實驗是"探索性研究"

## 📞 支援

如果有問題，檢查：
1. Python版本 >= 3.8
2. 所有依賴都已安裝
3. 資料路徑正確
4. 記憶體足夠 (建議16GB+)

Good luck! 🚀
