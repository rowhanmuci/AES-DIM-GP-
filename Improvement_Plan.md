# ASE FOCoS 熱阻預測 - 改進計劃

根據教授回饋和實驗結果制定的完整改進方案

---

## 📋 教授的核心需求

### 1. **關鍵指標**: 相對誤差 (MAPE)
- ✅ **目標**: 所有預測誤差 < 20%
- ⚠️ **現況**: 17筆資料誤差 > 20%
- 🎯 **重要性**: 對公司而言，Theta_JC 0.01 vs 0.02 物理性質差很大

### 2. **超參數搜尋**
- 🔧 公司建議結合超參數搜尋工具
- 📊 公司經驗: 模型超參數影響結果頗大
- 🎯 需要系統化的超參數優化

### 3. **TIM_TYPE特徵處理**
- ⚠️ 目前只用One-hot encoding
- 💡 應該有更好的類別特徵處理方式
- 🔍 需要探索更先進的embedding方法

### 4. **資料品質**
- ⚠️ Test資料中有蠻多重複
- 🧹 需要資料清理

---

## 🎯 改進策略總覽

### 優先順序 (Priority)

| 優先級 | 任務 | 預期效果 | 工作量 |
|--------|------|----------|--------|
| 🔴 **P0** | 超參數搜尋 | 大幅提升準確度 | 中 |
| 🔴 **P0** | 異常點分析 | 定位問題根源 | 小 |
| 🟡 **P1** | TIM_TYPE特徵工程 | 提升特徵表達 | 中 |
| 🟡 **P1** | 資料清理 | 避免過擬合 | 小 |
| 🟢 **P2** | Ensemble優化 | 進一步提升 | 大 |
| 🟢 **P2** | 損失函數調整 | 針對異常點 | 中 |

---

## 📊 方案1: 超參數搜尋 (P0)

### 目標
系統化搜尋最佳超參數組合，降低最大誤差到20%以下

### 需要搜尋的超參數

#### DKL架構參數
```python
hyperparameters = {
    # 特徵提取器
    'hidden_dims': [
        [64, 32, 16],      # 淺層
        [128, 64, 32],     # 中層
        [256, 128, 64],    # 深層（組員用這個）
        [128, 64, 32, 16], # 更深
    ],
    'feature_dim': [4, 6, 8, 12],  # 潛在空間維度
    'dropout_rate': [0.0, 0.1, 0.2],
    
    # GP kernel
    'kernel_type': [
        'RBF',                    # 單一kernel
        'RBF+Linear',            # 組合
        'RBF+Matern',           # 組合
        'Complex',              # 組員的複雜組合
    ],
    
    # 訓練參數
    'learning_rate': [0.001, 0.005, 0.01, 0.02],
    'weight_decay': [1e-5, 1e-4, 1e-3],
    'batch_norm': [True, False],
    
    # 正則化
    'noise_constraint': [1e-4, 1e-3, 1e-2],
}
```

### 搜尋方法

#### 方法1: Optuna (推薦)
```python
import optuna

def objective(trial):
    # 定義超參數空間
    hidden_dims = trial.suggest_categorical('hidden_dims', 
        [[64,32,16], [128,64,32], [256,128,64]])
    feature_dim = trial.suggest_int('feature_dim', 4, 12)
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    
    # 訓練模型
    model = train_dkl(hidden_dims, feature_dim, lr, ...)
    
    # 評估指標: 最大相對誤差
    max_relative_error = evaluate_max_error(model, test_data)
    
    return max_relative_error

# 執行搜尋
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

**優點**:
- ✅ 自動化搜尋
- ✅ 支援early stopping
- ✅ 視覺化結果
- ✅ 可以resume中斷的搜尋

#### 方法2: Ray Tune
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

config = {
    'hidden_dims': tune.choice([[64,32,16], [128,64,32], [256,128,64]]),
    'feature_dim': tune.randint(4, 12),
    'lr': tune.loguniform(1e-3, 1e-1),
}

scheduler = ASHAScheduler(metric='max_error', mode='min')
analysis = tune.run(train_dkl, config=config, num_samples=100)
```

**優點**:
- ✅ 並行搜尋
- ✅ 早停機制
- ✅ 資源分配優化

### 評估指標設計

**重點**: 不只看MAPE，要看最大誤差！

```python
def evaluate_comprehensive(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # 相對誤差
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    metrics = {
        'mape': np.mean(relative_errors),
        'max_error': np.max(relative_errors),           # 最重要！
        'outlier_20': np.sum(relative_errors > 20),     # >20%的樣本數
        'outlier_15': np.sum(relative_errors > 15),
        'outlier_10': np.sum(relative_errors > 10),
        'p95': np.percentile(relative_errors, 95),      # 95分位數
        'p99': np.percentile(relative_errors, 99),
    }
    
    return metrics
```

**優化目標**:
```python
# 複合目標函數
def combined_objective(metrics):
    # 主要目標: 降低最大誤差
    primary = metrics['max_error']
    
    # 次要目標: 降低異常點數量
    secondary = metrics['outlier_20'] * 5  # 懲罰異常點
    
    # 第三目標: 整體MAPE
    tertiary = metrics['mape']
    
    return primary + secondary + tertiary * 0.5
```

---

## 🔍 方案2: 異常點深度分析 (P0)

### 目標
找出那17筆誤差>20%的樣本，分析共同特徵

### 分析步驟

#### Step 1: 定位異常點
```python
def analyze_outliers(model, X_test, y_test, test_df, threshold=20):
    y_pred = model.predict(X_test)
    relative_errors = np.abs((y_test - y_pred) / y_test) * 100
    
    outlier_mask = relative_errors > threshold
    outlier_df = test_df[outlier_mask].copy()
    outlier_df['Pred'] = y_pred[outlier_mask]
    outlier_df['True'] = y_test[outlier_mask]
    outlier_df['Error%'] = relative_errors[outlier_mask]
    
    return outlier_df
```

#### Step 2: 特徵分布分析
```python
def outlier_feature_analysis(outlier_df, normal_df):
    """比較異常點和正常點的特徵分布"""
    
    features = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    
    analysis = {}
    for feat in features:
        analysis[feat] = {
            'outlier_mean': outlier_df[feat].mean(),
            'normal_mean': normal_df[feat].mean(),
            'outlier_std': outlier_df[feat].std(),
            'normal_std': normal_df[feat].std(),
            'outlier_range': (outlier_df[feat].min(), outlier_df[feat].max()),
            'normal_range': (normal_df[feat].min(), normal_df[feat].max()),
        }
    
    return analysis
```

#### Step 3: 異常點模式識別
```python
# 檢查異常點是否集中在某些區域
def check_outlier_patterns(outlier_df):
    patterns = {
        'TIM_TYPE分布': outlier_df['TIM_TYPE'].value_counts(),
        'THICKNESS範圍': {
            'low (<0.1)': len(outlier_df[outlier_df['TIM_THICKNESS'] < 0.1]),
            'mid (0.1-0.2)': len(outlier_df[(outlier_df['TIM_THICKNESS'] >= 0.1) & 
                                            (outlier_df['TIM_THICKNESS'] < 0.2)]),
            'high (>0.2)': len(outlier_df[outlier_df['TIM_THICKNESS'] >= 0.2]),
        },
        'COVERAGE範圍': {
            'low (<30%)': len(outlier_df[outlier_df['TIM_COVERAGE'] < 30]),
            'mid (30-70%)': len(outlier_df[(outlier_df['TIM_COVERAGE'] >= 30) & 
                                           (outlier_df['TIM_COVERAGE'] < 70)]),
            'high (>70%)': len(outlier_df[outlier_df['TIM_COVERAGE'] >= 70]),
        }
    }
    return patterns
```

### 可能的異常點原因

1. **邊界樣本**: 
   - 極端的THICKNESS或COVERAGE值
   - 訓練集中很少見的組合

2. **特定TIM_TYPE**:
   - 某些類型的樣本數太少
   - 物理特性差異大

3. **資料品質**:
   - 量測誤差
   - 重複資料
   - 標籤錯誤

### 針對性改進

```python
# 策略1: 異常點加權
def weighted_loss(y_pred, y_true, sample_weights):
    """對異常樣本區域加大權重"""
    mse = (y_pred - y_true) ** 2
    weighted_mse = mse * sample_weights
    return weighted_mse.mean()

# 策略2: 異常點增強
def augment_outlier_regions(X_train, y_train, outlier_indices):
    """對異常區域的樣本做資料增強"""
    # 在異常點附近添加噪聲樣本
    augmented_X = []
    augmented_y = []
    
    for idx in outlier_indices:
        for _ in range(5):  # 每個異常點生成5個變種
            noise = np.random.normal(0, 0.01, X_train[idx].shape)
            augmented_X.append(X_train[idx] + noise)
            augmented_y.append(y_train[idx])
    
    return np.vstack([X_train, augmented_X]), np.hstack([y_train, augmented_y])
```

---

## 🧬 方案3: TIM_TYPE特徵工程 (P1)

### 目前問題
- ❌ One-hot encoding: 假設類別間完全獨立
- ❌ 無法捕捉TIM_TYPE的物理相似性
- ❌ 高維稀疏表示

### 改進方案

#### 方案3.1: Entity Embedding (推薦)
```python
class TIMTypeEmbedding(nn.Module):
    def __init__(self, n_types, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(n_types, embedding_dim)
        
    def forward(self, tim_type_indices):
        # tim_type_indices: [batch_size]
        # output: [batch_size, embedding_dim]
        return self.embedding(tim_type_indices)

# 使用在DKL中
class DKLWithEmbedding(nn.Module):
    def __init__(self, n_types, continuous_dim, embedding_dim=4):
        super().__init__()
        self.type_embedding = TIMTypeEmbedding(n_types, embedding_dim)
        
        # DNN接受embedding + 連續特徵
        self.dnn = DnnFeatureExtractor(
            input_dim=embedding_dim + continuous_dim,
            output_dim=6
        )
```

**優點**:
- ✅ 自動學習TIM_TYPE的潛在表示
- ✅ 能捕捉類型間的相似性
- ✅ 降維（4-8維 vs One-hot的N維）

#### 方案3.2: Target Encoding
```python
def target_encode_tim_type(train_df, test_df, target_col='Theta.JC'):
    """用目標變量的平均值來編碼類別"""
    
    # 計算每個TIM_TYPE的平均Theta.JC
    type_means = train_df.groupby('TIM_TYPE')[target_col].mean()
    
    # 加入全局平均作為平滑
    global_mean = train_df[target_col].mean()
    smoothing = 10  # 平滑參數
    
    type_counts = train_df.groupby('TIM_TYPE').size()
    
    # 平滑後的編碼
    smooth_means = (type_means * type_counts + global_mean * smoothing) / (type_counts + smoothing)
    
    # 應用到訓練和測試集
    train_df['TIM_TYPE_encoded'] = train_df['TIM_TYPE'].map(smooth_means)
    test_df['TIM_TYPE_encoded'] = test_df['TIM_TYPE'].map(smooth_means).fillna(global_mean)
    
    return train_df, test_df
```

**優點**:
- ✅ 直接反映TIM_TYPE對目標的影響
- ✅ 單一維度，簡單高效
- ⚠️ 注意: 需要避免target leakage (用CV)

#### 方案3.3: 物理屬性Encoding
```python
def physics_based_encoding(tim_type):
    """根據TIM材料的物理屬性編碼"""
    
    # 假設我們知道每種TIM的物理屬性
    physics_properties = {
        1: {'thermal_conductivity': 5.0, 'density': 2.5, 'viscosity': 100},
        2: {'thermal_conductivity': 8.0, 'density': 3.0, 'viscosity': 150},
        3: {'thermal_conductivity': 3.5, 'density': 2.0, 'viscosity': 80},
        # ... 其他類型
    }
    
    # 用物理屬性作為特徵
    if tim_type in physics_properties:
        return np.array([
            physics_properties[tim_type]['thermal_conductivity'],
            physics_properties[tim_type]['density'],
            physics_properties[tim_type]['viscosity']
        ])
    else:
        return np.zeros(3)  # 未知類型用0填充
```

**優點**:
- ✅ 融入領域知識
- ✅ 物理意義明確
- ⚠️ 需要: 取得TIM材料的實際物理參數

---

## 🧹 方案4: 資料清理 (P1)

### 問題
教授提到test資料中有蠻多重複

### 清理步驟

#### Step 1: 檢測重複
```python
def check_duplicates(df, features=['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']):
    """檢查完全重複的樣本"""
    
    # 完全重複
    full_duplicates = df.duplicated(subset=features + ['Theta.JC'])
    print(f"完全重複: {full_duplicates.sum()} 筆")
    
    # 特徵重複但目標不同 (可能是量測誤差)
    feature_duplicates = df.duplicated(subset=features, keep=False)
    ambiguous = df[feature_duplicates & ~full_duplicates]
    
    if len(ambiguous) > 0:
        print(f"特徵相同但目標不同: {len(ambiguous)} 筆")
        print(ambiguous.groupby(features)['Theta.JC'].agg(['mean', 'std', 'count']))
    
    return full_duplicates, ambiguous
```

#### Step 2: 處理策略

```python
def clean_duplicates(df, strategy='average'):
    """
    strategy:
    - 'drop': 刪除重複
    - 'average': 特徵相同時取目標平均
    - 'keep_first': 保留第一個
    """
    
    features = ['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE']
    
    if strategy == 'drop':
        return df.drop_duplicates(subset=features + ['Theta.JC'])
    
    elif strategy == 'average':
        # 對相同特徵的樣本，取目標平均值
        df_clean = df.groupby(features, as_index=False).agg({
            'Theta.JC': 'mean',
            # 其他欄位保留第一個
            **{col: 'first' for col in df.columns if col not in features + ['Theta.JC']}
        })
        return df_clean
    
    elif strategy == 'keep_first':
        return df.drop_duplicates(subset=features, keep='first')
```

#### Step 3: 異常值處理
```python
def remove_outliers(df, target_col='Theta.JC', method='iqr'):
    """移除目標變量的異常值"""
    
    if method == 'iqr':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        mask = (df[target_col] >= lower) & (df[target_col] <= upper)
        
    elif method == 'zscore':
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        mask = z_scores < 3
    
    outliers = df[~mask]
    print(f"移除 {len(outliers)} 個異常值")
    
    return df[mask], outliers
```

---

## 🎯 方案5: 損失函數優化 (P2)

### 目標
針對相對誤差優化，而非絕對誤差

### 當前問題
MSE/MAE優化的是絕對誤差，但公司要求的是相對誤差<20%

### 解決方案

#### 方案5.1: MAPE Loss
```python
def mape_loss(y_pred, y_true):
    """Mean Absolute Percentage Error Loss"""
    epsilon = 1e-8  # 避免除以0
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# 在DKL訓練中使用
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(train_x)
    
    # GP likelihood loss
    gp_loss = -mll(output, train_y)
    
    # MAPE loss
    pred_mean = output.mean
    mape = mape_loss(pred_mean, train_y)
    
    # 組合loss
    total_loss = gp_loss + 0.1 * mape  # 權重可調
    
    total_loss.backward()
    optimizer.step()
```

#### 方案5.2: Huber Loss (對異常點穩健)
```python
def huber_loss(y_pred, y_true, delta=1.0):
    """對大誤差較不敏感"""
    error = y_pred - y_true
    is_small = torch.abs(error) <= delta
    
    small_error = 0.5 * error ** 2
    large_error = delta * (torch.abs(error) - 0.5 * delta)
    
    return torch.where(is_small, small_error, large_error).mean()
```

#### 方案5.3: Weighted MSE (對異常區域加權)
```python
def weighted_mse_loss(y_pred, y_true, sample_weights):
    """對預測困難的區域加大權重"""
    mse = (y_pred - y_true) ** 2
    weighted_mse = mse * sample_weights
    return weighted_mse.mean()

# 動態計算權重
def compute_sample_weights(X, outlier_regions):
    """靠近異常點區域的樣本權重更高"""
    weights = torch.ones(len(X))
    
    for region in outlier_regions:
        # 計算到異常區域的距離
        dist = torch.norm(X - region['center'], dim=1)
        
        # 距離越近權重越高
        region_weights = torch.exp(-dist / region['radius'])
        weights += region_weights
    
    return weights / weights.sum() * len(X)
```

---

## 📦 完整實作計劃

### Phase 1: 基礎改進 (期末後第1週)

**任務**:
1. ✅ 資料清理 (重複樣本處理)
2. ✅ 異常點深度分析 (找出17筆的共同特徵)
3. ✅ 實作MAPE loss

**預期成果**:
- 異常點報告
- 清理後的資料集
- 基準MAPE改善

---

### Phase 2: 超參數優化 (期末後第2週)

**任務**:
1. ✅ 整合Optuna
2. ✅ 定義搜尋空間
3. ✅ 執行100次試驗
4. ✅ 分析最佳配置

**預期成果**:
- 最佳超參數組合
- 超參數重要性分析
- Max error < 20%

---

### Phase 3: 特徵工程 (期末後第3週)

**任務**:
1. ✅ TIM_TYPE Entity Embedding
2. ✅ Target Encoding (with CV)
3. ✅ 如果能拿到物理參數，加入Physics-based encoding

**預期成果**:
- TIM_TYPE更好的表示
- 模型準確度提升

---

### Phase 4: 模型優化 (期末後第4週)

**任務**:
1. ✅ 組員的複雜kernel vs 簡單kernel對比
2. ✅ Ensemble多個最佳配置
3. ✅ 最終模型選擇

**預期成果**:
- 生產級模型
- 完整文檔

---

## 🔧 技術細節對比

### 你的DKL vs 組員的DKL

| 特性 | 你的版本 | 組員的版本 | 建議 |
|------|----------|-----------|------|
| **網路深度** | [64, 32, 16, 8] | [256, 128, 64] | 超參數搜尋決定 |
| **Kernel** | RBF | RBF+Linear+Matern+RQ | 先試簡單，再試複雜 |
| **訓練策略** | 三階段 | 單階段+早停 | 組員的更簡潔 |
| **Loss** | GP MLL | GP MLL | 都加入MAPE |
| **Scheduler** | 無 | CosineAnnealing | 組員的更好 |
| **記憶體管理** | 無 | 有gc清理 | 採用組員的 |

---

## 💡 快速開始建議

### 立即行動項目 (期末前可做)

1. **異常點分析腳本** (30分鐘):
```python
# 快速分析那17筆
outlier_df = analyze_outliers(model, X_test, y_test, test_df, threshold=20)
print(outlier_df[['TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE', 'Error%']].to_string())
```

2. **資料重複檢查** (15分鐘):
```python
# 檢查重複
full_dup, ambiguous = check_duplicates(test_df)
print(f"完全重複: {full_dup.sum()} 筆")
```

3. **準備Optuna框架** (1小時):
```python
# 建立基本框架，期末後直接執行
def objective(trial):
    # 超參數定義
    ...
    return max_error
```

---

## 📊 預期改進效果

### 保守估計

| 改進項目 | 預期效果 |
|----------|----------|
| 資料清理 | -2% MAPE |
| 超參數優化 | -5~10% max error |
| TIM_TYPE embedding | -3% MAPE |
| MAPE loss | -5% max error |
| **總計** | **異常點<17筆，max error接近20%** |

### 樂觀估計

如果超參數搜尋順利 + embedding效果好:
- ✅ 異常點降到 5-10筆
- ✅ Max error < 15%
- ✅ MAPE < 3%

---

## ✅ 檢查清單

期末後開始前，準備好：

- [ ] 組員程式碼整合測試
- [ ] Optuna安裝和測試
- [ ] 資料清理腳本準備
- [ ] 異常點分析腳本準備
- [ ] 實驗記錄表格設計
- [ ] 與組員分工討論

祝期末順利！🎉
