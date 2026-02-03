# Phase 3: Heteroscedastic DKL 實驗進度追蹤

> 開始時間：基於 Phase 2 的發現，引入 Noise Network 處理資料內部變異性

---

## 背景：Phase 2 關鍵發現

在 Phase 2K 分析中發現：
- **Type 3 + Coverage ≥ 0.8** 區域有 **100% 內部不一致性**
- 同樣的輸入條件，訓練資料的 Theta.JC 值差異極大
- 這解釋了為什麼這個區域難以預測

**結論**：需要讓模型「承認」某些區域本質上難以預測 → 引入 Heteroscedastic Noise Network

---

## 實驗總覽

| Phase | 方法 | MAPE | Max Error | >20% | >40% | 狀態 |
|-------|------|------|-----------|------|------|------|
| 2J | DKL Ensemble (weighted MAPE) | 8.04% | 48.1% | 5 | 3 | ✅ 之前最佳 |
| **3A** | **Heteroscedastic DKL** | **7.53%** | **36.3%** | **6** | **0** | ✅ **目前最佳** |
| 3B | 改進 Noise Network | 8.12% | 45.7% | 14 | 1 | ❌ 過度複雜 |
| 3C | MoE (學習 Gating) | 36.29% | 391.3% | 48 | 32 | ❌ Gating 學反 |
| 3D | MoE (規則 Gating) | ? | ? | ? | ? | 🔄 測試中 |

---

## Phase 3A: 基礎 Heteroscedastic DKL ✅ 目前最佳

### 架構

```
Input (9維) → ┌─ Feature Extractor [64→32→16] → Variational GP (100 inducing, RBF)
              └─ Noise Network [32→16→1] → log_noise → exp() → σ²(x)
                                                              ↓
                                          Final: mean=GP_mean, var=GP_var+σ²(x)
```

### Loss 函數

```python
# Heteroscedastic NLL
Loss = (1/N) Σ [0.5*log(σ²(x)) + 0.5*(y-μ)²/σ²(x)] + 0.1*KL_div

# 直覺：
# - 預測準 → σ²小（模型確定）
# - 預測不準 → σ²大（承認不確定）
# - log(σ²) 項防止所有點都說「不確定」
```

### 關鍵設定

| 參數 | 值 |
|------|-----|
| Feature Extractor | [64, 32] → 16 |
| Noise Network | [32, 16] → 1 |
| n_inducing | 100 |
| lr | 0.005 |
| noise_lr_ratio | 0.5 |
| Noise 範圍 | exp([-4, -1]) |
| KL weight | 0.1 |
| 去重 | ❌ 不去重，保留 5361 筆 |

### 結果

```
MAPE: 7.53% (vs Phase 2J: 8.04%)
Max Error: 36.3% (vs 48.1%) ← 改善 11.8%
Outliers >40%: 0 (vs 3) ← 消除極端異常
Outliers >20%: 6 (vs 5)

異常點 (全為 Type 3):
220/0.8: 36.3%, 240/1.0: 36.3%, 240/0.8: 35.5%
260/0.8: 31.9%, 300/0.8: 28.6%, 280/1.0: 23.7%
```

### 程式碼

`/mnt/user-data/outputs/phase3a_heteroscedastic_dkl.py`

---

## Phase 3B: 改進 Noise Network ❌ 失敗

### 改進嘗試

1. 加入交互特徵：`type3_high_cov = type_3 × (coverage≥0.8)`
2. 放寬 noise 範圍：exp([-6, 0])
3. 更深架構：[64, 32, 16]

### 結果

```
MAPE: 8.12% (vs 3A: 7.53%) ❌ 更差
Max Error: 45.7% (vs 3A: 36.3%) ❌ 更差
Outliers >20%: 14 (vs 3A: 6) ❌ 更差
```

### 失敗原因

- 過度複雜化
- Noise Network 學到的 noise 反而比 Type 1 小
- 新增了原本沒有的 Type 1, 2 異常點

### 結論

**簡單版本 (3A) 更優**，不需要額外的交互特徵

---

## Phase 3C: MoE (學習 Gating) ❌ 失敗

### 設計目標

用 Mixture of Experts 讓不同區域有不同處理：
- Expert 1: 處理正常區域 (Type 1, 2)
- Expert 2: 處理高變異區域 (Type 3 + 高 Coverage)
- Gating Network: 學習如何分配權重

### 架構

```
Input → SharedFeatureExtractor → ┬─ GatingNetwork → [w1, w2]
                                 ├─ Expert1 GP → mean1
                                 ├─ Expert2 GP → mean2
                                 └─ NoiseNetwork → noise
                                        ↓
                          Final = w1×mean1 + w2×mean2
```

### 防護措施

1. 共享 Feature Extractor
2. Gating 初始化偏向 Expert 1
3. Entropy 正則化（避免極端權重）
4. Gating 較低學習率

### 結果

```
MAPE: 36.29% ❌❌❌
Max Error: 391.3% ❌❌❌
Outliers >20%: 48
```

### 失敗原因

**Gating 學反了！**

```
期望：Type 3 + 高 Cov → 高 Expert2 權重
實際：Type 3 + 高 Cov → Expert2_w = 0.146 (很低！)
      Type 1, Cov=0.6 → Expert2_w = 0.907 (很高！完全反過來)
```

所有 Type 3 高 Coverage 預測成同一個值 (0.0491)

### 程式碼

`/mnt/user-data/outputs/phase3c_moe_dkl.py`

---

## Phase 3D: MoE (規則 Gating) 🔄 測試中

### 改進策略

放棄學習 Gating，改用固定規則：

```python
Type 1, 2:        w1=0.9, w2=0.1  # 主要用 Expert 1
Type 3, Cov<0.8:  w1=0.7, w2=0.3  # 混合
Type 3, Cov>=0.8: w1=0.3, w2=0.7  # 主要用 Expert 2
```

### 其他改動

1. Expert 初始化差異化（加入 offset）
2. 移除 Entropy 正則化
3. 簡化 Loss

### 結果

🔄 等待測試結果...

### 程式碼

`/mnt/user-data/outputs/phase3d_rule_moe.py`

---

## Phase 3A 種子搜尋 🔄 進行中

### 目標

測試種子 1-3000，找出：
- 最低 Max Error 的種子
- 最少 Outliers (>20%) 的種子

### 程式碼

`/mnt/user-data/outputs/phase3a_seed_search.py`

### 結果

🔄 等待測試結果...

---

## Loss 函數比較：Phase 2J vs Phase 3A

| 項目 | Phase 2J | Phase 3A |
|------|----------|----------|
| GP 類型 | ExactGP | VariationalGP (SVGP) |
| 主 Loss | -MLL + 0.1×MAPE | Hetero NLL + 0.1×KL |
| Sample Weights | ✅ (Type3+高Cov ×3) | ❌ |
| MAPE 項 | ✅ 直接優化 | ❌ 只用 NLL |
| Noise 建模 | GP likelihood | 獨立 Noise Network |
| 去重 | ✅ groupby mean | ❌ 完整 5361 筆 |

### Phase 3A 改善原因推測

1. **移除 MAPE Loss**：純 NLL 可能更適合 GP
2. **移除 sample weights**：讓 noise network 自動學習
3. **Heteroscedastic noise**：模型自己決定哪裡不確定
4. **不去重**：保留資料內部變異性資訊

---

## 待嘗試方向

### 高優先級

- [ ] Phase 3D 結果分析
- [ ] Phase 3A 種子搜尋結果分析
- [ ] Phase 3A 最佳種子 Ensemble

### 中優先級

- [ ] Phase 3A + weighted MAPE（結合兩者優點）
- [ ] Phase 3A 超參數微調

### 低優先級

- [ ] 其他 MoE 變體
- [ ] 更複雜的 Noise Network 設計

---

## 檔案索引

| 檔案 | 說明 |
|------|------|
| `phase3a_heteroscedastic_dkl.py` | ✅ 目前最佳模型 |
| `phase3a_seed_search.py` | 種子搜尋腳本 |
| `phase3b_improved_noise.py` | ❌ 失敗實驗 |
| `phase3c_moe_dkl.py` | ❌ 失敗實驗 |
| `phase3d_rule_moe.py` | 🔄 測試中 |

---

## 更新紀錄

| 日期 | 更新內容 |
|------|----------|
| 2026-02-03 | 建立文件，記錄 Phase 3A-3D |
