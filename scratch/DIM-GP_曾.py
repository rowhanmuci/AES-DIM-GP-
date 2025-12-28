import pandas as pd
import numpy as np
import glob
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
import gc
import torch
import torch.nn as nn
import gpytorch

# 設定顯示
plt_backend = 'Agg' # 防止在無介面環境報錯，雖然沒畫圖了但保留設定是好習慣
torch.set_default_dtype(torch.float64)

# 檢查 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算裝置: {device}")

# ==========================================
# 工具函式
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # print(f"-> 隨機種子已固定: {seed}") # 也可以註解掉讓輸出更乾淨

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def read_data_from_folder(folder_path, keyword):
    search_path = os.path.join(folder_path, '*.xlsx')
    files = glob.glob(search_path)
    target_files = [f for f in files if keyword in os.path.basename(f)]
    df_list = []
    for f in target_files:
        try:
            df = pd.read_excel(f)
            df['Source_File'] = os.path.basename(f)
            df_list.append(df)
        except Exception:
            pass
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

# ==========================================
# 模型定義
# ==========================================
class DnnFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(DnnFeatureExtractor, self).__init__()
        self.structure = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.structure(x)

class ComplexDKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(ComplexDKLModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        latent_dim = 6 
        self.covar_module = (
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)) +
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()) +
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=latent_dim)) +
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims=latent_dim))
        )
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ==========================================
# 主要執行流程 (Evaluation Only)
# ==========================================
def run_direct_pipeline(train_df, test_df, group_name):
    # 固定隨機種子
    set_seed(42)

    print(f"\n{'='*60}")
    print(f"執行實驗: {group_name}")
    print(f"訓練筆數: {len(train_df)} | 測試筆數: {len(test_df)}")
    print(f"{'='*60}")

    if train_df.empty:
        print("訓練資料不足，跳過。")
        return
    if test_df.empty:
        print("測試資料不足，跳過。")
        return

    # 合併處理
    train_df['dataset_type'] = 'train'
    test_df['dataset_type'] = 'test'
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 確保 TIM_TYPE 是數值型態
    full_df_encoded = full_df.copy()
    if 'TIM_TYPE' in full_df_encoded.columns:
        full_df_encoded['TIM_TYPE'] = full_df_encoded['TIM_TYPE'].astype(float)

    target_col = 'Theta.JC'
    ignore_cols = [target_col, 'Source_File', 'dataset_type', 'DNN', 'XGB', 'GP', 'DNN+GP', 'DNN+GP倒數', 'DNN+GP權重']
    drop_cols = [c for c in ignore_cols if c in full_df_encoded.columns]

    train_data = full_df_encoded[full_df_encoded['dataset_type'] == 'train']
    test_data = full_df_encoded[full_df_encoded['dataset_type'] == 'test']

    # 準備訓練與測試資料
    X_train_np = train_data.drop(columns=drop_cols).values.astype(np.float64)
    y_train_np = train_data[target_col].values.astype(np.float64)
    X_test_np = test_data.drop(columns=drop_cols).values.astype(np.float64)
    y_test_np = test_data[target_col].values.astype(np.float64)

    # 標準化
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_np)
    X_test_scaled = scaler_x.transform(X_test_np)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_np.reshape(-1, 1)).flatten()

    train_x = torch.from_numpy(X_train_scaled).to(device)
    train_y = torch.from_numpy(y_train_scaled).to(device)
    test_x = torch.from_numpy(X_test_scaled).to(device)

    # 建立模型
    feature_extractor = DnnFeatureExtractor(train_x.shape[1]).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ComplexDKLModel(train_x, train_y, likelihood, feature_extractor).to(device)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("  -> 模型訓練中...", end="\r") # 使用 \r 讓文字停留在同一行，不刷屏

    # 早停變數
    best_loss = float('inf')
    patience = 50        
    counter = 0
    max_epochs = 500     

    for i in range(max_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        curr_loss = loss.item()

        if curr_loss < best_loss:
            best_loss = curr_loss
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            # 訓練結束才印出
            print(f"  -> 早停觸發 (Epoch {i+1}), Best Loss: {best_loss:.4f}      ")
            break
    else:
        print(f"  -> 訓練完成 (Epoch {max_epochs}), Final Loss: {curr_loss:.4f}      ")

    # 預測 (Evaluation)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
        pred_dist = likelihood(model(test_x))
        y_pred_scaled = pred_dist.mean.cpu().numpy()

    # 還原預測值
    y_pred_final = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # 計算評估指標
    abs_percentage_error = np.abs((y_test_np - y_pred_final) / (y_test_np + 1e-8)) * 100
    mape = np.mean(abs_percentage_error)
    mae = mean_absolute_error(y_test_np, y_pred_final)
    max_err = max_error(y_test_np, y_pred_final)

    # 異常點統計 (>20%)
    high_error_mask = abs_percentage_error > 20.0
    num_high_error = np.sum(high_error_mask)
    total_samples = len(y_test_np)
    percent_high_error = (num_high_error / total_samples) * 100

    print(f"\n  [評估結果]")
    print(f"  MAPE      : {mape:.2f}%")
    print(f"  MAE       : {mae:.4f}")
    print(f"  Max Error : {max_err:.4f}")
    print(f"  異常點(>20%): {num_high_error} / {total_samples} ({percent_high_error:.2f}%)")

    # 列出異常點
    if num_high_error > 0:
        print("\n  [異常點明細 (前 20 筆)]:")
        outlier_df = test_df.iloc[high_error_mask].copy()
        outlier_df['Pred_Val'] = y_pred_final[high_error_mask]
        outlier_df['Error%'] = abs_percentage_error[high_error_mask]
        
        cols_to_show = ['Source_File', 'TIM_TYPE', 'TIM_THICKNESS', 'TIM_COVERAGE', 'Theta.JC', 'Pred_Val', 'Error%']
        cols_to_show = [c for c in cols_to_show if c in outlier_df.columns]
        
        print(outlier_df[cols_to_show].head(20).to_string(index=False))
        print("-" * 60)
    
    print(f"  -> 實驗完成。")
    clean_memory()

def main():
    train_dir = 'data/train'
    test_dir = 'data/test'

    if not os.path.exists(train_dir):
        print(f"錯誤: 找不到訓練資料夾 {train_dir}")
        return
    if not os.path.exists(test_dir):
        print(f"錯誤: 找不到測試資料夾 {test_dir}")
        return

    # 1. 讀取數據
    print("正在讀取資料...")
    df_train_above = read_data_from_folder(train_dir, 'Above')
    df_train_below = read_data_from_folder(train_dir, 'Below')
    
    df_test_above = read_data_from_folder(test_dir, 'Above')
    df_test_below = read_data_from_folder(test_dir, 'Below')

    # 2. 實驗: Above
    run_direct_pipeline(df_train_above, df_test_above, 'Above_Experiment')

    # 3. 實驗: Below
    run_direct_pipeline(df_train_below, df_test_below, 'Below_Experiment')

    # 4. 實驗: Combined (混合)
    print("\n>>> 混合訓練 (Combined)...")
    if not df_train_above.empty and not df_train_below.empty:
        df_train_mix = pd.concat([df_train_above, df_train_below], ignore_index=True)
    else:
        df_train_mix = pd.DataFrame()

    if not df_test_above.empty or not df_test_below.empty:
        test_dfs = []
        if not df_test_above.empty: test_dfs.append(df_test_above)
        if not df_test_below.empty: test_dfs.append(df_test_below)
        df_test_mix = pd.concat(test_dfs, ignore_index=True)
        
        run_direct_pipeline(df_train_mix, df_test_mix, 'Combined_Experiment')
    else:
        print("混合訓練跳過: 無測試資料。")

if __name__ == "__main__":
    main()