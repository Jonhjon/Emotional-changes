import pandas as pd
import os
import sys

# 設定輸出檔案
output_file = "analysis_report.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 將輸出同時導向到終端機和檔案
sys.stdout = Logger(output_file)

# 定義檔案路徑
files = {
    "combined_test": r"c:\Users\張銘傑\Desktop\Emotional-changes\archive\mental_health_combined_test.csv",
    "unbalanced": r"c:\Users\張銘傑\Desktop\Emotional-changes\archive\mental_heath_unbanlanced.csv"
}

def print_header(title):
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

def translate_status(df):
    if 'status' in df.columns:
        status_map = {
            'Normal': '正常 (Normal)',
            'Depression': '憂鬱 (Depression)',
            'Suicidal': '自殺傾向 (Suicidal)',
            'Anxiety': '焦慮 (Anxiety)'
        }
        # 為了不影響原始資料，我們在顯示時處理，或者建立一個新的映射欄位
        # 這裡直接回傳映射後的 Series 用於 groupby
        return df['status'].map(status_map).fillna(df['status'])
    return None

def analyze_file(name, path):
    print_header(f"檔案分析報告：{name}")
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"讀取錯誤 {path}: {e}")
        return None

    # 1. 基本統計
    print(f"資料形狀 (列, 欄): {df.shape}")
    print(f"欄位名稱: {df.columns.tolist()}")
    
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"缺失值總數: {missing}")
    if missing > 0:
        print("各欄位缺失值數量:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"重複資料筆數: {duplicates}")

    # 2. 標籤分佈
    if 'status' in df.columns:
        print("\n--- 標籤分佈 (status) ---")
        dist = df['status'].value_counts()
        print(dist)
        print("\n--- 標籤分佈 (百分比 %) ---")
        print(df['status'].value_counts(normalize=True) * 100)
    else:
        print("\n未找到 'status' 欄位。")

    return df

# 執行分析
dfs = {}
for name, path in files.items():
    dfs[name] = analyze_file(name, path)

# 3. 針對 Feature Engineered 檔案的深入分析
df_feat = dfs.get("feature_engineered")
if df_feat is not None:
    print_header("深入分析：特徵工程資料 (Feature Engineered Data)")
    
    # 建立中文狀態欄位用於顯示
    status_zh = translate_status(df_feat)
    
    # 數值特徵分析
    numerical_cols = ['word_count', 'polarity', 'subjectivity', 'noun_ratio', 'verb_ratio']
    existing_cols = [c for c in numerical_cols if c in df_feat.columns]
    
    if existing_cols and 'status' in df_feat.columns:
        print("\n--- 各狀態的平均值 (Mean) ---")
        print(df_feat.groupby(status_zh)[existing_cols].mean())
        
        print("\n--- 各狀態的中位數 (Median) ---")
        print(df_feat.groupby(status_zh)[existing_cols].median())

    # 布林關鍵字分析
    bool_cols = ['has_suicidal_keyword', 'has_stress_keyword', 'has_help_keyword']
    existing_bool = [c for c in bool_cols if c in df_feat.columns]
    
    if existing_bool and 'status' in df_feat.columns:
        print("\n--- 各狀態含有特定關鍵字的比例 (%) ---")
        # 轉換為數值以計算平均值 (即比例)
        for col in existing_bool:
            # 處理混合型態，將 True/False 字串或數值統一轉為 0 和 1
            df_feat[col] = df_feat[col].astype(str).map({'True': 1, 'False': 0, '1.0': 1, '0.0': 0})
            df_feat[col] = df_feat[col].fillna(0)

        print(df_feat.groupby(status_zh)[existing_bool].mean() * 100)

# 4. 文本長度分析
print_header("文本長度分析 (Text Length Analysis)")
for name, df in dfs.items():
    if df is not None and 'text' in df.columns and 'status' in df.columns:
        # 如果沒有 text_length 欄位則計算
        if 'text_length' not in df.columns:
            # 確保 text 為字串
            df['text_length'] = df['text'].astype(str).apply(len)
        
        status_zh = translate_status(df)
        print(f"\n[{name}] 各狀態平均文本長度:")
        print(df.groupby(status_zh)['text_length'].mean())

print(f"\n\n分析完成！報告已儲存至: {os.path.abspath(output_file)}")
