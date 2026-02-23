import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime
from GetDataLoader import GetDataLoader
from model import SentimentClassifier, LSTMSentimentClassifier, GRUSentimentClassifier, BiGRUSentimentClassifier
import json
# ===== 自動組合實驗設定 =====
AUTO_EXPERIMENT = True  # 設為 True 啟動自動組合實驗，False 則使用下方單一配置

# 組合實驗的參數範圍（基於最佳配置）
EXPERIMENT_CONFIGS = {
    'MODEL_NAMES': [
        # --- 原始參考 / 不使用 ---
        # 'intfloat/multilingual-e5-large',       # 560M 參數, 多語言模型
        # 'bert-base-chinese',                    # 102M 參數, BERT 中文版
        # 'hfl/chinese-bert-wwm-ext',             # 102M 參數, BERT 中文全詞遮罩
        # --- 現有清單 ---
        # 'uer/roberta-base-finetuned-dianping-chinese',  # 125M 參數, RoBERTa 中文點評微調版 (開箱即用)
        # 'hfl/chinese-roberta-wwm-ext',                  # 102M 參數, RoBERTa 中文全詞遮罩 (穩健 Baseline)
        'IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment',  # 110M 參數, Erlangshen RoBERTa 情感分析版
        # --- 新增推薦模型 ---
        # 'hfl/chinese-macbert-base',                     # 102M 參數, MacBERT (中文糾錯式預訓練，通常比 RoBERTa 準)
        # 'hfl/rbt3',                                     # 38M 參數, RBT3 (僅 3 層 Transformer，推論極快，適合大量數據)
        # 'hfl/chinese-electra-180g-base-discriminator',  # 102M 參數, ELECTRA (判別式架構，對模糊語意分辨能力佳)
    ],
    'HIDDEN_SIZES': [512],   # 最佳配置: 256（性能與效率平衡）
    'DROPOUT_RATES': [0.3],  # 最佳範圍: 0.2-0.3
    'SCHEDULERS': ['none'],  # 最佳配置: 固定學習率
    'MODEL_TYPES': ['simple'],  # 'simple', 'lstm', 'gru', 'bigru'
}
# ===== 單一實驗超參數設定（當 AUTO_EXPERIMENT = False 時使用）=====
# MODEL_NAME = "intfloat/multilingual-e5-large"  # 預訓練模型
with open('Key.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
USE_AUTH_TOKEN = data.get('Key')
TRAIN_PATH = r'archive\mental_heath_unbanlanced.csv'  # 訓練資料路徑
TEST_PATH = r'archive\mental_heath_unbanlanced.csv'    # 測試資料路徑
VAL_PATH = None  # 驗證資料路徑 (None 表示從訓練資料中分割)

BATCH_SIZE = 200
LEARNING_RATE = 2e-5
NUM_EPOCHS = 50
# HIDDEN_SIZE = 256
NUM_CLASSES = 4  # 正面/負面

# 實驗結果主資料夾
EXPERIMENTS_DIR = 'Experimental_results'
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# ===== 設定裝置 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== 訓練函數 =====
def train_epoch(model, dataloader, optimizer, device):
    """單個 epoch 的訓練"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # 將資料移到 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向傳播
        logits = model(input_ids, attention_mask)
        
        # 計算損失
        loss = criterion(logits, labels)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        # 記錄損失和預測
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新進度條
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# ===== 評估函數 =====
def evaluate(model, dataloader, device):
    """評估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向傳播
            logits = model(input_ids, attention_mask)
            
            # 計算損失
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

# ===== 繪製訓練曲線 =====
def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_dir):
    """繪製訓練歷史曲線"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 損失曲線
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 準確率曲線
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n訓練歷史曲線已儲存至 {save_path}")
    plt.close()


# ===== 繪製混淆矩陣 =====
def plot_confusion_matrix(y_true, y_pred, save_dir, class_names=['Negative', 'Positive']):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩陣已儲存至 {save_path}")
    plt.close()

# ===== 單次實驗執行函數 =====
def run_single_experiment(config, exp_num=None, total_exps=None):
    """執行單次實驗"""
    MODEL_NAME = config['MODEL_NAME']
    HIDDEN_SIZE = config['HIDDEN_SIZE']
    DROPOUT_RATE = config['DROPOUT_RATE']
    SCHEDULER_TYPE = config['SCHEDULER_TYPE']
    MODEL_TYPE = config['MODEL_TYPE']
    USE_SCHEDULER = (SCHEDULER_TYPE != 'none')
    T_MAX = NUM_EPOCHS
    
    print("="*60)
    if exp_num and total_exps:
        print(f"🔬 實驗 {exp_num}/{total_exps}")
    print("情緒分類模型訓練")
    print("="*60)
    print(f"配置: 模型={MODEL_NAME.split('/')[-1]}, Hidden={HIDDEN_SIZE}, Dropout={DROPOUT_RATE}, Scheduler={SCHEDULER_TYPE}")
    print("="*60)
    
    # 0. 創建實驗資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{timestamp}_{MODEL_TYPE}_{MODEL_NAME.split('/')[-1]}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n實驗資料夾: {exp_dir}")
    
    # 1. 載入資料
    print("\n[1/5] 載入資料...")
    data_loader = GetDataLoader(MODEL_NAME, use_auth_token=USE_AUTH_TOKEN)
    train_dataloader, val_dataloader, test_dataloader = data_loader.get_DataLoader(
        data_sum=-1,
        batch_size=BATCH_SIZE,
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=TEST_PATH
    )
    print(f"訓練集批次數: {len(train_dataloader)}")
    print(f"驗證集批次數: {len(val_dataloader)}")
    print(f"測試集批次數: {len(test_dataloader)}")
    
    # 2. 初始化模型
    print("\n[2/5] 初始化模型...")
    print(f"Tokenizer vocab size: {len(data_loader.tokenizer)}")
    
    if MODEL_TYPE == 'gru':
        model = GRUSentimentClassifier(
            model_name=MODEL_NAME,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
            use_auth_token=USE_AUTH_TOKEN,
            tokenizer=data_loader.tokenizer
        )
        print(f"使用模型: GRU Sentiment Classifier")
    elif MODEL_TYPE == 'bigru':
        model = BiGRUSentimentClassifier(
            model_name=MODEL_NAME,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
            use_auth_token=USE_AUTH_TOKEN,
            tokenizer=data_loader.tokenizer
        )
        print(f"使用模型: Bidirectional GRU Sentiment Classifier")
    elif MODEL_TYPE == 'lstm':
        model = LSTMSentimentClassifier(
            model_name=MODEL_NAME,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
            use_auth_token=USE_AUTH_TOKEN,
            tokenizer=data_loader.tokenizer
        )
        print(f"使用模型: LSTM Sentiment Classifier")
    else:
        model = SentimentClassifier(
            model_name=MODEL_NAME,
            hidden_size=HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            dropout_rate=DROPOUT_RATE,
            use_auth_token=USE_AUTH_TOKEN,
            tokenizer=data_loader.tokenizer
        )
        print(f"使用模型: Simple Sentiment Classifier")
    
    print(f"Model vocab size: {model.feature_extractor.model.config.vocab_size}")
    
    model.to(device)
    print(f"模型已移至 {device}")
    print(f"模型參數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 3. 設定優化器跟損失函數
    print("\n[3/5] 設定優化器...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"優化器: AdamW, 學習率: {LEARNING_RATE}")
    
    # 設定學習率調度器
    scheduler = None
    if USE_SCHEDULER:
        if SCHEDULER_TYPE == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
            print(f"學習率調度器: CosineAnnealingLR (T_max={T_MAX})")
        elif SCHEDULER_TYPE == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            print(f"學習率調度器: StepLR (step_size=10, gamma=0.1)")
        elif SCHEDULER_TYPE == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
            print(f"學習率調度器: ReduceLROnPlateau (patience=3)")
        elif SCHEDULER_TYPE == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            print(f"學習率調度器: ExponentialLR (gamma=0.95)")
    
    # 4. 訓練模型
    print("\n[4/5] 開始訓練...")
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # 訓練
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 驗證
        val_loss, val_acc, _, _ = evaluate(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1} 結果:")
        print(f"  訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")
        print(f"  驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(exp_dir, 'best_sentiment_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ 新的最佳模型！驗證準確率: {val_acc:.4f}")
        
        # 更新學習率調度器
        if scheduler is not None:
            if SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_acc)  # ReduceLROnPlateau 需要 metric
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  當前學習率: {current_lr:.2e}")
        
        # 更新學習率調度器
        if scheduler is not None:
            if SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_acc)  # ReduceLROnPlateau 需要 metric
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  當前學習率: {current_lr:.2e}")
    
    # 5. 測試模型
    print("\n[5/5] 測試最佳模型...")
    model_path = os.path.join(exp_dir, 'best_sentiment_model.pth')
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_dataloader, device)
    
    print(f"\n{'='*60}")
    print("最終測試結果")
    print(f"{'='*60}")
    print(f"測試損失: {test_loss:.4f}")
    print(f"測試準確率: {test_acc:.4f}")
    print("\n詳細分類報告:")
    print(classification_report(test_labels, test_preds, 
                            #    target_names=['Negative', 'Positive'],
                               digits=4))
    
    # 6. 繪製結果
    print("\n繪製訓練結果...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs, exp_dir)
    plot_confusion_matrix(test_labels, test_preds, exp_dir)
    
    # 7. 保存結果報告
    report_path = os.path.join(exp_dir, 'training_report.txt')
    with open(report_path, 'w', encoding='utf-8',) as f:
        f.write("情緒分類模型訓練報告\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型類型: {MODEL_TYPE}\n")
        f.write(f"預訓練模型: {MODEL_NAME}\n")
        f.write(f"批次大小: {BATCH_SIZE}\n")
        f.write(f"學習率: {LEARNING_RATE}\n")
        f.write(f"訓練輪數: {NUM_EPOCHS}\n")
        f.write(f"隱藏層大小: {HIDDEN_SIZE}\n\n")
        f.write(f"最佳驗證準確率: {best_val_acc:.4f}\n")
        f.write(f"測試準確率: {test_acc:.4f}\n\n")
        f.write("詳細分類報告:\n")
        f.write(classification_report(test_labels, test_preds, 
                                    # target_names=['Negative', 'Positive'],
                                     digits=4))
    
    print(f"\n訓練報告已儲存至 {report_path}")
    
    # 8. 保存 CSV 記錄
    # 計算評估指標
    precision = precision_score(test_labels, test_preds, average='weighted')
    recall = recall_score(test_labels, test_preds, average='weighted')
    f1 = f1_score(test_labels, test_preds, average='weighted')
    
    # 準備記錄資料
    record = {
        '實驗時間': timestamp,
        '大型語言模型編碼器': MODEL_NAME,
        '分類模型類型': MODEL_TYPE,
        '準確度 (Accuracy)': f'{test_acc:.4f}',
        '精確度 (Precision)': f'{precision:.4f}',
        '召回率 (Recall)': f'{recall:.4f}',
        'F1分數 (F1-Score)': f'{f1:.4f}',
        '訓練輪數': NUM_EPOCHS,
        '批次大小': BATCH_SIZE,
        '學習率': LEARNING_RATE,
        '隱藏層大小': HIDDEN_SIZE,
        'Dropout率': DROPOUT_RATE,
        '學習率調度器': SCHEDULER_TYPE if USE_SCHEDULER else 'None'
    }
    
    # 儲存到 experiments 資料夾的 CSV
    csv_path = os.path.join(EXPERIMENTS_DIR, 'training_records.csv')
    df_record = pd.DataFrame([record])
    
    # 如果檔案已存在，則附加（append）；否則創建新檔案
    if os.path.exists(csv_path):
        # 使用 mode='a' 直接附加到檔案尾端，不會覆蓋原有資料
        df_record.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        # 檔案不存在，創建新檔案並寫入標題
        df_record.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8-sig')
    
    print(f"\n實驗記錄已儲存至 {csv_path}")
    print(f"\n所有實驗結果已儲存至: {exp_dir}")
    print("\n訓練完成！")
    print("="*60)
    
    return {
        'exp_dir': exp_dir,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'config': config
    }

# ===== 自動組合實驗主函數 =====
def run_all_experiments():
    """自動執行所有參數組合的實驗"""
    import itertools
    import time
    
    print("\n" + "🚀"*40)
    print("🚀 自動組合實驗系統啟動")
    print("🚀"*40)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 生成所有參數組合
    combinations = list(itertools.product(
        EXPERIMENT_CONFIGS['MODEL_NAMES'],
        EXPERIMENT_CONFIGS['HIDDEN_SIZES'],
        EXPERIMENT_CONFIGS['DROPOUT_RATES'],
        EXPERIMENT_CONFIGS['SCHEDULERS'],
        EXPERIMENT_CONFIGS['MODEL_TYPES']
    ))
    
    total_experiments = len(combinations)
    print(f"📊 總共將執行 {total_experiments} 個實驗組合\n")
    print("實驗組合:")
    print(f"  - 模型: {len(EXPERIMENT_CONFIGS['MODEL_NAMES'])} 種")
    print(f"  - Hidden Size: {len(EXPERIMENT_CONFIGS['HIDDEN_SIZES'])} 種")
    print(f"  - Dropout Rate: {len(EXPERIMENT_CONFIGS['DROPOUT_RATES'])} 種")
    print(f"  - Scheduler: {len(EXPERIMENT_CONFIGS['SCHEDULERS'])} 種")
    print(f"  - Model Type: {len(EXPERIMENT_CONFIGS['MODEL_TYPES'])} 種")
    print("\n" + "="*80 + "\n")
    
    # 執行所有實驗
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    for idx, (model_name, hidden_size, dropout_rate, scheduler, model_type) in enumerate(combinations, 1):
        config = {
            'MODEL_NAME': model_name,
            'HIDDEN_SIZE': hidden_size,
            'DROPOUT_RATE': dropout_rate,
            'SCHEDULER_TYPE': scheduler,
            'MODEL_TYPE': model_type
        }
        
        print(f"\n{'='*80}")
        print(f"開始實驗 {idx}/{total_experiments}")
        print(f"{'='*80}")
        
        try:
            result = run_single_experiment(config, exp_num=idx, total_exps=total_experiments)
            results.append(result)
            successful += 1
            print(f"\n✅ 實驗 {idx}/{total_experiments} 完成")
        except Exception as e:
            failed += 1
            print(f"\n❌ 實驗 {idx}/{total_experiments} 失敗: {e}")
            import traceback
            traceback.print_exc()
        
        # 顯示進度
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = avg_time * (total_experiments - idx)
        print(f"\n📊 進度: {idx}/{total_experiments} ({idx/total_experiments*100:.1f}%)")
        print(f"⏱️  已用時: {elapsed/60:.1f} 分鐘")
        print(f"⏱️  預估剩餘: {remaining/60:.1f} 分鐘")
        print(f"✅ 成功: {successful} | ❌ 失敗: {failed}")
    
    # 總結
    total_time = time.time() - start_time
    print("\n" + "🎉"*40)
    print("🎉 所有實驗完成！")
    print("🎉"*40)
    print(f"\n總結:")
    print(f"  📊 總實驗數: {total_experiments}")
    print(f"  ✅ 成功: {successful}")
    print(f"  ❌ 失敗: {failed}")
    print(f"  ⏱️  總耗時: {total_time/3600:.2f} 小時 ({total_time/60:.1f} 分鐘)")
    print(f"  📅 完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 查看所有實驗結果: {EXPERIMENTS_DIR}/training_records.csv")
    print("="*80)
    
    return results


# ===== 主函數（原始 main 函數改為單次實驗，新增自動模式） =====
# def main():  # 原始單次實驗函數已重構為 run_single_experiment
#     pass


if __name__ == "__main__":
    # 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if AUTO_EXPERIMENT:
        # 執行所有組合實驗
        run_all_experiments()
    else:
        # 執行單一實驗（需要手動設定參數）
        single_config = {
            'MODEL_NAME': 'intfloat/multilingual-e5-large',
            'HIDDEN_SIZE': 256,
            'DROPOUT_RATE': 0.3,
            'SCHEDULER_TYPE': 'cosine',
            'MODEL_TYPE': 'simple',
        }
        run_single_experiment(single_config)