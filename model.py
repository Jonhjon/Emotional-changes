import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ===== 特徵提取器（使用預訓練模型） =====
class FeatureExtractor(nn.Module):
    """
    使用預訓練的 Transformer 模型提取文本特徵
    凍結預訓練模型參數，只提取特徵向量
    
    支援三種池化方式:
    - 'cls': 使用 [CLS] token (BERT 標準做法，推薦)
    - 'mean': 平均池化 (對所有有效 token 取平均)
    - 'last': 使用最後一個有效 token
    """
    def __init__(self, model_name, use_auth_token=None, pooling='cls'):
        super(FeatureExtractor, self).__init__()
        
        # 載入預訓練模型
        self.model = AutoModel.from_pretrained(model_name, use_auth_token=use_auth_token)
        
        # 設定池化方式
        self.pooling = pooling
        
        # 設定 pad_token_id (處理多種情況)
        if self.model.config.pad_token_id is None:
            if hasattr(self.model.config, 'eos_token_id') and self.model.config.eos_token_id is not None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
            elif hasattr(self.model.config, 'unk_token_id') and self.model.config.unk_token_id is not None:
                self.model.config.pad_token_id = self.model.config.unk_token_id
            else:
                # 設為 0 (通常是 [PAD] token 的 ID)
                self.model.config.pad_token_id = 0
        
        # 凍結模型參數（不進行微調）
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask=None):
        """
        提取文本的特徵向量
        根據 pooling 方式返回不同的特徵表示
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs['last_hidden_state'].to(torch.float32)
        
        if self.pooling == 'cls':
            # 使用 [CLS] token (第一個位置)
            # BERT 等模型在預訓練時優化 [CLS] 作為句子表示
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'mean':
            # 平均池化：對所有有效 token 取平均
            # 這種方法能充分利用所有 token 的信息
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self.pooling == 'last':
            # 使用最後一個有效 token
            if attention_mask is not None:
                sequence_lengths = torch.eq(attention_mask, 0).int().argmax(-1) - 1
                sequence_lengths = torch.clamp(sequence_lengths, min=0)
                return hidden_states[
                    torch.arange(hidden_states.shape[0], device=hidden_states.device), 
                    sequence_lengths
                ]
            else:
                return hidden_states[:, -1, :]
        
        else:
            raise ValueError(f"不支援的池化方式: {self.pooling}。請使用 'cls', 'mean', 或 'last'。")

# ===== 情緒分類模型（整合特徵提取 + 分類器） =====
class SentimentClassifier(nn.Module):
    """
    完整的情緒分類模型
    包含特徵提取器和分類器兩部分
    """
    def __init__(self, model_name, hidden_size=256, num_classes=2, dropout_rate=0.3, use_auth_token=None, tokenizer=None, pooling='cls'):
        super(SentimentClassifier, self).__init__()
        
        # 特徵提取器（凍結參數）
        self.feature_extractor = FeatureExtractor(model_name, use_auth_token, pooling=pooling)
        
        # 如果提供了 tokenizer,調整 embedding 大小
        if tokenizer is not None:
            self.feature_extractor.model.resize_token_embeddings(len(tokenizer))
            # 重新凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # 獲取特徵維度（預訓練模型的 hidden size）
        self.feature_dim = self.feature_extractor.model.config.hidden_size
        
        # 分類器部分（可訓練）
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向傳播
        input_ids: tokenized 文本
        attention_mask: 注意力遮罩
        """
        # 提取特徵
        features = self.feature_extractor(input_ids, attention_mask)
        
        # 分類
        logits = self.classifier(features)
        
        return logits
    
    def freeze_feature_extractor(self):
        """凍結特徵提取器參數"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """解凍特徵提取器參數（用於微調）"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
