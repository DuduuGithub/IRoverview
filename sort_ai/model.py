import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import logging
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

logger = logging.getLogger(__name__)

class NLRankingModel(nn.Module):
    """基于自然语言的重排序模型"""
    def __init__(self, bert_path=None):
        super().__init__()
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"NLRankingModel 初始化于设备: {self.device}")
        
        # 如果没有指定bert_path，使用默认路径
        if bert_path is None:
            sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
            bert_path = os.path.join(sort_ai_dir, 'bert', 'bert-base-uncased')
            logger.info(f"使用默认BERT模型路径: {bert_path}")
        
        # 加载BERT模型
        try:
            self.bert = BertModel.from_pretrained(bert_path)
            self.bert = self.bert.to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
            logger.info("BERT模型加载成功")
        except Exception as e:
            logger.error(f"加载BERT模型时出错: {str(e)}")
            raise
        
        # BiLSTM层处理BERT输出
        self.hidden_size = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,  # BiLSTM输出是双向的
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 相关性计算层
        self.relevance_calculator = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 将所有组件移到指定设备
        self.to(self.device)
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """编码文本序列"""
        # 确保输入是列表形式
        if isinstance(text, str):
            text = [text]
            
        # BERT分词
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # BERT编码
        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
            sequence_output = bert_outputs.last_hidden_state
        
        # BiLSTM处理
        lstm_output, _ = self.bilstm(sequence_output)
        
        # 通过注意力机制获取文本表示
        attn_output, _ = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # 获取序列的平均表示
        text_embedding = attn_output.mean(dim=1)
        
        return text_embedding
    
    def forward(self, query: str, doc_title: str, doc_abstract: str) -> torch.Tensor:
        """
        计算查询和文档的相关性得分
        
        Args:
            query: 用户输入的自然语言查询
            doc_title: 文档标题
            doc_abstract: 文档摘要
            
        Returns:
            torch.Tensor: 相关性得分 (0-1)
        """
        # 编码查询
        query_embedding = self.encode_text(query)
        
        # 编码文档标题
        title_embedding = self.encode_text(doc_title)
        
        # 编码文档摘要
        abstract_embedding = self.encode_text(doc_abstract)
        
        # 组合文档特征（标题和摘要）
        doc_embedding = torch.cat([
            title_embedding * 1.5,  # 标题权重更高
            abstract_embedding
        ], dim=-1)
        
        # 计算相关性得分
        relevance_score = self.relevance_calculator(doc_embedding)
        
        return relevance_score.squeeze(-1)
    
    def rerank_documents(self, query: str, documents: List[Dict], batch_size: int = 32) -> List[Tuple[Dict, float]]:
        """
        对文档列表进行重排序
        
        Args:
            query: 用户输入的自然语言查询
            documents: 文档列表，每个文档是包含title和abstract的字典
            batch_size: 批处理大小
            
        Returns:
            List[Tuple[Dict, float]]: 排序后的(文档,得分)列表
        """
        # 编码查询（只需要计算一次）
        query_embedding = self.encode_text(query)
        
        # 批量处理文档
        all_scores = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # 准备批次数据
            titles = [doc.get('title', '') for doc in batch_docs]
            abstracts = [doc.get('abstract', '') for doc in batch_docs]
            
            # 编码文档
            title_embeddings = self.encode_text(titles)
            abstract_embeddings = self.encode_text(abstracts)
            
            # 组合文档特征
            doc_embeddings = torch.cat([
                title_embeddings * 1.5,
                abstract_embeddings
            ], dim=-1)
            
            # 计算相关性得分
            scores = self.relevance_calculator(doc_embeddings)
            all_scores.extend(scores.squeeze(-1).cpu().tolist())
        
        # 将文档和得分打包并排序
        doc_scores = list(zip(documents, all_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores