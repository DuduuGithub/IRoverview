import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MultiFieldAttention(nn.Module):
    """字段间注意力机制"""
    def __init__(self, hidden_size):
        super(MultiFieldAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, field_vectors):
        """
        field_vectors: [batch_size, num_fields, hidden_size]
        """
        attn_output, _ = self.attention(field_vectors, field_vectors, field_vectors)
        return self.norm(field_vectors + attn_output)

class LogicalOperatorFusion(nn.Module):
    """逻辑运算符融合层"""
    def __init__(self, hidden_size):
        super(LogicalOperatorFusion, self).__init__()
        self.hidden_size = hidden_size
        
        # AND运算的融合层
        self.and_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # OR运算的融合层
        self.or_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # NOT运算的转换层
        self.not_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()  # 使用tanh来实现反转效果
        )
    
    def forward(self, field_vectors, logical_ops):
        """
        field_vectors: [batch_size, num_fields, hidden_size]
        logical_ops: 包含逻辑运算信息的字典
        """
        batch_size = field_vectors.size(0)
        num_fields = field_vectors.size(1)
        
        # 处理NOT操作
        if 'not' in logical_ops:
            for field_idx in logical_ops['not']:
                field_vectors[:, field_idx] = self.not_transform(field_vectors[:, field_idx])
        
        # 处理AND和OR操作
        fused_vectors = []
        skip_indices = set()
        
        # 处理AND操作
        if 'and' in logical_ops:
            for i, j in logical_ops['and']:
                if i in skip_indices or j in skip_indices:
                    continue
                combined = torch.cat([field_vectors[:, i], field_vectors[:, j]], dim=-1)
                fused = self.and_fusion(combined)
                fused_vectors.append(fused)
                skip_indices.add(i)
                skip_indices.add(j)
        
        # 处理OR操作
        if 'or' in logical_ops:
            for i, j in logical_ops['or']:
                if i in skip_indices or j in skip_indices:
                    continue
                combined = torch.cat([field_vectors[:, i], field_vectors[:, j]], dim=-1)
                fused = self.or_fusion(combined)
                fused_vectors.append(fused)
                skip_indices.add(i)
                skip_indices.add(j)
        
        # 添加未参与运算的字段
        for i in range(num_fields):
            if i not in skip_indices:
                fused_vectors.append(field_vectors[:, i])
        
        # 合并所有结果
        return torch.stack(fused_vectors, dim=1)

class IRRankingModel(nn.Module):
    def __init__(self, doc_feature_dim=768):
        super(IRRankingModel, self).__init__()
        
        # 检索词编码器
        self.query_encoder = BertModel.from_pretrained("D:/bert/bert-base-chinese")
        
        # 字段间注意力层
        self.field_attention = MultiFieldAttention(doc_feature_dim)
        
        # 逻辑运算符融合层
        self.logical_fusion = LogicalOperatorFusion(doc_feature_dim)
        
        # 字段权重（可学习）
        self.field_weights = nn.Parameter(torch.ones(1) / 1)
        
        # 文档特征编码
        self.title_weight = nn.Parameter(torch.tensor(0.6))
        self.abstract_weight = nn.Parameter(torch.tensor(0.4))
        
        # 交互层
        self.interaction_layer = nn.Sequential(
            nn.Linear(doc_feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # 最终得分融合层
        self.score_fusion = nn.Sequential(
            nn.Linear(doc_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def encode_query_fields(self, query_fields):
        """编码多个查询字段"""
        field_vectors = []
        for field_name, field_data in query_fields.items():
            field_output = self.query_encoder(
                input_ids=field_data['input_ids'],
                attention_mask=field_data['attention_mask']
            )
            field_vectors.append(field_output.last_hidden_state[:, 0, :])
        
        field_vectors = torch.stack(field_vectors, dim=1)
        attended_fields = self.field_attention(field_vectors)
        
        return attended_fields, field_vectors
    
    def encode_document(self, title_vector, abstract_vector):
        """编码文档信息"""
        doc_vector = (self.title_weight * title_vector + 
                     self.abstract_weight * abstract_vector)
        return doc_vector
    
    def compute_relevance(self, query_vector, doc_vector):
        """计算相关性得分"""
        interaction_features = torch.cat([query_vector, doc_vector], dim=-1)
        return self.interaction_layer(interaction_features)
    
    def forward(self, query_fields, title_vectors, abstract_vectors, logical_ops=None):
        """
        前向传播
        query_fields: 字典，包含每个字段的input_ids和attention_mask
        logical_ops: 字典，包含字段间的逻辑关系
            {
                'and': [(0,1), (2,3)],  # 字段索引对，表示AND关系
                'or': [(4,5)],          # 字段索引对，表示OR关系
                'not': [6]              # 字段索引列表，表示NOT关系
            }
        """
        # 1. 编码检索词字段并进行字段间注意力交互
        attended_fields, original_fields = self.encode_query_fields(query_fields)
        
        # 2. 应用逻辑运算符融合
        if logical_ops:
            query_vectors = self.logical_fusion(attended_fields, logical_ops)
        else:
            query_vectors = attended_fields
        
        # 3. 编码文档
        batch_size = title_vectors.size(0)
        doc_vectors = self.encode_document(title_vectors, abstract_vectors)
        
        # 4. 计算每个查询向量与文档的相关性
        relevance_scores = []
        for i in range(query_vectors.size(1)):
            query_vector = query_vectors[:, i]
            score = self.compute_relevance(
                query_vector.unsqueeze(1).expand(-1, batch_size, -1),
                doc_vectors
            )
            relevance_scores.append(score)
        
        # 5. 合并所有相关性得分
        all_scores = torch.cat(relevance_scores, dim=-1)
        final_score = self.score_fusion(all_scores)
        
        return final_score, all_scores 