import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

@dataclass
class SearchTerm:
    """检索词条"""
    field: str          # 检索字段：'title', 'abstract', 'keywords'等
    content: str        # 检索内容
    operation: str      # 操作类型：'match'（默认）, 'not_match'

@dataclass
class LogicalGroup:
    """逻辑组"""
    terms: List[Union[SearchTerm, 'LogicalGroup']]  # 检索词条或子逻辑组
    operation: str      # 组内逻辑：'and', 'or'

class SearchExpression:
    """检索式解析器"""
    def __init__(self):
        self.field_mapping = {
            'title': 0,
            'abstract': 1,
            'keywords': 2,
            'author': 3,
            # 可以添加更多字段
        }
    
    def parse(self, expression: LogicalGroup) -> Dict:
        """
        解析检索式，返回字段信息和逻辑关系
        
        示例检索式：
        LogicalGroup(
            terms=[
                SearchTerm(field='title', content='机器学习', operation='match'),
                LogicalGroup(
                    terms=[
                        SearchTerm(field='abstract', content='深度学习', operation='match'),
                        SearchTerm(field='keywords', content='图像处理', operation='not_match')
                    ],
                    operation='or'
                )
            ],
            operation='and'
        )
        """
        query_fields = {}
        logical_ops = {
            'and': [],
            'or': [],
            'not': []
        }
        field_contents = {}
        
        def process_group(group: LogicalGroup, parent_idx: int = None) -> List[int]:
            term_indices = []
            
            for term in group.terms:
                if isinstance(term, SearchTerm):
                    # 处理检索词条
                    field_idx = len(field_contents)
                    field_contents[field_idx] = {
                        'field': term.field,
                        'content': term.content
                    }
                    term_indices.append(field_idx)
                    
                    # 记录NOT操作
                    if term.operation == 'not_match':
                        logical_ops['not'].append(field_idx)
                        
                elif isinstance(term, LogicalGroup):
                    # 递归处理子组
                    sub_indices = process_group(term, parent_idx)
                    term_indices.extend(sub_indices)
            
            # 处理组内逻辑关系
            if len(term_indices) > 1:
                for i in range(len(term_indices) - 1):
                    if group.operation == 'and':
                        logical_ops['and'].append((term_indices[i], term_indices[i + 1]))
                    elif group.operation == 'or':
                        logical_ops['or'].append((term_indices[i], term_indices[i + 1]))
            
            return term_indices
        
        process_group(expression)
        
        # 构建query_fields
        for idx, content in field_contents.items():
            field = content['field']
            if field not in query_fields:
                query_fields[field] = {
                    'contents': [],
                    'indices': []
                }
            query_fields[field]['contents'].append(content['content'])
            query_fields[field]['indices'].append(idx)
        
        return {
            'query_fields': query_fields,
            'logical_ops': logical_ops,
            'field_contents': field_contents
        }

class MultiFieldAttention(nn.Module):
    """字段间注意力机制"""
    def __init__(self, hidden_size):
        super(MultiFieldAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        
        # 添加字段融合层
        self.field_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, field_vectors):
        """
        输入: field_vectors [batch_size, num_fields, hidden_size]
        输出: 
        - attended_fields: 每个字段的注意力结果 [batch_size, num_fields, hidden_size]
        - fused_vector: 融合后的单个向量 [batch_size, hidden_size]
        """
        # 1. 计算多头注意力
        attended_output, _ = self.attention(
            field_vectors, field_vectors, field_vectors
        )
        attended_fields = self.norm(field_vectors + attended_output)
        
        # 2. 融合所有字段信息
        fused_vector = self.field_fusion(attended_fields.mean(dim=1))
        
        return attended_fields, fused_vector

class LogicalOperatorFusion(nn.Module):
    """逻辑运算符融合层"""
    def __init__(self, hidden_size):
        super(LogicalOperatorFusion, self).__init__()
        self.hidden_size = hidden_size
        
        # NOT运算的转换层
        self.not_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()  # 使用tanh来帮助学习"相反"的语义
        )
        
        # AND运算的融合层
        self.and_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # OR运算的融合层
        self.or_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, field_vectors, logical_ops):
        """
        按照逻辑运算顺序依次处理，最终得到一个向量
        field_vectors: [batch_size, num_fields, hidden_size]
        logical_ops: 包含逻辑运算信息的字典
        return: [batch_size, hidden_size]
        """
        batch_size = field_vectors.size(0)
        num_fields = field_vectors.size(1)
        
        # 存储中间结果，初始包含所有原始向量
        intermediate_vectors = field_vectors.clone()  # [batch_size, num_fields, hidden_size]
        
        # 1. 首先处理NOT操作
        if 'not' in logical_ops:
            for field_idx in logical_ops['not']:
                intermediate_vectors[:, field_idx] = self.not_transform(field_vectors[:, field_idx])
        
        # 用于追踪已处理的字段
        processed_indices = set()
        # 存储最终的逻辑运算结果
        result_vectors = []
        
        # 2. 处理AND操作
        if 'and' in logical_ops:
            for i, j in logical_ops['and']:
                if i in processed_indices or j in processed_indices:
                    continue
                # 获取两个向量
                vec1 = intermediate_vectors[:, i]  # [batch_size, hidden_size]
                vec2 = intermediate_vectors[:, j]  # [batch_size, hidden_size]
                # 拼接并通过AND融合层
                combined = torch.cat([vec1, vec2], dim=-1)  # [batch_size, hidden_size*2]
                result = self.and_fusion(combined)  # [batch_size, hidden_size]
                result_vectors.append(result)
                processed_indices.add(i)
                processed_indices.add(j)
        
        # 3. 处理OR操作
        if 'or' in logical_ops:
            for i, j in logical_ops['or']:
                if i in processed_indices or j in processed_indices:
                    continue
                # 获取两个向量
                vec1 = intermediate_vectors[:, i]
                vec2 = intermediate_vectors[:, j]
                # 拼接并通过OR融合层
                combined = torch.cat([vec1, vec2], dim=-1)
                result = self.or_fusion(combined)
                result_vectors.append(result)
                processed_indices.add(i)
                processed_indices.add(j)
        
        # 4. 处理未参与运算的字段
        for i in range(num_fields):
            if i not in processed_indices:
                result_vectors.append(intermediate_vectors[:, i])
        
        # 如果有多个结果，取平均；如果只有一个结果，直接返回
        if len(result_vectors) > 1:
            final_vector = torch.stack(result_vectors, dim=1).mean(dim=1)  # [batch_size, hidden_size]
        else:
            final_vector = result_vectors[0]  # [batch_size, hidden_size]
        
        return final_vector

class IRRankingModel(nn.Module):
    def __init__(self, doc_feature_dim=768):
        super(IRRankingModel, self).__init__()
        
        # 检索词编码器
        self.query_encoder = BertModel.from_pretrained("D:/bert/bert-base-chinese")
        
        # 文档编码器（与检索词共享参数）
        self.doc_encoder = self.query_encoder
        
        # 检索式解析器
        self.search_parser = SearchExpression()
        
        # 逻辑运算符融合层
        self.logical_fusion = LogicalOperatorFusion(doc_feature_dim)
        
        # 文档特征权重（可学习）
        self.title_weight = nn.Parameter(torch.tensor(0.6))
        self.abstract_weight = nn.Parameter(torch.tensor(0.4))
        
        # 相关性计算层
        self.relevance_layer = nn.Sequential(
            nn.Linear(doc_feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def encode_query_fields(self, query_fields: Dict) -> torch.Tensor:
        """
        编码检索式中的所有字段
        query_fields: {
            'field_name': {
                'contents': [content1, content2, ...],
                'indices': [idx1, idx2, ...]
            }
        }
        """
        field_vectors = []
        field_indices = []
        
        # 按字段分别编码
        for field_name, field_data in query_fields.items():
            for content in field_data['contents']:
                # 对每个内容进行编码
                encoded = self.query_encoder(
                    **self.tokenizer(content, return_tensors='pt', padding=True)
                )
                field_vectors.append(encoded.last_hidden_state[:, 0, :])
                field_indices.extend(field_data['indices'])
        
        # 按原始索引重排序
        field_vectors = torch.stack(field_vectors)
        sorted_indices = torch.argsort(torch.tensor(field_indices))
        field_vectors = field_vectors[sorted_indices]
        
        return field_vectors
    
    def encode_document(self, title_vectors, abstract_vectors):
        """
        简化的文档编码：直接加权组合
        title_vectors: [batch_size, hidden_size]
        abstract_vectors: [batch_size, hidden_size]
        return: [batch_size, hidden_size]
        """
        # 使用可学习的权重组合标题和摘要
        doc_vector = (self.title_weight * title_vectors + 
                     self.abstract_weight * abstract_vectors)
        return doc_vector
    
    def compute_relevance(self, query_vector, doc_vector):
        """
        计算查询向量和文档向量的相关性
        query_vector: [batch_size, hidden_size]
        doc_vector: [batch_size, hidden_size]
        """
        # 拼接查询向量和文档向量
        interaction = torch.cat([query_vector, doc_vector], dim=-1)
        # 计算相关性得分
        relevance_score = self.relevance_layer(interaction)
        return relevance_score
    
    def forward(self, search_expression: LogicalGroup, title_vectors, abstract_vectors):
        """
        前向传播
        search_expression: 检索式
        title_vectors: [batch_size, hidden_size]
        abstract_vectors: [batch_size, hidden_size]
        return: [batch_size, 1]
        """
        # 1. 解析检索式
        parsed = self.search_parser.parse(search_expression)
        
        # 2. 编码所有检索字段
        query_vectors = self.encode_query_fields(parsed['query_fields'])  # [batch_size, num_fields, hidden_size]
        
        # 3. 进行逻辑运算，得到最终的查询向量
        query_vector = self.logical_fusion(query_vectors, parsed['logical_ops'])  # [batch_size, hidden_size]
        
        # 4. 编码文档
        doc_vector = self.encode_document(title_vectors, abstract_vectors)  # [batch_size, hidden_size]
        
        # 5. 计算相关性得分
        score = self.compute_relevance(query_vector, doc_vector)  # [batch_size, 1]
        
        return score

# 使用示例：
"""
# 构建检索式：(标题包含"机器学习") AND (摘要包含"深度学习" OR NOT 关键词包含"图像处理")
search_expr = LogicalGroup(
    terms=[
        SearchTerm(
            field='title',
            content='机器学习',
            operation='match'
        ),
        LogicalGroup(
            terms=[
                SearchTerm(
                    field='abstract',
                    content='深度学习',
                    operation='match'
                ),
                SearchTerm(
                    field='keywords',
                    content='图像处理',
                    operation='not_match'
                )
            ],
            operation='or'
        )
    ],
    operation='and'
)

# 使用模型
scores, details = model(search_expr, title_vectors, abstract_vectors)
""" 