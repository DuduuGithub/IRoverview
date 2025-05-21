import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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

class SearchExpressionEncoder(nn.Module):
    """检索式编码器"""
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.hidden_size = self.bert.config.hidden_size
        
        # 逻辑运算符的嵌入层
        self.operator_embeddings = nn.Embedding(4, self.hidden_size)  # [AND, OR, NOT, *]
        
        # 逻辑组合层
        self.logic_combine = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def encode_term(self, term_text, operator=None):
        """编码单个检索词项"""
        # 获取BERT编码
        term_encoding = self.bert(term_text).last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        
        if operator is not None:
            # 获取运算符嵌入
            op_embedding = self.operator_embeddings(operator)  # [batch_size, hidden_size]
            # 组合词项和运算符
            return self.logic_combine(torch.cat([term_encoding, op_embedding], dim=-1))
        
        return term_encoding
    
    def combine_terms(self, term1, term2, operator):
        """组合两个词项"""
        op_embedding = self.operator_embeddings(operator)
        combined = self.logic_combine(torch.cat([term1, term2], dim=-1))
        return combined * op_embedding  # 使用运算符调制组合结果

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, doc):
        # 确保输入形状正确 [batch_size, seq_len, hidden_size]
        if len(query.shape) == 2:
            query = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        if len(doc.shape) == 2:
            doc = doc.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
        # 交叉注意力，让查询和文档之间进行更深入的交互
        query = self.query_proj(query)
        key = self.key_proj(doc)
        value = self.value_proj(doc)
        attn_output, _ = self.attention(query, key, value)
        return self.dropout(self.norm(attn_output + query)).squeeze(1)  # [batch_size, hidden_size]

class MultiViewMatching(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.exact_matching = nn.Linear(hidden_size, hidden_size)
        self.semantic_matching = nn.Linear(hidden_size * 3, hidden_size)
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, query_vec, doc_vec):
        # 精确匹配
        exact_match = self.exact_matching(query_vec * doc_vec)
        # 语义匹配
        semantic_match = self.semantic_matching(
            torch.cat([query_vec, doc_vec, query_vec * doc_vec], dim=-1)
        )
        # 组合不同视角的匹配结果
        combined = torch.cat([exact_match, semantic_match], dim=-1)
        return self.combine(combined)

class HierarchicalEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 修改GRU的隐藏层大小，确保输出维度正确
        self.word_level = nn.GRU(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.sent_level = nn.GRU(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # 添加最终的投影层，确保输出维度为hidden_size
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # 词级别编码
        word_hidden, _ = self.word_level(x)  # [batch_size, seq_len, hidden_size]
        # 句子级别编码
        sent_hidden, _ = self.sent_level(word_hidden)  # [batch_size, seq_len, hidden_size]
        # 池化得到最终表示
        pooled = self.pooling(sent_hidden.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_size]
        # 投影到正确的维度
        return self.final_proj(pooled)  # [batch_size, hidden_size]

class IRRankingModel(nn.Module):
    """改进的语义排序模型"""
    def __init__(self, bert_path="D:/bert/bert-base-uncased"):
        super().__init__()
        
        # 检查CUDA是否可用并强制使用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            logger.info(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("警告：CUDA不可用，将使用CPU进行训练（这会显著降低训练速度）")
            logger.warning("请检查：")
            logger.warning("1. 是否安装了CUDA和cuDNN")
            logger.warning("2. 是否安装了GPU版本的PyTorch")
            logger.warning("3. 环境变量是否正确设置")
        
        logger.info(f"IRRankingModel 初始化于设备: {self.device}")
        
        # 确保BERT模型加载到正确的设备上
        try:
            self.bert = BertModel.from_pretrained(bert_path)
            self.bert = self.bert.to(self.device)
            logger.info(f"BERT模型已加载到: {self.device}")
        except Exception as e:
            logger.error(f"加载BERT模型时出错: {str(e)}")
            raise
            
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        
        # 检索式编码器
        self.expression_encoder = SearchExpressionEncoder(self.bert)
        
        # 文档编码器（共享BERT）
        self.doc_encoder = self.bert
        
        # 新增：层次化编码器
        self.hierarchical_encoder = HierarchicalEncoder(self.hidden_size)
        
        # 新增：交叉注意力层
        self.cross_attention = CrossAttention(self.hidden_size)
        
        # 新增：多视角匹配层
        self.multi_view_matching = MultiViewMatching(self.hidden_size)
        
        # 相关性计算层（更复杂的结构）
        self.relevance_calculator = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1)
        )
        
        # 将所有模块移动到指定设备
        self.expression_encoder = self.expression_encoder.to(self.device)
        self.hierarchical_encoder = self.hierarchical_encoder.to(self.device)
        self.cross_attention = self.cross_attention.to(self.device)
        self.multi_view_matching = self.multi_view_matching.to(self.device)
        self.relevance_calculator = self.relevance_calculator.to(self.device)
        self = self.to(self.device)
        
        logger.info(f"所有模型组件已初始化完成并移动到: {self.device}")
        
        # 打印CUDA内存使用情况（如果使用GPU）
        if self.device.type == 'cuda':
            logger.info(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            logger.info(f"GPU内存缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
    
    def encode_text(self, text):
        """编码文本，使用层次化编码"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # 获取序列表示
            sequence_output = outputs.last_hidden_state
            # 使用层次化编码器进一步处理
            text_embedding = self.hierarchical_encoder(sequence_output)
        
        return text_embedding
    
    def encode_search_expression(self, expression):
        """编码检索式
        expression: 包含检索词和逻辑关系的结构化数据或纯文本查询
        """
        terms = []
        operators = []
        
        def process_expression(expr):
            if isinstance(expr, str):  # 处理纯文本查询
                # 分词处理
                words = expr.strip().split()
                if not words:
                    logger.warning(f"查询文本为空: {expr}")
                    return
                
                # 对每个词进行编码
                for word in words:
                    if len(word) > 1:  # 忽略单字符词
                        term_encoding = self.encode_text([word])
                        terms.append(term_encoding)
                        operators.append(0)  # 默认使用AND
            
            elif isinstance(expr, SearchTerm):
                if not expr.content.strip():
                    logger.warning(f"SearchTerm内容为空: {expr}")
                    return
                
                term_encoding = self.encode_text([expr.content])
                terms.append(term_encoding)
                if expr.operation == 'not_match':
                    operators.append(2)  # NOT
                else:
                    operators.append(0)  # AND
            
            elif isinstance(expr, LogicalGroup):
                for i, term in enumerate(expr.terms):
                    process_expression(term)
                    if i < len(expr.terms) - 1 and terms:  # 只在有terms时添加操作符
                        operators.append(0 if expr.operation == 'and' else 1)
        
        try:
            # 处理输入
            if isinstance(expression, str):
                process_expression(expression)
            else:
                process_expression(expression)
            
            # 检查是否有有效的词项
            if not terms:
                logger.warning("没有找到有效的查询词项，尝试整体编码查询")
                # 尝试整体编码查询文本
                if isinstance(expression, str) and expression.strip():
                    return self.encode_text([expression.strip()])
                elif isinstance(expression, SearchTerm) and expression.content.strip():
                    return self.encode_text([expression.content.strip()])
                else:
                    logger.error("无法处理空查询")
                    return torch.zeros(1, self.hidden_size).to(self.device)
            
            # 组合所有词项
            combined_encoding = terms[0]
            for i in range(1, len(terms)):
                combined_encoding = self.expression_encoder.combine_terms(
                    combined_encoding, terms[i], 
                    torch.tensor([operators[i-1]]).to(self.device)
                )
            
            return combined_encoding
            
        except Exception as e:
            logger.error(f"编码检索式时出错: {str(e)}")
            # 返回一个空的编码向量
            return torch.zeros(1, self.hidden_size).to(self.device)
    
    def forward(self, query_expr, doc_text):
        """
        计算检索式和文档的相关性分数，使用增强的匹配机制
        """
        # 编码检索式
        if isinstance(query_expr, (str, list)):
            query_encoding = self.encode_text(query_expr if isinstance(query_expr, list) else [query_expr])
        else:
            query_encoding = self.encode_search_expression(query_expr)
        
        # 编码文档
        if isinstance(doc_text, (str, list)):
            doc_encoding = self.encode_text(doc_text if isinstance(doc_text, list) else [doc_text])
        else:
            logger.warning(f"Unexpected doc_text type: {type(doc_text)}")
            doc_encoding = self.encode_text([str(doc_text)])
        
        # 1. 交叉注意力交互
        cross_features = self.cross_attention(query_encoding, doc_encoding)
        
        # 2. 多视角匹配
        matching_score = self.multi_view_matching(query_encoding, doc_encoding)
        
        # 3. 组合所有特征
        concat_features = torch.cat([
            query_encoding,
            doc_encoding,
            cross_features
        ], dim=-1)
        
        # 4. 计算最终相关性分数
        relevance_score = self.relevance_calculator(concat_features)
        
        return relevance_score.squeeze(-1)
    
    def get_document_scores(self, query_expr: str, doc_texts: list) -> torch.Tensor:
        """
        批量计算多个文档的相关性分数
        """
        # 编码检索式（只需计算一次）
        query_encoding = self.encode_search_expression(query_expr)
        
        # 批量编码文档
        doc_encodings = []
        batch_size = 32  # 可以根据GPU内存调整
        
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i + batch_size]
            batch_encodings = self.encode_text(batch_texts)
            doc_encodings.append(batch_encodings)
        
        doc_encodings = torch.cat(doc_encodings, dim=0)
        
        # 计算相关性分数
        scores = []
        for doc_encoding in doc_encodings:
            concat_features = torch.cat([
                query_encoding.expand(1, -1),
                doc_encoding.unsqueeze(0)
            ], dim=-1)
            score = self.relevance_calculator(concat_features)
            scores.append(score)
        
        return torch.cat(scores, dim=0).squeeze(-1)

class RelevanceScore(nn.Module):
    """相关性分数计算，考虑点击顺序和停留时间"""
    def __init__(self):
        super().__init__()
        self.time_weight = nn.Parameter(torch.tensor([0.5]))  # 可学习的时间权重
    
    def forward(self, click_positions, dwell_times):
        """
        计算综合相关性分数
        click_positions: 点击位置 [batch_size]
        dwell_times: 停留时间（秒）[batch_size]
        """
        # 将点击位置转换为分数（越早点击分数越高）
        position_scores = 1.0 / (click_positions + 1)
        
        # 将停留时间归一化（使用sigmoid函数将时间映射到0-1区间）
        time_scores = torch.sigmoid(dwell_times / 300.0)  # 300秒作为参考时间
        
        # 组合两种分数
        combined_scores = (1 - self.time_weight) * position_scores + self.time_weight * time_scores
        
        return combined_scores

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