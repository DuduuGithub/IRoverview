import os
import sys
import torch
import logging
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sort_ai.model_backup import NLRankingModel
from Database.model import Work, RerankSession, UserBehavior
from .model_trainer_backup import ModelTrainer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RankingService:
    """排序服务类，用于对检索结果进行重排序"""
    
    def __init__(self, model_path=None):
        """
        初始化排序服务
        
        Args:
            model_path: 模型文件路径，如果为None则使用默认的best_model.pth
        """
        # 获取sort_ai目录的路径
        self.sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 如果没有指定模型路径，使用默认路径
        if model_path is None:
            model_path = os.path.join(self.sort_ai_dir, 'models', 'best_model.pth')
        
        # 初始化模型
        self.trainer = ModelTrainer()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 加载模型
        if not self.trainer.load_model(model_path):
            raise RuntimeError("加载模型失败")
        
        # 将模型设置为评估模式
        self.trainer.model.eval()
        logger.info("排序服务初始化完成")
    
    def process_doc_text(self, doc: Dict) -> str:
        """
        处理文档文本，确保格式统一
        
        Args:
            doc: 包含title和abstract的文档字典
        
        Returns:
            str: 处理后的文档文本
        """
        # 处理标题
        title = str(doc.get('title', '')).strip()
        
        # 处理摘要
        abstract = str(doc.get('abstract', '')).strip()
        if not abstract and 'abstract_inverted_index' in doc:
            abstract = str(doc['abstract_inverted_index']).strip()
        
        # 清理文本
        def clean_text(text):
            return ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.,!?-()[]{}:;"\'')
        
        title = clean_text(title)
        abstract = clean_text(abstract)
        
        # 使用特殊标记分隔标题和摘要
        return f"[TITLE] {title} [ABSTRACT] {abstract}".strip()
    
    def rerank_documents(self, query: str, documents: List[Dict], 
                        batch_size: int = 32) -> List[Tuple[Dict, float]]:
        """
        对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            documents: 文档列表，每个文档是一个字典，包含必要的字段（至少要有title和abstract）
            batch_size: 批处理大小
            
        Returns:
            List[Tuple[Dict, float]]: 排序后的(文档,得分)列表，按得分降序排列
        """
        if not documents:
            return []
        
        try:
            # 将模型设置为评估模式
            self.trainer.model.eval()
            
            # 处理所有文档文本
            doc_texts = [self.process_doc_text(doc) for doc in documents]
            
            # 批量计算文档得分
            with torch.no_grad():
                scores = []
                for i in range(0, len(doc_texts), batch_size):
                    batch_texts = doc_texts[i:i + batch_size]
                    batch_scores = self.trainer.model.get_document_scores(query, batch_texts)
                    scores.extend(batch_scores.cpu().numpy())
            
            # 将文档和得分打包并排序
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores
            
        except Exception as e:
            logger.error(f"重排序过程出错: {str(e)}")
            # 如果出错，返回原始顺序
            return list(zip(documents, [0.0] * len(documents)))
    
    def rerank_search_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """
        重排序搜索结果，并更新结果中的rank和relevance_score字段
        
        Args:
            query: 用户查询
            search_results: 搜索结果列表，每个结果是一个字典，包含文档信息
            
        Returns:
            List[Dict]: 重排序后的搜索结果列表
        """
        try:
            # 对文档进行重排序
            doc_scores = self.rerank_documents(query, search_results)
            
            # 更新排序结果
            ranked_results = []
            for rank, (doc, score) in enumerate(doc_scores, 1):
                doc_copy = doc.copy()
                doc_copy['rank'] = rank
                doc_copy['relevance_score'] = float(score)
                ranked_results.append(doc_copy)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"重排序搜索结果时出错: {str(e)}")
            return search_results  # 如果出错，返回原始结果 