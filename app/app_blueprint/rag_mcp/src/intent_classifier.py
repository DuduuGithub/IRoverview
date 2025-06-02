import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Literal
import pandas as pd
from pathlib import Path
import re
from pydantic import BaseModel, Field

from .api_requests import APIProcessor

# 设置日志格式
_log = logging.getLogger(__name__)

# 意图类型
class IntentType:
    GENERAL_SEARCH = "GENERAL_SEARCH"  # 一般性检索，查找与主题相关的文档
    QUESTION_ANSWERING = "QUESTION_ANSWERING"  # 问答，根据上下文回答具体问题
    DOCUMENT_ANALYSIS = "DOCUMENT_ANALYSIS"  # 文档分析，分析指定的文档或比较多个文档
    PDF_PROCESSING = "PDF_PROCESSING"  # PDF处理，处理上传的PDF文件
    PDF_ANALYSIS = "PDF_ANALYSIS"  # PDF分析，深入分析PDF文件内容
    PDF_SIMILAR_LITERATURE = "PDF_SIMILAR_LITERATURE"  # 查找与PDF类似的文献
    PDF_COMPARATIVE_ANALYSIS = "PDF_COMPARATIVE_ANALYSIS"  # 比较PDF与其他文献
    CONVERSATION_CONTINUATION = "CONVERSATION_CONTINUATION"  # 对话延续，基于前面的对话继续提问

# 查询实体模式
class QueryEntities(BaseModel):
    """查询中识别的实体"""
    topics: List[str] = Field(default_factory=list, description="主题关键词")
    document_references: List[str] = Field(default_factory=list, description="引用的文档ID或标题")
    analysis_type: Optional[str] = Field(None, description="分析类型，如'compare', 'summarize', 'critique'等")
    has_pdf: bool = Field(False, description="是否涉及PDF处理")
    
    model_config = {
        "extra": "forbid",  # 等同于设置 additionalProperties: false
    }

# 检索策略参数
class RetrievalStrategy(BaseModel):
    """检索策略参数"""
    retrieval_method: str = Field("hybrid", description="检索方法：hybrid, vector, bm25")
    reranking_approach: str = Field("jina", description="重排序方法：jina, llm, none")
    top_n: int = Field(50, description="返回的结果数量")
    llm_reranking_sample_size: int = Field(100, description="LLM重排序样本大小")
    return_parent_pages: bool = Field(False, description="是否返回父页面")
    
    model_config = {
        "extra": "forbid",  # 等同于设置 additionalProperties: false
    }

# 意图分类结果
class IntentClassification(BaseModel):
    """意图分类结果"""
    intent: str = Field(..., description="意图类型")
    confidence: float = Field(..., description="置信度分数")
    entities: QueryEntities = Field(default_factory=QueryEntities, description="识别的实体")
    suggested_strategy: RetrievalStrategy = Field(default_factory=RetrievalStrategy, description="建议的检索策略")
    
    model_config = {
        "extra": "forbid",  # 等同于设置 additionalProperties: false
    }

class IntentClassifier:
    """查询意图分类器"""
    
    def __init__(self, 
                 api_provider: str = "openai",
                 model: str = "gpt-4o-2024-08-06",
                 subset_path: Optional[Path] = None):
        """
        初始化意图分类器
        
        Args:
            api_provider: API提供者，默认为OpenAI
            model: 使用的模型，默认为gpt-4o
            subset_path: 数据集子集路径，用于提取元数据
        """
        self.api_processor = APIProcessor(provider=api_provider)
        self.model = model
        self.subset_path = subset_path
        self.docs_df = None
        
        if subset_path and os.path.exists(subset_path):
            try:
                self.docs_df = pd.read_csv(subset_path)
                _log.info(f"成功加载数据集子集，包含 {len(self.docs_df)} 条记录")
            except Exception as e:
                _log.error(f"加载数据集子集时出错: {str(e)}")
        
    def classify_intent(self, 
                        query: str, 
                        conversation_history: List[Dict] = None, 
                        selected_docs: List[str] = None,
                        has_pdf: bool = False) -> IntentClassification:
        """
        识别查询意图
        
        Args:
            query: 用户查询
            conversation_history: 对话历史记录
            selected_docs: 用户选择的文档
            has_pdf: 是否上传了PDF文件
            
        Returns:
            意图分类结果
        """
        # 提取文档引用
        extracted_docs = self._extract_document_references(query)
        
        # 准备对话历史上下文
        history_context = self._prepare_conversation_history(conversation_history)
        
        # 如果有用户选择的文档，添加到提取的文档中
        if selected_docs:
            extracted_docs.extend([doc for doc in selected_docs if doc not in extracted_docs])
        
        # 构建分类提示
        system_prompt = self._build_intent_classification_prompt()
        user_prompt = self._build_intent_classification_user_prompt(
            query=query,
            history=history_context,
            selected_docs=extracted_docs,
            has_pdf=has_pdf
        )
        
        _log.info(f"发送意图分类请求，查询: '{query}'")
        
        # 调用API进行意图分类
        try:
            classification_result = self.api_processor.send_message(
                model=self.model,
                temperature=0.1,  # 使用低温度以获得确定性结果
                system_content=system_prompt,
                human_content=user_prompt,
                is_structured=True,
                response_format=IntentClassification
            )
            
            _log.info(f"意图分类结果: {classification_result['intent']}, 置信度: {classification_result['confidence']}")
            
            # 添加提取的文档ID到实体中
            if extracted_docs and 'entities' in classification_result and 'document_references' in classification_result['entities']:
                # 确保不重复添加已有的文档引用
                current_refs = set(classification_result['entities']['document_references'])
                for doc in extracted_docs:
                    if doc not in current_refs:
                        classification_result['entities']['document_references'].append(doc)
            
            return IntentClassification(**classification_result)
            
        except Exception as e:
            _log.error(f"意图分类出错: {str(e)}")
            # 返回默认分类结果
            return IntentClassification(
                intent=IntentType.GENERAL_SEARCH,
                confidence=0.5,
                entities=QueryEntities(
                    document_references=extracted_docs
                ),
                suggested_strategy=RetrievalStrategy()
            )
    
    def _extract_document_references(self, query: str) -> List[str]:
        """
        从查询中提取文档引用（ID或标题）
        
        Args:
            query: 用户查询
            
        Returns:
            文档ID列表
        """
        doc_ids = []
        
        # 提取格式为 Paper[123456] 的文档ID
        paper_ids = re.findall(r'Paper\[(\d+)\]', query)
        if paper_ids:
            doc_ids.extend(paper_ids)
        
        # 提取格式为 "123456" 的文档ID
        numeric_ids = re.findall(r'"(\d+)"', query)
        if numeric_ids:
            doc_ids.extend(numeric_ids)
        
        # 如果有数据集，尝试匹配文档标题
        if self.docs_df is not None:
            # 1. 检查数据集中的文档ID
            if 'document_id' in self.docs_df.columns:
                document_ids = sorted(self.docs_df['document_id'].dropna().astype(str).unique(), key=len, reverse=True)
                for doc_id in document_ids:
                    if doc_id not in doc_ids and str(doc_id) in query:
                        doc_ids.append(str(doc_id))
            
            # 2. 检查查询中是否包含文档标题
            if 'title' in self.docs_df.columns:
                # 提取引号中的文本作为潜在标题
                quoted_texts = re.findall(r'"([^"]*)"', query)
                for text in quoted_texts:
                    # 尝试与标题匹配
                    for _, row in self.docs_df.iterrows():
                        title = str(row.get('title', ''))
                        # 避免匹配非常短的标题
                        if len(title) < 5:
                            continue
                        
                        # 如果文本包含标题或标题包含文本
                        if (text.lower() in title.lower() or title.lower() in text.lower()) and len(text) > 15:
                            doc_id = str(row['document_id'])
                            if doc_id not in doc_ids:
                                doc_ids.append(doc_id)
        
        return doc_ids
    
    def _prepare_conversation_history(self, history: List[Dict] = None) -> str:
        """
        准备对话历史上下文
        
        Args:
            history: 对话历史记录
            
        Returns:
            格式化的对话历史字符串
        """
        if not history or len(history) == 0:
            return "No previous conversation"
        
        # 只取最近的5轮对话
        recent_history = history[-10:]
        
        # 格式化对话历史
        formatted_history = []
        for msg in recent_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if content:
                formatted_history.append(f"{role.capitalize()}: {content[:100]}..." if len(content) > 100 else f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _build_intent_classification_prompt(self) -> str:
        """构建意图分类系统提示"""
        return """
You are an expert intent classification system for an academic research assistant.
Your task is to analyze user queries and classify them into specific intent categories.

You will receive information about:
1. The current user query
2. Recent conversation history (if any)
3. Currently selected documents (if any)
4. Whether a PDF has been uploaded

Based on this information, determine the most appropriate intent category and recommend retrieval strategies.

Be precise in your analysis and consider the context of the full conversation.

Important rules for classification:
- If the user is asking to find or search for papers/documents on a topic, classify as GENERAL_SEARCH
- If the user is asking a specific question expecting a direct answer, classify as QUESTION_ANSWERING
- If the user wants to analyze specific document(s) they've referred to, classify as DOCUMENT_ANALYSIS
- If the user wants to process an uploaded PDF, classify as PDF_PROCESSING
- If the user wants to analyze the content of the PDF in detail (summary, key points, etc.), classify as PDF_ANALYSIS
- If the user wants to find literature similar to their PDF, classify as PDF_SIMILAR_LITERATURE
- If the user wants to compare the PDF with other academic literature, classify as PDF_COMPARATIVE_ANALYSIS
- If the user's query builds on previous conversation without explicitly mentioning intent, classify as CONVERSATION_CONTINUATION

For each intent, extract relevant entities like topics, document references, and analysis types.
Also suggest an optimal retrieval strategy based on the intent.

You must follow the output schema precisely.
"""

    def _build_intent_classification_user_prompt(self, query: str, history: str, selected_docs: List[str], has_pdf: bool) -> str:
        """构建意图分类用户提示"""
        selected_docs_str = ", ".join(selected_docs) if selected_docs else "None"
        
        return f"""
Analyze the following user query and classify its intent:

User Query: "{query}"

Context:
- Recent conversation history: {history}
- Current selected documents: {selected_docs_str}
- PDF uploaded: {"Yes" if has_pdf else "No"}

Classify into one of the following intents:
1. GENERAL_SEARCH - User wants to find documents on a topic
2. QUESTION_ANSWERING - User asks a specific question expecting an answer
3. DOCUMENT_ANALYSIS - User wants analysis of specific document(s)
4. PDF_PROCESSING - User wants to process an uploaded PDF
5. PDF_ANALYSIS - User wants to analyze PDF content (summary, key points)
6. PDF_SIMILAR_LITERATURE - User wants to find literature similar to their PDF
7. PDF_COMPARATIVE_ANALYSIS - User wants to compare PDF with other literature
8. CONVERSATION_CONTINUATION - User continues previous discussion
"""

    def extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词，用于检索增强
        
        Args:
            query: 用户查询
            
        Returns:
            关键词列表
        """
        system_prompt = """
You are a keyword extraction system for academic search queries.
Your task is to extract the most important search keywords from a user query.

Extract only the most important 3-5 academic concepts, technical terms, or research topics that would be useful for searching academic papers.
- Focus on technical terms, research methodologies, specific concepts
- Ignore common words, general terms, and non-technical language
- Return only the keywords themselves without any explanation or additional text
"""

        user_prompt = f"""
Extract the most important academic search keywords from this query:

"{query}"

Return only a JSON array of keywords without explanation. Example: ["deep learning", "transformer architecture", "attention mechanism"]
"""

        try:
            response = self.api_processor.send_message(
                system_content=system_prompt,
                human_content=user_prompt,
                temperature=0.1
            )
            
            # 尝试解析JSON响应
            if response.startswith('[') and response.endswith(']'):
                try:
                    keywords = json.loads(response)
                    return keywords if isinstance(keywords, list) else []
                except json.JSONDecodeError:
                    pass
                    
            # 如果无法解析为JSON，尝试正则表达式提取
            keywords_match = re.search(r'\[(.*)\]', response)
            if keywords_match:
                keywords_str = keywords_match.group(1)
                keywords = [k.strip(' "\'') for k in keywords_str.split(',')]
                return keywords
                
            return []
            
        except Exception as e:
            _log.error(f"提取关键词时出错: {str(e)}")
            return [] 