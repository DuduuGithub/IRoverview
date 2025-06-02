import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import glob

from .intent_classifier import IntentClassifier, IntentType, IntentClassification
from .retrieval import HybridRetriever, VectorRetriever
from .questions_processing import QuestionsProcessor
from .api_requests import APIProcessor
from .pdf_parsing import PDFParser
from .text_splitter import TextSplitter
from .ingestion import VectorDBIngestor

# 设置日志格式
_log = logging.getLogger(__name__)

class IntentProcessor:
    """查询意图处理器"""
    
    def __init__(self, 
                 data_dir: Path,
                 vector_db_dir: Path,
                 documents_dir: Path,
                 upload_folder: Path,
                 subset_path: Optional[Path] = None,
                 bm25_db_dir: Optional[Path] = None,
                 api_provider: str = "openai",
                 model: str = "gpt-4o-2024-08-06"):
        """
        初始化意图处理器
        
        Args:
            data_dir: 数据目录
            vector_db_dir: 向量数据库目录
            documents_dir: 文档目录
            upload_folder: 上传目录
            subset_path: 数据集子集路径
            bm25_db_dir: BM25索引目录，如未指定则使用vector_db_dir
            api_provider: API提供者
            model: 使用的模型
        """
        self.data_dir = data_dir
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.upload_folder = upload_folder
        self.subset_path = subset_path
        self.api_provider = api_provider
        self.model = model
        self.bm25_db_dir = bm25_db_dir if bm25_db_dir is not None else vector_db_dir
        
        # 初始化意图分类器
        self.intent_classifier = IntentClassifier(
            api_provider=api_provider,
            model=model,
            subset_path=subset_path
        )
        
        # 初始化API处理器
        self.api_processor = APIProcessor(provider=api_provider)
        
        # 初始化检索器
        try:
            _log.info("初始化混合检索器")
            self.retriever = HybridRetriever(
                vector_db_dir=vector_db_dir,
                documents_dir=documents_dir,
                bm25_db_dir=self.bm25_db_dir,
                reranking_strategy="jina"  # 使用Jina作为重排序策略
            )
        except Exception as e:
            _log.error(f"初始化检索器时出错: {str(e)}")
            self.retriever = None
        
        _log.info("意图处理器初始化完成")
    
    def process_query(
        self, 
        query: str, 
        conversation_history = None, 
        selected_docs = None,
        has_pdf = False,
        pdf_path = None,
        similarity_threshold: float = 0.0,
        use_preselected_docs: bool = False
    ):
        """
        处理用户查询并返回相关答案。
        
        Args:
            query: 用户查询
            conversation_history: 对话历史
            selected_docs: 用户选择的文档
            has_pdf: 是否有已上传的PDF文件
            pdf_path: PDF文件路径
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉
            use_preselected_docs: 是否使用预先选择的文档
            
        Returns:
            处理结果
        """
        _log.info("="*50)
        _log.info(f"开始处理查询: '{query}'")
        _log.info("="*50)
        
        # 文献综述相关关键词
        review_keywords = [
            "综述", "文献综述", "literature review", "review", "survey", 
            "overview", "state of the art", "总结", "汇总", "生成一篇", "生成综述", 
            "撰写综述", "写一篇综述", "写一篇关于", "创建综述"
        ]
        
        # 对比分析相关关键词
        comparison_keywords = [
            "对比", "比较", "分析", "compare", "comparison", "analyze", "analysis",
            "similarities", "differences", "异同", "相同点", "不同点", "区别", 
            "这些", "these", "those", "这六篇", "六篇", "与", "和", "及"
        ]
        
        # 检测是否是文献综述请求
        is_literature_review = any(keyword in query.lower() for keyword in review_keywords)
        
        # 检测是否是对比分析请求
        is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
        
        # 增强文献综述检测：检查是否包含"生成"、"创建"等词与特定领域组合
        domain_keywords = ["deep learning", "machine learning", "artificial intelligence", "深度学习", "机器学习", "人工智能", "自然语言处理", "NLP", "计算机视觉", "computer vision"]
        generate_keywords = ["生成", "写", "创建", "制作", "撰写"]
        
        # 检查请求是否为生成特定领域的综述
        if any(gen_kw in query.lower() for gen_kw in generate_keywords) and any(domain in query.lower() for domain in domain_keywords):
            is_literature_review = True
            _log.info(f"[处理流程] 检测到生成特定领域综述请求: '{query}'")
        
        # 如果是文献综述请求，不论是否有预选文档，都使用预选文档处理方法
        # 为没有预选文档的情况检索相关文档
        if is_literature_review:
            _log.info(f"[处理流程] 检测到文献综述请求，使用预选文档处理方法")
            
            # 如果没有预选文档，先检索相关文档
            if not selected_docs:
                # 提取关键词
                _log.info(f"[处理流程] 没有预选文档，先检索相关文档")
                keywords = self.intent_classifier.extract_keywords(query)
                _log.info(f"[关键词提取] 提取的关键词: {keywords}")
                
                # 增强查询
                enhanced_query = " ".join(keywords) if keywords else query
                _log.info(f"[查询增强] 使用关键词增强查询: '{enhanced_query}'")
                
                # 检索相关文档
                retrieval_result = self._process_general_search(enhanced_query, None, similarity_threshold)
                
                # 从检索结果中提取文档ID
                retrieved_docs = []
                if retrieval_result and 'results' in retrieval_result:
                    results = retrieval_result['results']
                    # 按评分排序并取前10个文档
                    results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    top_results = results[:10]  # 限制为前10个文档
                    
                    # 提取文档ID并去重
                    doc_ids = set()
                    for result in top_results:
                        doc_id = result.get('document_id')
                        if doc_id and doc_id not in doc_ids:
                            doc_ids.add(doc_id)
                            retrieved_docs.append(doc_id)
                    
                    _log.info(f"[处理流程] 为综述检索到 {len(retrieved_docs)} 篇相关文档: {retrieved_docs}")
                
                # 使用检索到的文档作为预选文档
                selected_docs = retrieved_docs if retrieved_docs else selected_docs
            
            # 使用预选文档处理
            return self._process_with_preselected_docs(query, selected_docs, similarity_threshold, has_pdf, pdf_path)
        
        # 如果使用预选文档，直接进入问答流程，不进行意图分类
        if use_preselected_docs and selected_docs:
            _log.info(f"[预选文档] 使用 {len(selected_docs)} 篇预选文档处理查询")
            return self._process_with_preselected_docs(query, selected_docs, similarity_threshold, has_pdf, pdf_path)
        
        # 检测是否涉及PDF
        has_uploaded_pdf_keyword = "上传" in query or "uploaded" in query.lower() or "pdf" in query.lower()
        
        # 如果是比较请求且有PDF上传，则转而使用PDF对比分析方法
        if is_comparison and has_pdf and pdf_path and (has_uploaded_pdf_keyword or "文献" in query):
            _log.info(f"[预选文档处理] 检测到对比分析请求和上传的PDF文件，使用PDF对比分析方法")
            return self._process_pdf_comparative_analysis(query, pdf_path, selected_docs)
        
        # 分类意图
        intent_result = self.intent_classifier.classify_intent(query, conversation_history, selected_docs, has_pdf)
        
        # 直接使用对象属性，而不是当作字典处理
        intent_type = intent_result.intent  # IntentClassification对象有intent属性
        confidence = intent_result.confidence  # IntentClassification对象有confidence属性
        
        _log.info(f"[意图识别] 识别的意图: {intent_type}, 置信度: {confidence}")
        
        # 处理特定意图
        if intent_type == "GENERAL_SEARCH":
            _log.info("[处理流程] 开始处理一般检索请求")
            
            # 提取关键词
            keywords = self.intent_classifier.extract_keywords(query)
            _log.info(f"[关键词提取] 提取的关键词: {keywords}")
            
            # 增强查询
            enhanced_query = " ".join(keywords) if keywords else query
            _log.info(f"[查询增强] 使用关键词增强查询: '{enhanced_query}'")
            
            # 处理查询
            result = self._process_general_search(enhanced_query, selected_docs, similarity_threshold)
            
            _log.info(f"[检索结果] 总计找到 {len(result.get('results', []))} 条结果")
            if result.get('results'):
                top_3 = result.get('results')[:3]
                _log.info(f"[检索结果] 前3条结果摘要:")
                for i, item in enumerate(top_3):
                    _log.info(f"  [{i+1}] 标题: {item.get('title', 'N/A')}")
                    _log.info(f"      分数: {item.get('score', 0):.4f}")
                    _log.info(f"      页码: {item.get('page', 'N/A')}")
                    text_preview = item.get('text', '')[:100] + '...' if len(item.get('text', '')) > 100 else item.get('text', '')
                    _log.info(f"      内容: {text_preview}")
            
            return result
        
        elif intent_type == "QUESTION_ANSWERING":
            _log.info("[处理流程] 开始处理问答请求")
            return self._process_question_answering(query, selected_docs, similarity_threshold)
        
        elif intent_type == "DOCUMENT_ANALYSIS":
            _log.info("[处理流程] 开始处理文档分析请求")
            return self._process_document_analysis(query, selected_docs, similarity_threshold)
        
        elif intent_type == "PDF_PROCESSING" and has_pdf and pdf_path:
            _log.info(f"[处理流程] 开始处理PDF请求，文件路径: {pdf_path}")
            return self._process_pdf_request(query, pdf_path)
        
        elif intent_type == "PDF_ANALYSIS" and has_pdf and pdf_path:
            _log.info(f"[处理流程] 开始处理PDF分析请求，文件路径: {pdf_path}")
            return self._process_pdf_analysis(query, pdf_path)
            
        elif intent_type == "PDF_SIMILAR_LITERATURE" and has_pdf and pdf_path:
            _log.info(f"[处理流程] 开始处理查找类似文献请求，文件路径: {pdf_path}")
            return self._process_pdf_similar_literature(query, pdf_path, similarity_threshold)
            
        elif intent_type == "PDF_COMPARATIVE_ANALYSIS" and has_pdf and pdf_path:
            _log.info(f"[处理流程] 开始处理PDF对比分析请求，文件路径: {pdf_path}")
            return self._process_pdf_comparative_analysis(query, pdf_path, selected_docs)
            
        elif intent_type == "CONVERSATION_CONTINUATION":
            _log.info("[处理流程] 开始处理对话延续请求")
            return self._process_conversation_continuation(query, conversation_history)
            
        else:
            # 默认回退到一般检索
            _log.info(f"[处理流程] 意图 '{intent_type}' 不适用于当前上下文或置信度不足，回退到一般检索")
            return self._process_general_search(query, selected_docs, similarity_threshold)
    
    def _process_general_search(
        self, 
        query: str, 
        selected_docs = None, 
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        处理一般搜索查询
        
        Args:
            query: 用户查询文本
            selected_docs: 用户指定的文档列表
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉
            
        Returns:
            处理结果字典
        """
        _log.info(f"[一般检索] 开始处理查询: '{query}'")
        _log.info(f"[一般检索] 使用相似度阈值: {similarity_threshold}")
        
        if selected_docs:
            _log.info(f"[一般检索] 使用用户指定的文档: {selected_docs}")
            # 按文档ID检索
            selected_doc_ids = selected_docs
            
            # 初始化检索结果
            all_results = []
            
            # 对每个指定的文档执行检索
            for doc_id in selected_doc_ids:
                _log.info(f"[一般检索] 从文档ID '{doc_id}' 检索内容")
                
                try:
                    # 检索并将结果合并
                    doc_results = self.retriever.retrieve_by_document_id(
                        document_id=doc_id, 
                        query=query, 
                        top_n=50,
                        similarity_threshold=similarity_threshold
                    )
                    
                    _log.info(f"[一般检索] 从文档ID '{doc_id}' 找到 {len(doc_results)} 条结果")
                    all_results.extend(doc_results)
                except Exception as e:
                    _log.error(f"从文档ID '{doc_id}' 检索时出错: {str(e)}", exc_info=True)
            
            # 按相关性分数排序
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # 截取前50个结果
            results = all_results[:50]
            _log.info(f"[一般检索] 总共找到并合并了 {len(results)} 条结果")
            
        else:
            _log.info("[一般检索] 没有指定文档，执行全文检索")
            # 执行全文检索
            results = self.retriever.retrieve_by_query(
                query=query, 
                top_n=50,
                similarity_threshold=similarity_threshold
            )
            _log.info(f"[一般检索] 全文检索找到 {len(results)} 条结果")
        
        # 补全检索结果元数据
        for i, result in enumerate(results):
            # 确保所有结果都有基本元数据，即使字段为空
            if 'document_id' not in result or not result['document_id']:
                result['document_id'] = f"doc_{i+1}"
            if 'title' not in result or not result['title']:
                # 尝试从文本中提取标题
                text = result.get('text', '')
                if text.startswith('Title:'):
                    title_end = text.find('\n')
                    if title_end > 0:
                        result['title'] = text[6:title_end].strip()
                    else:
                        result['title'] = text[6:100].strip()
                else:
                    result['title'] = f"Document {i+1}"
            if 'authors' not in result or not result['authors']:
                result['authors'] = ["Unknown Author"]
            if 'year' not in result or not result['year']:
                result['year'] = "Unknown Year"
            if 'source' not in result or not result['source']:
                result['source'] = "Unknown Source"
            if 'score' not in result:
                result['score'] = 0.5  # 默认分数
        
        # 打印检索结果详情
        if results:
            _log.info("[检索结果详情] ===========================")
            for i, result in enumerate(results[:5]):
                _log.info(f"结果 #{i+1}:")
                _log.info(f"  文档ID: {result.get('document_id', 'N/A')}")
                _log.info(f"  标题: {result.get('title', 'N/A')}")
                _log.info(f"  年份: {result.get('year', 'N/A')}")
                _log.info(f"  作者: {', '.join(result.get('authors', []))}")
                _log.info(f"  相似度分数: {result.get('score', 0):.4f}")
                
                # 打印文本摘要
                text = result.get('text', '')
                text_preview = f"{text[:150]}..." if len(text) > 150 else text
                _log.info(f"  内容摘要: {text_preview}")
                _log.info("  " + "-"*40)
                
            _log.info("[检索结果详情] ===========================")
        
        # 构建返回结果
        response = {
            "query": query,
            "result_type": "search",
            "results": results,
            "total_results": len(results),
            "answer": None,
            "references": results
        }
        
        return response
    
    def _process_question_answering(self, question: str, selected_docs = None, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        处理问答请求
        
        Args:
            question: 用户问题
            selected_docs: 用户选择的文档列表
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉
            
        Returns:
            回答结果
        """
        _log.info(f"[问答处理] 开始处理问答请求: '{question}'")
        
        try:
            # 设置检索策略参数
            return_parent_pages = False
            reranking_approach = "jina"  # 默认使用jina重排序
            top_n = 50
            llm_reranking_sample_size = 100
            
            # 初始化问题处理器
            processor = QuestionsProcessor(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                parent_document_retrieval=return_parent_pages,
                llm_reranking=(reranking_approach == "llm"),
                llm_reranking_sample_size=llm_reranking_sample_size,
                top_n_retrieval=top_n,
                api_provider=self.api_provider,
                answering_model=self.model,
                subset_path=self.subset_path
            )
            
            # 如果有指定的文档引用
            if selected_docs and len(selected_docs) > 0:
                _log.info(f"[问答处理] 使用指定文档处理问题: {selected_docs}")
                
                if len(selected_docs) == 1:
                    # 单文档问答
                    result = processor.get_answer_for_document(
                        document_id=selected_docs[0],
                        question=question,
                        schema="free_response"
                    )
                else:
                    # 多文档比较问答
                    _log.info(f"[问答处理] 执行多文档比较问答: {selected_docs}")
                    result = processor.process_comparative_question(
                        question=question,
                        documents=selected_docs,
                        schema="free_response"
                    )
            else:
                # 全局检索问答
                _log.info("[问答处理] 使用全局检索处理问题")
                result = processor.get_answer_without_document(
                    question=question,
                    schema="free_response"
                )
                
            # 处理引用结果
            if 'references' in result and result['references']:
                processed_refs = result['references']
            else:
                # 如果没有返回引用，添加空引用列表
                processed_refs = []
            
            _log.info(f"[问答处理] 生成回答: '{result.get('final_answer', '')[:100]}...'")
            
            # 检查回答内容
            final_answer = result.get('final_answer', '')
            if "not contain sufficient information" in final_answer or "not enough information" in final_answer:
                _log.warning("[问答处理] 检测到'没有足够信息'的回答，尝试直接回答")
                # 调用API生成更通用的回答
                try:
                    system_prompt = """You are a helpful AI research assistant. 
When asked about topics, provide a general introduction and overview even if specific papers weren't found.
Use your general knowledge to give a helpful response."""
                    
                    user_prompt = f"The user asked: {question}\n\nPlease provide a general introduction to this topic."
                    
                    # 调用API获取更通用的回答
                    api_processor = APIProcessor(provider=self.api_provider)
                    response = api_processor.send_message(
                        model=self.model,
                        system_content=system_prompt,
                        human_content=user_prompt
                    )
                    
                    # 更新回答
                    final_answer = response
                    _log.info(f"[问答处理] 生成通用回答: '{final_answer[:100]}...'")
                except Exception as e:
                    _log.error(f"[问答处理] 生成通用回答时出错: {str(e)}")
            
            return {
                'answer': final_answer if final_answer else result.get('final_answer', '无法生成回答'),
                'references': processed_refs,
                'reasoning_summary': result.get('reasoning_summary', ''),
                'status': 'success',
                'result_type': 'answer',
                'query': question
            }
            
        except Exception as e:
            _log.error(f"处理问答请求时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"处理问题时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'answer',
                'query': question
            }
    
    def _process_document_analysis(self, query: str, selected_docs = None, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        处理文档分析请求
        
        Args:
            query: 用户查询
            selected_docs: 用户选择的文档列表
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉
            
        Returns:
            分析结果
        """
        _log.info(f"[文档分析] 开始处理文档分析请求: '{query}'")
        
        try:
            # 设置检索策略参数
            return_parent_pages = False
            reranking_approach = "jina"  # 默认使用jina重排序
            top_n = 50
            llm_reranking_sample_size = 100
            
            # 检查是否有指定文档
            if not selected_docs or len(selected_docs) == 0:
                _log.warning("[文档分析] 文档分析请求中未找到文档引用，回退到问答处理")
                # 回退到问答处理
                return self._process_question_answering(query, selected_docs, similarity_threshold)
                
            _log.info(f"[文档分析] 处理文档: {selected_docs}")
            
            # 分析类型（从查询中推断）- 增强类型检测
            analysis_type = "summarize"  # 默认为总结
            
            # 文献综述相关关键词
            review_keywords = [
                "综述", "文献综述", "literature review", "review", "survey", 
                "overview", "state of the art", "总结", "汇总"
            ]
            
            # 比较相关关键词
            compare_keywords = [
                "compare", "comparison", "differences", "similarities", "versus", "vs", 
                "contrast", "比较", "对比", "区别", "不同"
            ]
            
            # 总结相关关键词
            summary_keywords = [
                "summarize", "summary", "概括", "摘要", "总结"
            ]
            
            # 检测查询类型
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in review_keywords):
                analysis_type = "review"  # 文献综述
                _log.info("[文档分析] 检测到文献综述请求")
            elif any(keyword in query_lower for keyword in compare_keywords):
                analysis_type = "compare"  # 比较分析
            elif any(keyword in query_lower for keyword in summary_keywords):
                analysis_type = "summarize"  # 文本摘要
                
            _log.info(f"[文档分析] 分析类型: {analysis_type}")
            
            # 如果是文献综述请求，使用预选文档专用处理逻辑
            if analysis_type == "review" and len(selected_docs) > 0:
                _log.info(f"[文档分析] 使用预选文档专用逻辑处理文献综述请求")
                return self._process_with_preselected_docs(query, selected_docs, similarity_threshold, False, None)
            
            # 对于其他类型分析，使用标准处理逻辑
            # 初始化问题处理器
            processor = QuestionsProcessor(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                parent_document_retrieval=return_parent_pages,
                llm_reranking=(reranking_approach == "llm"),
                llm_reranking_sample_size=llm_reranking_sample_size,
                top_n_retrieval=top_n,
                api_provider=self.api_provider,
                answering_model=self.model,
                subset_path=self.subset_path
            )
            
            # 执行文档分析
            if len(selected_docs) == 1:
                # 单文档分析
                _log.info(f"[文档分析] 执行单文档分析: {selected_docs[0]}")
                result = processor.get_answer_for_document(
                    document_id=selected_docs[0],
                    question=f"{analysis_type} this document: {query}",
                    schema="free_response"
                )
            else:
                # 多文档比较分析
                _log.info(f"[文档分析] 执行多文档比较分析: {selected_docs}")
                
                # 根据分析类型调整查询
                if analysis_type.lower() == "compare":
                    enhanced_query = f"Compare these documents: {query}"
                elif analysis_type.lower() == "summarize":
                    enhanced_query = f"Summarize these documents: {query}"
                else:
                    enhanced_query = query
                    
                result = processor.process_comparative_question(
                    question=enhanced_query,
                    documents=selected_docs,
                    schema="free_response"
                )
            
            _log.info(f"[文档分析] 生成分析结果: '{result.get('final_answer', '')[:100]}...'")
                
            return {
                'answer': result.get('final_answer', '无法生成分析'),
                'references': result.get('references', []),
                'reasoning_summary': result.get('reasoning_summary', ''),
                'status': 'success',
                'result_type': 'analysis',
                'analysis_type': analysis_type,
                'document_ids': selected_docs,
                'query': query
            }
            
        except Exception as e:
            _log.error(f"[文档分析] 处理文档分析请求时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"分析文档时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'analysis',
                'document_ids': selected_docs if selected_docs else [],
                'query': query
            }
    
    def _process_pdf_request(self, query: str, pdf_path: str) -> Dict[str, Any]:
        """
        处理PDF处理请求
        
        Args:
            query: 用户查询
            pdf_path: PDF文件路径
            
        Returns:
            处理结果
        """
        _log.info(f"[PDF处理] 处理PDF请求: {pdf_path}")
        
        try:
            # 1. 解析PDF
            output_dir = self.data_dir / 'parsed_reports'
            output_dir.mkdir(exist_ok=True)
            
            pdf_parser = PDFParser(output_dir=output_dir)
            parsed_result = pdf_parser.parse_and_export(input_doc_paths=[pdf_path])
            
            _log.info("[PDF处理] PDF解析完成")
            
            # 2. 分割文本
            text_splitter = TextSplitter()
            chunks_dir = self.documents_dir
            text_splitter.split_all_reports(output_dir, chunks_dir)
            
            _log.info("[PDF处理] 文本分割完成")
            
            # 3. 向量化
            ingestor = VectorDBIngestor()
            ingestor.process_reports(chunks_dir, self.vector_db_dir)
            
            _log.info("[PDF处理] 向量化完成")
            
            return {
                'answer': "PDF文件已成功处理。现在您可以询问关于PDF内容的问题，查找相似文献，或进行对比分析。",
                'references': [],
                'reasoning_summary': '',
                'status': 'success',
                'result_type': 'pdf_processing',
                'query': query
            }
                
        except Exception as e:
            _log.error(f"[PDF处理] 处理PDF请求时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"处理PDF时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'pdf_analysis',
                'query': query
            }
    
    def _process_pdf_analysis(self, query: str, pdf_path: str) -> Dict[str, Any]:
        """
        处理PDF分析请求，详细分析PDF内容
        
        Args:
            query: 用户查询
            pdf_path: PDF文件路径
            
        Returns:
            处理结果
        """
        _log.info(f"[PDF分析] 分析PDF内容: {pdf_path}")
        
        try:
            # 读取已解析的PDF数据
            output_dir = self.data_dir / 'parsed_reports'
            pdf_filename = os.path.basename(pdf_path)
            pdf_id = os.path.splitext(pdf_filename)[0]
            
            parsed_file_path = output_dir / f"{pdf_id}.json"
            if not os.path.exists(parsed_file_path):
                # PDF尚未解析，先进行解析
                pdf_parser = PDFParser(output_dir=output_dir)
                # 将字符串路径转换为Path对象
                from pathlib import Path
                pdf_path_obj = Path(pdf_path)
                _log.info(f"[PDF分析] PDF尚未解析，开始解析: {pdf_path}")
                parse_result = pdf_parser.parse_and_export(input_doc_paths=[pdf_path_obj])
                _log.info(f"[PDF分析] PDF解析完成，成功: {parse_result[0]}, 失败: {parse_result[1]}")
                
                # 获取PDF文件名和预期的JSON文件路径
                pdf_filename = os.path.basename(pdf_path)
                
                # 在output_dir中查找最新生成的JSON文件
                json_files = glob.glob(os.path.join(output_dir, "*.json"))
                if not json_files:
                    _log.error("[PDF分析] 没有找到解析后的JSON文件")
                    return {
                        'answer': "解析PDF文件失败，无法找到相似文献。",
                        'references': [],
                        'reasoning_summary': '',
                        'status': 'error',
                        'result_type': 'pdf_analysis'
                    }
                
                # 按创建时间排序，获取最新的文件
                json_files.sort(key=os.path.getctime, reverse=True)
                latest_json = json_files[0]
                _log.info(f"[PDF分析] 使用解析后的JSON文件: {latest_json}")
                
                # 加载JSON文件
                with open(latest_json, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                
                # 提取PDF关键内容作为查询向量
                pdf_title = parsed_data.get("metainfo", {}).get("title", "")
                # 如果没有标题，尝试从文件名提取
                if not pdf_title:
                    pdf_title = os.path.splitext(pdf_filename)[0].replace("_", " ")
                
                pdf_abstract = ""
            else:
                # 加载已解析的PDF数据
                with open(parsed_file_path, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                _log.info(f"[PDF分析] 已加载解析的PDF数据: {parsed_file_path}")
            
                # 提取PDF关键内容作为查询向量
                pdf_title = parsed_data.get("metainfo", {}).get("title", "")
                pdf_abstract = ""
            
            _log.info(f"[PDF分析] 提取的PDF标题: '{pdf_title}'")
            
            # 尝试提取摘要（通常在前2页）
            for page in parsed_data.get("content", {}).get("pages", [])[:2]:
                pdf_abstract += page.get("text", "") + " "
            
            # 限制内容长度
            if len(pdf_abstract) > 3000:
                pdf_abstract = pdf_abstract[:3000]
            
            # 使用LLM分析PDF内容
            system_prompt = """
You are an academic research assistant specialized in analyzing scientific papers and documents.
Your task is to provide a detailed analysis of the PDF document content provided.
Based on the specific query, focus on:
1. Key findings and contributions
2. Methodology used
3. Results and their implications
4. Theoretical framework
5. Gaps and limitations identified
6. Future research directions

Organize your analysis in a clear, structured format with appropriate headings and sections.
Cite specific parts of the document to support your analysis.
If the query asks for specific information not contained in the document, clearly state that limitation.
"""

            user_prompt = f"""
PDF content:
```
{pdf_abstract}
```

User query: {query}

Provide a comprehensive, well-structured analysis of this academic document addressing the user's query.
"""
            
            try:
                response = self.api_processor.send_message(
                    model=self.model,
                    system_content=system_prompt,
                    human_content=user_prompt
                )
                
                _log.info(f"[PDF分析] 生成分析结果: '{response[:100]}...'")
                
                return {
                    'answer': response,
                    'references': [{"source": "Uploaded PDF", "text": pdf_abstract[:300] + "..."}],
                    'reasoning_summary': '',
                    'status': 'success',
                    'result_type': 'pdf_analysis',
                    'query': query
                }
            except Exception as e:
                _log.error(f"[PDF分析] 分析PDF内容时出错: {str(e)}")
                return {
                    'answer': f"分析PDF内容时出错: {str(e)}",
                    'references': [],
                    'reasoning_summary': '',
                    'status': 'error',
                    'result_type': 'pdf_analysis',
                    'query': query
                }
                
        except Exception as e:
            _log.error(f"[PDF分析] 处理PDF分析请求时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"分析PDF时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'pdf_analysis',
                'query': query
            }
    
    def _process_pdf_similar_literature(self, query: str, pdf_path: str, similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        处理查找与PDF类似的文献的请求
        
        Args:
            query: 用户查询
            pdf_path: PDF文件路径
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉，默认为0.3
            
        Returns:
            处理结果
        """
        _log.info(f"[PDF相似文献] 查找与PDF类似的文献: {pdf_path}")
        _log.info(f"[PDF相似文献] 使用相似度阈值: {similarity_threshold}")
        
        try:
            # 读取已解析的PDF数据
            output_dir = self.data_dir / 'parsed_reports'
            pdf_filename = os.path.basename(pdf_path)
            pdf_id = os.path.splitext(pdf_filename)[0]
            
            parsed_file_path = output_dir / f"{pdf_id}.json"
            if not os.path.exists(parsed_file_path):
                # PDF尚未解析，先进行解析
                pdf_parser = PDFParser(output_dir=output_dir)
                # 将字符串路径转换为Path对象
                from pathlib import Path
                pdf_path_obj = Path(pdf_path)
                _log.info(f"[PDF相似文献] PDF尚未解析，开始解析: {pdf_path}")
                parse_result = pdf_parser.parse_and_export(input_doc_paths=[pdf_path_obj])
                _log.info(f"[PDF相似文献] PDF解析完成，成功: {parse_result[0]}, 失败: {parse_result[1]}")
                
                # 获取PDF文件名和预期的JSON文件路径
                pdf_filename = os.path.basename(pdf_path)
                
                # 在output_dir中查找最新生成的JSON文件
                json_files = glob.glob(os.path.join(output_dir, "*.json"))
                if not json_files:
                    _log.error("[PDF相似文献] 没有找到解析后的JSON文件")
                    return {
                        'answer': "解析PDF文件失败，无法找到相似文献。",
                        'references': [],
                        'reasoning_summary': '',
                        'status': 'error',
                        'result_type': 'pdf_similar_literature'
                    }
                
                # 按创建时间排序，获取最新的文件
                json_files.sort(key=os.path.getctime, reverse=True)
                latest_json = json_files[0]
                _log.info(f"[PDF相似文献] 使用解析后的JSON文件: {latest_json}")
                
                # 加载JSON文件
                with open(latest_json, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                
                # 提取PDF关键内容作为查询向量
                pdf_title = parsed_data.get("metainfo", {}).get("title", "")
                # 如果没有标题，尝试从文件名提取
                if not pdf_title:
                    pdf_title = os.path.splitext(pdf_filename)[0].replace("_", " ")
                
                pdf_abstract = ""
            else:
                # 加载已解析的PDF数据
                with open(parsed_file_path, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                _log.info(f"[PDF相似文献] 已加载解析的PDF数据: {parsed_file_path}")
            
                # 提取PDF关键内容作为查询向量
                pdf_title = parsed_data.get("metainfo", {}).get("title", "")
                pdf_abstract = ""
            
            _log.info(f"[PDF相似文献] 提取的PDF标题: '{pdf_title}'")
            
            # 尝试提取摘要（通常在前2页）
            for page in parsed_data.get("content", {}).get("pages", [])[:2]:
                pdf_abstract += page.get("text", "") + " "
            
            # 限制内容长度
            if len(pdf_abstract) > 3000:
                pdf_abstract = pdf_abstract[:3000]
            
            _log.info(f"[PDF相似文献] 提取的PDF摘要长度: {len(pdf_abstract)}字符，前100字符: '{pdf_abstract[:100]}...'")
            
            # 使用LLM提取关键概念和主题
            system_prompt = """
You are a research assistant specialized in extracting key academic concepts and research topics from scientific papers.
Your task is to analyze the title and abstract of a paper and extract:
1. The main research topic
2. Key technical concepts
3. Methodologies used
4. Subject domain
5. Research questions addressed

Generate a concise summary of these elements that can be used as a search query to find similar papers.
"""

            user_prompt = f"""
Paper Title: {pdf_title}

Paper Abstract:
{pdf_abstract}

Extract the key research concepts, methodologies, and subject domain from this paper.
Format your response as a concise search query (about 3-5 sentences) that would help find similar academic papers.
"""
            
            try:
                # 获取增强查询
                _log.info("[PDF相似文献] 开始生成搜索查询...")
                search_query = self.api_processor.send_message(
                    model=self.model,
                    system_content=system_prompt,
                    human_content=user_prompt,
                    temperature=0.3
                )
                
                _log.info(f"[PDF相似文献] 生成的搜索查询: '{search_query}'")
                
                # 使用增强查询进行检索，使用传入的相似度阈值
                _log.info(f"[PDF相似文献] 开始检索相似文献，使用阈值{similarity_threshold}...")
                search_results = self._process_general_search(search_query, None, similarity_threshold)
                
                # 记录搜索结果数量
                result_count = len(search_results.get('results', [])) if search_results and 'results' in search_results else 0
                _log.info(f"[PDF相似文献] 检索完成，找到 {result_count} 条结果")
                
                if result_count > 0:
                    # 记录前几条结果的相似度分数
                    top_results = search_results.get('results', [])[:5]  # 只记录前5条
                    for i, res in enumerate(top_results):
                        _log.info(f"[PDF相似文献] 结果 #{i+1}: 标题='{res.get('title', 'N/A')}', 分数={res.get('score', 0):.6f}")
                else:
                    _log.info(f"[PDF相似文献] 未找到任何相似文献，可能相似度阈值({similarity_threshold})过高")
                
                # 追加解释到结果中
                if search_results and 'results' in search_results and search_results['results']:
                    explanation = f"基于您上传的PDF文件，我识别了以下关键主题和概念：\n\n{search_query}\n\n以下是与您上传的PDF内容相似的文献："
                    # 确保search_results中的answer字段不为None
                    if 'answer' not in search_results or search_results['answer'] is None:
                        search_results['answer'] = ''
                    search_results['answer'] = explanation + "\n\n" + search_results['answer']
                    search_results['result_type'] = 'pdf_similar_literature'
                
                return search_results
                
            except Exception as e:
                _log.error(f"[PDF相似文献] 查找相似文献时出错: {str(e)}")
                return {
                    'answer': f"查找相似文献时出错: {str(e)}",
                    'references': [],
                    'reasoning_summary': '',
                    'status': 'error',
                    'result_type': 'pdf_similar_literature',
                    'query': query
                }
                
        except Exception as e:
            _log.error(f"[PDF相似文献] 处理查找相似文献请求时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"查找相似文献时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'pdf_similar_literature',
                'query': query
            }
    
    def _process_pdf_comparative_analysis(self, query: str, pdf_path: str, selected_docs: List[str] = None) -> Dict[str, Any]:
        """
        处理PDF与其他文献的比较分析请求
        
        Args:
            query: 用户查询
            pdf_path: PDF文件路径
            selected_docs: 用于对比的选定文档
            
        Returns:
            处理结果
        """
        _log.info(f"[PDF对比分析] 进行PDF对比分析: {pdf_path}")
        
        try:
            # 读取已解析的PDF数据
            output_dir = self.data_dir / 'parsed_reports'
            pdf_filename = os.path.basename(pdf_path)
            pdf_id = os.path.splitext(pdf_filename)[0]
            
            parsed_file_path = output_dir / f"{pdf_id}.json"
            if not os.path.exists(parsed_file_path):
                # PDF尚未解析，先进行解析
                pdf_parser = PDFParser(output_dir=output_dir)
                # 将字符串路径转换为Path对象
                from pathlib import Path
                pdf_path_obj = Path(pdf_path)
                parse_result = pdf_parser.parse_and_export(input_doc_paths=[pdf_path_obj])
                _log.info(f"[PDF对比分析] PDF解析完成，成功: {parse_result[0]}, 失败: {parse_result[1]}")
                
                # 获取PDF文件名和预期的JSON文件路径
                pdf_filename = os.path.basename(pdf_path)
                
                # 在output_dir中查找最新生成的JSON文件
                json_files = glob.glob(os.path.join(output_dir, "*.json"))
                if not json_files:
                    _log.error("[PDF对比分析] 没有找到解析后的JSON文件")
                    return {
                        'answer': "解析PDF文件失败，无法进行对比分析。",
                        'references': [],
                        'reasoning_summary': '',
                        'status': 'error',
                        'result_type': 'pdf_comparative_analysis'
                    }
                
                # 按创建时间排序，获取最新的文件
                json_files.sort(key=os.path.getctime, reverse=True)
                latest_json = json_files[0]
                _log.info(f"[PDF对比分析] 使用解析后的JSON文件: {latest_json}")
                
                # 加载JSON文件
                with open(latest_json, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                
                # 提取PDF关键内容作为查询向量
                pdf_title = parsed_data.get("metainfo", {}).get("title", "")
                # 如果没有标题，尝试从文件名提取
                if not pdf_title:
                    pdf_title = os.path.splitext(pdf_filename)[0].replace("_", " ")
                
                pdf_content = ""
            else:
                # 加载已解析的PDF数据
                with open(parsed_file_path, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                _log.info(f"[PDF对比分析] 已加载解析的PDF数据: {parsed_file_path}")
            
                # 提取PDF关键内容
                pdf_title = parsed_data.get("metainfo", {}).get("title", "")
                pdf_content = ""
            
            _log.info(f"[PDF对比分析] 提取的PDF标题: '{pdf_title}'")
            
            # 提取PDF内容（取前几页）
            for page in parsed_data.get("content", {}).get("pages", [])[:5]:
                pdf_content += page.get("text", "") + " "
            
            # 限制长度
            if len(pdf_content) > 5000:
                pdf_content = pdf_content[:5000]
            
            comparison_docs = []
            
            # 如果用户选择了特定文档进行比较
            if selected_docs and len(selected_docs) > 0:
                _log.info(f"[PDF对比分析] 使用用户选择的文档进行对比: {selected_docs}")
                
                # 检索每个选定文档的内容
                for doc_id in selected_docs:
                    try:
                        doc_results = self.retriever.retrieve_by_document_id(
                            document_id=doc_id,
                            query=pdf_title + " " + pdf_content[:200],  # 添加缺少的query参数
                            top_n=10
                        )
                        
                        if doc_results:
                            # 提取文档标题
                            doc_title = doc_results[0].get('title', f"文档 {doc_id}")
                            
                            # 合并文档内容
                            doc_content = "\n".join([result.get('text', '') for result in doc_results])
                            
                            # 限制长度
                            if len(doc_content) > 5000:
                                doc_content = doc_content[:5000]
                                
                            comparison_docs.append({
                                'title': doc_title,
                                'content': doc_content,
                                'document_id': doc_id
                            })
                    except Exception as e:
                        _log.error(f"[PDF对比分析] 检索文档 {doc_id} 时出错: {str(e)}")
            
            # 如果没有选定的文档，或者选定的文档获取失败，则查找相似文档
            if not comparison_docs:
                _log.info("[PDF对比分析] 没有选定的文档用于对比，尝试查找相似文档")
                
                # 使用提取的PDF内容查找相似文档
                similar_results = self.retriever.retrieve_by_query(
                    query=pdf_content[:1000],  # 使用PDF内容的前部分作为查询
                    top_n=5
                )
                
                # 将相似文档添加到对比列表
                for result in similar_results:
                    doc_id = result.get('document_id', '')
                    if doc_id:
                        # 检索完整文档内容
                        try:
                            doc_results = self.retriever.retrieve_by_document_id(
                                document_id=doc_id,
                                query=pdf_content[:300],  # 添加缺少的query参数
                                top_n=10
                            )
                            
                            if doc_results:
                                # 提取文档标题
                                doc_title = doc_results[0].get('title', f"文档 {doc_id}")
                                
                                # 合并文档内容
                                doc_content = "\n".join([r.get('text', '') for r in doc_results])
                                
                                # 限制长度
                                if len(doc_content) > 5000:
                                    doc_content = doc_content[:5000]
                                    
                                # 避免重复添加
                                if not any(d['document_id'] == doc_id for d in comparison_docs):
                                    comparison_docs.append({
                                        'title': doc_title,
                                        'content': doc_content,
                                        'document_id': doc_id
                                    })
                        except Exception as e:
                            _log.error(f"[PDF对比分析] 检索文档 {doc_id} 时出错: {str(e)}")
            
            # 限制对比文档数量
            comparison_docs = comparison_docs[:3]  # 最多比较3篇文献
            
            if not comparison_docs:
                return {
                    'answer': "未能找到合适的文档进行比较。请尝试选择特定文档进行对比分析。",
                    'references': [],
                    'reasoning_summary': '',
                    'status': 'error',
                    'result_type': 'pdf_comparative_analysis',
                    'query': query
                }
            
            # 构建分析提示
            system_prompt = """
You are an academic research assistant specialized in comparative analysis of scientific literature.
Your task is to compare the uploaded PDF with other academic documents and identify:
1. Key similarities and differences in research approaches
2. Complementary findings and how they relate to each other
3. Contradictions or disagreements among the documents
4. Methodological differences and their implications
5. The progression of ideas or concepts across the documents

Provide a structured, comprehensive comparative analysis with clear sections and examples from each document.
Be specific and cite content from the documents to support your analysis.
Conclude with a synthesis that highlights the collective insights from all documents.
"""

            # 构建用户提示
            user_prompt = f"""
Uploaded PDF: {pdf_title}

PDF Content:
```
{pdf_content}
```

Comparison Documents:
"""
            
            # 添加比较文档
            for i, doc in enumerate(comparison_docs):
                user_prompt += f"""
Document {i+1}: {doc['title']} (ID: {doc['document_id']})
```
{doc['content']}
```
"""
            
            user_prompt += f"""
User Query: {query}

Please provide a detailed comparative analysis of the uploaded PDF in relation to the comparison documents, focusing on the user's specific query.
"""
            
            try:
                # 生成对比分析
                response = self.api_processor.send_message(
                    model=self.model,
                    system_content=system_prompt,
                    human_content=user_prompt
                )
                
                _log.info(f"[PDF对比分析] 生成对比分析结果: '{response[:100]}...'")
                
                # 准备引用
                references = [{"source": "Uploaded PDF", "text": pdf_content[:300] + "..."}]
                
                for doc in comparison_docs:
                    references.append({
                        "source": f"{doc['title']} (ID: {doc['document_id']})",
                        "text": doc['content'][:300] + "..."
                    })
                
                return {
                    'answer': response,
                    'references': references,
                    'reasoning_summary': '',
                    'status': 'success',
                    'result_type': 'pdf_comparative_analysis',
                    'query': query
                }
                
            except Exception as e:
                _log.error(f"[PDF对比分析] 生成对比分析时出错: {str(e)}")
                return {
                    'answer': f"生成对比分析时出错: {str(e)}",
                    'references': [],
                    'reasoning_summary': '',
                    'status': 'error',
                    'result_type': 'pdf_comparative_analysis',
                    'query': query
                }
                
        except Exception as e:
            _log.error(f"[PDF对比分析] 处理PDF对比分析请求时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"进行PDF对比分析时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'pdf_comparative_analysis',
                'query': query
            }
    
    def _process_retrieval_results(self, results: List[Dict]) -> List[Dict]:
        """
        处理检索结果
        
        Args:
            results: 检索结果列表
            
        Returns:
            处理后的结果列表
        """
        processed_results = []
        
        for item in results:
            processed_item = {
                'document_id': item.get('document_id', ''),
                'score': item.get('score', 0),
                'text': item.get('text', '')[:300],  # 限制文本长度
                'page': item.get('page', 0)
            }
            
            # 尝试添加更多元数据
            if self.subset_path and os.path.exists(self.subset_path):
                try:
                    import pandas as pd
                    docs_df = pd.read_csv(self.subset_path)
                    doc_id = processed_item['document_id']
                    
                    if 'document_id' in docs_df.columns:
                        doc_info = docs_df[docs_df['document_id'].astype(str) == str(doc_id)]
                        if not doc_info.empty:
                            doc_row = doc_info.iloc[0]
                            # 添加标题和作者信息
                            if 'title' in doc_row:
                                processed_item['title'] = doc_row['title']
                            if 'authors' in doc_row:
                                processed_item['authors'] = doc_row['authors']
                            if 'year' in doc_row:
                                processed_item['year'] = doc_row['year']
                            if 'venue' in doc_row:
                                processed_item['venue'] = doc_row['venue']
                except Exception as e:
                    _log.warning(f"添加元数据时出错: {str(e)}")
            
            processed_results.append(processed_item)
        
        return processed_results
    
    def _process_conversation_continuation(self, query: str, conversation_history = None) -> Dict[str, Any]:
        """
        处理对话延续请求，直接使用LLM进行回复而不进行检索
        
        Args:
            query: 用户查询
            conversation_history: 对话历史
            
        Returns:
            对话回复结果
        """
        _log.info("[对话延续] 使用对话历史直接回复，无需检索")
        
        try:
            # 构建对话历史上下文
            formatted_history = []
            
            # 仅使用最近的10轮对话
            if conversation_history:
                recent_history = conversation_history[-10:]
                for msg in recent_history:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role and content:
                        formatted_history.append({
                            "role": "user" if role == "user" else "assistant",
                            "content": content
                        })
            
            # 添加当前问题
            formatted_history.append({
                "role": "user",
                "content": query
            })
            
            # 系统提示
            system_prompt = """You are a helpful AI research assistant with expertise in academic literature and scientific topics.
Please respond directly to the user's query based on our conversation history.
Be thorough yet concise in your responses.
If you're unsure about something, acknowledge the limitations.
"""
            
            _log.info(f"[对话延续] 发送LLM请求，历史消息数量: {len(formatted_history)}")
            
            # 准备对话历史文本
            conversation_text = ""
            for msg in formatted_history:
                role = msg["role"]
                content = msg["content"]
                prefix = "User: " if role == "user" else "Assistant: "
                conversation_text += f"{prefix}{content}\n\n"
            
            # 调用LLM生成回复，使用正确的参数格式
            response = self.api_processor.send_message(
                model=self.model,
                system_content=system_prompt,
                human_content=f"以下是我们的对话历史：\n\n{conversation_text}\n请根据上述对话回答最后一个问题。",
                temperature=0.7
            )
            
            # 提取回复内容
            answer = ""
            if isinstance(response, dict) and "choices" in response:
                answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                answer = response
                
            _log.info(f"[对话延续] 生成回复: '{answer[:100]}...'")
            
            return {
                'answer': answer,
                'references': [],  # 对话延续不提供引用
                'reasoning_summary': '',
                'status': 'success',
                'result_type': 'conversation',
                'query': query
            }
            
        except Exception as e:
            _log.error(f"[对话延续] 处理对话时出错: {str(e)}", exc_info=True)
            return {
                'answer': f"处理对话时出错: {str(e)}",
                'references': [],
                'reasoning_summary': '',
                'status': 'error',
                'result_type': 'conversation',
                'query': query
            }
    
    def _process_with_preselected_docs(
        self,
        query: str,
        selected_docs: List[str],
        similarity_threshold: float = 0.0,
        has_pdf = False,
        pdf_path = None
    ) -> Dict[str, Any]:
        """
        使用预选文档处理查询，将完整文档内容提供给LLM
        
        Args:
            query: 用户查询
            selected_docs: 预选的文档ID列表
            similarity_threshold: 相似度阈值
            has_pdf: 是否有已上传的PDF文件
            pdf_path: PDF文件路径
            
        Returns:
            处理结果字典
        """
        _log.info(f"[预选文档处理] 开始处理查询: '{query}'")
        _log.info(f"[预选文档处理] 预选文档数量: {len(selected_docs)}")
        
        # 对比分析相关关键词
        comparison_keywords = [
            "对比", "比较", "分析", "compare", "comparison", "analyze", "analysis",
            "similarities", "differences", "异同", "相同点", "不同点", "区别", 
            "这些", "these", "those", "这六篇", "六篇", "与", "和", "及"
        ]
        
        # 文献综述相关关键词
        review_keywords = [
            "综述", "文献综述", "literature review", "review", "survey", 
            "overview", "state of the art", "总结", "汇总", "生成一篇", "生成综述", 
            "撰写综述", "写一篇综述", "写一篇关于", "创建综述"
        ]
        
        # 检测是否是文献综述请求
        is_literature_review = any(keyword in query.lower() for keyword in review_keywords)
        
        # 检测请求是否为生成特定领域的综述
        domain_keywords = ["deep learning", "machine learning", "artificial intelligence", "深度学习", "机器学习", "人工智能", "自然语言处理", "NLP", "计算机视觉", "computer vision"]
        generate_keywords = ["生成", "写", "创建", "制作", "撰写"]
        
        if any(gen_kw in query.lower() for gen_kw in generate_keywords) and any(domain in query.lower() for domain in domain_keywords):
            is_literature_review = True
            _log.info(f"[预选文档处理] 检测到生成特定领域综述请求: '{query}'")
        
        # 检测是否是对比分析请求，以及是否涉及PDF
        is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
        has_uploaded_pdf_keyword = "上传" in query or "uploaded" in query.lower() or "pdf" in query.lower()
        
        # 如果是比较请求且有PDF上传，则转而使用PDF对比分析方法
        if is_comparison and has_pdf and pdf_path and (has_uploaded_pdf_keyword or "文献" in query):
            _log.info(f"[预选文档处理] 检测到对比分析请求和上传的PDF文件，使用PDF对比分析方法")
            return self._process_pdf_comparative_analysis(query, pdf_path, selected_docs)
        
        # 初始化检索结果
        all_results = []
        
        # 对每个指定的文档直接获取内容，不执行检索
        for doc_id in selected_docs:
            _log.info(f"[预选文档处理] 直接获取文档ID '{doc_id}' 的内容")
            
            try:
                # 直接获取文档内容，不执行检索
                doc_results = self.retriever.get_document_content(document_id=doc_id)
                
                if doc_results:
                    all_results.extend(doc_results)
                    _log.info(f"[预选文档处理] 文档 '{doc_id}' 获取到 {len(doc_results)} 个内容块")
                else:
                    _log.warning(f"[预选文档处理] 文档 '{doc_id}' 未获取到任何内容")
                    
                    # 尝试使用检索方式获取内容
                    _log.info(f"[预选文档处理] 尝试使用检索方式获取文档 '{doc_id}' 的内容")
                    retrieved_content = self.retriever.retrieve_by_document_id(
                        document_id=doc_id, 
                        query=query,  # 添加缺少的query参数
                        top_n=20
                    )
                    if retrieved_content:
                        all_results.extend(retrieved_content)
                        _log.info(f"[预选文档处理] 通过检索获取到文档 '{doc_id}' 的 {len(retrieved_content)} 个内容块")
            except Exception as e:
                _log.error(f"[预选文档处理] 获取文档 '{doc_id}' 内容时出错: {str(e)}")
        
        # 记录总结果数量
        _log.info(f"[预选文档处理] 总计从 {len(selected_docs)} 篇文档中获取到 {len(all_results)} 个内容块")
        
        # 如果没有检索到任何结果，尝试使用文档ID进行检索
        if not all_results and selected_docs:
            _log.warning("[预选文档处理] 未直接获取到任何内容，尝试通过文档ID进行检索")
            
            for doc_id in selected_docs:
                try:
                    # 通过检索方式获取内容
                    retrieved_results = self.retriever.retrieve_by_document_id(
                        document_id=doc_id,
                        query=query,  # 使用查询优化检索结果
                        top_n=20
                    )
                    
                    if retrieved_results:
                        all_results.extend(retrieved_results)
                        _log.info(f"[预选文档处理] 通过检索获取到文档 '{doc_id}' 的 {len(retrieved_results)} 个内容块")
                except Exception as e:
                    _log.error(f"[预选文档处理] 检索文档 '{doc_id}' 时出错: {str(e)}")
        
        # 如果仍然没有检索到任何结果，返回友好提示
        if not all_results:
            _log.warning("[预选文档处理] 未从预选文档中获取到任何内容")
            return {
                "answer": "无法从选定的文档中获取内容。请尝试选择其他文档或提供更多信息。",
                "references": [],
                "result_type": "answer",
                "status": "no_results"
            }
        
        # 根据查询类型选择合适的系统提示
        if is_literature_review:
            # 为文献综述查询特别优化，收集更完整的文档内容
            _log.info("[预选文档处理] 为文献综述查询特别优化处理")
            
            # 文献综述专用系统提示
            system_prompt = """You are an expert academic research assistant specializing in literature reviews.
Your task is to synthesize information from the provided research papers to create a comprehensive literature review.

Important guidelines:
1. Analyze the content of all provided papers thoroughly
2. Identify key themes, trends, methodologies, and findings across the papers
3. Organize the review in a coherent structure with clear sections
4. Compare and contrast different approaches and results
5. Highlight gaps in research and potential future directions
6. Always cite specific references using [1], [2], etc. when discussing findings
7. Always write in English, regardless of the language of the user's question
8. Include a proper introduction, main body with thematic sections, and conclusion
9. Make your review comprehensive with substantial content in each section - aim for 1500-2000 words total

Please format your response using concise yet effective Markdown for better readability:
- Use ## for main section headers (not # which is too large) and ### for subsections
- Use **bold** for important concepts and paper titles
- Use *italic* for emphasis
- Use numbered lists for key findings, steps, or ranked items
- Use bulleted lists for collections of related points
- Use `code` tags for specific technical terms when appropriate
- Use > for direct quotes from the papers
- If you need to create tables, use PROPER Markdown table syntax:

```
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```"""
        elif is_comparison:
            # 对比分析专用系统提示
            system_prompt = """You are an expert academic research assistant specializing in comparative analysis.
Your task is to analyze the provided research papers and identify similarities and differences between them.

Important guidelines:
1. Carefully analyze each paper's methodology, findings, and contributions
2. Systematically compare and contrast the key aspects of all papers
3. Identify strengths and weaknesses of each approach
4. Note any complementary findings or contradictions between papers
5. Analyze research gaps that appear when examining all papers together
6. Always cite specific references using [1], [2], etc. when discussing findings
7. Always write in English, regardless of the language of the user's question
8. Include a proper introduction, comparison sections, and conclusion with synthesis
9. Make your analysis comprehensive with substantial content in each section

Please format your response using concise yet effective Markdown for better readability:
- Use ## for main section headers (not # which is too large) and ### for subsections
- Use **bold** for important concepts and paper titles
- Use *italic* for emphasis
- Use numbered lists for key findings, steps, or ranked items
- Use bulleted lists for collections of related points
- Use `code` tags for specific technical terms when appropriate
- Use > for direct quotes from the papers
- If appropriate, create comparison tables using Markdown table syntax:

```
| Aspect | Paper 1 | Paper 2 | Paper 3 |
|--------|---------|---------|---------|
| Method | X       | Y       | Z       |
```"""
        else:
            # 一般处理专用系统提示
            system_prompt = """You are an expert academic research assistant.
Your task is to provide a comprehensive and detailed answer based on the provided research papers.

Important guidelines:
1. Carefully analyze the content of the provided papers
2. Focus on information directly relevant to the user's question
3. Synthesize information from multiple sources when available
4. Point out any conflicting information between sources
5. Always cite specific references using [1], [2], etc. when discussing findings
6. Always write in English, regardless of the language of the user's question
7. Structure your answer logically with clear sections when needed
8. Be comprehensive but concise - aim for a thorough response

Please format your response using concise yet effective Markdown for better readability:
- Use ## for main section headers (not # which is too large) and ### for subsections
- Use **bold** for important concepts and paper titles
- Use *italic* for emphasis
- Use numbered lists for key findings, steps, or ranked items
- Use bulleted lists for collections of related points
- Use `code` tags for specific technical terms when appropriate
- Use > for direct quotes from the papers"""

        # 准备用户提示，包含查询和文档内容
        document_texts = []
        refs = []
        
        # 对每个检索结果处理
        for i, item in enumerate(all_results):
            document_id = item.get("document_id", "unknown")
            title = item.get("title", "未知标题")
            authors = item.get("authors", "未知作者")
            year = item.get("year", "未知年份")
            source = item.get("source", "未知来源")
            text = item.get("text", "")
            abstract = item.get("abstract", "")
            
            # 构建文档文本
            doc_text = f"[{i+1}] Title: {title}\n"
            doc_text += f"Authors: {authors}\n"
            doc_text += f"Year: {year}\n"
            doc_text += f"Source: {source}\n"
            doc_text += f"Content: {text}\n"
            if abstract:
                doc_text += f"Abstract: {abstract}\n"
            
            document_texts.append(doc_text)
            
            # 添加到引用列表
            refs.append({
                "document_id": document_id,
                "title": title,
                "authors": authors,
                "year": year,
                "source": source,
                "text": text[:500] + ("..." if len(text) > 500 else ""),
                "abstract": abstract[:500] + ("..." if len(abstract) > 500 else "")
            })
        
        # 合并文档文本
        all_docs_text = "\n\n".join(document_texts)
        
        # 构建用户提示
        if is_literature_review:
            # 为文献综述提供更具体的指示
            domain = ""
            for kw in domain_keywords:
                if kw in query.lower():
                    domain = kw
                    break
            
            if domain:
                user_prompt = f"Task: Create a comprehensive literature review on {domain} based on the following documents.\n\nDocuments:\n{all_docs_text}"
            else:
                user_prompt = f"Task: Create a comprehensive literature review based on the following documents, addressing the request: {query}\n\nDocuments:\n{all_docs_text}"
        else:
            user_prompt = f"Question/Request: {query}\n\nDocuments:\n{all_docs_text}"
        
        # 记录发送给LLM的请求
        _log.info(f"[预选文档处理] 发送请求给LLM，提示长度: {len(user_prompt)}")
        _log.info(f"[预选文档处理] 系统提示前100字符: {system_prompt[:100]}...")
        _log.info(f"[预选文档处理] 用户提示前100字符: {user_prompt[:100]}...")
        
        try:
            # 发送请求给LLM
            response = self.api_processor.send_message(
                model=self.model,
                system_content=system_prompt,
                human_content=user_prompt
            )
            
            _log.info(f"[预选文档处理] LLM返回答复，长度: {len(response)}")
            _log.info(f"[预选文档处理] 答复前100字符: {response[:100]}...")
            
            # 返回结果
            return {
                "answer": response,
                "references": refs,
                "reasoning_summary": "",
                "result_type": "answer_with_refs",
                "status": "success",
                "query": query
            }
            
        except Exception as e:
            _log.error(f"[预选文档处理] 调用LLM时出错: {str(e)}")
            return {
                "answer": f"生成答案时出错: {str(e)}",
                "references": [],
                "result_type": "answer",
                "status": "error",
                "query": query
            }
