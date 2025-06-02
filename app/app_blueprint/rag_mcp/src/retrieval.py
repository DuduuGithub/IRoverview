import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from .reranking import LLMReranker, JinaReranker

_log = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir
        
    def retrieve_by_query(self, query: str, top_n: int = 10) -> List[Dict]:
        """
        根据查询检索文档
        
        Args:
            query: 查询文本
            top_n: 返回结果数量
            
        Returns:
            检索结果列表
        """
        _log.info(f"BM25检索，查询: '{query}'，top_n: {top_n}")
        
        # 检查BM25索引目录是否存在
        if not self.bm25_db_dir or not self.bm25_db_dir.exists():
            _log.error(f"BM25索引目录不存在: {self.bm25_db_dir}")
            return []
            
        _log.info(f"使用BM25索引目录: {self.bm25_db_dir}")
        
        # 所有检索结果
        all_results = []
        
        # 遍历所有文档
        for document_path in self.documents_dir.glob("*.json"):
            try:
                # 加载文档
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                # 获取文档ID
                document_id = document.get("metainfo", {}).get("document_id", document_path.stem)
                
                # 加载对应的BM25索引
                bm25_path = self.bm25_db_dir / f"{document_path.stem}.pkl"
                if not bm25_path.exists():
                    # 尝试直接使用文档ID作为文件名
                    bm25_path = self.bm25_db_dir / f"{document_id}.pkl"
                    if not bm25_path.exists():
                        _log.warning(f"找不到BM25索引: {bm25_path}")
                        continue
                    
                _log.debug(f"找到BM25索引: {bm25_path}")
                with open(bm25_path, 'rb') as f:
                    bm25_index = pickle.load(f)
                
                # 获取文档内容
                chunks = document.get("content", {}).get("chunks", [])
                if not chunks:
                    _log.warning(f"文档 {document_id} 没有内容块")
                    continue
                
                # 获取BM25分数
                tokenized_query = query.split()
                scores = bm25_index.get_scores(tokenized_query)
                
                # 获取前top_n个结果的索引
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:min(top_n, len(scores))]
                
                # 提取结果
                for index in top_indices:
                    score = float(scores[index])
                    if score <= 0:
                        continue
                        
                    chunk = chunks[index]
                    metainfo = document.get("metainfo", {})
                    
                    # 构建结果
                    result = {
                        "document_id": document_id,
                        "score": score,
                        "page": chunk.get("page", 0),
                        "text": chunk.get("text", ""),
                        "title": metainfo.get("title", ""),
                        "source": metainfo.get("source", ""),
                        "year": metainfo.get("year", ""),
                        "authors": metainfo.get("authors", []),
                        "metadata": metainfo
                    }
                    all_results.append(result)
            except Exception as e:
                _log.error(f"处理文档 {document_path} 时出错: {str(e)}", exc_info=True)
                continue
        
        # 按分数排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 返回前top_n个结果
        _log.info(f"BM25检索找到 {len(all_results)} 条结果")
        return all_results[:top_n]
        
    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
                    
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")
            
        # Load corresponding BM25 index
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)
            
        # Get the document content and BM25 index
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        # Get BM25 scores for the query
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)
        
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]
        
        retrieval_results = []
        seen_pages = set()
        
        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "score": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "score": score,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
        
        return retrieval_results



class VectorRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, reranking_strategy="jina", use_llm_reranking=False):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.llm = self._set_up_llm()
        self.reranking_strategy = reranking_strategy
        self.use_llm_reranking = use_llm_reranking
        
        # 初始化重排序器
        self.reranker = None
        if self.reranking_strategy == "jina":
            try:
                _log.info("初始化Jina重排序器")
                self.reranker = JinaReranker()
            except Exception as e:
                _log.error(f"初始化Jina重排序器失败: {str(e)}")
        elif self.reranking_strategy == "llm" or self.use_llm_reranking:
            try:
                _log.info("初始化LLM重排序器")
                self.reranker = LLMReranker(self.llm)
            except Exception as e:
                _log.error(f"初始化LLM重排序器失败: {str(e)}")
        else:
            _log.info(f"未使用重排序器，reranking_strategy={self.reranking_strategy}")

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm
    
    @staticmethod
    def set_up_llm():
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def _load_dbs(self):
        all_dbs = []
        # 检查目录是否存在
        if not self.documents_dir.exists():
            _log.warning(f"文档目录不存在: {self.documents_dir}，将创建该目录")
            os.makedirs(self.documents_dir, exist_ok=True)
            
        if not self.vector_db_dir.exists():
            _log.warning(f"向量数据库目录不存在: {self.vector_db_dir}，将创建该目录")
            os.makedirs(self.vector_db_dir, exist_ok=True)
            
        # 获取JSON文档路径列表
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        if not all_documents_paths:
            _log.warning(f"文档目录中没有找到任何JSON文件: {self.documents_dir}")
            _log.warning("请先通过RAG界面上传PDF文档并让系统处理它们")
            return []
            
        # 获取向量数据库文件列表
        vector_db_files_list = list(self.vector_db_dir.glob('*.faiss'))
        if not vector_db_files_list:
            _log.warning(f"向量数据库目录中没有找到任何FAISS文件: {self.vector_db_dir}")
            _log.warning("请先通过RAG界面上传PDF文档并让系统处理它们")
            return []
            
        # 创建向量数据库文件映射
        vector_db_files = {db_path.stem: db_path for db_path in vector_db_files_list}
        _log.info(f"找到 {len(vector_db_files)} 个向量数据库文件")
        
        # 遍历所有文档路径
        docs_with_vectors = 0
        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files:
                _log.warning(f"找不到文档 {document_path.name} 对应的向量数据库文件")
                continue
                
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"加载JSON文件 {document_path.name} 时出错: {e}")
                continue
            
            # 验证文档是否符合预期的架构
            if not (isinstance(document, dict) and "metainfo" in document and "content" in document):
                _log.warning(f"跳过 {document_path.name}: 不符合预期的架构。")
                continue
            
            try:
                vector_db = faiss.read_index(str(vector_db_files[stem]))
                docs_with_vectors += 1
            except Exception as e:
                _log.error(f"读取向量数据库 {document_path.name} 时出错: {e}")
                continue
                
            report = {
                "name": stem,
                "vector_db": vector_db,
                "document": document,
                "document_id": document.get("metainfo", {}).get("document_id", stem)
            }
            all_dbs.append(report)
        
        _log.info(f"成功加载 {len(all_dbs)}/{len(all_documents_paths)} 个文档")
        if not all_dbs:
            _log.error("没有可用的文档数据库。请通过RAG界面上传PDF文档并处理它们。")
            
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(f"Report '{report.get('name')}' is missing 'metainfo'!")
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        
        actual_top_n = min(top_n, len(chunks))
        
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)
    
        retrieval_results = []
        seen_pages = set()
        
        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "score": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"]
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "score": distance,
                    "page": chunk["page"],
                    "text": chunk["text"]
                }
                retrieval_results.append(result)
            
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "score": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages

    def retrieve_by_query(self, query: str, llm_reranking_sample_size: int = None, top_n: int = 5, similarity_threshold: float = 0.0) -> List[Dict]:
        """
        根据查询检索所有文档中的相关内容
        
        Args:
            query: 查询文本
            llm_reranking_sample_size: LLM重排序样本大小
            top_n: 返回结果数量
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉
            
        Returns:
            检索结果列表
        """
        if not self.all_dbs:
            _log.error("没有可用的文档数据库")
            raise ValueError("No document databases available")
            
        _log.info(f"执行向量检索，查询: '{query}'，top_n: {top_n}，相似度阈值: {similarity_threshold}")
            
        # 获取查询的向量表示
        _log.info("将查询转换为向量表示")
        try:
            embedding = self.llm.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            )
            embedding = embedding.data[0].embedding
            embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
            _log.info(f"成功获取查询向量，维度: {len(embedding)}")
        except Exception as e:
            _log.error(f"获取向量表示失败: {str(e)}", exc_info=True)
            raise
        
        all_results = []
        all_raw_scores = [] # 记录所有文档的原始分数
        
        # 遍历所有文档，进行向量检索
        _log.info(f"开始检索{len(self.all_dbs)}个文档...")
        for report_idx, report in enumerate(self.all_dbs):
            try:
                document = report["document"]
                vector_db = report["vector_db"]
                chunks = document["content"]["chunks"]
                metainfo = document.get("metainfo", {})
                document_id = metainfo.get("document_id", "")
                
                _log.info(f"[向量检索] 检索文档 {report_idx+1}/{len(self.all_dbs)}: '{document_id}'")
                
                # 搜索向量相似度
                distances, indices = vector_db.search(x=embedding_array, k=1)
                distance = distances[0][0]
                idx = indices[0][0]
                
                # 记录原始相似度分数
                similarity = float(distance)
                all_raw_scores.append({"document_id": document_id, "score": similarity})
                _log.info(f"[向量检索] 文档 '{document_id}' 的最高相似度块: {similarity:.6f}")
                
                # 获取最相似的块内容
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    
                    # 应用相似度阈值过滤
                    if similarity_threshold > 0 and similarity < similarity_threshold:
                        _log.info(f"[向量检索] 文档 '{document_id}' 的相似度分数 {similarity:.6f} 低于阈值 {similarity_threshold}，跳过")
                        continue
                    else:
                        _log.info(f"[向量检索] 文档 '{document_id}' 的相似度分数 {similarity:.6f} 高于阈值 {similarity_threshold}，保留")
                    
                    # 优先使用原始文档ID，确保与数据库中的ID一致
                    real_document_id = document_id
                    if not real_document_id or real_document_id.startswith("doc_"):
                        # 尝试从文件名获取真实ID
                        filename = os.path.basename(report.get("name", ""))
                        if filename:
                            real_document_id = filename
                    
                    # 创建检索结果
                    result = {
                        "document_id": real_document_id,
                        "score": similarity,
                        "page": chunk.get("page", 0),
                        "text": chunk.get("text", ""),
                        "title": metainfo.get("title", ""),
                        "source": metainfo.get("source", ""),
                        "year": metainfo.get("year", ""),
                        "authors": metainfo.get("authors", []),
                        "metadata": metainfo,
                        # 确保ID与document_id一致，用于Jina重排序
                        "id": real_document_id
                    }
                    all_results.append(result)
            except Exception as e:
                _log.warning(f"[向量检索] 处理文档时出错: {str(e)}")
        
        _log.info(f"[向量检索] 初始检索返回 {len(all_results)} 条结果，共检查 {len(self.all_dbs)} 个文档数据库")
        
        # 记录所有原始分数的统计信息
        if all_raw_scores:
            scores = [item["score"] for item in all_raw_scores]
            _log.info(f"[向量检索] 相似度分数统计 - 最低: {min(scores):.6f}, 最高: {max(scores):.6f}, 平均: {sum(scores)/len(scores):.6f}, 中位数: {sorted(scores)[len(scores)//2]:.6f}")
            
            # 记录所有分数的分布情况
            below_threshold = len([s for s in scores if s < similarity_threshold])
            _log.info(f"[向量检索] 相似度分数分布 - 低于阈值({similarity_threshold}): {below_threshold}/{len(scores)}, 比例: {below_threshold*100/len(scores):.2f}%")
        
        if not all_results:
            _log.warning(f"[向量检索] 没有找到符合相似度阈值 {similarity_threshold} 的结果")
            return []
            
        # 对结果进行排序
        all_results.sort(key=lambda x: x['score'], reverse=True)
        _log.info(f"[向量检索] 排序后，相似度前3位分数: " + 
                 ", ".join([f"{res['score']:.6f}" for res in all_results[:3]]) if len(all_results) >= 3 else "不足3条结果")
        
        # 只返回前top_n个结果
        limited_results = all_results[:top_n]
        _log.info(f"[向量检索] 最终返回 {len(limited_results)} 条结果（限制为top_{top_n}）")

        # 添加重排序的结果
        if self.reranker and self.reranking_strategy:
            _log.info(f"[向量检索] 使用 {self.reranking_strategy} 重排序器对检索结果进行重排序")
            try:
                # 使用重排序器重新排序结果
                reranked_results = self.reranker.rerank(query, limited_results, top_n=top_n)
                _log.info(f"[向量检索] 重排序完成，返回 {len(reranked_results)} 条结果")
                return reranked_results
            except Exception as e:
                _log.error(f"[向量检索] 重排序出错: {str(e)}")
                _log.info("[向量检索] 由于重排序错误，返回原始排序结果")
                return limited_results
        else:
            _log.info("[向量检索] 没有使用重排序，返回原始排序结果")
            return limited_results

    def retrieve_by_document_id(self, document_id: str, query: str, llm_reranking_sample_size: int = None, top_n: int = 3, return_parent_pages: bool = False, similarity_threshold: float = 0.0) -> List[Dict]:
        """
        根据文档ID和查询检索相关内容
        
        Args:
            document_id: 文档ID
            query: 查询文本
            llm_reranking_sample_size: LLM重排序样本大小
            top_n: 返回结果数量
            return_parent_pages: 是否返回父页面
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤掉
            
        Returns:
            检索结果列表
        """
        _log.info(f"按文档ID检索，ID: '{document_id}'，查询: '{query}'")
        
        # 检查是否为文献综述类请求
        is_review_request = any(keyword in query.lower() for keyword in ["文献综述", "综述", "review", "survey", "literature"])
        
        # 获取查询的向量表示
        embedding = self.llm.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        embedding = embedding.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        
        # 查找匹配的文档
        target_report = None
        for report in self.all_dbs:
            report_doc_id = report.get("document_id", "")
            if not report_doc_id:
                metainfo = report.get("document", {}).get("metainfo", {})
                report_doc_id = metainfo.get("document_id", report.get("name", ""))
            
            if str(report_doc_id) == str(document_id):
                target_report = report
                break
        
        if target_report is None:
            _log.warning(f"未找到ID为 '{document_id}' 的文档")
            return []
        
        # 获取文档内容和向量数据库
        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]
        metainfo = document.get("metainfo", {})
    
        retrieval_results = []
        seen_pages = set()
        
        # 如果是文献综述请求，直接返回文档中所有的内容块
        if is_review_request:
            _log.info(f"检测到文献综述请求，为文档 {document_id} 返回所有内容")
            
            if return_parent_pages:
                # 返回所有页面
                for page in pages:
                    page_number = page.get("page", 0)
                    if page_number not in seen_pages:
                        seen_pages.add(page_number)
                        result = {
                            "document_id": document_id,
                            "score": 1.0,  # 给予最高分数
                            "page": page_number,
                            "text": page.get("text", ""),
                            "title": metainfo.get("title", ""),
                            "source": metainfo.get("source", ""),
                            "year": metainfo.get("year", ""),
                            "authors": metainfo.get("authors", []),
                            "metadata": metainfo
                        }
                        retrieval_results.append(result)
            else:
                # 返回所有内容块
                for chunk in chunks:
                    result = {
                        "document_id": document_id,
                        "score": 1.0,  # 给予最高分数
                        "page": chunk.get("page", 0),
                        "text": chunk.get("text", ""),
                        "title": metainfo.get("title", ""),
                        "source": metainfo.get("source", ""),
                        "year": metainfo.get("year", ""),
                        "authors": metainfo.get("authors", []),
                        "metadata": metainfo
                    }
                    retrieval_results.append(result)
                    
            # 对于文献综述请求，我们只需要返回内容，不需要进行重排序
            retrieval_results.sort(key=lambda x: x.get("page", 0))  # 按页码排序
            _log.info(f"为文献综述请求返回 {len(retrieval_results)} 条结果")
            # 增加返回的结果数量，确保有足够的内容用于生成详细的文献综述
            return retrieval_results[:top_n * 5]  # 返回更多结果用于综述
        
        # 以下是正常检索逻辑
        actual_retrieve = min(llm_reranking_sample_size or len(chunks), len(chunks))
        distances, indices = vector_db.search(x=embedding_array, k=actual_retrieve)
        
        for distance, index in zip(distances[0], indices[0]):
            if 0 <= index < len(chunks):
                chunk = chunks[index]
                
                # 获取相似度分数
                similarity = float(distance)
                
                # 应用相似度阈值过滤
                if similarity_threshold > 0 and similarity < similarity_threshold:
                    continue
                
                if return_parent_pages:
                    # 查找父页面
                    page_number = chunk.get("page", 0)
                    if page_number not in seen_pages:
                        seen_pages.add(page_number)
                        parent_page = next((p for p in pages if p.get("page") == page_number), None)
                        
                        if parent_page:
                            result = {
                                "document_id": document_id,
                                "score": similarity,
                                "page": page_number,
                                "text": parent_page.get("text", ""),
                                "title": metainfo.get("title", ""),
                                "source": metainfo.get("source", ""),
                                "year": metainfo.get("year", ""),
                                "authors": metainfo.get("authors", []),
                                "metadata": metainfo
                            }
                            retrieval_results.append(result)
            else:
                    # 返回块内容
                result = {
                    "document_id": document_id,
                        "score": similarity,
                        "page": chunk.get("page", 0),
                        "text": chunk.get("text", ""),
                        "title": metainfo.get("title", ""),
                        "source": metainfo.get("source", ""),
                        "year": metainfo.get("year", ""),
                        "authors": metainfo.get("authors", []),
                        "metadata": metainfo
                }
                retrieval_results.append(result)
        
        if not retrieval_results:
            _log.warning(f"未找到与查询 '{query}' 相关的结果")
            return []
            
        # 执行重排序
        if self.reranking_strategy == "llm" and self.use_llm_reranking:
            # 使用LLM重排序
            llm_reranker = LLMReranker()
            reranked_results = llm_reranker.rerank(query, retrieval_results)
            
            # 应用相似度阈值过滤
            if similarity_threshold > 0:
                reranked_results = [r for r in reranked_results if r["score"] >= similarity_threshold]
                _log.info(f"应用相似度阈值 {similarity_threshold} 后，剩余 {len(reranked_results)} 条结果")
                
            # 截取前top_n个结果
            return reranked_results[:top_n]
            
        elif self.reranking_strategy == "jina":
            # 使用Jina重排序
            jina_reranker = JinaReranker()
            reranked_docs = jina_reranker.rerank(query, retrieval_results, top_n=top_n)
            
            # 记录分数分布信息
            if reranked_docs:
                scores = [r.get("score", 0) for r in reranked_docs]
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 0
                avg_score = sum(scores) / len(scores) if scores else 0
                _log.info(f"Jina重排序结果分数分布：最小={min_score:.4f}, 最大={max_score:.4f}, 平均={avg_score:.4f}")
            
            # 应用相似度阈值过滤
            if similarity_threshold > 0:
                results_before = len(reranked_docs)
                reranked_docs = [r for r in reranked_docs if r.get("score", 0) >= similarity_threshold]
                filtered_count = results_before - len(reranked_docs)
                _log.info(f"应用相似度阈值 {similarity_threshold} 后，剩余 {len(reranked_docs)} 条结果，过滤掉 {filtered_count} 条")
                
            # 截取前top_n个结果
            return reranked_docs[:top_n]
        
        else:
            # 不进行重排序，直接按相似度分数排序
            retrieval_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 应用相似度阈值过滤
            if similarity_threshold > 0:
                retrieval_results = [r for r in retrieval_results if r["score"] >= similarity_threshold]
                _log.info(f"应用相似度阈值 {similarity_threshold} 后，剩余 {len(retrieval_results)} 条结果")
                
            # 截取前top_n个结果
            return retrieval_results[:top_n]

    def get_document_content(self, document_id: str) -> List[Dict]:
        """
        直接获取文档内容而不执行检索
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档内容列表
        """
        _log.info(f"[直接获取内容] 获取文档 '{document_id}' 的内容")
        
        # 查找匹配的文档
        target_report = None
        for report in self.all_dbs:
            report_doc_id = report.get("document_id", "")
            if not report_doc_id:
                metainfo = report.get("document", {}).get("metainfo", {})
                report_doc_id = metainfo.get("document_id", report.get("name", ""))
            
            if str(report_doc_id) == str(document_id):
                target_report = report
                break
        
        if target_report is None:
            _log.warning(f"[直接获取内容] 未找到ID为 '{document_id}' 的文档")
            return []
        
        # 获取文档内容
        document = target_report["document"]
        chunks = document["content"]["chunks"]
        metainfo = document.get("metainfo", {})
        
        # 直接返回所有内容块，不做任何限制
        results = []
        for chunk in chunks:
                            result = {
                                "document_id": document_id,
                "score": 1.0,  # 给予最高分数
                        "page": chunk.get("page", 0),
                        "text": chunk.get("text", ""),
                        "title": metainfo.get("title", ""),
                        "source": metainfo.get("source", ""),
                        "year": metainfo.get("year", ""),
                        "authors": metainfo.get("authors", []),
                        "metadata": metainfo
                }
        results.append(result)
        
        _log.info(f"[直接获取内容] 成功获取文档 '{document_id}' 的 {len(results)} 个内容块")
        return results  # 返回所有内容块，不做限制


class HybridRetriever:
    def __init__(self, vector_db_dir: Path, documents_dir: Path, bm25_db_dir: Path = None, max_tokens=1048576, reranking_strategy="jina"):
        """
        混合检索器实现 - 注意：根据设计调整，现在只使用向量检索，BM25已弃用
        但为了与现有代码兼容，保留了相同的参数结构
        """
        _log.info("初始化混合检索器(仅向量检索模式)")
        # 强制使用jina作为重排策略，忽略参数中可能传入的其他值
        reranking_strategy = "jina"
        self.vector_retriever = VectorRetriever(
            vector_db_dir=vector_db_dir, 
            documents_dir=documents_dir, 
            reranking_strategy=reranking_strategy,  # 确保使用jina重排序
            use_llm_reranking=False  # 关闭LLM重排序
        )
        
        # BM25检索器参数保留但不再使用
        self.bm25_retriever = None
        self.reranking_strategy = reranking_strategy
        
        # 初始化合适的reranker
        if reranking_strategy is not None and reranking_strategy.lower() == "jina":
            self.reranker = JinaReranker()
        else:
            self.reranker = None
        
    def retrieve_by_query(
        self, 
        query: str, 
        top_n: int = 50,
        llm_reranking_sample_size: int = 100,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """通过查询检索文档 - 只使用向量检索"""
        
        _log.info(f"[混合检索(向量模式)] 开始检索查询: '{query}'")
        
        # 直接调用向量检索
        results = self.vector_retriever.retrieve_by_query(
            query=query,
            llm_reranking_sample_size=llm_reranking_sample_size,
            top_n=top_n,
            similarity_threshold=similarity_threshold
        )
        
        _log.info(f"[混合检索(向量模式)] 返回 {len(results)} 条结果")
        
        # 标记来源为hybrid
        for doc in results:
            doc['source'] = 'hybrid'
            
        return results
    
    def retrieve_by_document_id(
        self, 
        document_id: str, 
        query: str, 
        top_n: int = 50,
        llm_reranking_sample_size: int = 100,
        return_parent_pages: bool = False,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """按文档ID检索 - 只使用向量检索"""
        return self.vector_retriever.retrieve_by_document_id(
            document_id=document_id,
            query=query,
            top_n=top_n,
            llm_reranking_sample_size=llm_reranking_sample_size,
            return_parent_pages=return_parent_pages,
            similarity_threshold=similarity_threshold
        )

    def get_document_content(self, document_id: str) -> List[Dict]:
        """直接获取文档内容而不执行检索
        
        Args:
            document_id: 文档ID
            
        Returns:
            文档内容列表
        """
        _log.info(f"[混合检索] 直接获取文档 '{document_id}' 内容")
        results = self.vector_retriever.get_document_content(document_id)
        
        # 标记来源为hybrid
        for doc in results:
            doc['source'] = 'hybrid'
            
        return results
