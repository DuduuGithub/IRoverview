import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import logging
from . import prompts as prompts_module
from concurrent.futures import ThreadPoolExecutor

# 设置详细的日志格式
_log = logging.getLogger(__name__)

class JinaReranker:
    def __init__(self):
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        _log.info("初始化Jina重排序器")
        
    def get_headers(self):
        load_dotenv()
        jina_api_key = "jina_740f81178a7e4e49a6604937637818580StptpMPUHvkLZmPjSjcQL48Q6Rh"    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n=50):
        """
        使用Jina API对文档进行重排序
        
        Args:
            query: 查询字符串
            documents: 包含文档的列表，每个文档应该是一个字典，包含document_id/id和text字段
            top_n: 返回的最大结果数量
            
        Returns:
            重排序后的文档列表
        """
        
        # 验证输入参数
        if not documents:
            return []
            
        if not query or not query.strip():
            return documents[:top_n]
        
        # 准备API请求的文档格式
        docs_for_api = []
        doc_index_map = {}  # 新增：存储索引到文档的映射
        
        for i, doc in enumerate(documents):
            # 尝试获取文档ID，优先使用document_id字段
            doc_id = doc.get("document_id") or doc.get("id") or doc.get("doc_id") or f"doc_{i}"
            doc_text = doc.get("text", "")
            
            # 如果没有文本，尝试使用其他字段
            if not doc_text:
                doc_text = doc.get("abstract", "") or doc.get("snippet", "") or doc.get("content", "") or f"Empty document {i}"
            
            # 确保文档有文本内容
            if not doc_text.strip():
                continue
                
            # 保存索引到文档的映射关系
            doc_index_map[i] = doc
            
            docs_for_api.append({
                "id": str(doc_id),  # 确保ID是字符串类型
                "text": doc_text
            })
        
        if not docs_for_api:
            return documents[:top_n]
        
        # 构建请求数据
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": docs_for_api,
            "return_documents": False  # 只返回ID和分数，不返回文档内容
        }

        try:
            response = requests.post(url=self.url, headers=self.headers, json=data, timeout=30)
            
            if response.status_code != 200:
                _log.error(f"Jina API返回错误码: {response.status_code}, 响应: {response.text}")
                return documents[:top_n]
                
            json_response = response.json()
            
            # 添加详细日志以查看响应结构
            _log.info(f"Jina API响应结构: {json_response.keys()}")
            
            # 修改这里：更灵活地处理API响应
            results_field = None
            for field in ['results', 'documents', 'data', 'hits']:
                if field in json_response and json_response[field]:
                    results_field = field
                    break
            
            if not results_field:
                return documents[:top_n]
            
            reranked_ids_with_scores = json_response[results_field]
            
            # 更灵活地处理文档ID映射
            doc_map = {}
            for doc in documents:
                # 尝试所有可能的ID字段
                for id_field in ["document_id", "id", "doc_id"]:
                    if id_field in doc and doc[id_field]:
                        # 存储多种ID格式的映射
                        doc_id = str(doc[id_field])
                        doc_map[doc_id] = doc
                        # 同时尝试添加不带前缀的ID形式（处理可能的格式差异）
                        if "/" in doc_id:
                            doc_map[doc_id.split("/")[-1]] = doc
                        break
            
            # 根据API返回的顺序重新排列文档
            reranked_docs = []
            
                # 记录处理过程
            _log.info(f"处理API返回的结果，示例条目: {reranked_ids_with_scores[0] if reranked_ids_with_scores else 'N/A'}")
            
            for result in reranked_ids_with_scores:
                # 更灵活地处理ID和分数字段
                doc_id = None
                index = None
                score = 0.5  # 默认分数
                
                # 尝试获取索引字段（新增）
                if 'index' in result:
                    index = result['index']
                    _log.info(f"找到索引: {index}")
                
                # 尝试所有可能的ID字段名
                for id_field in ['id', 'document_id', 'doc_id', 'documentId']:
                    if id_field in result:
                        doc_id = str(result[id_field])
                        break
                    
                # 尝试所有可能的分数字段名
                for score_field in ['score', 'relevance', 'relevance_score', 'similarity']:
                    if score_field in result:
                        score = float(result[score_field])
                        break
                
                # 首先尝试使用索引找到文档
                doc = None
                if index is not None and index in doc_index_map:
                    doc = doc_index_map[index]
                # 如果没有索引或索引没有匹配，尝试使用文档ID
                elif doc_id and doc_id in doc_map:
                    doc = doc_map[doc_id]
                # 尝试不同形式的ID
                elif doc_id:
                    for mapped_id in doc_map:
                        if mapped_id.endswith(doc_id) or doc_id.endswith(mapped_id):
                            doc = doc_map[mapped_id]
                            break
                
                if doc:
                    # 获取原始文档并添加分数
                    reranked_doc = doc.copy()  # 创建副本避免修改原始文档
                    reranked_doc['score'] = score
                    reranked_docs.append(reranked_doc)
                else:
                    _log.warning(f"未找到匹配的原始文档 - 索引: {index}, ID: {doc_id}")
            

            # 如果没有找到任何匹配的文档，返回原始文档
            if not reranked_docs:

                return documents[:top_n]
            
            return reranked_docs

        except Exception as e:
            _log.error(f"调用Jina API出错: {str(e)}", exc_info=True)
            # 返回原始顺序的文档，但最多top_n个
            return documents[:top_n]

class LLMReranker:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts_module.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts_module.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts_module.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts_module.RetrievalRankingMultipleBlocks

      
    def set_up_llm(self):
        load_dotenv()
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return llm
    
    def get_rank_for_single_block(self, query, retrieved_document):

        # 截断文本以保持日志简洁
        doc_preview = retrieved_document[:100] + "..." if len(retrieved_document) > 100 else retrieved_document
        
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'
        
        try:
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block
            )

            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            
            return response_dict
            
        except Exception as e:
            _log.error(f"调用LLM进行文本块排名时出错: {str(e)}", exc_info=True)
            # 返回默认值
            return {"relevance_score": 0.0, "reasoning": f"Error during LLM call: {str(e)}"}

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text[:100]}...\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )

        try:
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_multiple_blocks
            )

            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            
            block_rankings = response_dict.get('block_rankings', [])

            # 检查分数是否都为0，如果是，则应用默认分数分布
            all_scores = [rank.get('relevance_score', 0) for rank in block_rankings]
            if all(score == 0 for score in all_scores):

                # 应用从0.9到0.5的线性分布
                for i, rank in enumerate(block_rankings):
                    default_score = 0.9 - (0.4 * i / len(block_rankings))
                    rank['relevance_score'] = round(default_score, 2)
                    rank['reasoning'] += " (Applied default scoring due to zero scores)"
            
            return response_dict
            
        except Exception as e:
            _log.error(f"调用LLM进行多个文本块排名时出错: {str(e)}", exc_info=True)
            # 返回默认值，使用从0.9到0.5的线性分布的分数
            default_rankings = []
            for i in range(len(retrieved_documents)):
                default_score = 0.9 - (0.4 * i / len(retrieved_documents))
                default_rankings.append({
                    "relevance_score": round(default_score, 2),
                    "reasoning": f"Default ranking due to error in LLM call: {str(e)}"
                })
            return {"block_rankings": default_rankings}

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        _log.info(f"开始重新排序 {len(documents)} 个文档，批次大小: {documents_batch_size}")
        
        # Create batches of documents
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        _log.info(f"创建了 {len(doc_batches)} 个批次")
        
        vector_weight = 1 - llm_weight
        
        if documents_batch_size == 1:
            _log.info("使用单文档处理模式")
            
            def process_single_doc(doc):
                # Get ranking for single document
                doc_id = doc.get('document_id', 'unknown')
                _log.info(f"处理文档 ID: {doc_id}")
                
                ranking = self.get_rank_for_single_block(query, doc['text'])
                
                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # 使用"score"键代替"distance"键
                original_score = doc.get('score', 0)
                
                # 计算组合分数
                combined_score = round(
                    llm_weight * ranking["relevance_score"] + 
                    vector_weight * original_score,
                    4
                )
                doc_with_score["combined_score"] = combined_score
                
                _log.info(f"文档 {doc_id} 的组合分数: {combined_score} (LLM: {ranking['relevance_score']}, 向量: {original_score})")
                return doc_with_score

            # Process all documents in parallel using single-block method
            _log.info("启动并行处理")
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))
                
        else:
            _log.info("使用批处理模式")
            
            def process_batch(batch):
                _log.info(f"处理批次，包含 {len(batch)} 个文档")
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])
                
                if len(block_rankings) < len(batch):
                    _log.warning(f"警告: 期望 {len(batch)} 个排名但只获得 {len(block_rankings)} 个")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        doc_id = doc.get('document_id', 'unknown')
                        page = doc.get('page', 'unknown')
                        _log.warning(f"文档 {doc_id} 页面 {page} 缺少排名")
                    
                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0, 
                            "reasoning": "Default ranking due to missing LLM response"
                        })
                
                for doc, rank in zip(batch, block_rankings):
                    doc_id = doc.get('document_id', 'unknown')
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    
                    # 使用"score"键代替"distance"键
                    original_score = doc.get('score', 0)
                    
                    combined_score = round(
                        llm_weight * rank["relevance_score"] + 
                        vector_weight * original_score,
                        4
                    )
                    doc_with_score["combined_score"] = combined_score
                    
                    _log.info(f"文档 {doc_id} 的组合分数: {combined_score} (LLM: {rank['relevance_score']}, 向量: {original_score})")
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            _log.info("启动并行批处理")
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        _log.info(f"排序完成，共 {len(all_results)} 个结果")
        
        # 确保所有结果都使用"score"键
        for result in all_results:
            result["score"] = result["combined_score"]
            
        return all_results
