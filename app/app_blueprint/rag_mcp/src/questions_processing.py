import json
import os
import re
import concurrent.futures
import threading
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from tqdm import tqdm
import logging

# 设置详细的日志格式
_log = logging.getLogger(__name__)

# 导入其他依赖和模块
from .retrieval import VectorRetriever, HybridRetriever
from .api_requests import APIProcessor
import pandas as pd


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        llm_reranking: bool = False,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "openai",
        answering_model: str = "gpt-4o-2024-08-06",
        full_context: bool = False
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            document_id = result.get('document_id', 'unknown')
            page_number = result.get('page', 0)
            text = result.get('text', '')
            score = result.get('score', 0.0)
            
            # 处理文本，检测并转换倒排索引
            processed_text = self._process_text_content(text)
            
            context_parts.append(f'Text from document {document_id}, page {page_number}, relevance {score:.4f}: \n"""\n{processed_text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _process_text_content(self, text: str) -> str:
        """
        处理文本内容，检测并转换倒排索引
        
        Args:
            text: 原始文本内容
            
        Returns:
            处理后的文本
        """
        # 检查文本是否包含标题、关键词等部分
        if "Title:" in text:
            # 分离内容各部分
            parts = {}
            current_section = None
            
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Title:"):
                    current_section = "title"
                    parts[current_section] = line[6:].strip()
                elif line.startswith("Abstract:"):
                    current_section = "abstract"
                    parts[current_section] = ""
                elif line.startswith("Keywords:"):
                    current_section = "keywords"
                    parts[current_section] = line[9:].strip()
                elif line.startswith("Authors:"):
                    current_section = "authors"
                    parts[current_section] = line[8:].strip()
                elif current_section:
                    parts[current_section] = parts.get(current_section, "") + " " + line
            
            # 检查摘要是否是倒排索引格式
            if parts.get("abstract") and "{" in parts["abstract"] and "}" in parts["abstract"]:
                try:
                    # 尝试解析JSON格式的倒排索引
                    import json
                    abstract_text = parts["abstract"].strip()
                    # 如果是完整的JSON对象
                    if abstract_text.startswith("{") and abstract_text.endswith("}"):
                        try:
                            index_data = json.loads(abstract_text)
                            # 将倒排索引转换为文本
                            converted_text = self._parse_inverted_index(index_data)
                            if converted_text:
                                parts["abstract"] = converted_text
                        except json.JSONDecodeError:
                            # 如果不是有效的JSON，保持原样
                            pass
                    # 如果是包含abstract_inverted_index的结构
                    elif "abstract_inverted_index" in abstract_text:
                        try:
                            # 尝试提取JSON部分
                            json_start = abstract_text.find("{")
                            json_end = abstract_text.rfind("}") + 1
                            if json_start >= 0 and json_end > json_start:
                                json_str = abstract_text[json_start:json_end]
                                index_data = json.loads(json_str)
                                # 将倒排索引转换为文本
                                converted_text = self._parse_inverted_index(index_data)
                                if converted_text:
                                    parts["abstract"] = converted_text
                        except (json.JSONDecodeError, ValueError):
                            # 如果提取或解析失败，保持原样
                            pass
                except Exception as e:
                    _log.warning(f"处理倒排索引时出错: {str(e)}")
            
            # 处理关键词部分 - 如果是逗号分隔的单词列表，尝试将其转换为更可读的格式
            if "keywords" in parts and parts["keywords"]:
                keywords_text = parts["keywords"]
                if "," in keywords_text:
                    # 尝试将逗号分隔的单词连接成句子
                    keywords = [k.strip() for k in keywords_text.split(",")]
                    # 过滤掉空字符串和单个字符
                    keywords = [k for k in keywords if len(k) > 1]
                    if keywords:
                        parts["keywords"] = "Keywords: " + ", ".join(keywords)
            
            # 重新构建文本
            processed_text = ""
            if "title" in parts:
                processed_text += f"Title: {parts['title']}\n\n"
            if "authors" in parts:
                processed_text += f"Authors: {parts['authors']}\n\n"
            if "abstract" in parts:
                processed_text += f"Abstract: {parts['abstract']}\n\n"
            if "keywords" in parts:
                processed_text += f"{parts['keywords']}\n\n"
                
            return processed_text.strip()
        
        # 检查是否仅包含逗号分隔的单词列表
        if "," in text and text.count(",") > 5:
            words = [w.strip() for w in text.split(",")]
            # 如果有很多短词，这可能是一个单词列表
            if all(len(w) < 15 for w in words) and len(words) > 10:
                # 尝试将其构建为更可读的句子
                return ". ".join([w.capitalize() for w in words if w]) + "."
            
        return text
        
    def _parse_inverted_index(self, inverted_index):
        """
        将倒排索引转换为普通文本
        
        Args:
            inverted_index: 倒排索引数据
            
        Returns:
            str: 转换后的文本
        """
        try:
            # 如果是包含abstract_inverted_index字段的对象
            if isinstance(inverted_index, dict) and "abstract_inverted_index" in inverted_index:
                inverted_index = inverted_index["abstract_inverted_index"]
                
            # 确保是字典格式
            if not isinstance(inverted_index, dict):
                return str(inverted_index)
                
            # 获取所有单词和它们的位置
            words_positions = []
            for word, positions in inverted_index.items():
                if isinstance(positions, list):
                    for pos in positions:
                        if isinstance(pos, int):
                            words_positions.append((word, pos))
            
            # 没有有效的位置数据
            if not words_positions:
                return str(inverted_index)
                
            # 按照位置排序
            words_positions.sort(key=lambda x: x[1])
            
            # 拼接文本
            text = ' '.join([wp[0] for wp in words_positions])
            return text
        except Exception as e:
            _log.warning(f"解析倒排索引失败: {str(e)}")
            return str(inverted_index)

    def _extract_references(self, pages_list: list, document_id: str) -> list:
        """
        Extract references from pages list
        
        Args:
            pages_list: List of page numbers
            document_id: Document ID
            
        Returns:
            List of references
        """
        # Load document data
        if self.subset_path is None:
            raise ValueError("subset_path is required for processing references.")
            
        # Load dataset
        try:
            docs_df = pd.read_csv(self.subset_path)
        except Exception as e:
            print(f"Warning: Cannot load subset file: {str(e)}")
            docs_df = None
            
        refs = []
        for page in pages_list:
            # Add more information if document data is available
            if docs_df is not None and not docs_df.empty:
                doc_info = docs_df[docs_df['document_id'] == document_id]
                if not doc_info.empty:
                    doc_row = doc_info.iloc[0]
                    # Add different reference formats based on document type
                    if 'title' in doc_row and 'authors' in doc_row and 'year' in doc_row:
                        refs.append({
                            "document_id": document_id,
                            "page": page,
                            "title": doc_row.get('title', ''),
                            "authors": doc_row.get('authors', ''),
                            "year": doc_row.get('year', '')
                        })
                        continue
            
            # Basic reference format
            refs.append({
                "document_id": document_id,
                "page": page
            })
            
        return refs

    def _extract_documents_from_subset(self, question_text: str) -> list[str]:
        """
        Extract document IDs from question by matching with documents in the subset file
        
        Args:
            question_text: Question text
            
        Returns:
            List of document IDs
        """
        if not hasattr(self, 'docs_df'):
            if self.subset_path is None:
                raise ValueError("subset_path must be provided to use subset extraction")
                
            try:
                self.docs_df = pd.read_csv(self.subset_path)
            except Exception as e:
                print(f"Warning: Cannot load subset file: {str(e)}")
                return []
        
        found_documents = []
        
        # 1. Check if question contains document IDs
        if 'document_id' in self.docs_df.columns:
            document_ids = sorted(self.docs_df['document_id'].dropna().unique(), key=len, reverse=True)
            
            for doc_id in document_ids:
                if str(doc_id) in question_text:
                    found_documents.append(str(doc_id))
                    question_text = question_text.replace(str(doc_id), '')
        
        # 2. Check if question contains document titles
        if 'title' in self.docs_df.columns and not found_documents:
            titles = sorted(self.docs_df['title'].dropna().unique(), key=len, reverse=True)
            
            for title in titles:
                # Avoid matching very short titles
                if len(str(title)) < 5:
                    continue
                    
                if str(title).lower() in question_text.lower():
                    # Find corresponding document ID
                    doc_row = self.docs_df[self.docs_df['title'] == title]
                    if not doc_row.empty:
                        doc_id = str(doc_row.iloc[0]['document_id'])
                        if doc_id not in found_documents:
                            found_documents.append(doc_id)
        
        # 3. If no documents found using the above methods, try to extract text in quotes as potential titles
        if not found_documents:
            quoted_text = re.findall(r'"([^"]*)"', question_text)
            for text in quoted_text:
                if 'title' in self.docs_df.columns:
                    # Try fuzzy matching titles
                    for _, row in self.docs_df.iterrows():
                        title = str(row.get('title', ''))
                        if text.lower() in title.lower() or title.lower() in text.lower():
                            doc_id = str(row['document_id'])
                            if doc_id not in found_documents:
                                found_documents.append(doc_id)
        
        return found_documents

    def get_answer_for_document(self, document_id: str, question: str, schema: str) -> dict:
        """
        Generate answer for a specific document
        
        Args:
            document_id: Document ID
            question: Question text
            schema: Answer schema
            
        Returns:
            Answer dictionary
        """
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )

        if self.full_context:
            retrieval_results = retriever.retrieve_all(document_id)
        else:           
            retrieval_results = retriever.retrieve_by_document_id(
                document_id=document_id,
                query=question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                return_parent_pages=self.return_parent_pages
            )
        
        if not retrieval_results:
            raise ValueError(f"No relevant context found for document {document_id}")
        
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        
        self.response_data = self.openai_processor.response_data
        
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["references"] = self._extract_references(validated_pages, document_id)
            
        return answer_dict

    def get_answer_without_document(self, question: str, schema: str) -> dict:
        """
        Perform retrieval and answer generation based on question without specifying document ID
        
        Args:
            question: Question text
            schema: Answer schema
            
        Returns:
            Answer dictionary
        """
        _log.info(f"执行get_answer_without_document, 问题: {question}, 模式: {schema}")
        
        if self.llm_reranking:
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        
        # Perform global retrieval based on question
        retrieval_results = retriever.retrieve_by_query(
            query=question,
            llm_reranking_sample_size=self.llm_reranking_sample_size,
            top_n=self.top_n_retrieval
        )
        
        if not retrieval_results:
            _log.warning("未找到相关上下文")
            raise ValueError("No relevant context found")
        
        _log.info(f"检索到 {len(retrieval_results)} 个结果")
        
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        
        self.response_data = self.openai_processor.response_data
        
        # 检查回答是否包含"不含足够信息"等字样
        final_answer = answer_dict.get("final_answer", "")
        
        # 查找原始回答中表示信息不足或无法生成全面回答的模式
        insufficient_patterns = [
            "does not contain sufficient information",
            "not enough information",
            "not contain enough context",
            "not enough context",
            "insufficient information",
            "insufficient context",
            "I don't have enough information",
            "The provided context does not",
            "The documents provided do not",
            "cannot generate a comprehensive",
            "insufficient details"
        ]
        
        if isinstance(final_answer, str) and (
            any(pattern in final_answer.lower() for pattern in insufficient_patterns) or
            "no relevant information" in final_answer.lower() or
            "information is not provided" in final_answer.lower() or
            "information is not available" in final_answer.lower() or
            "cannot answer" in final_answer.lower() or
            "n/a" == final_answer.lower()
        ):
            _log.warning(f"原始回答表示信息不足: {final_answer[:100]}...")
            
            try:
                # 尝试使用通用知识回答问题
                general_response = self.openai_processor.send_message(
                    model=self.answering_model,
                    system_content="""You are a helpful assistant with general knowledge about academic research and scientific topics. 
                    Your task is to provide a general answer to the question based on your knowledge,
                    but make it clear that your answer is based on general knowledge rather than specific document content.
                    Begin your answer with "Based on general knowledge..." and provide concise, factual information.""",
                    human_content=f"The user asked: {question}\n\nPlease provide a general knowledge answer."
                )
                
                _log.info(f"生成了一个基于通用知识的回答")
                
                # 更新回答
                answer_dict["final_answer"] = general_response
                answer_dict["reasoning_summary"] = "This answer is based on general knowledge rather than specific document content."
                
            except Exception as e:
                _log.error(f"尝试生成通用知识回答时出错: {str(e)}")
        
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            
            # Get document ID from retrieval results
            if retrieval_results and len(retrieval_results) > 0:
                document_id = retrieval_results[0].get('document_id', 'unknown')
                answer_dict["references"] = self._extract_references(validated_pages, document_id)
            else:
                answer_dict["references"] = []
        
        return answer_dict

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break
        
        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def process_question(self, question: str, schema: str):
        """
        处理单个问题
        
        Args:
            question: 问题文本
            schema: 回答模式
            
        Returns:
            处理结果
        """
        if self.new_challenge_pipeline:
            extracted_documents = self._extract_documents_from_subset(question)
        else:
            # 兼容旧方法，从引号中提取可能的文档ID
            extracted_documents = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_documents) == 0:
            # 没有找到指定文档，尝试全局搜索
            return self.get_answer_without_document(question=question, schema=schema)
        
        if len(extracted_documents) == 1:
            document_id = extracted_documents[0]
            answer_dict = self.get_answer_for_document(document_id=document_id, question=question, schema=schema)
            return answer_dict
        else:
            # 如果有多个文档ID，可能是比较问题
            return self.process_comparative_question(question, extracted_documents, schema)

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """Calculate statistics about processed questions."""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = "") -> dict:
        total_questions = len(questions_list)
        # Add index to each question so we know where to write the answer details
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
        else:
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, team_email=team_email, submission_name=submission_name, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        Post-process answers for submission format:
        1. Convert page indices from one-based to zero-based
        2. Clear references for N/A answers
        3. Format answers according to submission schema
        4. Include step_by_step_analysis from answer details
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "document_id": ref["document_id"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, team_email: str = "", submission_name: str = "", pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', team_email: str = "79250515615@yandex.com", submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval", submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, documents: List[str], schema: str) -> dict:
        """
        Process a question comparing multiple documents
        
        Args:
            question: Question text
            documents: List of document IDs
            schema: Answer schema
            
        Returns:
            Comparative answer dictionary
        """
        # Step 1: Check documents
        if not documents or len(documents) < 2:
            raise ValueError("At least two documents are required for comparative analysis")
        
        # Step 2: Process each document individually
        individual_answers = {}
        aggregated_references = []
        
        def process_document_question(document: str) -> tuple[str, dict]:
            """Process question for a single document"""
            print(f"Processing document: {document}")
            
            answer_dict = self.get_answer_for_document(
                document_id=document,
                question=question,
                schema=schema
            )
            
            return document, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_document = {
                executor.submit(process_document_question, document): document 
                for document in documents
            }
            
            for future in concurrent.futures.as_completed(future_to_document):
                try:
                    document, answer_dict = future.result()
                    individual_answers[document] = answer_dict
                    
                    document_references = answer_dict.get("references", [])
                    aggregated_references.extend(document_references)
                except Exception as e:
                    document = future_to_document[future]
                    print(f"Error processing document {document}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("document_id"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer
    