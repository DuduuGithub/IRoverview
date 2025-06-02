import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.
        
        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化VectorDBIngestor")
        self.llm = self._set_up_llm()

    def _set_up_llm(self):
        load_dotenv()
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            self.logger.info("设置OpenAI客户端")
            llm = OpenAI(
                api_key=api_key,
                timeout=60,
                max_retries=3
            )
            return llm
        except Exception as e:
            self.logger.error(f"OpenAI客户端设置失败: {str(e)}")
            raise

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(3))
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[float]:
        """获取文本的嵌入向量
        
        Args:
            text: 输入文本或文本列表
            model: 使用的嵌入模型名称
            
        Returns:
            嵌入向量列表
        """
        if isinstance(text, str) and not text.strip():
            self.logger.warning("输入文本为空字符串")
            raise ValueError("输入文本不能为空字符串")
        
        if isinstance(text, list) and len(text) == 0:
            self.logger.warning("输入文本列表为空")
            raise ValueError("输入文本列表不能为空")
        
        try:
            # 处理输入文本
            if isinstance(text, list):
                # 过滤空文本
                text = [t for t in text if t and t.strip()]
                if not text:
                    self.logger.warning("过滤后的文本列表为空")
                    raise ValueError("过滤后的文本列表为空")
                # 将长文本列表分成小块
                text_chunks = []
                for t in text:
                    text_chunks.extend([t[i:i + 1024] for i in range(0, len(t), 1024)])
            else:
                text_chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
            
            self.logger.info(f"获取 {len(text_chunks)} 个文本块的嵌入向量")
            embeddings = []
            for chunk in text_chunks:
                response = self.llm.embeddings.create(input=chunk, model=model)
                embeddings.extend([embedding.embedding for embedding in response.data])
            
            self.logger.info(f"成功获取 {len(embeddings)} 个嵌入向量")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"获取嵌入向量失败: {str(e)}")
            raise

    def _create_vector_db(self, embeddings: List[float]):
        """创建向量数据库索引
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            FAISS索引对象
        """
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dimension = len(embeddings[0])
            index = faiss.IndexFlatIP(dimension)  # Cosine distance
            index.add(embeddings_array)
            return index
        except Exception as e:
            self.logger.error(f"创建向量数据库失败: {str(e)}")
            raise
    
    def _process_report(self, report: dict, report_path: str):
        """处理单个报告
        
        Args:
            report: 报告字典
            report_path: 报告文件路径，用于日志
            
        Returns:
            FAISS索引对象
        """
        try:
            # 检查报告结构
            if 'content' not in report:
                self.logger.error(f"报告 {report_path} 缺少content字段")
                raise ValueError(f"报告 {report_path} 缺少content字段")
                
            if 'chunks' not in report['content']:
                self.logger.error(f"报告 {report_path} 缺少chunks字段")
                raise ValueError(f"报告 {report_path} 缺少chunks字段")
            
            chunks = report['content']['chunks']
            if not chunks:
                self.logger.error(f"报告 {report_path} 没有文本块")
                raise ValueError(f"报告 {report_path} 没有文本块")
                
            # 提取文本块
            text_chunks = []
            for chunk in chunks:
                if 'text' in chunk and chunk['text'] and chunk['text'].strip():
                    text_chunks.append(chunk['text'])
            
            if not text_chunks:
                self.logger.error(f"报告 {report_path} 没有有效的文本块")
                raise ValueError(f"报告 {report_path} 没有有效的文本块")
                
            self.logger.info(f"处理报告 {report_path}，共 {len(text_chunks)} 个文本块")
            
            # 获取嵌入向量
            embeddings = self._get_embeddings(text_chunks)
            
            # 创建索引
            index = self._create_vector_db(embeddings)
            return index
        except Exception as e:
            self.logger.error(f"处理报告 {report_path} 失败: {str(e)}")
            raise

    def process_single_report(self, report_file_path: Path, output_dir: Path) -> bool:
        """处理单个报告文件并创建向量索引
        
        Args:
            report_file_path: 报告JSON文件的路径
            output_dir: 输出向量索引文件的目录
            
        Returns:
            bool: 处理是否成功
        """
        self.logger.info(f"处理单个报告文件: {report_file_path}")
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查文件是否存在
        if not report_file_path.exists():
            self.logger.error(f"报告文件不存在: {report_file_path}")
            return False
            
        try:
            # 读取JSON文件
            try:
                with open(report_file_path, 'r', encoding='utf-8') as file:
                    report_data = json.load(file)
            except json.JSONDecodeError as e:
                self.logger.error(f"解析JSON文件失败 {report_file_path}: {str(e)}")
                return False
            except Exception as e:
                self.logger.error(f"读取文件失败 {report_file_path}: {str(e)}")
                return False
            
            # 处理报告
            try:
                index = self._process_report(report_data, str(report_file_path))
            except Exception as e:
                self.logger.error(f"处理报告失败 {report_file_path}: {str(e)}")
                return False
            
            # 获取SHA1名称
            if "metainfo" not in report_data or "sha1_name" not in report_data["metainfo"]:
                # 使用文件名作为备用
                sha1_name = report_file_path.stem
                self.logger.warning(f"在 {report_file_path} 中找不到SHA1名称，使用文件名")
            else:
                sha1_name = report_data["metainfo"]["sha1_name"]
            
            # 保存索引
            try:
                faiss_file_path = output_dir / f"{sha1_name}.faiss"
                faiss.write_index(index, str(faiss_file_path))
                self.logger.info(f"成功处理并保存索引: {report_file_path}")
                return True
            except Exception as e:
                self.logger.error(f"保存索引失败 {report_file_path}: {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"处理文件时出错 {report_file_path}: {str(e)}")
            return False
            
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """处理所有报告并创建向量索引
        
        Args:
            all_reports_dir: 包含所有报告JSON文件的目录
            output_dir: 输出向量索引文件的目录
        """
        # 确保目录存在
        if not all_reports_dir.exists():
            self.logger.error(f"报告目录不存在: {all_reports_dir}")
            raise FileNotFoundError(f"报告目录不存在: {all_reports_dir}")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有JSON文件
        all_report_paths = list(all_reports_dir.glob("*.json"))
        if not all_report_paths:
            self.logger.warning(f"在 {all_reports_dir} 中没有找到JSON文件")
            return

        self.logger.info(f"处理 {len(all_report_paths)} 个报告文件")
        
        success_count = 0
        failed_files = []
        
        for report_path in tqdm(all_report_paths, desc="处理报告"):
            try:
                # 处理单个报告文件
                if self.process_single_report(report_path, output_dir):
                    success_count += 1
                else:
                    failed_files.append(report_path)
            except Exception as e:
                self.logger.error(f"处理文件时出错 {report_path}: {str(e)}")
                failed_files.append(report_path)
        
        if failed_files:
            failed_count = len(failed_files)
            self.logger.warning(f"处理了 {len(all_report_paths)} 个文件，其中 {failed_count} 个处理失败")
            if failed_count == len(all_report_paths):
                self.logger.error("所有文件处理失败")
                raise RuntimeError("所有文件处理失败")
        else:
            self.logger.info(f"成功处理 {success_count} 个文件")
            
        return success_count, len(failed_files)