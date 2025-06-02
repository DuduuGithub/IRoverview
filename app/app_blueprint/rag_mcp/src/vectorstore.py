"""
向量数据库处理模块

提供向量数据库创建和管理功能，扩展ingestion.py中的功能
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt

# 导入基础实现
from .ingestion import VectorDBIngestor as BaseVectorDBIngestor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDBIngestor(BaseVectorDBIngestor):
    """扩展向量数据库创建工具"""
    
    def _set_up_llm(self):
        """重写基类方法，硬编码API密钥"""
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm
    
    def create_vector_db(
        self,
        documents_dir: Path,
        output_dir: Path,
        document_id: str,
        embedding_model: str = "text-embedding-3-large"
    ) -> Path:
        """
        为单个文档创建向量数据库
        
        Args:
            documents_dir: 文档目录
            output_dir: 输出目录
            document_id: 文档ID
            embedding_model: 嵌入模型名称
            
        Returns:
            向量数据库路径
        """
        logger.info(f"为文档 {document_id} 创建向量数据库")
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找文档
        doc_path = documents_dir / f"{document_id}.json"
        if not doc_path.exists():
            raise FileNotFoundError(f"找不到文档: {doc_path}")
        
        # 读取文档
        with open(doc_path, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
        
        # 重构为满足_process_report的格式
        # 注意：这里我们需要调整document_data的结构以适应process_reports
        
        # 获取所有文本块
        chunks = document_data.get('chunks', [])
        if not chunks:
            raise ValueError(f"文档 {document_id} 没有文本块")
        
        # 构建适合_process_report的格式
        report = {
            'content': {
                'chunks': chunks
            },
            'metainfo': {
                'sha1_name': document_id
            }
        }
        
        # 使用基类的方法处理
        index = self._process_report(report)
        
        # 保存向量数据库
        db_path = output_dir / f"{document_id}.faiss"
        faiss.write_index(index, str(db_path))
        
        logger.info(f"向量数据库创建完成: {db_path}")
        return db_path 