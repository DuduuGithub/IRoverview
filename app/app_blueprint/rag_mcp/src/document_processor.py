"""
DocumentProcessor类

提供PDF文档处理功能，包括PDF解析、分块和存储。
集成pdf_parsing.py中的功能，作为API接口的后端。
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .pdf_parsing import PDFParser
from .text_splitter import TextSplitter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """处理PDF文档并将其分块存储"""
    
    def __init__(self):
        """初始化DocumentProcessor"""
        logger.info("初始化DocumentProcessor")
    
    def process_pdf(
        self,
        pdf_path: str,
        output_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        处理PDF文档，提取文本，分块并保存结果
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出JSON文件路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            
        Returns:
            处理结果字典，包含页面和块信息
        """
        logger.info(f"开始处理PDF: {pdf_path}")
        
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 解析PDF
        parser = PDFParser(output_dir=output_dir)
        pdf_paths = [Path(pdf_path)]
        
        # 提取文档ID
        document_id = Path(pdf_path).stem
        if '_' in document_id:
            # 处理"uuid_filename.pdf"格式
            parts = document_id.split('_', 1)
            if len(parts) > 1:
                document_id = parts[0]
        
        # 处理文档
        try:
            logger.info(f"解析PDF文件: {pdf_path}")
            parser.parse_and_export(pdf_paths)
            
            # 找到解析后的JSON文件
            parsed_json_path = output_dir / f"{Path(pdf_path).stem}.json"
            if not parsed_json_path.exists():
                raise FileNotFoundError(f"找不到解析后的JSON文件: {parsed_json_path}")
            
            # 读取解析后的JSON
            with open(parsed_json_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            # 创建文本分割器
            logger.info(f"分割文本，块大小: {chunk_size}，重叠: {chunk_overlap}")
            splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # 提取所有页面的文本
            pages = []
            all_text = ""
            
            # 从document_data中提取页面
            for page in document_data.get('content', {}).get('pages', []):
                page_num = page.get('page', 0)
                page_text = page.get('text', '')
                all_text += page_text + "\n\n"
                pages.append({
                    "page": page_num,
                    "text": page_text
                })
            
            # 分割文本
            chunks = splitter.split_text(all_text)
            chunked_texts = []
            
            for i, chunk in enumerate(chunks):
                chunked_texts.append({
                    "chunk_id": i,
                    "text": chunk
                })
            
            # 准备结果
            result = {
                "document_id": document_id,
                "pages": pages,
                "chunks": chunked_texts
            }
            
            # 保存处理后的结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"PDF处理完成: {pdf_path} -> {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"处理PDF时发生错误: {str(e)}")
            # 创建基本结果以避免错误
            result = {
                "document_id": document_id,
                "pages": [],
                "chunks": []
            }
            
            # 尝试保存基本结果
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as write_error:
                logger.error(f"保存结果时发生错误: {str(write_error)}")
            
            return result 