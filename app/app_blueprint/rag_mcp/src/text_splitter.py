import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from tqdm import tqdm

class TextSplitter():
    def _get_serialized_tables_by_page(self, tables: List[Dict]) -> Dict[int, List[Dict]]:
        """Group serialized tables by page number"""
        tables_by_page = {}
        for table in tables:
            if 'serialized' not in table:
                continue
                
            page = table['page']
            if page not in tables_by_page:
                tables_by_page[page] = []
            
            table_text = "\n".join(
                block["information_block"] 
                for block in table["serialized"]["information_blocks"]
            )
            
            tables_by_page[page].append({
                "page": page,
                "text": table_text,
                "table_id": table["table_id"],
                "length_tokens": self.count_tokens(table_text)
            })
            
        return tables_by_page

    def _split_report(self, file_content: Dict[str, any], serialized_tables_report_path: Optional[Path] = None) -> Dict[str, any]:
        """Split report into chunks, preserving markdown tables in content and optionally including serialized tables.
        
        Args:
            file_content: 报告JSON内容
            serialized_tables_report_path: 可选的序列化表格文件路径
            
        Returns:
            更新后的报告内容，包含分块后的文本
        """
        logger = logging.getLogger(__name__)
        
        chunks = []
        chunk_id = 0
        
        # 处理序列化表格文件
        tables_by_page = {}
        if serialized_tables_report_path is not None and serialized_tables_report_path.exists():
            try:
                with open(serialized_tables_report_path, 'r', encoding='utf-8') as f:
                    parsed_report = json.load(f)
                tables_by_page = self._get_serialized_tables_by_page(parsed_report.get('tables', []))
            except Exception as e:
                logger.warning(f"无法加载序列化表格文件 {serialized_tables_report_path}: {str(e)}")
        
        # 处理内容结构
        try:
            # 检查content结构是否存在
            if 'content' not in file_content:
                logger.warning("文件内容中缺少'content'字段，创建空白结构")
                file_content['content'] = {'pages': []}
            
            # 确保content是字典类型
            if not isinstance(file_content['content'], dict):
                logger.warning("content字段不是字典类型，尝试转换")
                content_text = str(file_content['content'])
                file_content['content'] = {'pages': [{'page': 1, 'text': content_text}]}
            
            # 检查content中是否包含pages
            if 'pages' not in file_content['content']:
                logger.warning("无法在内容中找到pages结构，创建单一页面")
                
                # 创建一个包含所有内容的单一页面
                all_text = ""
                
                # 尝试从content中提取文本内容
                if 'content' in file_content['content'] and isinstance(file_content['content']['content'], list):
                    # 如果有嵌套的content
                    try:
                        pages = []
                        for page_idx, page in enumerate(file_content['content']['content']):
                            if isinstance(page, dict) and 'page' in page:
                                pages.append(page)
                            else:
                                # 创建新的页面结构
                                pages.append({'page': page_idx + 1, 'text': str(page)})
                        file_content['content']['pages'] = pages
                    except Exception as e:
                        logger.warning(f"处理嵌套content时出错: {str(e)}")
                        all_text = json.dumps(file_content.get('content', {}))
                        file_content['content']['pages'] = [{'page': 1, 'text': all_text}]
                else:
                    # 如果没有嵌套的content
                    all_text = json.dumps(file_content.get('content', {}))
                    file_content['content']['pages'] = [{'page': 1, 'text': all_text}]
            
            # 确保pages是列表类型
            if not isinstance(file_content['content']['pages'], list):
                logger.warning("pages字段不是列表类型，尝试转换")
                pages_text = str(file_content['content']['pages'])
                file_content['content']['pages'] = [{'page': 1, 'text': pages_text}]
            
            # 处理每个页面
            for page_idx, page in enumerate(file_content['content']['pages']):
                # 确保page是字典类型
                if not isinstance(page, dict):
                    page = {'page': page_idx + 1, 'text': str(page)}
                    file_content['content']['pages'][page_idx] = page
                
                # 获取页码
                page_num = page.get('page', page_idx + 1)
                
                # 确保有文本字段
                if 'text' not in page:
                    # 尝试从其他字段获取文本
                    if isinstance(page, dict):
                        page_text = " ".join([str(v) for k, v in page.items() if k != 'page' and v])
                    else:
                        page_text = str(page)
                    page['text'] = page_text
                    logger.warning(f"为页面 {page_num} 创建了合成文本")
                
                # 分割页面文本
                page_chunks = self._split_page(page)
                for chunk in page_chunks:
                    chunk['id'] = chunk_id
                    chunk['type'] = 'content'
                    chunk_id += 1
                    chunks.append(chunk)
                
                # 添加序列化表格(如果存在)
                if tables_by_page and page_num in tables_by_page:
                    for table in tables_by_page[page_num]:
                        table['id'] = chunk_id
                        table['type'] = 'serialized_table'
                        chunk_id += 1
                        chunks.append(table)
        
        except Exception as e:
            logger.error(f"分割报告时出错: {str(e)}")
            # 创建最小文本块以避免完全失败
            if not chunks and 'content' in file_content:
                try:
                    if isinstance(file_content['content'], dict):
                        simple_text = json.dumps(file_content['content'])[:1000]  # 限制长度
                    else:
                        simple_text = str(file_content['content'])[:1000]  # 限制长度
                        
                    chunks.append({
                        'id': 0,
                        'page': 1,
                        'type': 'content',
                        'text': f"解析错误，部分内容: {simple_text}...",
                        'length_tokens': self.count_tokens(simple_text)
                    })
                except Exception as e:
                    logger.error(f"创建最小文本块失败: {str(e)}")
                    chunks.append({
                        'id': 0,
                        'page': 1,
                        'type': 'content',
                        'text': "内容解析错误",
                        'length_tokens': 3
                    })
        
        # 更新文件内容
        if 'content' not in file_content:
            file_content['content'] = {}
        file_content['content']['chunks'] = chunks
        return file_content

    def count_tokens(self, string: str, encoding_name="o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name)

        tokens = encoding.encode(string)
        token_count = len(tokens)

        return token_count

    def _split_page(self, page: Dict[str, any], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Dict[str, any]]:
        """Split page text into chunks. The original text includes markdown tables."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(page['text'])
        chunks_with_meta = []
        for chunk in chunks:
            chunks_with_meta.append({
                "page": page['page'],
                "length_tokens": self.count_tokens(chunk),
                "text": chunk
            })
        return chunks_with_meta

    def split_all_reports(self, all_report_dir: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None):
        """Split all reports in a directory and save results
        
        Args:
            all_report_dir: Directory with parsed JSON reports
            output_dir: Directory to save the split reports
            serialized_tables_dir: Optional directory with serialized tables
        """
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all JSON files
        all_report_paths = list(all_report_dir.glob("*.json"))
        
        for report_path in tqdm(all_report_paths, desc="Splitting reports"):
            self.split_single_report(report_path, output_dir, serialized_tables_dir)
        
    def split_single_report(self, report_path: Path, output_dir: Path, serialized_tables_dir: Optional[Path] = None) -> bool:
        """拆分单个报告文件并保存结果
        
        Args:
            report_path: 解析后的JSON报告文件路径
            output_dir: 保存拆分报告的目录
            serialized_tables_dir: 可选的序列化表格目录
            
        Returns:
            bool: 处理是否成功
        """
        logger = logging.getLogger(__name__)
        
        # 检查文件是否存在
        if not report_path.exists():
            logger.error(f"报告文件不存在: {report_path}")
            return False
            
        try:
            # 读取JSON文件
            with open(report_path, 'r', encoding='utf-8') as f:
                file_content = json.load(f)
            
            # 获取序列化表格文件路径（如果有）
            serialized_tables_report_path = None
            if serialized_tables_dir:
                # 使用相同的文件名在序列化表格目录中查找
                serialized_tables_report_path = serialized_tables_dir / report_path.name
            
            # 分割文本
            updated_content = self._split_report(file_content, serialized_tables_report_path)
            
            # 保存结果
            output_path = output_dir / report_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(updated_content, f, ensure_ascii=False, indent=2)
                
            logger.info(f"成功分割报告: {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"分割报告时出错 {report_path}: {str(e)}", exc_info=True)
            return False
