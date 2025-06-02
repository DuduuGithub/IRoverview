import os
import time
import logging
import re
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Union, Any
import warnings
import traceback
from datetime import datetime
import hashlib

# from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
# from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult

# 添加 PyPDF2 导入
try:
    import PyPDF2
except ImportError:
    warnings.warn("PyPDF2 未安装，本地解析功能将不可用。请使用 pip install PyPDF2 安装。")

_log = logging.getLogger(__name__)

def _process_chunk(pdf_paths, pdf_backend, output_dir, num_threads, metadata_lookup, debug_data_path):
    """Helper function to process a chunk of PDFs in a separate process."""
    # Create a new parser instance for this process
    parser = PDFParser(
        pdf_backend=pdf_backend,
        output_dir=output_dir,
        num_threads=num_threads,
        csv_metadata_path=None,  # Metadata lookup is passed directly
        first_page_only=True,  # 默认只处理第一页
        use_local_parser=True  # 使用本地解析器
    )
    parser.metadata_lookup = metadata_lookup
    parser.debug_data_path = debug_data_path
    parser.parse_and_export(pdf_paths)
    return f"Processed {len(pdf_paths)} PDFs."

class PDFParser:
    def __init__(
        self,
        pdf_backend=DoclingParseV2DocumentBackend,
        output_dir: Path = Path("./parsed_pdfs"),
        num_threads: int = None,
        csv_metadata_path: Path = None,
        first_page_only: bool = True,  # 默认只处理第一页
        use_local_parser: bool = True,  # 使用本地解析器
    ):
        self.pdf_backend = pdf_backend
        # 确保output_dir是字符串
        self.output_dir = str(output_dir) if output_dir else "./parsed_pdfs"
        self.num_threads = num_threads
        self.metadata_lookup = {}
        self.debug_data_path = None
        self.first_page_only = first_page_only  # 保存为实例属性
        self.use_local_parser = use_local_parser  # 保存为实例属性
        
        if csv_metadata_path:
            self.metadata_lookup = self._parse_csv_metadata(csv_path=csv_metadata_path)
        
        # 创建文档转换器
        self.doc_converter = self._create_document_converter()

        if self.num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)

    @staticmethod
    def _parse_csv_metadata(csv_path: Path) -> dict:
        """Parse CSV file and create a lookup dictionary with sha1 as key."""
        import csv
        metadata_lookup = {}
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Handle both old and new CSV formats for company name
                company_name = row.get('company_name', row.get('name', '')).strip('"')
                metadata_lookup[row['sha1']] = {
                    'company_name': company_name
                }
        return metadata_lookup

    def _create_document_converter(self) -> "DocumentConverter": # type: ignore
        """Creates and returns a DocumentConverter with default pipeline options."""
        from docling.document_converter import DocumentConverter, FormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        
        pipeline_options = PdfPipelineOptions()
        
        # 禁用OCR功能，加快处理速度
        pipeline_options.do_ocr = False
        _log.info("OCR功能已禁用，以提高处理速度")
        
        # 禁用表格结构识别，避免使用TableModel04_rs模型
        pipeline_options.do_table_structure = False
        _log.info("表格结构识别功能已禁用，不再需要TableModel04_rs模型")
        
        # 记录页面处理设置
        if self.first_page_only:
            _log.info("设置仅处理第一页 - 将在后续处理中实现")
            
            # 注意：我们不再尝试设置页面范围属性，
            # 而是在process_documents方法中实现页面过滤
        
        format_options = {
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
                backend=self.pdf_backend
            )
        }
        
        return DocumentConverter(format_options=format_options)

    def convert_documents(self, input_doc_paths: List[Path]) -> Iterable[ConversionResult]:
        conv_results = self.doc_converter.convert_all(source=input_doc_paths)
        return conv_results
    
    def process_documents(self, conv_results: Iterable[ConversionResult]):
        if self.output_dir is not None:
            # 确保output_dir存在
            os.makedirs(self.output_dir, exist_ok=True)
        success_count = 0
        failure_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                processor = JsonReportProcessor(metadata_lookup=self.metadata_lookup, debug_data_path=self.debug_data_path)
                
                # Normalize the document data to ensure sequential pages
                data = conv_res.document.export_to_dict()
                
                # 如果设置了只处理第一页，则在这里过滤内容
                if self.first_page_only and 'content' in data:
                    _log.info("只保留PDF的第一页内容")
                    # 查找第一页（page=1，页码通常从1开始）
                    first_page = next((page for page in data['content'] if page.get('page') == 1), None)
                    if first_page:
                        # 只保留第一页
                        data['content'] = [first_page]
                    else:
                        _log.warning("未找到第一页内容，将处理所有页面")
                
                normalized_data = self._normalize_page_sequence(data)
                
                processed_report = processor.assemble_report(conv_res, normalized_data)
                doc_filename = conv_res.input.file.stem
                if self.output_dir is not None:
                    # 使用os.path.join而不是/操作符
                    output_file_path = os.path.join(self.output_dir, f"{doc_filename}.json")
                    with open(output_file_path, "w", encoding="utf-8") as fp:
                        json.dump(processed_report, fp, indent=2, ensure_ascii=False)
            else:
                failure_count += 1
                _log.info(f"Document {conv_res.input.file} failed to convert.")

        _log.info(f"Processed {success_count + failure_count} docs, of which {failure_count} failed")
        return success_count, failure_count

    def _normalize_page_sequence(self, data: dict) -> dict:
        """Ensure that page numbers in content are sequential by filling gaps with empty pages."""
        if 'content' not in data:
            return data
        
        # Create a copy of the data to modify
        normalized_data = data.copy()
        
        # Get existing page numbers and find max page
        existing_pages = {page['page'] for page in data['content']}
        max_page = max(existing_pages)
        
        # Create template for empty page
        empty_page_template = {
            "content": [],
            "page_dimensions": {}  # or some default dimensions if needed
        }
        
        # Create new content array with all pages
        new_content = []
        for page_num in range(1, max_page + 1):
            # Find existing page or create empty one
            page_content = next(
                (page for page in data['content'] if page['page'] == page_num),
                {"page": page_num, **empty_page_template}
            )
            new_content.append(page_content)
        
        normalized_data['content'] = new_content
        return normalized_data

    def parse_with_pypdf(self, input_doc_paths: List[Path]) -> List[Dict]:
        """
        使用PyPDF2解析PDF文件，避免依赖在线模型
        
        Args:
            input_doc_paths: 输入PDF文件路径列表
            
        Returns:
            解析结果列表
        """
        if not input_doc_paths:
            return []
            
        _log.info(f"使用PyPDF2本地解析器处理 {len(input_doc_paths)} 个文档")
        
        results = []
        success_docs = []
        failed_docs = []
        
        for doc_path in input_doc_paths:
            try:
                # 确保doc_path是Path对象
                if not isinstance(doc_path, Path):
                    doc_path = Path(doc_path)
                
                _log.info(f"处理文档: {doc_path}")
                
                # 读取PDF文件
                with open(doc_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    # 计算SHA1作为唯一标识符
                    file_hash = hashlib.sha1()
                    file.seek(0)
                    while chunk := file.read(8192):
                        file_hash.update(chunk)
                    sha1_name = file_hash.hexdigest()
                    
                    # 准备页面内容
                    pages = []
                    
                    # 如果只处理第一页
                    page_range = range(1) if self.first_page_only else range(total_pages)
                    
                    for i in page_range:
                        page = pdf_reader.pages[i]
                        text = page.extract_text() or f"页面 {i+1} 无可提取文本"
                        pages.append({
                            "page": i + 1,
                            "text": text
                        })
                    
                    # 创建结果文档
                    result = {
                        "metainfo": {
                            "sha1_name": sha1_name,
                            "file_name": doc_path.name,
                            "creation_time": datetime.now().isoformat(),
                            "total_pages": total_pages,
                            "processed_pages": len(pages)
                        },
                        "content": {
                            "pages": pages
                        }
                    }
                    
                    # 生成输出路径
                    output_path = os.path.join(self.output_dir, f"{sha1_name}.json")
                    
                    # 保存结果
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                        
                    _log.info(f"成功处理文档: {doc_path}, 保存到 {output_path}")
                    results.append(result)
                    success_docs.append(doc_path)
                    
            except Exception as e:
                _log.error(f"处理文档 {doc_path} 时出错: {str(e)}")
                traceback.print_exc()
                failed_docs.append(doc_path)
                
        _log.info(f"处理完成: 成功 {len(success_docs)} 个, 失败 {len(failed_docs)} 个")
        
        if failed_docs:
            _log.error(f"无法处理 {len(failed_docs)} 个文档，共 {len(input_doc_paths)} 个。")
            _log.error("失败的文档路径:")
            for doc in failed_docs:
                _log.error(str(doc))
            
            # 如果所有文档都失败了
            if len(failed_docs) == len(input_doc_paths):
                raise RuntimeError(f"所有 {len(input_doc_paths)} 个文档处理失败。")
                
        return results

    def parse_and_export(self, input_doc_paths: List[Path]) -> Tuple[int, int]:
        """
        解析PDF文档并导出为JSON
        
        Args:
            input_doc_paths: 输入PDF文件路径列表
            
        Returns:
            (成功处理数量, 失败处理数量)
        """
        # 确保所有路径都是Path对象
        from pathlib import Path
        input_doc_paths = [Path(path) if not isinstance(path, Path) else path for path in input_doc_paths]
        
        total_docs = len(input_doc_paths)
        _log.info(f"开始处理 {total_docs} 个文档")
        
        if self.use_local_parser:
            # 使用PyPDF2解析
            try:
                results = self.parse_with_pypdf(input_doc_paths)
                return len(results), total_docs - len(results)
            except Exception as e:
                _log.error(f"使用本地解析器处理PDF时出错: {str(e)}")
                traceback.print_exc()
                raise
        else:
            # 使用原有的docling解析
            try:
                from docling.document_converter import DocumentConverter, DocumentFormat
                from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
                
                # 获取文档路径
                doc_paths = [str(path) for path in input_doc_paths]
                
                # 创建转换器
                converter = DocumentConverter()
                
                # 如果需要限制页面范围且仅处理第一页
                if self.first_page_only:
                    self.doc_converter.pipeline_options.page_range = (0, 0)  # 只处理第一页
                
                # 执行转换
                conv_results = list(converter.convert_all(
                    doc_paths=doc_paths,
                    doc_format=DocumentFormat.PDF,
                    pipeline_options=self.doc_converter.pipeline_options,
                    raises_on_error=False
                ))
                
                # 处理文档
                doc_success, doc_failure = self.process_documents(conv_results=conv_results)
                
                # 检查结果
                if doc_failure > 0:
                    _log.error(f"无法转换 {doc_failure} 个文档，共 {doc_failure + doc_success} 个。")
                    if doc_failure + doc_success == 0:
                        raise RuntimeError("未能处理任何文档")
                    if doc_success == 0:
                        raise RuntimeError(f"所有 {total_docs} 个文档处理失败。")
                
                return doc_success, doc_failure
                
            except ImportError as e:
                _log.error(f"导入docling库时出错: {str(e)}")
                _log.info("回退到本地解析器")
                self.use_local_parser = True
                return self.parse_and_export(input_doc_paths)
            except Exception as e:
                _log.error(f"处理PDF文件时出错: {str(e)}")
                traceback.print_exc()
                raise

    def parse_and_export_parallel(
        self,
        input_doc_paths: List[Path] = None,
        doc_dir: Path = None,
        optimal_workers: int = 10,
        chunk_size: int = None
    ):
        """Parse PDF files in parallel using multiple processes.
        
        Args:
            input_doc_paths: List of paths to PDF files to process
            doc_dir: Directory containing PDF files (used if input_doc_paths is None)
            optimal_workers: Number of worker processes to use. If None, uses CPU count.
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Get input paths if not provided
        if input_doc_paths is None and doc_dir is not None:
            input_doc_paths = list(doc_dir.glob("*.pdf"))

        total_pdfs = len(input_doc_paths)
        _log.info(f"Starting parallel processing of {total_pdfs} documents")
        
        cpu_count = multiprocessing.cpu_count()
        
        # Calculate optimal workers if not specified
        if optimal_workers is None:
            optimal_workers = min(cpu_count, total_pdfs)
        
        if chunk_size is None:
            # Calculate chunk size (ensure at least 1)
            chunk_size = max(1, total_pdfs // optimal_workers)
        
        # Split documents into chunks
        chunks = [
            input_doc_paths[i : i + chunk_size]
            for i in range(0, total_pdfs, chunk_size)
        ]

        start_time = time.time()
        processed_count = 0
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Schedule all tasks
            futures = [
                executor.submit(
                    _process_chunk,
                    chunk,
                    self.pdf_backend,
                    self.output_dir,
                    self.num_threads,
                    self.metadata_lookup,
                    self.debug_data_path
                )
                for chunk in chunks
            ]
            
            # Wait for completion and log results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_count += int(result.split()[1])  # Extract number from "Processed X PDFs"
                    _log.info(f"{'#'*50}\n{result} ({processed_count}/{total_pdfs} total)\n{'#'*50}")
                except Exception as e:
                    _log.error(f"Error processing chunk: {str(e)}")
                    raise

        elapsed_time = time.time() - start_time
        _log.info(f"Parallel processing completed in {elapsed_time:.2f} seconds.")


class JsonReportProcessor:
    def __init__(self, metadata_lookup: dict = None, debug_data_path: Path = None):
        self.metadata_lookup = metadata_lookup or {}
        self.debug_data_path = debug_data_path

    def assemble_report(self, conv_result, normalized_data=None):
        """Assemble the report using either normalized data or raw conversion result."""
        data = normalized_data if normalized_data is not None else conv_result.document.export_to_dict()
        assembled_report = {}
        assembled_report['metainfo'] = self.assemble_metainfo(data)
        assembled_report['content'] = self.assemble_content(data)
        assembled_report['tables'] = self.assemble_tables(conv_result.document.tables, data)
        assembled_report['pictures'] = self.assemble_pictures(data)
        self.debug_data(data)
        return assembled_report
    
    def assemble_metainfo(self, data):
        metainfo = {}
        sha1_name = data['origin']['filename'].rsplit('.', 1)[0]
        metainfo['sha1_name'] = sha1_name
        metainfo['pages_amount'] = len(data.get('pages', []))
        metainfo['text_blocks_amount'] = len(data.get('texts', []))
        metainfo['tables_amount'] = len(data.get('tables', []))
        metainfo['pictures_amount'] = len(data.get('pictures', []))
        metainfo['equations_amount'] = len(data.get('equations', []))
        metainfo['footnotes_amount'] = len([t for t in data.get('texts', []) if t.get('label') == 'footnote'])
        
        # Add CSV metadata if available
        if self.metadata_lookup and sha1_name in self.metadata_lookup:
            csv_meta = self.metadata_lookup[sha1_name]
            metainfo['company_name'] = csv_meta['company_name']
            
        return metainfo

    def process_table(self, table_data):
        # Implement your table processing logic here
        return 'processed_table_content'

    def debug_data(self, data):
        if self.debug_data_path is None:
            return
        doc_name = data['name']
        path = self.debug_data_path / f"{doc_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)    
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def expand_groups(self, body_children, groups):
        expanded_children = []

        for item in body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'groups':
                    group = groups[ref_num]
                    group_id = ref_num
                    group_name = group.get('name', '')
                    group_label = group.get('label', '')

                    for child in group['children']:
                        child_copy = child.copy()
                        child_copy['group_id'] = group_id
                        child_copy['group_name'] = group_name
                        child_copy['group_label'] = group_label
                        expanded_children.append(child_copy)
                else:
                    expanded_children.append(item)
            else:
                expanded_children.append(item)

        return expanded_children
    
    def _process_text_reference(self, ref_num, data):
        """Helper method to process text references and create content items.
        
        Args:
            ref_num (int): Reference number for the text item
            data (dict): Document data dictionary
            
        Returns:
            dict: Processed content item with text information
        """
        text_item = data['texts'][ref_num]
        item_type = text_item['label']
        content_item = {
            'text': text_item.get('text', ''),
            'type': item_type,
            'text_id': ref_num
        }
        
        # Add 'orig' field only if it differs from 'text'
        orig_content = text_item.get('orig', '')
        if orig_content != text_item.get('text', ''):
            content_item['orig'] = orig_content

        # Add additional fields if they exist
        if 'enumerated' in text_item:
            content_item['enumerated'] = text_item['enumerated']
        if 'marker' in text_item:
            content_item['marker'] = text_item['marker']
            
        return content_item
    
    def assemble_content(self, data):
        pages = {}
        # Expand body children to include group references
        body_children = data['body']['children']
        groups = data.get('groups', [])
        expanded_body_children = self.expand_groups(body_children, groups)

        # Process body content
        for item in expanded_body_children:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)

                if ref_type == 'texts':
                    text_item = data['texts'][ref_num]
                    content_item = self._process_text_reference(ref_num, data)

                    # Add group information if available
                    if 'group_id' in item:
                        content_item['group_id'] = item['group_id']
                        content_item['group_name'] = item['group_name']
                        content_item['group_label'] = item['group_label']

                    # Get page number from prov
                    if 'prov' in text_item and text_item['prov']:
                        page_num = text_item['prov'][0]['page_no']

                        # Initialize page if not exists
                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': text_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)

                elif ref_type == 'tables':
                    table_item = data['tables'][ref_num]
                    content_item = {
                        'type': 'table',
                        'table_id': ref_num
                    }

                    if 'prov' in table_item and table_item['prov']:
                        page_num = table_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': table_item['prov'][0].get('bbox', {})
                            }

                        pages[page_num]['content'].append(content_item)
                
                elif ref_type == 'pictures':
                    picture_item = data['pictures'][ref_num]
                    content_item = {
                        'type': 'picture',
                        'picture_id': ref_num
                    }
                    
                    if 'prov' in picture_item and picture_item['prov']:
                        page_num = picture_item['prov'][0]['page_no']

                        if page_num not in pages:
                            pages[page_num] = {
                                'page': page_num,
                                'content': [],
                                'page_dimensions': picture_item['prov'][0].get('bbox', {})
                            }
                        
                        pages[page_num]['content'].append(content_item)

        sorted_pages = [pages[page_num] for page_num in sorted(pages.keys())]
        return sorted_pages

    def assemble_tables(self, tables, data):
        assembled_tables = []
        for i, table in enumerate(tables):
            table_json_obj = table.model_dump()
            table_md = self._table_to_md(table_json_obj)
            table_html = table.export_to_html()
            
            table_data = data['tables'][i]
            table_page_num = table_data['prov'][0]['page_no']
            table_bbox = table_data['prov'][0]['bbox']
            table_bbox = [
                table_bbox['l'],
                table_bbox['t'], 
                table_bbox['r'],
                table_bbox['b']
            ]
            
            # Get rows and columns from the table data structure
            nrows = table_data['data']['num_rows']
            ncols = table_data['data']['num_cols']

            ref_num = table_data['self_ref'].split('/')[-1]
            ref_num = int(ref_num)

            table_obj = {
                'table_id': ref_num,
                'page': table_page_num,
                'bbox': table_bbox,
                '#-rows': nrows,
                '#-cols': ncols,
                'markdown': table_md,
                'html': table_html,
                'json': table_json_obj
            }
            assembled_tables.append(table_obj)
        return assembled_tables

    def _table_to_md(self, table):
        # Extract text from grid cells
        table_data = []
        for row in table['data']['grid']:
            table_row = [cell['text'] for cell in row]
            table_data.append(table_row)
        
        # Check if the table has headers
        if len(table_data) > 1 and len(table_data[0]) > 0:
            try:
                md_table = tabulate(
                    table_data[1:], headers=table_data[0], tablefmt="github"
                )
            except ValueError:
                md_table = tabulate(
                    table_data[1:],
                    headers=table_data[0],
                    tablefmt="github",
                    disable_numparse=True,
                )
        else:
            md_table = tabulate(table_data, tablefmt="github")
        
        return md_table

    def assemble_pictures(self, data):
        assembled_pictures = []
        for i, picture in enumerate(data['pictures']):
            children_list = self._process_picture_block(picture, data)
            
            ref_num = picture['self_ref'].split('/')[-1]
            ref_num = int(ref_num)
            
            picture_page_num = picture['prov'][0]['page_no']
            picture_bbox = picture['prov'][0]['bbox']
            picture_bbox = [
                picture_bbox['l'],
                picture_bbox['t'], 
                picture_bbox['r'],
                picture_bbox['b']
            ]
            
            picture_obj = {
                'picture_id': ref_num,
                'page': picture_page_num,
                'bbox': picture_bbox,
                'children': children_list,
            }
            assembled_pictures.append(picture_obj)
        return assembled_pictures
    
    def _process_picture_block(self, picture, data):
        children_list = []
        
        for item in picture['children']:
            if isinstance(item, dict) and '$ref' in item:
                ref = item['$ref']
                ref_type, ref_num = ref.split('/')[-2:]
                ref_num = int(ref_num)
                
                if ref_type == 'texts':
                    content_item = self._process_text_reference(ref_num, data)
                        
                    children_list.append(content_item)

        return children_list
