"""
向量数据库创建工具
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# 导入数据库组件
from Database.config import db
from Database.model import Work, Author, WorkConcept, Concept, WorkAuthorship
from flask import Flask

# 导入RAG组件 - 将相对导入修改为绝对导入
from app.app_blueprint.rag_mcp.src.text_splitter import TextSplitter
from app.app_blueprint.rag_mcp.src.ingestion import VectorDBIngestor, BM25Ingestor

def create_app():
    """创建Flask应用实例"""
    app = Flask(__name__)
    app.config.from_object('Database.config')
    db.init_app(app)
    return app

def _parse_inverted_index(inverted_index):
    """将倒排索引转换为普通文本
    
    Args:
        inverted_index: JSON格式的倒排索引
        
    Returns:
        str: 拼接后的普通文本
    """
    if not isinstance(inverted_index, dict):
        return str(inverted_index)
    
    # 获取所有单词和它们的位置
    words_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    words_positions.append((word, pos))
    
    # 按照位置排序
    words_positions.sort(key=lambda x: x[1])
    
    # 拼接文本
    text = ' '.join([wp[0] for wp in words_positions])
    return text

def get_works_from_db(limit=1000, offset=0):
    """从数据库获取学术作品数据"""
    try:
        works = Work.query.limit(limit).offset(offset).all()
        result = []
        
        for work in works:
            # 获取作品的抽象内容
            abstract = _parse_inverted_index(work.abstract_inverted_index) if work.abstract_inverted_index else ""
            
            # 获取作者信息
            authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
            authors = []
            for authorship in authorships:
                if authorship.author_id:
                    author = Author.query.get(authorship.author_id)
                    if author:
                        authors.append({
                            'id': author.id,
                            'name': author.display_name,
                            'position': authorship.author_position
                        })
            
            # 获取概念信息
            work_concepts = WorkConcept.query.filter_by(work_id=work.id).all()
            concepts = []
            for work_concept in work_concepts:
                concept = Concept.query.get(work_concept.concept_id)
                if concept:
                    concepts.append({
                        'id': concept.id,
                        'name': concept.display_name,
                        'score': work_concept.score
                    })
            
            # 构建文档数据
            work_data = {
                'id': work.id,
                'title': work.title or work.display_name or "",
                'abstract': abstract,
                'publication_year': work.publication_year,
                'publication_date': work.publication_date.isoformat() if work.publication_date else None,
                'type': work.type,
                'doi': work.doi,
                'language': work.language or "en",
                'cited_by_count': work.cited_by_count,
                'authors': authors,
                'concepts': concepts
            }
            
            result.append(work_data)
        
        logger.info(f"成功获取{len(result)}个作品")
        return result
    except Exception as e:
        logger.error(f"获取作品时发生错误: {str(e)}")
        return []

def prepare_documents_for_vectorization(works):
    """准备文档用于向量化处理"""
    documents = []
    
    for work in works:
        # 构建文本内容
        content = f"Title: {work['title']}\n\n"
        
        if work['authors']:
            authors_str = ", ".join([author.get('name', '') for author in work['authors']])
            content += f"Authors: {authors_str}\n\n"
        
        if work['abstract']:
            content += f"Abstract: {work['abstract']}\n\n"
        
        if work['concepts']:
            concepts_str = ", ".join([concept.get('name', '') for concept in work['concepts']])
            content += f"Keywords: {concepts_str}\n\n"
        
        # 构建文档数据
        document = {
            "metainfo": {
                "sha1_name": work['id'],
                "company_name": work['id']  # 使用ID作为公司名，以便兼容现有系统
            },
            "content": {
                "pages": [
                    {
                        "page": 0,
                        "text": content
                    }
                ]
            }
        }
        
        documents.append(document)
    
    return documents

def index_works(limit=1000, output_dir=None, include_bm25=True):
    """从数据库索引学术作品到向量存储
    
    Args:
        limit: 最大索引数量
        output_dir: 输出目录
        include_bm25: 是否包含BM25索引
    
    Returns:
        索引结果统计
    """
    logger.info(f"开始索引作品，最大数量: {limit}")
    
    # 创建Flask应用上下文
    app = create_app()
    
    try:
        with app.app_context():
            # 创建目录结构
            if output_dir is None:
                # 将输出目录设为rag_mcp目录下的data
                output_dir = Path(__file__).parent.parent / "data"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 设置子目录
            raw_documents_dir = output_dir / "raw_documents"
            chunked_documents_dir = output_dir / "chunked_documents"
            vector_db_dir = output_dir / "vector_dbs"
            bm25_db_dir = output_dir / "bm25_dbs"
            
            # 获取作品数据
            logger.info("从数据库获取作品...")
            works = get_works_from_db(limit=limit)
            logger.info(f"获取到 {len(works)} 个作品")
            
            if not works:
                logger.warning("没有找到作品，索引终止")
                return {
                    'success': False,
                    'message': '没有找到作品',
                    'indexed_count': 0
                }
            
            # 准备文档
            logger.info("准备文档...")
            documents = prepare_documents_for_vectorization(works)
            
            # 创建subset.csv文件
            subset_data = []
            for work in works:
                subset_data.append({
                    "sha1": work["id"],
                    "company_name": work["id"]
                })
            
            subset_df = pd.DataFrame(subset_data)
            subset_path = output_dir / "subset.csv"
            subset_df.to_csv(subset_path, index=False)
            
            # 保存原始文档
            logger.info("保存原始文档...")
            raw_documents_dir.mkdir(parents=True, exist_ok=True)
            for doc in documents:
                doc_id = doc["metainfo"]["sha1_name"]
                with open(raw_documents_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
            
            # 文本分块
            logger.info("文本分块...")
            chunked_documents_dir.mkdir(parents=True, exist_ok=True)
            text_splitter = TextSplitter()
            text_splitter.split_all_reports(raw_documents_dir, chunked_documents_dir)
            
            # 创建向量数据库
            logger.info("创建向量数据库...")
            vector_db_dir.mkdir(parents=True, exist_ok=True)
            vdb_ingestor = VectorDBIngestor()
            vdb_ingestor.process_reports(chunked_documents_dir, vector_db_dir)
            
            # 创建BM25索引
            if include_bm25:
                logger.info("创建BM25索引...")
                bm25_db_dir.mkdir(parents=True, exist_ok=True)
                bm25_ingestor = BM25Ingestor()
                bm25_ingestor.process_reports(chunked_documents_dir, bm25_db_dir)
            
            # 整理结果
            result = {
                'success': True,
                'message': f'成功索引 {len(works)} 个作品',
                'indexed_count': len(works),
                'output_dirs': {
                    'raw_documents': str(raw_documents_dir),
                    'chunked_documents': str(chunked_documents_dir),
                    'vector_dbs': str(vector_db_dir),
                }
            }
            
            if include_bm25:
                result['output_dirs']['bm25_dbs'] = str(bm25_db_dir)
            
            logger.info(f"索引完成，成功索引 {len(works)} 个作品")
            return result
    
    except Exception as e:
        logger.error(f"索引过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'success': False,
            'message': f'索引失败: {str(e)}',
            'indexed_count': 0
        }
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='IRoverview向量数据库创建工具')
    
    # 参数
    parser.add_argument('--limit', type=int, default=1000, help='最大索引数量')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--skip-bm25', action='store_true', help='跳过BM25索引创建')
    
    args = parser.parse_args()
    
    # 执行索引
    result = index_works(
        limit=args.limit,
        output_dir=args.output_dir,
        include_bm25=not args.skip_bm25
    )
    
    # 打印结果
    if result['success']:
        logger.info(f"索引成功: {result['message']}")
        logger.info(f"索引数量: {result['indexed_count']}")
        if 'output_dirs' in result:
            logger.info("输出目录:")
            for key, path in result['output_dirs'].items():
                logger.info(f"  {key}: {path}")
    else:
        logger.error(f"索引失败: {result['message']}")

if __name__ == "__main__":
    main()

