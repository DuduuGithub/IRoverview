import os
import sys
import json
import random
from datetime import datetime, timedelta
import logging
import uuid
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.config import db
from Database.model import (
    Work, Author, WorkAuthorship, SearchSession, 
    SearchResult, UserBehavior, Concept, WorkConcept, WorkReferencedWork
)
from app.app_blueprint.search.search_utils import (
    record_search_session, record_search_results,
    record_document_click, record_dwell_time
)
from flask import Flask

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 减少SQLAlchemy的日志输出
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

class DataImporter:
    def __init__(self, dataset_dir='yale_dataset', items_per_page=10):
        """
        初始化数据导入器
        
        Args:
            dataset_dir: 数据集目录
            items_per_page: 每页显示的文档数量
        """
        # 使用绝对路径
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_dir))
        self.corpus_dir = os.path.join(self.dataset_dir, 'corpus_new')
        self.items_per_page = items_per_page
        
        # 确保目录存在
        if not os.path.exists(self.corpus_dir):
            logger.error(f"数据集目录不存在: {self.corpus_dir}")
            raise FileNotFoundError(f"数据集目录不存在: {self.corpus_dir}")
        
        # 创建Flask应用上下文
        self.app = Flask(__name__)
        self.app.config.from_object('Database.config')
        # 关闭SQLAlchemy的echo模式
        self.app.config['SQLALCHEMY_ECHO'] = False
        db.init_app(self.app)

    def import_corpus(self):
        """导入文献集到数据库"""
        logger.info("开始导入文献集...")
        
        with self.app.app_context():
            try:
                # 读取Parquet文件
                file_path = os.path.join(self.corpus_dir, 'corpus.parquet')
                logger.info(f"尝试读取文件: {file_path}")
                
                if not os.path.exists(file_path):
                    logger.error(f"数据文件不存在: {file_path}")
                    return
                
                # 使用pandas读取Parquet文件
                logger.info("开始读取Parquet文件...")
                df = pd.read_parquet(file_path)
                logger.info(f"DataFrame列名: {df.columns.tolist()}")
                logger.info(f"DataFrame总行数: {len(df)}")
                
                # 检查数据内容
                logger.info("\n检查第一条记录的所有字段:")
                first_row = df.iloc[0]
                for col in df.columns:
                    logger.info(f"{col}: {first_row[col]}")
                    logger.info(f"类型: {type(first_row[col])}")
                
                # 取前400条记录
                df = df.head(400)
                total_rows = len(df)
                logger.info(f"\n将处理前 {total_rows} 条记录")
                
                # 检查citations列的内容
                logger.info("\n检查citations列的内容:")
                for idx, row in df.head(5).iterrows():
                    citations = row.get('citations', None)
                    logger.info(f"\n记录 {idx}:")
                    logger.info(f"citations类型: {type(citations)}")
                    logger.info(f"citations内容: {citations}")
                    
                    if citations is not None:
                        if isinstance(citations, str):
                            logger.info("尝试解析字符串形式的citations...")
                            try:
                                parsed = eval(citations)
                                logger.info(f"解析结果: {parsed}")
                            except:
                                logger.info("eval解析失败，尝试其他方法...")
                                try:
                                    parsed = citations.strip('[]').split(',')
                                    logger.info(f"split解析结果: {parsed}")
                                except:
                                    logger.info("split解析也失败")
                        elif isinstance(citations, (list, tuple)):
                            logger.info(f"citations已经是列表形式: {citations}")
                
                # 第一阶段：导入所有文献基本信息
                logger.info("\n第一阶段：导入文献基本信息...")
                for index, row in df.iterrows():
                    try:
                        # 获取文献ID
                        doc_id = str(row['corpusid'])
                        
                        # 检查文献是否已存在
                        existing_work = Work.query.filter_by(id=doc_id).first()
                        if existing_work:
                            logger.info(f"文献已存在: {doc_id}")
                            continue
                        
                        # 处理引用列表（仅计数）
                        citations = row['citations']
                        # logger.info(f"\n处理文献 {doc_id} 的引用:")
                        # logger.info(f"citations类型: {type(citations)}")
                        # logger.info(f"citations原始值: {citations}")

                        # 处理numpy数组类型的citations
                        if hasattr(citations, '__array__'):  # 检查是否为numpy数组
                            #logger.info("citations是numpy数组，转换为列表")
                            citations = citations.tolist()  # 转换为Python列表
                        elif isinstance(citations, str):
                            try:
                                citations = eval(citations)
                            except:
                                citations = citations.strip('[]').split(',')
                                citations = [cid.strip() for cid in citations if cid.strip()]
                        
                        # 确保所有引用ID都是字符串类型
                        citations = [str(cid) for cid in citations]
                        cited_by_count = len(citations)
                        # logger.info(f"处理后的citations: {citations[:5]}...")  # 只显示前5个
                        # logger.info(f"引用数量: {cited_by_count}")
                        
                        # 创建Work记录
                        work = Work(
                            id=doc_id,
                            title=str(row['title']),
                            display_name=str(row['title']),
                            publication_year=None,  # 数据中没有年份信息
                            cited_by_count=cited_by_count,
                            abstract_inverted_index=str(row['abstract'])
                        )
                        db.session.add(work)
                        
                        # 从标题和摘要中提取关键词作为概念
                        keywords = set()
                        title = str(row['title'])
                        abstract = str(row['abstract'])
                        
                        # 添加标题中的关键词
                        if title:
                            keywords.update(word.lower() for word in title.split() if len(word) > 3)
                        
                        # 添加摘要中的关键词
                        if abstract:
                            keywords.update(word.lower() for word in abstract.split() if len(word) > 3)
                        
                        # 过滤掉常见词和数字
                        keywords = {k for k in keywords if not k.isdigit()}
                        
                        # 为每个关键词创建概念
                        for keyword in keywords:
                            concept = Concept.query.filter_by(display_name=keyword).first()
                            if not concept:
                                concept = Concept(
                                    id=f"C{uuid.uuid4().hex[:8]}",
                                    display_name=keyword,
                                    level=1
                                )
                                db.session.add(concept)
                                db.session.flush()
                            
                            # 创建概念关联
                            work_concept = WorkConcept(
                                work_id=work.id,
                                concept_id=concept.id,
                                score=1.0
                            )
                            db.session.add(work_concept)
                        
                        # 每10条记录提交一次
                        if (index + 1) % 10 == 0:
                            db.session.commit()
                            logger.info(f"已导入 {index + 1}/{total_rows} 篇文献")
                    
                    except Exception as e:
                        logger.error(f"处理第 {index + 1} 条记录时出错: {str(e)}")
                        db.session.rollback()
                        continue
                
                # 提交剩余记录
                db.session.commit()
                
                # 第二阶段：处理引用关系
                logger.info("\n第二阶段：处理引用关系...")
                imported_ids = set(str(row['corpusid']) for _, row in df.iterrows())
                logger.info(f"已导入的文献ID列表: {sorted(list(imported_ids))[:5]}...")
                logger.info(f"已导入的文献ID数量: {len(imported_ids)}")
                
                refs_added = 0
                citations_found = 0
                valid_citations_found = 0
                
                for index, row in df.iterrows():
                    try:
                        doc_id = str(row['corpusid'])
                        citations = row['citations']
                        
                        # 处理numpy数组类型的citations
                        if hasattr(citations, '__array__'):  # 检查是否为numpy数组
                            citations = citations.tolist()  # 转换为Python列表
                        elif isinstance(citations, str):
                            try:
                                citations = eval(citations)
                            except:
                                citations = citations.strip('[]').split(',')
                                citations = [cid.strip() for cid in citations if cid.strip()]
                        
                        # 确保所有引用ID都是字符串类型
                        citations = [str(cid) for cid in citations]
                        citations_found += len(citations)
                        
                        if citations:
                            logger.info(f"\n处理文献 {doc_id} 的引用关系:")
                            logger.info(f"- 引用列表: {citations[:5]}...")
                            logger.info(f"- 引用数量: {len(citations)}")
                        
                        # 只处理已导入文献之间的引用关系
                        valid_citations = [cid for cid in citations if cid in imported_ids]
                        valid_citations_found += len(valid_citations)
                        
                        if valid_citations:
                            logger.info(f"- 有效引用数量: {len(valid_citations)}")
                            logger.info(f"- 有效引用列表: {valid_citations[:5]}...")
                        
                            for cited_id in valid_citations:
                                # 检查引用关系是否已存在
                                existing_ref = WorkReferencedWork.query.filter_by(
                                    work_id=doc_id,
                                    referenced_work_id=cited_id
                                ).first()
                                
                                if not existing_ref:
                                    # 创建引用关系记录
                                    ref = WorkReferencedWork(
                                        work_id=doc_id,
                                        referenced_work_id=cited_id
                                    )
                                    db.session.add(ref)
                                    refs_added += 1
                                    logger.info(f"添加引用关系: {doc_id} -> {cited_id}")
                        
                        # 每10条记录提交一次
                        if (index + 1) % 10 == 0:
                            db.session.commit()
                            logger.info(f"已处理 {index + 1}/{total_rows} 篇文献的引用关系")
                            if refs_added > 0:
                                logger.info(f"当前已添加 {refs_added} 条引用关系")
                    
                    except Exception as e:
                        logger.error(f"处理文献 {doc_id} 的引用关系时出错: {str(e)}")
                        db.session.rollback()
                        continue
                
                # 提交剩余的引用关系
                if refs_added > 0:
                    db.session.commit()
                
                # 统计最终结果
                works_count = Work.query.count()
                concepts_count = Concept.query.count()
                refs_count = WorkReferencedWork.query.count()
                logger.info(f"\n引用关系统计：")
                logger.info(f"- 找到的总引用数: {citations_found}")
                logger.info(f"- 有效的引用数: {valid_citations_found}")
                logger.info(f"- 成功添加的引用关系: {refs_added}")
                
                logger.info(f"\n导入完成！统计信息：")
                logger.info(f"- 文献数量: {works_count}")
                logger.info(f"- 概念数量: {concepts_count}")
                logger.info(f"- 引用关系: {refs_count}")
                logger.info(f"- 本次添加的引用关系: {refs_added}")
                
            except Exception as e:
                logger.error(f"导入文献集出错: {str(e)}")
                db.session.rollback()

    def generate_search_sessions(self, num_sessions=None):
        """生成搜索会话数据
        
        Args:
            num_sessions: 要生成的会话数量，如果为None则使用所有查询
        """
        logger.info("开始生成搜索会话...")
        
        with self.app.app_context():
            try:
                # 读取查询文件
                query_file = os.path.join(self.dataset_dir, 'query', 'query.parquet')
                if not os.path.exists(query_file):
                    logger.error(f"查询文件不存在: {query_file}")
                    return
                
                # 读取查询数据
                queries_df = pd.read_parquet(query_file)
                logger.info(f"成功读取查询文件，共 {len(queries_df)} 条查询")
                
                # 如果指定了数量，随机选择部分查询
                if num_sessions and num_sessions < len(queries_df):
                    queries_df = queries_df.sample(n=num_sessions)
                
                # 处理每个查询
                for index, row in queries_df.iterrows():
                    try:
                        query_text = str(row.get('query', ''))
                        if not query_text:
                            continue
                        
                        # 记录搜索会话
                        session_id = record_search_session(query_text)
                        if not session_id:
                            continue
                        
                        # 获取相关文献ID
                        relevant_doc_ids = row.get('corpusids', [])
                        if isinstance(relevant_doc_ids, str):
                            try:
                                relevant_doc_ids = eval(relevant_doc_ids)
                            except:
                                relevant_doc_ids = []
                        
                        # 确保relevant_doc_ids是列表
                        if not isinstance(relevant_doc_ids, list):
                            relevant_doc_ids = [relevant_doc_ids]
                        
                        # 构造结果数据
                        results = []
                        for doc_id in relevant_doc_ids:
                            work = Work.query.get(str(doc_id))
                            if work:
                                results.append({
                                    'id': work.id,
                                    'relevance_score': row.get('quality', 1.0),  # 使用quality作为相关性分数
                                    'query_text': query_text
                                })
                        
                        # 记录搜索结果
                        if results:
                            record_search_results(session_id, results, 1, self.items_per_page)
                            
                            # 模拟用户行为
                            for result in results[:5]:  # 假设用户只看前5个结果
                                if random.random() < 0.6:  # 60%的概率点击
                                    record_document_click(session_id, result['id'])
                                    # 随机停留时间30-600秒
                                    dwell_time = random.randint(30, 600)
                                    record_dwell_time(session_id, result['id'], dwell_time)
                        
                        if (index + 1) % 100 == 0:
                            logger.info(f"已处理 {index + 1}/{len(queries_df)} 个查询")
                    
                    except Exception as e:
                        logger.error(f"处理第 {index + 1} 个查询时出错: {str(e)}")
                        continue
                
                logger.info(f"搜索会话生成完成，共生成 {len(queries_df)} 个会话")
                
            except Exception as e:
                logger.error(f"生成搜索会话出错: {str(e)}")
                db.session.rollback()

def main():
    """主函数，执行完整的数据导入和处理流程"""
    try:
        importer = DataImporter()
        
        # 导入文献集（只导入前5个示例）
        with importer.app.app_context():
            # 先清空数据库
            logger.info("清空数据库...")
            db.drop_all()
            db.create_all()
            logger.info("数据库已重置")
        
        # 导入文献
        importer.import_corpus()
        
        # 生成搜索会话（只使用前3个查询）
        importer.generate_search_sessions(num_sessions=3)
        
        logger.info("示例数据处理完成！")
        
    except Exception as e:
        logger.error(f"数据处理出错: {str(e)}")

if __name__ == '__main__':
    main() 