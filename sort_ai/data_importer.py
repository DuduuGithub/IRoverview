import os
import sys
import json
import random
from datetime import datetime, timedelta
import logging
import uuid
import pandas as pd
import math
import numpy as np
import itertools

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.config import db
from Database.model import (
    Work, Author, WorkAuthorship, SearchSession, 
    SearchResult, UserBehavior, Concept, WorkConcept, WorkReferencedWork,
    RerankSession, Topic, WorkTopic
)
from app.app_blueprint.search.search_utils import (
    record_search_session, record_search_results,
    record_document_click, record_dwell_time
)
from app.app_blueprint.search.basicSearch.search import search  # 修改导入的函数名
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

    def import_all_data(self):
        """按顺序导入所有数据"""
        with self.app.app_context():
            try:
                # 1. 清空数据库
                logger.info("清空数据库...")
                db.drop_all()
                db.create_all()
                logger.info("数据库已重置")

                # 2. 导入文献集
                self.import_corpus()
                
                # 3. 导入检索记录
                self.import_search_sessions()
                
                # 4. 导入重排序记录
                self.import_rerank_sessions()
                
                logger.info("所有数据导入完成！")
                
            except Exception as e:
                logger.error(f"导入过程出错: {str(e)}")
                db.session.rollback()
                raise

    def import_corpus(self):
        """导入文献集到数据库"""
        logger.info("\n========== 开始导入文献集 ==========")

        with self.app.app_context():
            try:
                # 读取Parquet文件
                file_path = os.path.join(self.corpus_dir, 'corpus.parquet')
                if not os.path.exists(file_path):
                    logger.error(f"数据文件不存在: {file_path}")
                    return
                
                df = pd.read_parquet(file_path)
                df = df.head(300)  # 测试用，只取前300条
                total_docs = len(df)
                if total_docs == 0:
                    logger.error("没有可导入的文献记录")
                    return
                    
                logger.info(f"读取到 {total_docs} 条文献记录")
                
                # 导入文献基本信息
                concepts_count = 0
                citations_count = 0
                imported_count = 0
                
                for index, row in df.iterrows():
                    try:
                        # 开始新的事务
                        db.session.begin_nested()
                        
                        # 处理corpusid（numpy.ndarray类型）
                        if isinstance(row['corpusid'], np.ndarray):
                            if row['corpusid'].size == 0:
                                logger.warning(f"行 {index} 的corpusid为空数组，跳过")
                                db.session.rollback()
                                continue
                            doc_id = str(row['corpusid'].item())
                        else:
                            doc_id = str(row['corpusid'])
                            
                        # 检查文献是否已存在
                        existing_work = Work.query.get(doc_id)
                        if existing_work:
                            logger.debug(f"文献 {doc_id} 已存在，跳过")
                            db.session.rollback()
                            continue
                            
                        # 处理引用列表
                        citations = []
                        if 'corpusids' in row and not pd.isna(row['corpusids']):
                            citations = self._process_citations(row['corpusids'])
                        
                        # 创建Work记录
                        work = Work(
                            id=doc_id,
                            title=str(row['title']) if not pd.isna(row['title']) else '',
                            display_name=str(row['title']) if not pd.isna(row['title']) else '',
                            abstract_inverted_index=str(row['abstract']) if not pd.isna(row['abstract']) else ''
                        )
                        db.session.add(work)
                        db.session.flush()
                        
                        # 处理概念和引用关系
                        try:
                            new_concepts = self._process_work_concepts(work, row)
                            new_citations = self._process_work_citations(work, citations)
                            
                            concepts_count += new_concepts
                            citations_count += new_citations
                            imported_count += 1
                            
                            # 提交当前文献的事务
                            db.session.commit()
                            
                        except Exception as e:
                            logger.error(f"处理文献 {doc_id} 的概念或引用关系时出错: {str(e)}")
                            db.session.rollback()
                            continue
                        
                        if (index + 1) % 5 == 0:
                            progress = (index + 1) / total_docs * 100
                            logger.info(f"\n进度: {progress:.1f}% | 已处理 {index + 1}/{total_docs} 篇文献")
                            logger.info(f"- 已导入: {imported_count} 篇")
                            logger.info(f"- 当前概念数: {concepts_count}")
                            logger.info(f"- 当前引用关系数: {citations_count}")
                            if imported_count > 0:
                                logger.info(f"- 平均每篇文献的概念数: {concepts_count/imported_count:.1f}")
                                logger.info(f"- 平均每篇文献的引用数: {citations_count/imported_count:.1f}")
                    
                    except Exception as e:
                        logger.error(f"处理文献 {doc_id if 'doc_id' in locals() else '未知'} 时出错: {str(e)}")
                        db.session.rollback()
                        continue
                
                # 输出最终统计信息
                works_count = Work.query.count()
                total_concepts = Concept.query.count()
                total_citations = WorkReferencedWork.query.count()
                
                logger.info("\n文献导入完成！统计信息：")
                logger.info(f"- 总文献数: {works_count}")
                logger.info(f"- 总概念数: {total_concepts}")
                logger.info(f"- 总引用关系数: {total_citations}")
                
                if works_count > 0:
                    logger.info(f"- 平均每篇文献的概念数: {concepts_count/works_count:.1f}")
                    logger.info(f"- 平均每篇文献的引用数: {citations_count/works_count:.1f}")
                logger.info("====================================")
                
            except Exception as e:
                logger.error(f"导入文献集出错: {str(e)}")
                db.session.rollback()
                raise

    def import_query_related_works(self, query_df):
        """导入查询相关的所有文献
        
        Args:
            query_df: 包含查询和corpusids的DataFrame
            
        Returns:
            int: 导入的文献数量
        """
        logger.info("\n========== 开始导入查询相关文献 ==========")
        
        try:
            # 收集所有需要导入的文献ID
            all_corpus_ids = set()
            for _, row in query_df.iterrows():
                corpus_ids = self._process_corpus_ids(row['corpusids'])
                all_corpus_ids.update(corpus_ids)
            
            logger.info(f"从查询中收集到 {len(all_corpus_ids)} 个唯一文献ID")
            
            # 读取文献数据
            file_path = os.path.join(self.corpus_dir, 'corpus.parquet')
            if not os.path.exists(file_path):
                logger.error(f"数据文件不存在: {file_path}")
                return 0
            
            df = pd.read_parquet(file_path)
            
            # 检查和打印类型信息
            logger.info(f"corpusid的类型: {df['corpusid'].dtype}")
            sample_corpus_id = df['corpusid'].iloc[0] if len(df) > 0 else None
            logger.info(f"corpusid样例: {sample_corpus_id}, 类型: {type(sample_corpus_id)}")
            logger.info(f"all_corpus_ids中的ID类型: {type(next(iter(all_corpus_ids))) if all_corpus_ids else 'empty'}")
            
            # 将all_corpus_ids中的ID转换为与corpusid相同的类型
            if isinstance(sample_corpus_id, np.ndarray):
                # 如果corpusid是numpy数组，需要特殊处理
                def convert_id(x):
                    if isinstance(x, np.ndarray):
                        return x.item() if x.size > 0 else None
                    return x
                df['corpusid'] = df['corpusid'].apply(convert_id)
                all_corpus_ids = {str(id) for id in all_corpus_ids}  # 转换为字符串
            elif isinstance(sample_corpus_id, (int, np.int64)):
                # 如果corpusid是整数，将all_corpus_ids转换为整数
                all_corpus_ids = {int(id) for id in all_corpus_ids}
            else:
                # 默认转换为字符串
                df['corpusid'] = df['corpusid'].astype(str)
                all_corpus_ids = {str(id) for id in all_corpus_ids}
            
            # 再次检查类型
            logger.info(f"转换后corpusid的类型: {df['corpusid'].dtype}")
            logger.info(f"转换后all_corpus_ids中的ID类型: {type(next(iter(all_corpus_ids))) if all_corpus_ids else 'empty'}")
            
            # 过滤文献
            df = df[df['corpusid'].isin(all_corpus_ids)]
            total_docs = len(df)
            logger.info(f"找到 {total_docs} 条匹配的文献记录")
            
            if total_docs == 0:
                logger.warning("没有找到匹配的文献记录，请检查ID类型是否匹配")
                # 打印一些示例数据帮助调试
                logger.info(f"corpusids示例: {list(df['corpusid'].head())}")
                logger.info(f"all_corpus_ids示例: {list(itertools.islice(all_corpus_ids, 5))}")
                return 0
            
            # 导入文献
            imported_count = 0
            for index, row in df.iterrows():
                try:
                    # 处理corpusid（确保是字符串类型）
                    doc_id = str(row['corpusid'])
                        
                    # 检查文献是否已存在
                    if Work.query.get(doc_id):
                        logger.debug(f"文献 {doc_id} 已存在，跳过")
                        continue
                        
                    # 创建Work记录
                    work = Work(
                        id=doc_id,
                        title=str(row['title']) if not pd.isna(row['title']) else '',
                        display_name=str(row['title']) if not pd.isna(row['title']) else '',
                        abstract_inverted_index=str(row['abstract']) if not pd.isna(row['abstract']) else ''
                    )
                    db.session.add(work)
                    imported_count += 1
                    
                    # 每100条记录提交一次
                    if imported_count % 100 == 0:
                        db.session.commit()
                        logger.info(f"已导入 {imported_count}/{total_docs} 篇文献")
                    
                except Exception as e:
                    logger.error(f"处理文献 {doc_id if 'doc_id' in locals() else '未知'} 时出错: {str(e)}")
                    db.session.rollback()
                            continue
            
            # 提交剩余的记录
            db.session.commit()
            logger.info(f"文献导入完成，共导入 {imported_count} 篇新文献")
            return imported_count
            
        except Exception as e:
            logger.error(f"导入查询相关文献时出错: {str(e)}")
            db.session.rollback()
            raise

    def import_search_sessions(self):
        """导入检索会话记录和对应的重排序会话"""
        logger.info("\n========== 开始导入检索会话和重排序会话 ==========")

        with self.app.app_context():
            try:
                # 读取query.parquet文件
                query_file = os.path.join(self.dataset_dir, 'query/query.parquet')
                query_df = pd.read_parquet(query_file)
                logger.info(f"读取到 {len(query_df)} 条原始查询")
                
                # 读取转换后的查询
                transformed_file = os.path.join(self.dataset_dir, 'query_transformed.csv')
                transformed_df = pd.read_csv(transformed_file)
                logger.info(f"读取到 {len(transformed_df)} 条转换后的查询")
                
                # 合并查询数据
                merged_df = pd.merge(
                    transformed_df,
                    query_df,
                    left_on='index',
                    right_index=True,
                    how='inner'
                )
                
                merged_df = merged_df.head(300)
                total_queries = len(merged_df)
                logger.info(f"合并后有 {total_queries} 条有效查询")
                
                # 处理进度统计
                session_count = 0
                rerank_count = 0
                result_count = 0
                valid_docs_count = 0
                
                for idx, row in merged_df.iterrows():
                    try:
                        # 1. 创建搜索会话
                        search_session_id = f"search_{uuid.uuid4().hex}"
                        search_session = SearchSession(
                            session_id=search_session_id,
                            query_text=row['query_text'],
                            search_time=datetime.now()
                        )
                        db.session.add(search_session)
                        db.session.flush()
                        
                        # 2. 处理搜索结果
                        valid_docs = self._create_search_results(search_session, row['corpusids'], row['query_text'])
                        
                        # 3. 创建对应的重排序会话
                        if valid_docs:  # 只有当有有效的搜索结果时才创建重排序会话
                            rerank_session_id = f"rerank_{search_session_id}_{hash(str(row['query']))}"
                            rerank_session = RerankSession(
                                session_id=rerank_session_id,
                                search_session_id=search_session_id,
                                rerank_query=row['query'],  # 使用原始查询作为重排序查询
                                rerank_time=datetime.now()
                            )
                            db.session.add(rerank_session)
                            db.session.flush()
                            
                            # 4. 生成用户行为数据
                            behaviors = self._create_user_behaviors(rerank_session)
                            if behaviors:
                                rerank_count += 1
                        
                        session_count += 1
                        result_count += len(valid_docs)
                        valid_docs_count += len(valid_docs)
                        
                        # 每处理一条记录就提交一次，这样即使有错误也不会影响其他记录
                        db.session.commit()
                        
                        if session_count % 100 == 0:
                            progress = session_count / total_queries * 100
                            logger.info(f"\n进度: {progress:.1f}% | 已处理 {session_count}/{total_queries} 个查询")
                            logger.info(f"- 当前搜索会话数: {session_count}")
                            logger.info(f"- 当前重排序会话数: {rerank_count}")
                            logger.info(f"- 当前结果数: {result_count}")
                            if session_count > 0:
                                logger.info(f"- 平均每个查询的结果数: {result_count/session_count:.1f}")
                    
                    except Exception as e:
                        logger.error(f"处理查询 (索引: {idx}) 时出错: {str(e)}")
                        db.session.rollback()  # 回滚当前事务
                        continue
                
                logger.info(f"\n导入完成统计:")
                logger.info(f"- 搜索会话数: {session_count}")
                logger.info(f"- 重排序会话数: {rerank_count}")
                logger.info(f"- 总结果数: {result_count}")
                
            except Exception as e:
                logger.error(f"导入会话出错: {str(e)}")
                db.session.rollback()
                raise

    def _process_citations(self, citations):
        """处理引用数据
        
        Args:
            citations: 原始引用数据
            
        Returns:
            list: 处理后的引用ID列表
        """
        if hasattr(citations, '__array__'):
            citations = citations.tolist()
        elif isinstance(citations, str):
            try:
                citations = eval(citations)
            except:
                citations = citations.strip('[]').split(',')
                citations = [cid.strip() for cid in citations if cid.strip()]
        
        return [str(cid) for cid in citations]

    def _process_work_concepts(self, work, row):
        """处理文献的概念关联
        
        Args:
            work: Work实例
            row: 数据行
            
        Returns:
            int: 添加的概念数量
        """
        # 从标题和摘要中提取关键词
        keywords = set()
        title = str(row['title'])
        abstract = str(row['abstract'])
        
        if title:
            keywords.update(word.lower() for word in title.split() if len(word) > 3)
        if abstract:
            keywords.update(word.lower() for word in abstract.split() if len(word) > 3)
        
        # 过滤常见词和数字
        keywords = {k for k in keywords if not k.isdigit()}
        
        # 创建概念关联
        concepts_added = 0
        
        # 使用no_autoflush避免过早的自动刷新
        with db.session.no_autoflush:
            for keyword in keywords:
                try:
                    # 检查概念是否存在
                    concept = Concept.query.filter_by(display_name=keyword).first()
                    if not concept:
                        concept = Concept(
                            id=f"C{uuid.uuid4().hex[:8]}",
                            display_name=keyword,
                            level=1
                        )
                        db.session.add(concept)
                        db.session.flush()
                    
                    # 检查work-concept关联是否已存在
                    existing_relation = WorkConcept.query.filter_by(
                        work_id=work.id,
                        concept_id=concept.id
                    ).first()
                    
                    if not existing_relation:
                        work_concept = WorkConcept(
                            work_id=work.id,
                            concept_id=concept.id,
                            score=1.0
                        )
                        db.session.add(work_concept)
                        concepts_added += 1
                    
                except Exception as e:
                    logger.warning(f"处理概念 '{keyword}' 时出错: {str(e)}")
                    continue
                
        return concepts_added

    def _process_work_citations(self, work, citations):
        """处理文献的引用关系
        
        Args:
            work: Work实例
            citations: 引用ID列表
            
        Returns:
            int: 添加的引用关系数量
        """
        citations_added = 0
        for cited_id in citations:
            cited_work = Work.query.get(str(cited_id))
            if cited_work:
                ref = WorkReferencedWork(
                            work_id=work.id,
                    referenced_work_id=cited_id
                )
                db.session.add(ref)
                citations_added += 1
                
        return citations_added

    def _process_corpus_ids(self, corpus_ids):
        """处理语料库ID
        
        Args:
            corpus_ids: 原始语料库ID数据
            
        Returns:
            list: 处理后的ID列表
        """
        if corpus_ids is None:
            return []
            
        try:
            # 处理numpy数组
            if isinstance(corpus_ids, np.ndarray):
                # 确保数组不是空的
                if corpus_ids.size == 0:
                    return []
                # 如果是一维数组，直接转换
                if corpus_ids.ndim == 1:
                    return [str(cid) for cid in corpus_ids.tolist() if cid]
                # 如果是多维数组，先展平
                return [str(cid) for cid in corpus_ids.flatten().tolist() if cid]
            
            # 处理字符串
            if isinstance(corpus_ids, str):
                try:
                    corpus_ids = eval(corpus_ids)
                except:
                    corpus_ids = corpus_ids.strip('[]').split(',')
                    corpus_ids = [cid.strip() for cid in corpus_ids if cid.strip()]
            
            # 处理列表或元组
            if isinstance(corpus_ids, (list, tuple)):
                return [str(cid) for cid in corpus_ids if cid]
            
            logger.warning(f"无法处理的corpus_ids类型: {type(corpus_ids)}, 值: {corpus_ids}")
            return []
                
            except Exception as e:
            logger.error(f"处理corpus_ids时出错: {str(e)}, 值: {corpus_ids}")
            return []

    def _create_search_results(self, session, corpus_ids, query_text):
        """创建搜索结果记录
        
        Args:
            session: SearchSession实例
            corpus_ids: 文献ID列表
            query_text: 查询文本
            
        Returns:
            list: 有效的文档ID列表
        """
        try:
            logger.info(f"开始处理搜索结果，检索式: {query_text}")
            
            # 使用检索式进行检索
            from app.app_blueprint.search.basicSearch.search import search
            search_results = search(
                dictionary_file=None,
                postings_file=None,
                queries_file=None,
                output_file=None,
                sort_method='relevance',
                query_text=query_text,
                use_db=True
            )
            
            if not search_results:
                logger.warning(f"检索式 '{query_text}' 没有返回任何结果")
                return []
            
            # 获取正确的文档ID列表
            correct_doc_ids = self._process_corpus_ids(corpus_ids)
            if not correct_doc_ids:
                logger.warning(f"没有找到正确的文档ID")
                return []
            
            # 确保所有ID都是字符串类型
            search_results = [str(doc_id) for doc_id in search_results]
            correct_doc_ids = [str(doc_id) for doc_id in correct_doc_ids]
            
            # 限制结果数量为12个，并确保包含至少一个正确结果
            final_results = []
            correct_result_found = False
            
            # 首先添加前11个检索结果
            for doc_id in search_results[:11]:
                if doc_id in correct_doc_ids:
                    correct_result_found = True
                final_results.append(doc_id)
            
            # 如果还没有找到正确结果，且还有正确结果可用
            if not correct_result_found and correct_doc_ids:
                # 如果结果不足12个，直接添加一个正确结果
                if len(final_results) < 12:
                    final_results.append(correct_doc_ids[0])
                # 如果已有12个结果，替换最后一个
                else:
                    final_results[-1] = correct_doc_ids[0]
                correct_result_found = True
            
            # 如果结果不足12个，且还有更多检索结果，继续添加
            while len(final_results) < 12 and len(search_results) > len(final_results):
                next_doc = search_results[len(final_results)]
                if next_doc not in final_results:
                    final_results.append(next_doc)
            
            # 创建搜索结果记录
            valid_ids = []
            for rank_position, doc_id in enumerate(final_results, 1):
                # 检查文档是否存在
                if not Work.query.get(doc_id):
                    logger.debug(f"文档 {doc_id} 在数据库中不存在，跳过")
                    continue
                    
                result_record = SearchResult(
                    session_id=session.session_id,
                    entity_type='work',
                    entity_id=doc_id,
                    rank_position=rank_position,
                    relevance_score=0.9 if doc_id in correct_doc_ids else 0.5,  # 正确结果给更高的相关性分数
                    query_text=query_text,
                    result_page=math.ceil(rank_position / self.items_per_page),
                    result_position=((rank_position - 1) % self.items_per_page) + 1
                )
                db.session.add(result_record)
                valid_ids.append(doc_id)
            
            # 更新会话的结果总数
            session.total_results = len(valid_ids)
            logger.info(f"检索到 {len(valid_ids)} 个结果，包含正确结果: {correct_result_found}")
            
            return valid_ids
        except Exception as e:
            logger.error(f"创建搜索结果时出错: {str(e)}")
            raise

    def _create_rerank_session(self, search_session, rerank_query):
        """创建重排序会话
        
        Args:
            search_session: SearchSession实例
            rerank_query: 重排序查询文本
            
        Returns:
            RerankSession: 创建的重排序会话实例
        """
        # 检查是否有搜索结果
        if not SearchResult.query.filter_by(session_id=search_session.session_id).first():
            return None
        
        # 创建重排序会话
        rerank_session = RerankSession(
            session_id=f"rerank_{search_session.session_id}_{hash(rerank_query)}",
            search_session_id=search_session.session_id,
            rerank_query=rerank_query,
            rerank_time=datetime.now()
        )
        db.session.add(rerank_session)
        db.session.flush()
        
        return rerank_session

    def _create_user_behaviors(self, rerank_session):
        """为重排序会话创建用户行为记录
        
        Args:
            rerank_session: RerankSession实例
            
        Returns:
            list: 创建的用户行为记录列表
        """
        try:
            # 获取原始搜索会话
            search_session = SearchSession.query.filter_by(
                session_id=rerank_session.search_session_id
            ).first()
            
            if not search_session:
                logger.warning(f"找不到搜索会话: {rerank_session.search_session_id}")
                return []
            
            # 获取原始的corpusids
            query_df = pd.read_parquet(os.path.join(self.dataset_dir, 'query/query.parquet'))
            transformed_df = pd.read_csv(os.path.join(self.dataset_dir, 'query_transformed.csv'))
            
            # 合并查询数据
            merged_df = pd.merge(
                transformed_df,
                query_df,
                left_on='index',
                right_index=True,
                how='inner'
            )
            
            # 找到对应的行
            row = merged_df[merged_df['query_text'] == search_session.query_text].iloc[0]
            corpus_ids = self._process_corpus_ids(row['corpusids'])
            
            if not corpus_ids:
                logger.warning(f"没有找到corpus_ids: {search_session.query_text}")
                return []
            
            behaviors = []
            # 只处理前10个结果
            for position, doc_id in enumerate(corpus_ids[:10], 1):
                # 确保文档存在
                if not Work.query.get(str(doc_id)):
                    logger.warning(f"文档 {doc_id} 在works表中不存在，跳过")
                    continue
                    
                # 生成用户行为数据
                relevance_score = 0.9 if position <= 3 else 0.5
                dwell_time = self._generate_dwell_time(relevance_score, position)
                click_prob = relevance_score * (1.0 / math.sqrt(position))
                is_clicked = random.random() < click_prob
                current_time = datetime.now()
                
                # 创建用户行为记录
                    behavior = UserBehavior(
                session_id=rerank_session.search_session_id,
                rerank_session_id=rerank_session.session_id,
                document_id=str(doc_id),  # 使用原始的corpus_id
                rank_position=position,
                is_clicked=is_clicked,
                click_time=current_time if is_clicked else None,
                dwell_time=dwell_time if is_clicked else 0,
                behavior_time=current_time
                    )
                    db.session.add(behavior)
                behaviors.append(behavior)
            
            return behaviors
        except Exception as e:
            logger.error(f"创建用户行为记录时出错: {str(e)}")
            raise

    def _rerank_results(self, results, rerank_query):
        """重新排序搜索结果
        
        Args:
            results: 原始搜索结果列表
            rerank_query: 重排序查询文本
            
        Returns:
            list: 重排序后的结果列表
        """
        try:
            reranked = list(results)
            
            if "时间" in rerank_query or "最新" in rerank_query:
                reranked.sort(
                    key=lambda x: Work.query.get(x.entity_id).publication_year or 0,
                    reverse=True
                )
            elif "引用" in rerank_query:
                reranked.sort(
                    key=lambda x: Work.query.get(x.entity_id).cited_by_count or 0,
                    reverse=True
                )
            
            return reranked
        except Exception as e:
            logger.error(f"重排序结果时出错: {str(e)}")
            raise

    def _generate_dwell_time(self, relevance_score, position):
        """生成停留时间
        
        Args:
            relevance_score: 相关性分数
            position: 结果位置
            
        Returns:
            int: 生成的停留时间（秒）
        """
        min_time = 30
        max_time = 300
        base_time = min_time + (max_time - min_time) * relevance_score
        position_factor = 1.0 / math.sqrt(position)
        adjusted_time = base_time * position_factor
        variation = random.uniform(0.8, 1.2)
        final_time = int(adjusted_time * variation)
        
        return max(min_time, min(final_time, max_time))

    def _get_query_session_mapping(self):
        """获取查询索引到会话ID的映射
        
        Returns:
            dict: 查询索引到会话ID的映射
        """
        try:
            transformed_file = os.path.join(self.dataset_dir, 'query_transformed.csv')
            if not os.path.exists(transformed_file):
                logger.error(f"转换后的查询文件不存在: {transformed_file}")
                return {}
            
            # 读取转换后的查询文件
            transformed_df = pd.read_csv(transformed_file)
            logger.info(f"读取到 {len(transformed_df)} 条转换后的查询")
            
            # 获取所有搜索会话
            search_sessions = SearchSession.query.all()
            logger.info(f"数据库中有 {len(search_sessions)} 条搜索会话")
            
            # 创建查询文本到会话ID的映射
            query_to_session = {
                session.query_text: session.session_id 
                for session in search_sessions
            }
            
            # 创建索引到会话ID的映射
            mapping = {}
            for _, row in transformed_df.iterrows():
                query_text = row['query']  # 使用'query'列而不是'query_text'
                if query_text in query_to_session:
                    mapping[row['index']] = query_to_session[query_text]
                    
            logger.info(f"找到 {len(mapping)} 个有效的查询-会话映射")
            
            # 打印一些调试信息
            if not mapping:
                logger.warning("没有找到任何查询-会话映射！")
                logger.warning("转换后查询文件中的列名：" + str(transformed_df.columns.tolist()))
                logger.warning("示例查询：" + str(transformed_df['query'].head().tolist()))
                logger.warning("示例会话查询：" + str([s.query_text for s in search_sessions[:5]]))
            
            return mapping
            
        except Exception as e:
            logger.error(f"获取查询-会话映射时出错: {str(e)}")
            return {}

def main():
    """主函数，可以选择执行不同的功能"""
    try:
        importer = DataImporter()
        
        # 让用户选择要执行的功能
        print("\n请选择要执行的功能：")
        print("1. 导入文献集（这会先清空数据库）")
        print("2. 导入查询相关文献（不清空数据库）")
        print("3. 生成搜索会话重排序会话和用户行为")
        print("4. 全部执行（按顺序执行1-3）")
        print("5. 清除所有数据")
        choice = input("请输入选项（1-5）: ").strip()
        
        with importer.app.app_context():
            if choice == '1':
                # 清空数据库
                logger.info("清空数据库...")
                db.drop_all()
                db.create_all()
                logger.info("数据库已重置")
                
                # 导入文献
                importer.import_corpus()
                logger.info("文献导入完成！")
                
            elif choice == '2':
                # 导入查询相关文献
                logger.info("开始导入查询相关文献...")
                
                # 读取query.parquet文件
                query_file = os.path.join(importer.dataset_dir, 'query/query.parquet')
                query_df = pd.read_parquet(query_file)
                logger.info(f"读取到 {len(query_df)} 条原始查询")
                
                # 读取转换后的查询
                transformed_file = os.path.join(importer.dataset_dir, 'query_transformed.csv')
                transformed_df = pd.read_csv(transformed_file)
                logger.info(f"读取到 {len(transformed_df)} 条转换后的查询")
                
                # 合并查询数据
                merged_df = pd.merge(
                    transformed_df,
                    query_df,
                    left_on='index',
                    right_index=True,
                    how='inner'
                )
                
                # 让用户选择要处理的查询数量
                print("\n请选择要处理的查询数量：")
                print("1. 前10条查询")
                print("2. 前50条查询")
                print("3. 前100条查询")
                print("4. 所有查询")
                num_choice = input("请输入选项（1-4）: ").strip()
                
                if num_choice == '1':
                    merged_df = merged_df.head(10)
                elif num_choice == '2':
                    merged_df = merged_df.head(50)
                elif num_choice == '3':
                    merged_df = merged_df.head(100)
                
                # 导入相关文献
                imported_docs = importer.import_query_related_works(merged_df)
                logger.info(f"已导入 {imported_docs} 篇相关文献")
                
            elif choice == '3':
                # 生成搜索会话
                importer.import_search_sessions()
                logger.info("搜索会话生成完成！")
                
            elif choice == '4':
                # 清空数据库
                logger.info("清空数据库...")
                db.drop_all()
                db.create_all()
                logger.info("数据库已重置")
                
                # 导入文献
        importer.import_corpus()
                logger.info("文献导入完成！")
        
        # 生成搜索会话
                importer.import_search_sessions()
                logger.info("搜索会话生成完成！")
                
            elif choice == '5':
                # 清空数据库
                logger.info("清空数据库...")
                db.drop_all()
                db.create_all()
                logger.info("数据库已重置")
                
            else:
                logger.error("无效的选项！请输入1-5")
                return
        
    except Exception as e:
        logger.error(f"执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()