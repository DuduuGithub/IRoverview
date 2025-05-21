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
                df = df.head(600)
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
                        cited_by_count = len(citations)
                        
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
                        
                        # 只处理已导入文献之间的引用关系
                        valid_citations = [cid for cid in citations if cid in imported_ids]
                        valid_citations_found += len(valid_citations)
                        
                        if valid_citations:
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
                        
                        # 每10条记录提交一次
                        if (index + 1) % 10 == 0:
                            db.session.commit()
                            logger.info(f"已处理 {index + 1}/{total_rows} 篇文献的引用关系，添加了 {refs_added} 条引用")
                    
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
                logger.info(f"\n导入完成！统计信息：")
                logger.info(f"- 文献数量: {works_count}")
                logger.info(f"- 概念数量: {concepts_count}")
                logger.info(f"- 引用关系: {refs_count}")
                logger.info(f"- 本次添加的引用关系: {refs_added}")
                
            except Exception as e:
                logger.error(f"导入文献集出错: {str(e)}")
                db.session.rollback()

    def generate_search_sessions(self, num_sessions=50):
        """生成搜索会话数据，同时导入query_transformed.csv的内容
        
        Args:
            num_sessions: 要生成的会话数量，如果为None则使用所有查询
        """
        logger.info("开始生成搜索会话...")
        
        with self.app.app_context():
            try:
                # 获取已导入的文档ID列表
                imported_doc_ids = set(str(work.id) for work in Work.query.all())
                logger.info(f"已导入 {len(imported_doc_ids)} 篇文献")
                
                # 读取query_transformed.csv
                query_file = os.path.join(self.dataset_dir, 'query_transformed.csv')
                transformed_queries = pd.read_csv(query_file)  # 使用默认的逗号分隔符
                logger.info(f"读取到 {len(transformed_queries)} 条转换后的查询")
                
                # 读取原始查询数据（包含corpusids）
                query_parquet = os.path.join(self.dataset_dir, 'query/query.parquet')
                query_df = pd.read_parquet(query_parquet)
                logger.info(f"原始查询数据大小: {len(query_df)}")
                
                valid_query_info = []  # 存储有效查询的详细信息
                # 检查每个查询的正确结果是否都在已导入的文献中
                for idx, row in query_df.iterrows():
                    try:
                        corpus_ids = eval(row['corpusids']) if isinstance(row['corpusids'], str) else row['corpusids']
                        corpus_ids = [str(id) for id in corpus_ids]  # 确保ID是字符串类型
                        
                        # 检查所有正确结果是否都在已导入文献中
                        if all(doc_id in imported_doc_ids for doc_id in corpus_ids):
                            # 保存查询的原始索引和正确结果信息
                            valid_query_info.append({
                                'original_index': idx,
                                'corpus_ids': corpus_ids,
                                'correct_results_count': len(corpus_ids)
                            })
                    except Exception as e:
                        logger.error(f"处理查询 {idx} 的corpusids时出错: {str(e)}")
                        continue
                    
                logger.info(f"找到 {len(valid_query_info)} 个有效查询（所有正确结果都在已导入文献中）")
                
                # 显示有效查询的详细信息
                logger.info("\n有效查询详细信息:")
                for i, info in enumerate(valid_query_info[:3]):  # 只显示前3个作为示例
                    logger.info(f"查询 {i+1}:")
                    logger.info(f"  - 原始索引: {info['original_index']}")
                    logger.info(f"  - 正确结果数量: {info['correct_results_count']}")
                else:
                    logger.error("valid_transformed_queries 仍然为空！")
                
                # 获取有效查询的原始索引列表
                valid_original_indices = [info['original_index'] for info in valid_query_info]
                
                # 将这些信息保存到CSV文件中
                valid_queries_df = pd.DataFrame(valid_query_info)
                valid_queries_file = os.path.join(self.dataset_dir, 'valid_queries_info.csv')
                valid_queries_df.to_csv(valid_queries_file, index=False)
                logger.info(f"\n有效查询信息已保存到: {valid_queries_file}")
                
                # 只保留有效的查询
                valid_transformed_queries = transformed_queries[transformed_queries['index'].isin(valid_original_indices)]
                logger.info(f"匹配到的转换后查询数量: {len(valid_transformed_queries)}")
                
                if len(valid_transformed_queries) == 0:
                    logger.error("没有找到匹配的查询！")
                    return
                
                logger.info(f"将处理前150个有效查询")
                
                # 只处理前150条有效查询
                valid_transformed_queries = valid_transformed_queries.head(150)
                
                # 打印查询内容示例
                if not valid_transformed_queries.empty:
                    logger.info("\n查询示例:")
                    for idx, row in valid_transformed_queries.head(3).iterrows():
                        original_index = row['index']
                        query_info = next(info for info in valid_query_info if info['original_index'] == original_index)
                        logger.info(f"查询 {idx}:")
                        logger.info(f"  - 查询文本: {row['query']}")
                        logger.info(f"  - 正确结果数量: {query_info['correct_results_count']}")
                else:
                    logger.error("valid_transformed_queries 仍然为空！")
                
                # 处理每个查询
                for idx, row in valid_transformed_queries.iterrows():
                    try:
                        query_text = row['query']
                        logger.info(f"\n处理查询: {query_text}")
                        
                        # 记录搜索会话
                        session_id = record_search_session(query_text)
                        if not session_id:
                            logger.error("创建会话失败，跳过此查询")
                            continue
                        
                        # 使用实际的检索函数获取结果
                        search_results = search(query_text=query_text, use_db=True)
                        if not search_results:
                            logger.warning("未找到搜索结果，跳过此查询")
                            continue
                        
                        logger.info(f"搜索返回 {len(search_results)} 个结果")
                            
                        # 获取该查询的正确结果
                        query_info = next(info for info in valid_query_info if info['original_index'] == row['index'])
                        correct_results = query_info['corpus_ids']
                        correct_results = set(str(id) for id in correct_results)
                            
                        # 构造结果数据
                        results = []
                        correct_results_found = []  # 存储找到的正确结果
                        other_results = []  # 存储其他结果
                        
                        for doc_id in search_results:
                            work = Work.query.get(str(doc_id))
                            if work:
                                result = {
                                    'id': work.id,
                                    'relevance_score': 0.9 if str(doc_id) in correct_results else 0.5,
                                    'query_text': row['query']
                                }
                                
                                # 区分正确结果和其他结果
                                if str(doc_id) in correct_results:
                                    correct_results_found.append(result)
                                else:
                                    other_results.append(result)
                        
                        # 组合结果，确保最多10条记录
                        # 如果有正确结果，优先使用正确结果
                        if correct_results_found:
                            results = correct_results_found + other_results
                            results = results[:10]  # 只取前10条
                        else:
                            # 如果没有正确结果，取前9条其他结果，并添加一个正确结果作为第10条
                            if correct_results:  # 如果有正确答案
                                other_results = other_results[:9]  # 只取前9条
                                # 添加一个正确结果
                                correct_doc_id = next(iter(correct_results))
                                correct_work = Work.query.get(correct_doc_id)
                                if correct_work:
                                    other_results.append({
                                        'id': correct_work.id,
                                        'relevance_score': 0.9,
                                        'query_text': row['query']
                                    })
                            else:
                                other_results = other_results[:10]  # 如果没有正确答案，就取前10条
                            results = other_results
                        
                        # 记录搜索结果
                        if results:
                            logger.info(f"记录 {len(results)} 个搜索结果")
                            record_search_results(session_id, results, 1, self.items_per_page)
                            
                            # 模拟用户行为：对相关性分数高的文档更可能点击
                            sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
                            for result in sorted_results[:3]:
                                if random.random() < result['relevance_score']:
                                    record_document_click(session_id, result['id'])
                                    base_time = int(result['relevance_score'] * 300)
                                    dwell_time = random.randint(base_time, base_time + 120)
                                    record_dwell_time(session_id, result['id'], dwell_time)
                            
                            logger.info(f"- 正确结果数量: {len(correct_results_found)}")
                            logger.info(f"- 保存结果数量: {len(results)}")
                        
                        # 每50条查询输出一次进度
                        if (idx + 1) % 50 == 0:
                            logger.info(f"已处理 {idx + 1}/150 条查询")
                            # 提交事务
                            db.session.commit()
                    
                    except Exception as e:
                        logger.error(f"处理查询时出错: {str(e)}", exc_info=True)
                        db.session.rollback()
                    continue
                
                # 最后一次提交
                db.session.commit()
                
                # 验证生成的数据
                sessions_count = SearchSession.query.count()
                results_count = SearchResult.query.count()
                behaviors_count = UserBehavior.query.count()
                
                logger.info("\n搜索会话生成完成！统计信息：")
                logger.info(f"- 生成的会话数: {sessions_count}")
                logger.info(f"- 生成的搜索结果数: {results_count}")
                logger.info(f"- 生成的用户行为数: {behaviors_count}")
                
            except Exception as e:
                logger.error(f"生成搜索会话出错: {str(e)}", exc_info=True)
                db.session.rollback()
                raise

    def clear_search_records(self):
        """清除所有搜索相关的记录，包括搜索会话、搜索结果和用户行为"""
        logger.info("开始清除搜索相关记录...")
        
        with self.app.app_context():
            try:
                # 记录清除前的数量
                sessions_before = SearchSession.query.count()
                results_before = SearchResult.query.count()
                behaviors_before = UserBehavior.query.count()
                
                # 按顺序清除相关表（考虑外键约束）
                UserBehavior.query.delete()
                SearchResult.query.delete()
                SearchSession.query.delete()
                
                # 提交更改
                db.session.commit()
                
                logger.info("\n清除完成！统计信息：")
                logger.info(f"- 清除的会话数: {sessions_before}")
                logger.info(f"- 清除的搜索结果数: {results_before}")
                logger.info(f"- 清除的用户行为数: {behaviors_before}")
            
            except Exception as e:
                    logger.error(f"清除搜索记录时出错: {str(e)}")
                    db.session.rollback()
                    raise

def main():
    """主函数，可以选择执行不同的功能"""
    try:
        importer = DataImporter()
        
        # 让用户选择要执行的功能
        print("\n请选择要执行的功能：")
        print("1. 导入文献集（这会先清空数据库）")
        print("2. 生成搜索会话和用户行为")
        print("3. 全部执行（先导入文献，再生成搜索会话）")
        print("4. 清除搜索相关记录（保留文献数据）")
        choice = input("请输入选项（1/2/3/4）: ").strip()
        
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
                # 检查是否已有文献数据
                works_count = Work.query.count()
                if works_count == 0:
                    logger.error("错误：数据库中没有文献数据！请先导入文献集（选项1）")
                    return
                    
                # 生成搜索会话
                importer.generate_search_sessions(num_sessions=50)
                logger.info("搜索会话生成完成！")
                
            elif choice == '3':
                # 清空数据库
                logger.info("清空数据库...")
                db.drop_all()
                db.create_all()
                logger.info("数据库已重置")
                
                # 导入文献
                importer.import_corpus()
                logger.info("文献导入完成！")
        
        # 生成搜索会话
                importer.generate_search_sessions(num_sessions=50)
                logger.info("搜索会话生成完成！")
        
            elif choice == '4':
                # 清除搜索相关记录
                importer.clear_search_records()
        
            else:
                logger.error("无效的选项！请输入1、2、3或4")
                return
        
    except Exception as e:
        logger.error(f"执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()