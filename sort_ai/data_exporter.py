import os
import sys
import json
import pandas as pd
from flask import Flask
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.model import SearchSession, SearchResult, Work, UserBehavior, RerankSession
from Database.config import db

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object('Database.config')
db.init_app(app)

def export_training_data(output_dir='training_data'):
    """
    从数据库导出训练数据到文件
    
    Args:
        output_dir: 输出目录路径
    """
    # 获取sort_ai目录的路径
    sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(sort_ai_dir, output_dir)
    
    logger.info("开始导出训练数据...")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    with app.app_context():
        # 1. 导出搜索会话
        logger.info("导出搜索会话...")
        sessions = SearchSession.query.all()
        sessions_data = []
        for session in sessions:
            sessions_data.append({
                'session_id': session.session_id,
                'query_text': session.query_text,
                'search_time': session.search_time.isoformat() if session.search_time else None,
                'total_results': session.total_results
            })
        
        sessions_df = pd.DataFrame(sessions_data)
        sessions_df.to_csv(os.path.join(output_dir, 'sessions.csv'), index=False)
        logger.info(f"已导出 {len(sessions_data)} 条搜索会话")
        
        # 2. 导出文档信息
        logger.info("导出文档信息...")
        works = Work.query.all()
        works_data = []
        for work in works:
            works_data.append({
                'id': work.id,
                'title': work.title,
                'abstract_inverted_index': work.abstract_inverted_index,
                'publication_year': work.publication_year,
                'type': work.type,
                'language': work.language
            })
        
        works_df = pd.DataFrame(works_data)
        works_df.to_csv(os.path.join(output_dir, 'documents.csv'), index=False)
        logger.info(f"已导出 {len(works_data)} 个文档")
        
        # 3. 导出用户行为
        logger.info("导出用户行为...")
        behaviors = UserBehavior.query.all()
        behaviors_data = []
        for behavior in behaviors:
            behaviors_data.append({
                'session_id': behavior.session_id,
                'rerank_session_id': behavior.rerank_session_id,
                'document_id': behavior.document_id,
                'rank_position': behavior.rank_position,
                'is_clicked': behavior.is_clicked,
                'click_time': behavior.click_time.isoformat() if behavior.click_time else None,
                'dwell_time': behavior.dwell_time,
                'behavior_time': behavior.behavior_time.isoformat() if behavior.behavior_time else None
            })
        
        behaviors_df = pd.DataFrame(behaviors_data)
        behaviors_df.to_csv(os.path.join(output_dir, 'behaviors.csv'), index=False)
        logger.info(f"已导出 {len(behaviors_data)} 条用户行为")
        
        # 4. 导出搜索结果
        logger.info("导出搜索结果...")
        results = SearchResult.query.filter_by(entity_type='work').all()
        results_data = []
        for result in results:
            results_data.append({
                'session_id': result.session_id,
                'entity_id': result.entity_id,
                'rank_position': result.rank_position,
                'relevance_score': result.relevance_score,
                'query_text': result.query_text,
                'result_page': result.result_page,
                'result_position': result.result_position
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(output_dir, 'search_results.csv'), index=False)
        logger.info(f"已导出 {len(results_data)} 条搜索结果")
        
        # 5. 导出重排序会话
        logger.info("导出重排序会话...")
        rerank_sessions = RerankSession.query.all()
        rerank_data = []
        for session in rerank_sessions:
            rerank_data.append({
                'session_id': session.session_id,
                'search_session_id': session.search_session_id,
                'rerank_query': session.rerank_query,
                'rerank_time': session.rerank_time.isoformat() if session.rerank_time else None
            })
        
        rerank_df = pd.DataFrame(rerank_data)
        rerank_df.to_csv(os.path.join(output_dir, 'rerank_sessions.csv'), index=False)
        logger.info(f"已导出 {len(rerank_data)} 条重排序会话")
        
        # 6. 导出数据统计信息
        stats = {
            'total_sessions': len(sessions_data),
            'total_documents': len(works_data),
            'total_behaviors': len(behaviors_data),
            'total_results': len(results_data),
            'total_rerank_sessions': len(rerank_data),
            'export_time': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("\n数据导出完成！")
        logger.info(f"- 搜索会话数: {stats['total_sessions']}")
        logger.info(f"- 文档数: {stats['total_documents']}")
        logger.info(f"- 用户行为数: {stats['total_behaviors']}")
        logger.info(f"- 搜索结果数: {stats['total_results']}")
        logger.info(f"- 重排序会话数: {stats['total_rerank_sessions']}")
        logger.info(f"\n数据已保存到目录: {output_dir}")

def export_rerank_training_data(output_dir='training_data'):
    """
    导出重排序训练数据，包括：
    1. 搜索会话和结果
    2. 重排序查询
    3. 用户行为数据
    
    Args:
        output_dir: 输出目录路径
    """
    # 获取sort_ai目录的路径
    sort_ai_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(sort_ai_dir, output_dir)
    
    logger.info("开始导出重排序训练数据...")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    with app.app_context():
        try:
            # 1. 导出重排序会话及其关联的搜索会话
            logger.info("导出重排序会话数据...")
            rerank_data = []
            rerank_sessions = RerankSession.query.all()
            
            for rerank_session in rerank_sessions:
                search_session = rerank_session.search_session
                if not search_session:
                    continue
                    
                # 获取搜索结果
                search_results = SearchResult.query.filter_by(
                    session_id=search_session.session_id
                ).order_by(SearchResult.rank_position).all()
                
                # 获取用户行为数据
                behaviors = UserBehavior.query.filter_by(
                    rerank_session_id=rerank_session.session_id
                ).order_by(UserBehavior.rank_position).all()
                
                # 获取文档详情
                doc_details = []
                for result in search_results:
                    work = Work.query.get(result.entity_id)
                    if work:
                        doc_details.append({
                            'doc_id': work.id,
                            'title': work.title,
                            'abstract': work.abstract_inverted_index,
                            'rank_position': result.rank_position,
                            'relevance_score': result.relevance_score
                        })
                
                # 获取用户行为详情
                behavior_details = []
                for behavior in behaviors:
                    behavior_details.append({
                        'doc_id': behavior.document_id,
                        'rank_position': behavior.rank_position,
                        'is_clicked': behavior.is_clicked,
                        'dwell_time': behavior.dwell_time,
                        'click_time': behavior.click_time.isoformat() if behavior.click_time else None
                    })
                
                # 构建完整的训练样本
                rerank_data.append({
                    'rerank_session_id': rerank_session.session_id,
                    'search_session_id': search_session.session_id,
                    'search_query': search_session.query_text,
                    'rerank_query': rerank_session.rerank_query,
                    'rerank_time': rerank_session.rerank_time.isoformat() if rerank_session.rerank_time else None,
                    'documents': doc_details,
                    'behaviors': behavior_details
                })
            
            # 保存为JSON文件
            output_file = os.path.join(output_dir, 'rerank_training_data.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(rerank_data, f, ensure_ascii=False, indent=2)
            
            # 生成统计信息
            stats = {
                'total_sessions': len(rerank_data),
                'total_documents': sum(len(item['documents']) for item in rerank_data),
                'total_behaviors': sum(len(item['behaviors']) for item in rerank_data),
                'avg_docs_per_session': sum(len(item['documents']) for item in rerank_data) / len(rerank_data) if rerank_data else 0,
                'avg_behaviors_per_session': sum(len(item['behaviors']) for item in rerank_data) / len(rerank_data) if rerank_data else 0,
                'query_examples': [item['rerank_query'] for item in rerank_data[:5]],  # 保存前5个查询作为示例
                'export_time': datetime.now().isoformat()
            }
            
            # 保存统计信息
            stats_file = os.path.join(output_dir, 'rerank_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info("\n数据导出完成！")
            logger.info(f"- 重排序会话数: {stats['total_sessions']}")
            logger.info(f"- 文档总数: {stats['total_documents']}")
            logger.info(f"- 用户行为总数: {stats['total_behaviors']}")
            logger.info(f"- 平均每会话文档数: {stats['avg_docs_per_session']:.2f}")
            logger.info(f"- 平均每会话行为数: {stats['avg_behaviors_per_session']:.2f}")
            logger.info(f"\n示例查询:")
            for i, query in enumerate(stats['query_examples'], 1):
                logger.info(f"{i}. {query}")
            logger.info(f"\n数据已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"导出训练数据时出错: {str(e)}")
            raise

if __name__ == '__main__':
    # 让用户选择要执行的功能
    print("\n请选择要执行的功能：")
    print("1. 导出所有数据")
    print("2. 仅导出重排序训练数据")
    choice = input("请输入选项（1/2）: ").strip()
    
    if choice == '1':
        export_training_data()
    elif choice == '2':
        export_rerank_training_data()
    else:
        print("无效的选项！请输入1或2") 