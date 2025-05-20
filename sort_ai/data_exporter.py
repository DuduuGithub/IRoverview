import os
import sys
import json
import pandas as pd
from flask import Flask
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.model import SearchSession, SearchResult, UserBehavior, Work
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
                'work_id': work.id,
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
                'behavior_id': behavior.id,
                'session_id': behavior.session_id,
                'document_id': behavior.document_id,
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
                'result_id': result.id,
                'session_id': result.session_id,
                'entity_id': result.entity_id,
                'relevance_score': result.relevance_score,
                'rank': result.rank,
                'is_clicked': result.is_clicked,
                'dwell_time': result.dwell_time
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(os.path.join(output_dir, 'search_results.csv'), index=False)
        logger.info(f"已导出 {len(results_data)} 条搜索结果")
        
        # 5. 导出数据统计信息
        stats = {
            'total_sessions': len(sessions_data),
            'total_documents': len(works_data),
            'total_behaviors': len(behaviors_data),
            'total_results': len(results_data)
        }
        
        with open(os.path.join(output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("\n数据导出完成！")
        logger.info(f"- 搜索会话数: {stats['total_sessions']}")
        logger.info(f"- 文档数: {stats['total_documents']}")
        logger.info(f"- 用户行为数: {stats['total_behaviors']}")
        logger.info(f"- 搜索结果数: {stats['total_results']}")
        logger.info(f"\n数据已保存到目录: {output_dir}")

if __name__ == '__main__':
    export_training_data() 