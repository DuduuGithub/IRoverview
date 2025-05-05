import os
import sys
import json
import random
import math
from datetime import datetime, timedelta
import uuid

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.config import db
from Database.model import Document, SearchSession, SearchResult
from app.utils import generate_search_query
from flask import Flask

class YaleDataImporter:
    def __init__(self, dataset_dir='yale_dataset', items_per_page=10):
        """
        初始化数据导入器
        
        Args:
            dataset_dir: Yale数据集目录
            items_per_page: 每页显示的文档数量
        """
        self.dataset_dir = dataset_dir
        self.corpus_dir = os.path.join(dataset_dir, 'corpus')
        self.query_file = os.path.join(dataset_dir, 'queries.json')
        self.relevance_file = os.path.join(dataset_dir, 'relevance.json')
        self.items_per_page = items_per_page
        
        # 创建Flask应用上下文
        self.app = Flask(__name__)
        self.app.config.from_object('Database.config')
        db.init_app(self.app)

    def calculate_relevance_score(self, dwell_time, page_number):
        """
        计算文档相关性得分
        
        Args:
            dwell_time: 停留时间（秒）
            page_number: 文档所在页码
        
        Returns:
            float: 相关性得分
        """
        if dwell_time == 0:  # 未点击的文档
            return -1.0
        
        # 基础得分：根据停留时间计算
        base_score = math.log(dwell_time + 1) / math.log(301)  # 归一化到[0,1]区间
        
        # 页面衰减因子：考虑用户浏览到该页面的耐心程度
        page_decay = 1.0 / math.sqrt(page_number)
        
        # 最终得分：基础得分 * 页面衰减因子，结果范围在[-1, 1]之间
        return base_score * page_decay

    def import_corpus(self):
        """
        导入文献集到数据库
        """
        print("开始导入文献集...")
        with self.app.app_context():
            try:
                # 读取并导入每个文档
                for filename in os.listdir(self.corpus_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.corpus_dir, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            
                            # 创建文档记录
                            doc = Document(
                                title=doc_data.get('title', ''),
                                author=doc_data.get('author', ''),
                                content=doc_data.get('content', ''),
                                keywords=doc_data.get('keywords', ''),
                                publish_date=datetime.strptime(
                                    doc_data.get('publish_date', '2024-01-01'),
                                    '%Y-%m-%d'
                                )
                            )
                            db.session.add(doc)
                
                db.session.commit()
                print("文献集导入完成")
                
            except Exception as e:
                print(f"导入文献集出错: {str(e)}")
                db.session.rollback()

    def generate_search_sessions(self):
        """
        根据查询和相关性数据生成搜索会话
        """
        print("开始生成搜索会话...")
        with self.app.app_context():
            try:
                # 读取查询和相关性数据
                with open(self.query_file, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
                with open(self.relevance_file, 'r', encoding='utf-8') as f:
                    relevance = json.load(f)
                
                # 处理每个查询
                for query_id, query_text in queries.items():
                    # 生成会话ID
                    session_id = str(uuid.uuid4())
                    
                    # 获取相关文档
                    relevant_docs = relevance.get(query_id, [])
                    if not relevant_docs:
                        continue
                        
                    # 计算用户浏览的最大页数
                    max_rank = len(relevant_docs)
                    max_page = math.ceil(max_rank / self.items_per_page)
                    
                    # 创建搜索会话
                    session = SearchSession(
                        session_id=session_id,
                        search_time=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                        keyword=query_text,
                        search_type='basic',
                        total_results=max_rank
                    )
                    db.session.add(session)
                    
                    # 为每页的文档创建搜索结果记录
                    for rank, doc_id in enumerate(relevant_docs, 1):
                        page_number = math.ceil(rank / self.items_per_page)
                        
                        # 如果是相关文档，生成随机停留时间；否则设为0
                        dwell_time = random.randint(30, 300) if doc_id in relevant_docs else 0
                        
                        # 计算相关性得分
                        relevance_score = self.calculate_relevance_score(dwell_time, page_number)
                        
                        result = SearchResult(
                            session_id=session_id,
                            document_id=doc_id,
                            rank_position=rank,
                            is_clicked=(dwell_time > 0),
                            click_time=session.search_time + timedelta(minutes=random.randint(1, 10)) if dwell_time > 0 else None,
                            dwell_time=dwell_time,
                            relevance_score=relevance_score
                        )
                        db.session.add(result)
                
                db.session.commit()
                print("搜索会话生成完成")
                
            except Exception as e:
                print(f"生成搜索会话出错: {str(e)}")
                db.session.rollback()

    def generate_training_data(self, output_file='training_data.json'):
        """
        生成训练数据
        
        Args:
            output_file: 输出文件路径
        """
        print("开始生成训练数据...")
        training_data = []
        
        with self.app.app_context():
            try:
                # 获取所有搜索会话
                sessions = SearchSession.query.all()
                
                for session in sessions:
                    session_data = {
                        'session_id': session.session_id,
                        'query': session.keyword,
                        'documents': []
                    }
                    
                    # 获取该会话的所有搜索结果（包括未点击的）
                    results = SearchResult.query.filter_by(
                        session_id=session.session_id
                    ).order_by(SearchResult.rank_position).all()
                    
                    if not results:
                        continue
                    
                    # 获取最大页码
                    max_page = math.ceil(len(results) / self.items_per_page)
                    
                    # 处理每个搜索结果
                    for result in results:
                        doc = Document.query.get(result.document_id)
                        if not doc:
                            continue
                        
                        # 使用AI生成检索式
                        generated_query = generate_search_query(doc.content)
                        if not generated_query:
                            continue
                        
                        # 构建文档数据
                        doc_data = {
                            'document_id': doc.id,
                            'rank_position': result.rank_position,
                            'page_number': math.ceil(result.rank_position / self.items_per_page),
                            'relevance_score': result.relevance_score,
                            'is_clicked': result.is_clicked,
                            'dwell_time': result.dwell_time,
                            'generated_query': generated_query
                        }
                        session_data['documents'].append(doc_data)
                    
                    # 添加会话级别的信息
                    session_data['max_page'] = max_page
                    training_data.append(session_data)
                
                # 保存训练数据
                output_dir = os.path.join('sort_ai', 'data', 'processed')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=2)
                    
                print(f"训练数据已保存到: {output_path}")
                
            except Exception as e:
                print(f"生成训练数据出错: {str(e)}")

def main():
    """
    主函数，执行完整的数据导入和处理流程
    """
    try:
        importer = YaleDataImporter()
        
        # 导入文献集
        importer.import_corpus()
        
        # 生成搜索会话
        importer.generate_search_sessions()
        
        # 生成训练数据
        importer.generate_training_data()
        
        print("数据处理完成！")
        
    except Exception as e:
        print(f"数据处理出错: {str(e)}")

if __name__ == '__main__':
    main() 