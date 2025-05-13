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
from Database.model import Document, SearchSession, SearchResult, Work, Author, WorkAuthorship, UserBehavior
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

def import_works():
    """导入文献数据"""
    print("开始导入文献数据...")
    
    # 读取yale_dataset中的文献数据
    dataset_path = os.path.join(os.path.dirname(__file__), 'yale_dataset')
    works_imported = 0
    
    for filename in os.listdir(dataset_path):
        if not filename.endswith('.json'):
            continue
            
        with open(os.path.join(dataset_path, filename), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # 检查文献是否已存在
                existing_work = Work.query.filter_by(id=data['id']).first()
                if existing_work:
                    continue
                
                # 创建Work记录
                work = Work(
                    id=data['id'],
                    title=data.get('title', ''),
                    display_name=data.get('title', ''),
                    publication_year=data.get('year'),
                    cited_by_count=data.get('n_citation', 0),
                    abstract_inverted_index=data.get('abstract', '')
                )
                db.session.add(work)
                
                # 创建Author记录
                if 'authors' in data:
                    for i, author_name in enumerate(data['authors']):
                        author = Author.query.filter_by(display_name=author_name).first()
                        if not author:
                            author = Author(display_name=author_name)
                            db.session.add(author)
                            db.session.flush()  # 获取author.id
                        
                        # 创建WorkAuthorship记录
                        authorship = WorkAuthorship(
                            work_id=work.id,
                            author_id=author.id,
                            position=i + 1
                        )
                        db.session.add(authorship)
                
                works_imported += 1
                if works_imported % 100 == 0:
                    print(f"已导入 {works_imported} 篇文献...")
                    db.session.commit()
                
            except Exception as e:
                print(f"导入文献 {filename} 时出错: {str(e)}")
                db.session.rollback()
    
    db.session.commit()
    print(f"文献导入完成，共导入 {works_imported} 篇文献")
    return works_imported

def create_sample_search_records(num_sessions=5):
    """创建示例搜索记录"""
    print("开始创建示例搜索记录...")
    
    # 获取所有文献ID
    work_ids = [w.id for w in Work.query.all()]
    if not work_ids:
        print("没有可用的文献数据")
        return
    
    # 示例查询关键词
    sample_queries = [
        "machine learning applications",
        "deep neural networks",
        "natural language processing",
        "computer vision techniques",
        "artificial intelligence trends"
    ]
    
    sessions_created = 0
    for i in range(num_sessions):
        try:
            # 创建搜索会话
            query_text = random.choice(sample_queries)
            session_id = f"{int(datetime.utcnow().timestamp())}_{hash(query_text)}"
            
            session = SearchSession(
                session_id=session_id,
                search_time=datetime.utcnow(),
                query_text=query_text,
                total_results=random.randint(50, 200)
            )
            db.session.add(session)
            db.session.flush()
            
            # 为每个会话创建10-20个搜索结果
            num_results = random.randint(10, 20)
            selected_works = random.sample(work_ids, num_results)
            
            for rank, work_id in enumerate(selected_works, 1):
                # 创建搜索结果
                result = SearchResult(
                    session_id=session_id,
                    entity_type='work',
                    entity_id=work_id,
                    rank=rank,
                    relevance_score=random.uniform(0.5, 1.0),
                    query_text=query_text,
                    result_page=((rank - 1) // 10) + 1,
                    result_position=((rank - 1) % 10) + 1
                )
                db.session.add(result)
                
                # 随机选择一些文档创建浏览记录（30%的概率）
                if random.random() < 0.3:
                    dwell_time = random.randint(300, 600)  # 5-10分钟
                    behavior = UserBehavior(
                        session_id=session_id,
                        document_id=work_id,
                        dwell_time=dwell_time,
                        behavior_time=datetime.utcnow()
                    )
                    db.session.add(behavior)
            
            sessions_created += 1
            print(f"已创建第 {sessions_created} 个搜索会话的记录")
            db.session.commit()
            
        except Exception as e:
            print(f"创建搜索记录时出错: {str(e)}")
            db.session.rollback()
    
    print(f"示例搜索记录创建完成，共创建 {sessions_created} 个会话")

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
    # 导入文献数据
    num_works = import_works()
    
    if num_works > 0:
        # 创建示例搜索记录
        create_sample_search_records(5)  # 创建5个搜索会话作为示例
    else:
        print("没有导入任何文献，跳过创建搜索记录") 