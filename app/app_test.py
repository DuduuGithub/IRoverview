import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from werkzeug.security import generate_password_hash
from flask import Flask, redirect, url_for, jsonify
from app_blueprint.search import searcher_bp
from app_blueprint.reader import reader_bp
from Database.model import *
from Database.config import db
import Database.config 
from sqlalchemy.sql import text
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def createApp(debug=False):
    app = Flask(__name__,
               static_folder='static',
               static_url_path='/static')
               
    
    # 加载配置
    app.config.from_object(Database.config)
    
    # 只在调试模式下显示SQL语句
    app.config['SQLALCHEMY_ECHO'] = debug
    
    # 初始化数据库
    db.init_app(app)
    
    # 注册蓝图
    app.register_blueprint(searcher_bp, url_prefix='/search')
    app.register_blueprint(reader_bp, url_prefix='/reader')
    print(app.url_map)
    return app

app = createApp(debug=False)  # 设置为False来关闭SQL语句输出

# 根路由重定向到首页
@app.route('/')
def index():
    return redirect(url_for('searcher.search_page'))  # 修改为searcher蓝图的search_page路由

# 添加一个路径来查看数据库内容
@app.route('/check_db')
def check_db():
    works = Work.query.all()
    result = []
    for work in works:
        result.append({
            'id': work.id,
            'title': work.title,
            'year': work.publication_year
        })
    return jsonify({"works": result})

# 添加测试数据
def add_test_data():
    print("开始添加测试数据...")
    
    # 创建一些作者
    authors = [
        Author(id=f"A{i}", display_name=name, orcid=orcid)
        for i, (name, orcid) in enumerate([
            ("John Smith", "0000-0001-1111-1111"),
            ("Sarah Johnson", "0000-0002-2222-2222"),
            ("Michael Brown", "0000-0003-3333-3333"),
            ("Emily Davis", "0000-0004-4444-4444"),
            ("David Wilson", "0000-0005-5555-5555")
        ], 1)
    ]
    
    for author in authors:
        db.session.add(author)
    
    db.session.commit()
    print(f"已添加 {len(authors)} 个作者")
    
    # 创建一些机构
    institutions = [
        Institution(id=f"I{i}", display_name=name, country_code=code)
        for i, (name, code) in enumerate([
            ("Peking University", "CN"),
            ("Stanford University", "US"),
            ("University of Cambridge", "UK"),
            ("Massachusetts Institute of Technology", "US"),
            ("University of Tokyo", "JP")
        ], 1)
    ]
    
    for institution in institutions:
        db.session.add(institution)
    
    db.session.commit()
    print(f"已添加 {len(institutions)} 个机构")
    
    # 创建一些概念
    concepts = [
        Concept(id=f"C{i}", display_name=name, level=level)
        for i, (name, level) in enumerate([
            ("Machine Learning", 1),
            ("Deep Learning", 2),
            ("Algorithm", 1),
            ("Data Structure", 2),
            ("Artificial Intelligence", 1),
            ("Natural Language Processing", 2),
            ("Computer Vision", 2)
        ], 1)
    ]
    
    for concept in concepts:
        db.session.add(concept)
    
    db.session.commit()
    print(f"已添加 {len(concepts)} 个概念")
    
    # 创建一些主题
    topics = [
        Topic(id=f"T{i}", display_name=name)
        for i, name in enumerate([
            "Deep Neural Networks",
            "Graph Algorithms",
            "Optimization Methods",
            "Distributed Systems",
            "Big Data Analytics"
        ], 1)
    ]
    
    for topic in topics:
        db.session.add(topic)
    
    db.session.commit()
    print(f"已添加 {len(topics)} 个主题")
    
    # 创建一些论文作品
    works = [
        Work(
            id=f"W{i}",
            title=title,
            display_name=title,
            publication_year=year,
            cited_by_count=cited,
            abstract_inverted_index=abstract
        )
        for i, (title, year, cited, abstract) in enumerate([
            (
                "Research on Image Recognition Algorithm Based on Deep Learning", 
                2022, 
                150, 
                json.dumps({"deep": [1, 5], "learning": [2], "algorithm": [7, 15], "image": [3], "recognition": [4]})
            ),
            (
                "Survey of Optimization Algorithms in Natural Language Processing", 
                2021, 
                98, 
                json.dumps({"natural": [1], "language": [2], "processing": [3], "optimization": [5], "algorithm": [6], "survey": [8]})
            ),
            (
                "Analysis of Consistency Algorithms in Distributed Systems", 
                2023, 
                75, 
                json.dumps({"distributed": [1], "system": [2], "consistency": [4], "algorithm": [5], "analysis": [6]})
            ),
            (
                "Efficient Sorting Algorithms in Big Data Environment", 
                2020, 
                120, 
                json.dumps({"big": [1], "data": [2], "efficient": [4], "sorting": [5], "algorithm": [6]})
            ),
            (
                "Applications of Machine Learning in Medical Diagnosis", 
                2022, 
                88, 
                json.dumps({"machine": [1], "learning": [2], "medical": [4], "diagnosis": [5], "application": [7]})
            )
        ], 1)
    ]
    
    for work in works:
        db.session.add(work)
    
    db.session.commit()
    print(f"已添加 {len(works)} 篇论文")
    
    # 创建作者与论文的关联
    authorships = [
        WorkAuthorship(work_id=works[w_idx-1].id, author_id=authors[a_idx-1].id, author_position=f"{pos}")
        for pos, (w_idx, a_idx) in enumerate([
            (1, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (4, 1),
            (5, 2)
        ], 1)
    ]
    
    for authorship in authorships:
        db.session.add(authorship)
    
    db.session.commit()
    print(f"已添加 {len(authorships)} 个作者关联")
    
    # 创建概念与论文的关联
    work_concepts = [
        WorkConcept(work_id=works[w_idx-1].id, concept_id=concepts[c_idx-1].id)
        for w_idx, c_idx in [
            (1, 2),  # 深度学习
            (1, 3),  # 算法
            (1, 7),  # 计算机视觉
            (2, 3),  # 算法
            (2, 6),  # 自然语言处理
            (3, 3),  # 算法
            (3, 4),  # 数据结构
            (4, 3),  # 算法
            (4, 4),  # 数据结构
            (5, 1),  # 机器学习
            (5, 5)   # 人工智能
        ]
    ]
    
    for work_concept in work_concepts:
        db.session.add(work_concept)
    
    db.session.commit()
    print(f"已添加 {len(work_concepts)} 个概念关联")
    
    # 创建主题与论文的关联
    work_topics = [
        WorkTopic(work_id=works[w_idx-1].id, topic_id=topics[t_idx-1].id)
        for w_idx, t_idx in [
            (1, 1),  # 深度神经网络
            (2, 3),  # 优化算法
            (3, 4),  # 分布式系统
            (4, 5),  # 大数据分析
            (5, 1)   # 深度神经网络
        ]
    ]
    
    for work_topic in work_topics:
        db.session.add(work_topic)
    
    db.session.commit()
    print(f"已添加 {len(work_topics)} 个主题关联")
    
    print("测试数据添加完成！")
    
    # 验证数据
    verify_test_data()

# 验证测试数据
def verify_test_data():
    works_count = Work.query.count()
    authors_count = Author.query.count()
    concepts_count = Concept.query.count()
    
    print(f"数据库中有 {works_count} 篇论文, {authors_count} 个作者, {concepts_count} 个概念")
    
    # 检查包含"Algorithm"的论文
    algorithm_works = Work.query.filter(Work.title.like('%Algorithm%')).all()
    print(f"包含'Algorithm'的论文有 {len(algorithm_works)} 篇:")
    for work in algorithm_works:
        print(f"  - {work.title} (ID: {work.id})")
    
    # 检查概念为"Algorithm"的论文关联
    algorithm_concept = Concept.query.filter_by(display_name="Algorithm").first()
    if algorithm_concept:
        algo_works = db.session.query(Work).join(
            WorkConcept, Work.id == WorkConcept.work_id
        ).filter(
            WorkConcept.concept_id == algorithm_concept.id
        ).all()
        
        print(f"概念为'Algorithm'的论文有 {len(algo_works)} 篇:")
        for work in algo_works:
            print(f"  - {work.title} (ID: {work.id})")

def record_document_click(session_id, document_id):
    """记录文档点击（只插入一次，后续只更新时间）"""
    if not session_id or not document_id:
        logger.warning(f"无效的会话ID或文档ID: session_id={session_id}, document_id={document_id}")
        return
    try:
        behavior = UserBehavior.query.filter_by(
            session_id=session_id,
            document_id=document_id
        ).first()
        if behavior:
            # 只更新时间，不重置dwell_time
            behavior.behavior_time = datetime.now()
            logger.info(f"更新已存在的点击记录: session_id={session_id}, document_id={document_id}")
        else:
            # 只在没有记录时插入
            behavior = UserBehavior(
                session_id=session_id,
                document_id=document_id,
                dwell_time=0,
                behavior_time=datetime.now()
            )
            db.session.add(behavior)
            logger.info(f"创建新的点击记录: session_id={session_id}, document_id={document_id}")
        db.session.commit()
    except Exception as e:
        logger.error(f"记录文档点击失败: session_id={session_id}, document_id={document_id}, 错误={str(e)}", exc_info=True)
        db.session.rollback()

if __name__ == '__main__':
    with app.app_context():
        try:
            # 删除所有表
            db.drop_all()
            db.session.commit()
            
            # 创建所有表
            db.create_all()
            db.session.commit()
            print("数据库初始化完成！")
            
            # 添加测试数据
            add_test_data()
            print("测试数据已添加！")
            
        except Exception as e:
            print(f"数据库初始化失败: {str(e)}")
            db.session.rollback()
            raise
    
    app.run(debug=True)  # 这里的debug只控制Flask的调试模式，不影响SQL输出 