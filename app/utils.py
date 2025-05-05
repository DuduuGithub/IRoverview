# 连接数据库的基本功能
import sys
import os
from datetime import datetime
import math

from flask import jsonify
from sqlalchemy import text
# 将项目根目录添加到 sys.path,Python默认从当前文件所在的目录开始找，也就是app文件夹开始找
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Database.config import db
from Database.model import *
from sqlalchemy import or_
from flask import session
from flask_login import current_user
import json
import openai

def record_search_session(search_data, total_results):
    """
    记录搜索会话信息
    
    Args:
        search_data: 搜索请求数据
        total_results: 搜索结果总数
    
    Returns:
        str: 会话ID
    """
    try:
        # 生成会话ID（使用时间戳和随机数）
        session_id = f"{int(datetime.utcnow().timestamp())}_{hash(str(search_data))}"
        
        # 创建搜索会话记录
        search_session = SearchSession(
            session_id=session_id,
            search_time=datetime.utcnow(),
            keyword=search_data.get('keyword'),
            title_query=search_data.get('title'),
            author_query=search_data.get('author'),
            date_from=search_data.get('date_from'),
            date_to=search_data.get('date_to'),
            search_type=search_data.get('type', 'basic'),
            total_results=total_results
        )
        
        db.session.add(search_session)
        db.session.commit()
        
        return session_id
    except Exception as e:
        print(f"记录搜索会话出错: {str(e)}")
        db.session.rollback()
        return None

def record_search_results(session_id, documents):
    """
    记录搜索结果列表
    
    Args:
        session_id: 搜索会话ID
        documents: 文档列表
    """
    try:
        # 批量创建搜索结果记录
        search_results = []
        for rank, doc in enumerate(documents, 1):
            result = SearchResult(
                session_id=session_id,
                document_id=doc.id,
                rank_position=rank
            )
            search_results.append(result)
        
        db.session.bulk_save_objects(search_results)
        db.session.commit()
    except Exception as e:
        print(f"记录搜索结果出错: {str(e)}")
        db.session.rollback()

def record_document_click(session_id, document_id):
    """
    记录文档点击信息
    
    Args:
        session_id: 搜索会话ID
        document_id: 被点击的文档ID
    """
    try:
        # 获取当前会话中最大的点击顺序
        max_order = db.session.query(db.func.max(SearchResult.click_order))\
            .filter_by(session_id=session_id)\
            .scalar() or 0
            
        # 更新搜索结果记录
        result = SearchResult.query.filter_by(
            session_id=session_id,
            document_id=document_id
        ).first()
        
        if result:
            result.is_clicked = True
            result.click_time = datetime.utcnow()
            result.click_order = max_order + 1
            db.session.commit()
    except Exception as e:
        print(f"记录文档点击出错: {str(e)}")
        db.session.rollback()

def update_dwell_time(session_id, document_id, dwell_time):
    """
    更新文档停留时间
    
    Args:
        session_id: 搜索会话ID
        document_id: 文档ID
        dwell_time: 停留时间（秒）
    """
    try:
        result = SearchResult.query.filter_by(
            session_id=session_id,
            document_id=document_id
        ).first()
        
        if result:
            result.dwell_time = dwell_time
            db.session.commit()
    except Exception as e:
        print(f"更新停留时间出错: {str(e)}")
        db.session.rollback()

def get_citation_network_data(doc_id):
    """
    获取指定文档的引用网络数据
    
    Args:
        doc_id: 文档ID
        
    Returns:
        dict: 包含节点和边的网络数据
    """
    try:
        # 获取中心文档
        center_doc = Document.query.get(doc_id)
        if not center_doc:
            return None
            
        # 初始化节点和边的列表
        nodes = []
        edges = []
        node_ids = set()  # 用于跟踪已添加的节点

        # 添加中心节点
        nodes.append({
            'id': center_doc.id,
            'label': center_doc.title,
            'type': 'center',  # 中心节点类型
            'author': center_doc.author,
            'date': center_doc.publish_date.strftime('%Y-%m-%d') if center_doc.publish_date else None
        })
        node_ids.add(center_doc.id)

        # 添加引用的文献（向外的引用）
        for cited_doc in center_doc.citations:
            if cited_doc.id not in node_ids:
                nodes.append({
                    'id': cited_doc.id,
                    'label': cited_doc.title,
                    'type': 'cited',  # 被引用节点类型
                    'author': cited_doc.author,
                    'date': cited_doc.publish_date.strftime('%Y-%m-%d') if cited_doc.publish_date else None
                })
                node_ids.add(cited_doc.id)
            edges.append({
                'source': center_doc.id,
                'target': cited_doc.id,
                'type': 'cites'
            })

        # 添加引用该文献的文献（向内的引用）
        for citing_doc in center_doc.cited_by:
            if citing_doc.id not in node_ids:
                nodes.append({
                    'id': citing_doc.id,
                    'label': citing_doc.title,
                    'type': 'citing',  # 引用中心文献的节点类型
                    'author': citing_doc.author,
                    'date': citing_doc.publish_date.strftime('%Y-%m-%d') if citing_doc.publish_date else None
                })
                node_ids.add(citing_doc.id)
            edges.append({
                'source': citing_doc.id,
                'target': center_doc.id,
                'type': 'cites'
            })

        return {
            'nodes': nodes,
            'edges': edges
        }
        
    except Exception as e:
        print(f"获取引用网络数据出错: {str(e)}")
        return None

#全文搜索
def db_context_query(query, doc_type=None, date_from=None, date_to=None):
    """
    实现全文检索，查询文书标题或原文中包含关键字的记录，并支持高级搜索。
    支持部分匹配和多个关键词。
    """
    try:
        # 处理搜索关键词，添加通配符和 + 操作符
        search_terms = query.split()
        formatted_query = ' '.join([f'+*{term}*' for term in search_terms])
        
        # 构建基本的全文检索 SQL 查询
        sql = """
            SELECT Doc_id FROM Documents
            WHERE MATCH(Doc_title, Doc_simplifiedText, Doc_originalText) 
            AGAINST(:query IN BOOLEAN MODE)
            OR Doc_title LIKE :like_query
            OR Doc_simplifiedText LIKE :like_query
            OR Doc_originalText LIKE :like_query
        """
        params = {
            'query': formatted_query,
            'like_query': f'%{query}%'  # 添加 LIKE 查询作为备选
        }
        
        # 添加文档类型筛选条件（如果提供）
        if doc_type:
            sql += " AND Doc_type = :doc_type"
            params['doc_type'] = doc_type
        
        # 添加日期范围筛选条件（如果提供）
        if date_from:
            sql += " AND Doc_createdAt >= :date_from"
            params['date_from'] = date_from
        if date_to:
            sql += " AND Doc_createdAt <= :date_to"
            params['date_to'] = date_to
        
        # 添加排序条件
        sql += """ 
            ORDER BY MATCH(Doc_title, Doc_simplifiedText, Doc_originalText) 
            AGAINST(:query IN BOOLEAN MODE) DESC
        """
        
        print(f"Executing search with query: {formatted_query}")
        print(f"SQL: {sql}")
        print(f"Params: {params}")
        
        # 执行查询获取文档ID
        result = db.session.execute(text(sql), params)
        doc_ids = [row[0] for row in result.fetchall()]
        
        if not doc_ids:
            return []
            
        # 使用找到的文档ID从 Document中获取完整信息
        display_results = Document.query.filter(
            Document.Doc_id.in_(doc_ids)
        ).all()
        
        return display_results
        
    except Exception as e:
        print(f"全文搜索出错: {str(e)}")
        return []

def generate_search_query(text, api_key=None):
    """
    使用AI助手生成检索式
    
    Args:
        text: 文本内容
        api_key: OpenAI API密钥（可选）
        
    Returns:
        str: 生成的检索式，如果生成失败则返回None
    """
    try:
        # 设置API密钥
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API密钥未提供")
        openai.api_key = api_key

        # 调用API生成检索式
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的文献检索专家，擅长将文本内容转换为布尔检索式。"
                                            "请分析输入的文本，提取关键概念和术语，生成一个合适的布尔检索式。"
                                            "检索式应该包含AND、OR、NOT等布尔操作符，并使用括号表示优先级。"},
                {"role": "user", "content": f"请为以下文本生成一个布尔检索式：\n\n{text}"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成检索式出错: {str(e)}")
        return None

def get_search_session_data(min_dwell_time=30):
    """
    获取搜索会话数据，用于训练数据生成
    
    Args:
        min_dwell_time: 最小停留时间（秒）
        
    Returns:
        list: 会话数据列表
    """
    try:
        session_data = []
        
        # 获取所有搜索会话
        sessions = SearchSession.query.all()
        
        for session in sessions:
            # 获取会话中的搜索结果
            results = SearchResult.query.filter_by(session_id=session.session_id).all()
            
            if not results:
                continue
            
            # 获取点击的文档
            clicked_docs = [r for r in results if r.is_clicked and (r.dwell_time or 0) >= min_dwell_time]
            
            if not clicked_docs:
                continue
            
            # 收集会话数据
            session_info = {
                'session_id': session.session_id,
                'original_query': session.keyword,
                'clicks': []
            }
            
            # 收集点击数据
            for click in clicked_docs:
                doc = Document.query.get(click.document_id)
                if not doc:
                    continue
                    
                click_info = {
                    'document_id': doc.id,
                    'content': doc.content,
                    'rank_position': click.rank_position,
                    'dwell_time': click.dwell_time,
                    'click_order': click.click_order
                }
                session_info['clicks'].append(click_info)
            
            if session_info['clicks']:
                session_data.append(session_info)
        
        return session_data
        
    except Exception as e:
        print(f"获取会话数据出错: {str(e)}")
        return []

def calculate_relevance_score(dwell_time, page_number, items_per_page=10):
    """
    计算文档相关性得分
    
    Args:
        dwell_time (int): 停留时间（秒）
        page_number (int): 文档所在页码，如果为None则根据rank_position计算
        items_per_page (int): 每页显示的文档数量，默认为10
    
    Returns:
        float: 相关性得分，范围[-1, 1]
    """
    if dwell_time == 0:  # 未点击的文档
        return -1.0
    
    # 基础得分：根据停留时间计算，归一化到[0,1]区间
    # 使用对数函数将停留时间映射到[0,1]区间，最大停留时间假设为300秒
    base_score = math.log(dwell_time + 1) / math.log(301)
    
    # 页面衰减因子：考虑用户浏览到该页面的耐心程度
    # 使用平方根函数使得衰减更平缓
    page_decay = 1.0 / math.sqrt(page_number)
    
    # 最终得分：基础得分 * 页面衰减因子
    return base_score * page_decay

def update_search_result_score(search_result, items_per_page=10):
    """
    更新搜索结果的相关性得分
    
    Args:
        search_result (SearchResult): 搜索结果记录
        items_per_page (int): 每页显示的文档数量，默认为10
    """
    if not search_result:
        return
        
    # 计算页码
    page_number = math.ceil(search_result.rank_position / items_per_page)
    
    # 计算相关性得分
    score = calculate_relevance_score(
        dwell_time=search_result.dwell_time or 0,
        page_number=page_number,
        items_per_page=items_per_page
    )
    
    # 更新得分
    search_result.relevance_score = score