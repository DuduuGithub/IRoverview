from datetime import datetime
import math
from Database.model import SearchSession, SearchResult, Work
from Database.config import db

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
        
        # 获取搜索类型并映射到数据库允许的值
        search_type = search_data.get('search_type', 'basic')
        # 映射搜索类型到数据库允许的值
        type_mapping = {
            'basic': 'keyword',
            'advanced': 'advanced',
            'ai': 'semantic'
        }
        search_type = type_mapping.get(search_type, 'keyword')  # 默认使用keyword类型
        
        # 创建搜索会话记录
        search_session = SearchSession(
            session_id=session_id,
            search_time=datetime.utcnow(),
            keyword=search_data.get('keyword'),
            title_query=search_data.get('title'),
            author_query=search_data.get('author'),
            date_from=search_data.get('date_from'),
            date_to=search_data.get('date_to'),
            search_type=search_type,
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
        for rank, doc_id in enumerate(documents, 1):
            result = SearchResult(
                session_id=session_id,
                entity_id=doc_id,
                entity_type='work',  # 默认为work类型
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
            entity_id=document_id,
            entity_type='work'
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
            entity_id=document_id,
            entity_type='work'
        ).first()
        
        if result:
            result.dwell_time = dwell_time
            db.session.commit()
    except Exception as e:
        print(f"更新停留时间出错: {str(e)}")
        db.session.rollback()

def calculate_relevance_score(dwell_time, page_number, items_per_page=10):
    """
    计算文档相关性得分
    
    Args:
        dwell_time (int): 停留时间（秒）
        page_number (int): 文档所在页码
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
    db.session.commit() 