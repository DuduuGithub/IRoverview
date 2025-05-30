from datetime import datetime
import math
import logging
from Database.model import SearchSession, SearchResult, Work, UserBehavior
from Database.config import db
from sqlalchemy import text

from flask import request
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def record_search_session(query_text, total_results=0):
    """
    记录一次搜索会话
    
    Args:
        query_text: 检索式
        total_results: 搜索结果总数
    
    Returns:
        str: 会话ID，如果创建失败则返回 None
    """
    try:
        # 生成会话ID（使用时间戳和查询内容的哈希值）
        session_id = f"{int(datetime.utcnow().timestamp())}_{hash(query_text)}"
        
        # 检查是否已存在相同的会话
        existing_session = SearchSession.query.filter_by(session_id=session_id).first()
        if existing_session:
            logger.info(f"使用已存在的会话: session_id={session_id}")
            return session_id
        
        # 创建新的搜索会话
        search_session = SearchSession(
            session_id=session_id,
            search_time=datetime.utcnow(),
            query_text=query_text,
            total_results=total_results
        )
        
        # 添加并提交会话
        db.session.add(search_session)
        db.session.commit()
        
        logger.info(f"成功创建新会话: session_id={session_id}, query={query_text}")
        return session_id
        
    except Exception as e:
        logger.error(f"创建搜索会话失败: {str(e)}", exc_info=True)
        db.session.rollback()
        return None

def record_search_results(session_id, results, page, per_page):
    """记录搜索结果
    
    Args:
        session_id: 会话ID
        results: 搜索结果列表
        page: 当前页码
        per_page: 每页结果数
    """
    print(f"记录搜索结果: session_id={session_id}, page={page}, per_page={per_page}")
    if not session_id or not results:
        logger.warning("无效的会话ID或结果列表")
        return
        
    try:
        # 开始事务
        for i, result in enumerate(results):
            position = (page - 1) * per_page + i + 1
            
            # 检查是否已存在相同的结果记录
            existing_result = SearchResult.query.filter_by(
                session_id=session_id,
                entity_id=result['id']
            ).first()
            
            if existing_result:
                logger.info(f"结果记录已存在: session_id={session_id}, entity_id={result['id']}")
                continue
                
            search_result = SearchResult(
                session_id=session_id,
                entity_type='work',
                entity_id=result['id'],
                rank_position=position,
                relevance_score=result.get('relevance_score', 0.0),
                query_text=result.get('query_text', ''),
                result_page=page,
                result_position=i + 1
            )
            db.session.add(search_result)
            
            # 创建初始用户行为记录
            behavior = UserBehavior(
                session_id=session_id,
                document_id=result['id'],
                rank_position=position,
                is_clicked=False,
                dwell_time=0,
                behavior_time=datetime.utcnow()
            )
            db.session.add(behavior)
        
        db.session.commit()
        logger.info(f"成功记录搜索结果和初始用户行为: session_id={session_id}, page={page}, count={len(results)}")
    except Exception as e:
        logger.error(f"记录搜索结果失败: {str(e)}", exc_info=True)
        db.session.rollback()

def record_document_click(session_id, document_id):
    """记录文档点击
    
    Args:
        session_id: 会话ID
        document_id: 文档ID
    """
    if not session_id or not document_id:
        logger.warning("无效的会话ID或文档ID")
        return False
        
    try:
        # 查找或创建用户行为记录
        behavior = UserBehavior.query.filter_by(
            session_id=session_id,
            document_id=document_id
        ).first()
        
        current_time = datetime.utcnow()
        
        if behavior:
            behavior.is_clicked = True
            behavior.click_time = current_time
            behavior.behavior_time = current_time
        else:
            behavior = UserBehavior(
                session_id=session_id,
                document_id=document_id,
                is_clicked=True,
                click_time=current_time,
                dwell_time=0,
                behavior_time=current_time
            )
            db.session.add(behavior)
        
        db.session.commit()
        logger.info(f"成功记录文档点击: session_id={session_id}, document_id={document_id}")
        return True
        
    except Exception as e:
        logger.error(f"记录文档点击失败: {str(e)}", exc_info=True)
        db.session.rollback()
        return False

def record_dwell_time(session_id, document_id, dwell_time):
    """记录文档停留时间
    
    Args:
        session_id: 会话ID
        document_id: 文档ID
        dwell_time: 停留时间（秒）
    """
    if not session_id or not document_id or dwell_time <= 0:
        logger.warning("无效的参数")
        return False
        
    try:
        # 查找或创建用户行为记录
        behavior = UserBehavior.query.filter_by(
            session_id=session_id,
            document_id=document_id
        ).first()
        
        current_time = datetime.utcnow()
        
        if behavior:
            behavior.dwell_time = dwell_time
            behavior.behavior_time = current_time
        else:
            behavior = UserBehavior(
                session_id=session_id,
                document_id=document_id,
                dwell_time=dwell_time,
                behavior_time=current_time
            )
            db.session.add(behavior)
        
        db.session.commit()
        logger.info(f"成功记录停留时间: session_id={session_id}, document_id={document_id}, dwell_time={dwell_time}")
        return True
        
    except Exception as e:
        logger.error(f"记录停留时间失败: {str(e)}", exc_info=True)
        db.session.rollback()
        return False

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
    page_number = math.ceil(search_result.rank / items_per_page)
    
    # 计算相关性得分
    score = calculate_relevance_score(
        dwell_time=search_result.dwell_time or 0,
        page_number=page_number,
        items_per_page=items_per_page
    )
    
    # 更新得分
    search_result.relevance_score = score
    db.session.commit()

def get_search_history(session_id):
    """获取搜索历史
    
    Args:
        session_id (str): 会话ID
        
    Returns:
        dict: 搜索历史信息
    """
    try:
        session = SearchSession.query.filter_by(session_id=session_id).first()
        if not session:
            return None
            
        results = SearchResult.query.filter_by(session_id=session_id).all()
        
        return {
            'session': session.to_dict(),
            'results': [result.to_dict() for result in results]
        }
    except Exception as e:
        print(f"获取搜索历史失败: {str(e)}")
        return None 
        return None 

def record_rerank_operation(session_id, prompt, results):
    """记录重排序操作
    
    Args:
        session_id: 会话ID
        prompt: 重排序提示词
        results: 重排序后的结果列表
    """
    print(f"记录重排序操作: session_id={session_id}, prompt={prompt}")
    if not session_id or not results:
        logger.warning("无效的会话ID或结果列表")
        return
        
    try:
        # 更新会话信息
        session = SearchSession.query.filter_by(session_id=session_id).first()
        if session:
            session.last_rerank_time = datetime.utcnow()
            session.last_rerank_prompt = prompt
        
        # 更新搜索结果的排名
        for i, result in enumerate(results):
            search_result = SearchResult.query.filter_by(
                session_id=session_id,
                entity_id=result['id']
            ).first()
            
            if search_result:
                search_result.rerank_position = i + 1
                search_result.rerank_time = datetime.utcnow()
                search_result.rerank_prompt = prompt
        
        db.session.commit()
        logger.info(f"成功记录重排序操作: session_id={session_id}, prompt={prompt}")
        
    except Exception as e:
        logger.error(f"记录重排序操作失败: {str(e)}", exc_info=True)
        db.session.rollback() 