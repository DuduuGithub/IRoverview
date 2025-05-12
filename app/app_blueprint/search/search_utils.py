from datetime import datetime
import math
import logging
from Database.model import SearchSession, SearchResult, Work, UserBehavior
from Database.config import db
<<<<<<< HEAD
from sqlalchemy import text

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def record_search_session(query_text, total_results):
=======
from flask import request
import uuid

def record_search_session(query, total_results=0):
>>>>>>> 59ca80dec5c161d8c0053d10189343fbab80cad2
    """
    记录一次搜索会话
    
<<<<<<< HEAD
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
=======
    参数:
        query (str): 搜索查询
        total_results (int): 搜索结果总数
        
    返回:
        str: 会话ID
    """
    try:
        # 生成唯一的会话ID
        session_id = str(uuid.uuid4())
        
        # 创建会话记录
        session = SearchSession(
            session_id=session_id,
            search_time=datetime.now(),  # 使用search_time而不是timestamp
            keyword=query,  # 使用query作为keyword字段
            search_type='keyword',  # 默认设置为keyword类型
            user_id=None,  # 可以在用户系统中设置
            total_results=total_results,
            client_info={"ip_address": request.remote_addr} if request else None
        )
        
        # 保存到数据库
        db.session.add(session)
        db.session.commit()
        
        # 返回会话ID
>>>>>>> 59ca80dec5c161d8c0053d10189343fbab80cad2
        return session_id
        
    except Exception as e:
<<<<<<< HEAD
        logger.error(f"创建搜索会话失败: {str(e)}", exc_info=True)
        db.session.rollback()
=======
        print(f"记录搜索会话出错: {e}")
>>>>>>> 59ca80dec5c161d8c0053d10189343fbab80cad2
        return None

def record_search_results(session_id, results, page, per_page):
    """记录搜索结果
    
    Args:
        session_id: 会话ID
        results: 搜索结果列表
        page: 当前页码
        per_page: 每页结果数
    """
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
                entity_id=result['id'],
                result_page=page,
                result_position=i + 1
            ).first()
            
            if existing_result:
                logger.info(f"结果记录已存在: session_id={session_id}, entity_id={result['id']}")
                continue
                
            search_result = SearchResult(
                session_id=session_id,
                entity_type='work',
                entity_id=result['id'],
                rank=position,
                relevance_score=result.get('relevance_score', 0.0),
                query_text=result.get('query_text', ''),
                result_page=page,
                result_position=i + 1,
                is_clicked=False,
                dwell_time=0,
                click_time=None
            )
            db.session.add(search_result)
        
        db.session.commit()
        logger.info(f"成功记录搜索结果: session_id={session_id}, page={page}, count={len(results)}")
    except Exception as e:
        logger.error(f"记录搜索结果失败: {str(e)}", exc_info=True)
        db.session.rollback()

def record_document_click(session_id, document_id):
    """记录文档点击（使用 UPSERT 避免并发冲突）"""
    if not session_id or not document_id:
        logger.warning(f"无效的会话ID或文档ID: session_id={session_id}, document_id={document_id}")
        return
    try:
        # 使用原生SQL的UPSERT语法
        stmt = text("""
            INSERT INTO user_behaviors (session_id, document_id, dwell_time, behavior_time)
            VALUES (:session_id, :document_id, 0, :behavior_time)
            ON DUPLICATE KEY UPDATE
                behavior_time = :behavior_time
        """)
        db.session.execute(stmt, {
            'session_id': session_id,
            'document_id': document_id,
            'behavior_time': datetime.now()
        })
        db.session.commit()
        logger.info(f"成功记录/更新点击: session_id={session_id}, document_id={document_id}")
    except Exception as e:
        logger.error(f"记录文档点击失败: session_id={session_id}, document_id={document_id}, 错误={str(e)}", exc_info=True)
        db.session.rollback()

def record_dwell_time(session_id, document_id, dwell_time):
    """累加文档停留时间（使用 UPSERT 避免并发冲突）"""
    try:
        if not session_id or not document_id:
            logging.warning(f"无效的会话ID或文档ID: session_id={session_id}, document_id={document_id}")
            return False
        if dwell_time <= 0:
            logging.warning(f"无效的停留时间: {dwell_time}秒")
            return False
        if dwell_time > 1200:
            dwell_time = 1200
            logging.info(f"停留时间超过20分钟，已限制为1200秒")

        # 使用原生SQL的UPSERT语法，累加dwell_time
        stmt = text("""
            INSERT INTO user_behaviors (session_id, document_id, dwell_time, behavior_time)
            VALUES (:session_id, :document_id, :dwell_time, :behavior_time)
            ON DUPLICATE KEY UPDATE
                dwell_time = dwell_time + :dwell_time,
                behavior_time = :behavior_time
        """)
        db.session.execute(stmt, {
            'session_id': session_id,
            'document_id': document_id,
            'dwell_time': dwell_time,
            'behavior_time': datetime.now()
        })
        db.session.commit()
        logging.info(f"成功记录/更新停留时间: session_id={session_id}, document_id={document_id}, dwell_time增加={dwell_time}秒")
        return True
    except Exception as e:
        db.session.rollback()
        logging.error(f"记录停留时间失败: {str(e)}", exc_info=True)
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