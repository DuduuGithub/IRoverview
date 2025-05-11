from flask import Blueprint, render_template, request, jsonify
from Database.model import Work, Author, WorkAuthorship, SearchResult
from Database.config import db
from ..search.search_utils import (
    record_search_session,
    record_search_results,
    record_document_click,
    update_dwell_time,
    calculate_relevance_score,
    update_search_result_score
)
import sys
# 创建蓝图
reader_bp = Blueprint('reader', __name__,
                     template_folder='templates',
                     static_folder='static')

@reader_bp.route('/document/<doc_id>')
def document_detail(doc_id):
    try:
        work = Work.query.get(doc_id)
        if work:
            print(f"[INFO] 查到文档: {work.id} - {work.title}")
            message = f"查到文档: {work.id} - {work.title}"
        else:
            print(f"[WARN] 没查到文档: {doc_id}")
            message = f"没有查到文档: {doc_id}"
        # 获取作者
        authorships = WorkAuthorship.query.filter_by(work_id=doc_id).all()
        authors = []
        for authorship in authorships:
            if authorship.author_id:
                author = Author.query.get(authorship.author_id)
                if author:
                    authors.append(author.display_name)
        return render_template('reader/document_detail.html', 
                              work=work,
                              authors=authors,
                              message=message)
    except Exception as e:
        print(f"[ERROR] 查询文档失败: {e}")
        return jsonify({'error': str(e)}), 500

@reader_bp.route('/api/record-dwell-time', methods=['POST'])
def record_dwell_time():
    """记录文档停留时间"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        document_id = data.get('document_id')
        dwell_time = data.get('dwell_time')
        
        if not all([session_id, document_id, dwell_time]):
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 更新停留时间
        result = SearchResult.query.filter_by(
            session_id=session_id,
            document_id=document_id
        ).first()
        
        if result:
            result.dwell_time = dwell_time
            # 更新相关性得分
            update_search_result_score(result)
            db.session.commit()
            
        return jsonify({'message': '停留时间记录成功'})
        
    except Exception as e:
        print(f"记录停留时间出错: {str(e)}")
        return jsonify({'error': '记录停留时间失败'}), 500
