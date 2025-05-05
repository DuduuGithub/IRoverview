from flask import Blueprint, render_template, request, jsonify
from Database.model import Document, citation_network, SearchResult
from Database.config import db
from sqlalchemy import or_
from app.utils import get_citation_network_data, record_search_session, record_search_results, record_document_click, update_dwell_time, calculate_relevance_score, update_search_result_score
import re

searcher_bp = Blueprint('searcher', __name__,
                       template_folder='templates',
                       static_folder='static')

def highlight_text(text, keyword):
    if not text or not keyword:
        return text
    pattern = re.compile(f'({re.escape(keyword)})', re.IGNORECASE)
    return pattern.sub(r'<span class="highlight">\1</span>', text)

@searcher_bp.route('/search_page')
def search_page():
    return render_template('searcher/search_page.html')

@searcher_bp.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        keyword = data.get('keyword', '')
        page = data.get('page', 1)
        per_page = 10  # 每页显示的结果数

        if not keyword:
            return jsonify({
                'documents': [],
                'current_page': page,
                'total_pages': 0
            })

        # 构建搜索查询
        query = Document.query.filter(
            or_(
                Document.title.ilike(f'%{keyword}%'),
                Document.content.ilike(f'%{keyword}%'),
                Document.author.ilike(f'%{keyword}%')
            )
        )

        # 获取分页数据
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # 记录搜索会话
        session_id = record_search_session(data, pagination.total)
        if session_id:
            # 记录搜索结果
            record_search_results(session_id, pagination.items)
        
        # 格式化结果并添加高亮
        documents = []
        for doc in pagination.items:
            content_preview = doc.content[:200] + '...' if doc.content and len(doc.content) > 200 else doc.content
            documents.append({
                'id': doc.id,
                'title': highlight_text(doc.title, keyword),
                'author': highlight_text(doc.author, keyword),
                'publish_date': doc.publish_date.strftime('%Y-%m-%d') if doc.publish_date else None,
                'content': highlight_text(content_preview, keyword)
            })

        return jsonify({
            'documents': documents,
            'current_page': pagination.page,
            'total_pages': pagination.pages,
            'session_id': session_id  # 返回会话ID给前端
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@searcher_bp.route('/document/<int:doc_id>')
def document_detail(doc_id):
    try:
        session_id = request.args.get('session_id')
        document = Document.query.get_or_404(doc_id)
        
        # 记录文档点击
        if session_id:
            record_document_click(session_id, doc_id)
            
        return render_template('searcher/document_detail.html', 
                             document=document,
                             session_id=session_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@searcher_bp.route('/api/record-dwell-time', methods=['POST'])
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

@searcher_bp.route('/api/citation-network/<int:doc_id>')
def get_citation_network(doc_id):
    """获取文档的引用网络数据"""
    try:
        network_data = get_citation_network_data(doc_id)
        if network_data is None:
            return jsonify({'error': '未找到文档或获取引用网络失败'}), 404
        return jsonify(network_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 