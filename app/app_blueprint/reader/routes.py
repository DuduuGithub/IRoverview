from flask import Blueprint, render_template, request, jsonify
from Database.model import Work, Author, WorkAuthorship, SearchResult, YearlyStat
from Database.config import db
from ..search.search_utils import (
    record_search_session,
    record_search_results,
    record_document_click,
    record_dwell_time,
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
        # 获取session_id
        session_id = request.args.get('session_id')
        print(f"[INFO] 访问文档详情页: doc_id={doc_id}, session_id={session_id}")
        
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
                    
        # 构建文档数据
        document_data = {
            'id': work.id if work else None,
            'title': work.title if work else None,
            'authors': authors,
            'session_id': session_id  # 添加session_id到文档数据中
        }

        # 获取年度引用统计数据
        yearly_citations = YearlyStat.query.filter_by(
            entity_id=doc_id#,
            # entity_type='work'
        ).order_by(YearlyStat.year).all()
        
        print(f"[DEBUG] 查询到的年度引用数据数量: {len(yearly_citations)}")
        for citation in yearly_citations:
            print(f"[DEBUG] 年份: {citation.year}, 引用次数: {citation.cited_by_count}")
        
        # 处理年度引用数据，补全缺失年份
        if yearly_citations:
            min_year = min(stat.year for stat in yearly_citations)
            max_year = max(stat.year for stat in yearly_citations)
            
            # 创建完整的年份序列和对应的引用数
            complete_years = list(range(min_year, max_year + 1))
            existing_citations = {stat.year: stat.cited_by_count for stat in yearly_citations}
            
            citation_data = {
                'years': complete_years,
                'citations': [existing_citations.get(year, 0) for year in complete_years]
            }
            print(f"[DEBUG] 处理后的数据: {citation_data}")
        else:
            citation_data = {
                'years': [],
                'citations': []
            }
            print("[DEBUG] 没有找到年度引用数据")
                    
        return render_template('reader/document_detail.html', 
                             work=work,
                             authors=authors,
                             message=message,
                             document_data=document_data,
                             citation_data=citation_data)  # 传递document_data到模板
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
            
        # 记录停留时间
        if record_dwell_time(session_id, document_id, dwell_time):
            return jsonify({'message': '停留时间记录成功'})
        else:
            return jsonify({'error': '记录停留时间失败'}), 500
        
    except Exception as e:
        print(f"记录停留时间出错: {str(e)}")
        return jsonify({'error': '记录停留时间失败'}), 500
