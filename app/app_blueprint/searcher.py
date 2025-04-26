# 检索页面，即是主页面

# 搜索框是文书内容的全文搜索，并类似知网提供筛选条件(高级检索)
from flask import Blueprint,request, render_template, jsonify
from utils import db_context_query
from sqlalchemy import or_
from Database.model import *
from Database.config import db
from yearToyearTools.run import convert_to_gregorian

searcher_bp=Blueprint('searcher',__name__,template_folder='app/templates/searcher',static_folder='app/static/searcher',url_prefix='/searcher')

@searcher_bp.route('/')
def index():
    return render_template('searcher/search_page.html')

    

#筛选字段需更新
@searcher_bp.route('/search', methods=['POST'])
def search():
    try:
        
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
            
        print("Received search request:", data)
        
        search_type = data.get('type', 'basic')
        page = data.get('page', 1)
        per_page = 9
        
        # 构建基础查询
        query = DocumentDisplayView.query
        
        if search_type == 'advanced':
            # 精细搜索
            person = data.get('person', '').strip()
            title = data.get('title', '').strip()
            content = data.get('content', '').strip()
            
            # 只有当提供了搜索条件时才进行筛选
            if person or title or content:
                if person:
                    query = query.filter(
                        or_(
                            DocumentDisplayView.ContractorInfo.like(f'%{person}%'),
                            DocumentDisplayView.ParticipantInfo.like(f'%{person}%')
                        )
                    )
                if title:
                    query = query.filter(DocumentDisplayView.Doc_title.like(f'%{title}%'))
                if content:
                    content_results = db_context_query(content)
                    content_ids = [doc.Doc_id for doc in content_results]
                    if content_ids:
                        query = query.filter(DocumentDisplayView.Doc_id.in_(content_ids))
                    else:
                        return jsonify({
                            'documents': [],
                            'total_pages': 0,
                            'current_page': page
                        })
        else:
            # 基本搜索
            keyword = data.get('keyword', '').strip()
            start_date = data.get('startDate')
            end_date = data.get('endDate')
            doc_type = data.get('docType')
            
            # 只有当提供了搜索条件时才进行筛选
            if keyword or start_date or end_date or doc_type:
                if keyword:
                    query = query.filter(
                        or_(
                            DocumentDisplayView.Doc_id.like(f'%{keyword}%'),
                            DocumentDisplayView.Doc_title.like(f'%{keyword}%'),
                            DocumentDisplayView.ContractorInfo.like(f'%{keyword}%'),
                            DocumentDisplayView.ParticipantInfo.like(f'%{keyword}%')
                        )
                    )
                
                # 处理时间筛选
                if start_date:
                    # 转换为公历日期
                    standard_start = convert_to_gregorian(start_date)
                    if standard_start and ':' in standard_start:
                        year, month, day = map(int, standard_start.split(':'))
                        if month == 0:
                            standard_start = f"{year}-01-01"
                        else:
                            standard_start = f"{year}-{month:02d}-01"
                        query = query.filter(DocumentDisplayView.Doc_standardTime >= standard_start)
                
                if end_date:
                    # 转换为公历日期
                    standard_end = convert_to_gregorian(end_date)
                    if standard_end and ':' in standard_end:
                        year, month, day = map(int, standard_end.split(':'))
                        if month == 0:
                            standard_end = f"{year}-12-31"
                        else:
                            standard_end = f"{year}-{month:02d}-31"
                        query = query.filter(DocumentDisplayView.Doc_standardTime <= standard_end)
                
                if doc_type:
                    query = query.filter(DocumentDisplayView.Doc_type == doc_type)
        
        # 执行分页查询
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        total_pages = pagination.pages
        documents = pagination.items
        
        # 格式化结果
        def format_contractors(contractor_info):
            """格式化契约人信息"""
            if not contractor_info:
                return ''
            # 移除所有《》
            text = contractor_info.replace('《', '').replace('》', '')
            # 处理关系
            if '（' in text:
                names, relation = text.split('（')
                relation = relation.rstrip('）')
                # 只有在关系不为 null 且不为空时才显示
                if relation and relation.lower() != 'null':
                    return f"{names}, {relation}"
                return names
            return text
        
        def format_participants(participant_info):
            """格式化参与人信息"""
            if not participant_info:
                return []
            result = []
            # 分割每个参与人信息
            parts = participant_info.split('《')
            for part in parts:
                if not part:
                    continue
                # 处理每个参与人
                part = part.strip('》')
                if '（' in part:
                    name, role = part.split('（')
                    role = role.rstrip('）')
                    if name and role:  # 确保名字和角色都不为空
                        result.append(f"{name}（{role}）")
            return result
        
        formatted_results = [{
            'doc_id': doc.Doc_id,
            'title': doc.Doc_title,
            'type': doc.Doc_type,
            'summary': doc.Doc_summary,
            'contractors': format_contractors(doc.ContractorInfo),
            'participants': format_participants(doc.ParticipantInfo)
        } for doc in documents]
        
        return jsonify({
            'documents': formatted_results,
            'total_pages': total_pages,
            'current_page': page
        })
        
    except Exception as e:
        print(f"搜索文书时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


