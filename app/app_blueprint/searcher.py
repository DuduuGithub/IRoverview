# 检索页面，即是主页面

# 搜索框是文书内容的全文搜索，并类似知网提供筛选条件(高级检索)
from flask import Blueprint,request, render_template, jsonify, session
from sqlalchemy import or_
from Database.config import db
import openai
from Database.model import Work, Author, WorkAuthorship, SearchSession, SearchResult
import uuid
from datetime import datetime

searcher_bp = Blueprint('searcher', __name__, 
                       template_folder='../templates/searcher',
                       static_folder='../static/searcher')

@searcher_bp.route('/')
def index():
    return render_template('searcher/search_page.html')

    

#筛选字段需更新
@searcher_bp.route('/search', methods=['POST'])
def search():
    try:
        # 获取搜索参数
        keyword = request.form.get('keyword', '')
        title_query = request.form.get('title', '')
        author_query = request.form.get('author', '')
        date_from = request.form.get('date_from')
        date_to = request.form.get('date_to')
        search_type = request.form.get('search_type', 'keyword')
        
        # 创建搜索会话
        session_id = str(uuid.uuid4())
        search_session = SearchSession(
            session_id=session_id,
            keyword=keyword,
            title_query=title_query,
            author_query=author_query,
            date_from=datetime.strptime(date_from, '%Y-%m-%d') if date_from else None,
            date_to=datetime.strptime(date_to, '%Y-%m-%d') if date_to else None,
            search_type=search_type,
            client_info={"user_agent": request.user_agent.string}
        )
        
        db.session.add(search_session)
        
        # 执行搜索（这里以作品搜索为例）
        query = Work.query
        
        if keyword:
            query = query.filter(Work.title.ilike(f'%{keyword}%'))
        if title_query:
            query = query.filter(Work.title.ilike(f'%{title_query}%'))
        if author_query:
            # 通过作者署名关联查询
            query = query.join(WorkAuthorship).join(Author).filter(
                Author.display_name.ilike(f'%{author_query}%')
            )
        if date_from:
            query = query.filter(Work.publication_date >= date_from)
        if date_to:
            query = query.filter(Work.publication_date <= date_to)
            
        results = query.limit(50).all()  # 限制返回50条结果
        
        # 记录搜索结果
        for rank, work in enumerate(results, 1):
            search_result = SearchResult(
                session_id=session_id,
                entity_type='work',
                entity_id=work.id,
                rank_position=rank,
                relevance_score=1.0  # 这里可以根据实际相关性算法计算得分
            )
            db.session.add(search_result)
        
        # 更新搜索会话的结果数
        search_session.total_results = len(results)
        db.session.commit()
        
        # 准备返回数据
        results_data = []
        for work in results:
            # 获取作者信息
            authors = []
            for authorship in work.authorships:
                if authorship.author:
                    authors.append({
                        'name': authorship.author.display_name,
                        'institution': authorship.institution.display_name if authorship.institution else None
                    })
            
            results_data.append({
                'id': work.id,
                'title': work.title,
                'authors': authors,
                'publication_date': work.publication_date.strftime('%Y-%m-%d') if work.publication_date else None,
                'type': work.type,
                'cited_by_count': work.cited_by_count
            })
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'total_results': len(results),
            'results': results_data
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@searcher_bp.route('/record_click', methods=['POST'])
def record_click():
    try:
        session_id = request.form.get('session_id')
        entity_id = request.form.get('entity_id')
        
        # 更新搜索结果的点击信息
        search_result = SearchResult.query.filter_by(
            session_id=session_id,
            entity_id=entity_id
        ).first()
        
        if search_result:
            search_result.is_clicked = True
            search_result.click_time = datetime.now()
            search_result.click_order = (
                db.session.query(db.func.count(SearchResult.id))
                .filter(
                    SearchResult.session_id == session_id,
                    SearchResult.is_clicked == True
                ).scalar() + 1
            )
            db.session.commit()
            
        return jsonify({'status': 'success'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@searcher_bp.route('/ai-assist', methods=['POST'])
def ai_assist():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        user_query = data['query']
        
        # 调用OpenAI API构建检索式
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专业的文献检索助手，帮助用户将自然语言需求转换为规范的检索式。"},
                {"role": "user", "content": f"请将以下检索需求转换为布尔检索式：{user_query}"}
            ]
        )
        
        suggested_query = response.choices[0].message.content
        
        return jsonify({
            'suggested_query': suggested_query
        })
        
    except Exception as e:
        print(f"AI助手出错: {str(e)}")
        return jsonify({'error': str(e)}), 500


