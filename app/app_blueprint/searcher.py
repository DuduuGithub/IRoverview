# 检索页面，即是主页面

# 搜索框是文书内容的全文搜索，并类似知网提供筛选条件(高级检索)
from flask import Blueprint,request, render_template, jsonify
from sqlalchemy import or_
from Database.model import Document
from Database.config import db
import openai

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
        
        data = request.get_json()
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
            
        print("Received search request:", data)
        
        search_type = data.get('type', 'basic')
        page = data.get('page', 1)
        per_page = 10
        
        # 构建基础查询
        query = Document.query
        
        if search_type == 'basic':
            # 基本搜索
            keyword = data.get('keyword', '').strip()
            if keyword:
                query = query.filter(
                    or_(
                        Document.title.like(f'%{keyword}%'),
                        Document.author.like(f'%{keyword}%'),
                        Document.content.like(f'%{keyword}%'),
                        Document.keywords.like(f'%{keyword}%')
                    )
                )
                
        elif search_type == 'advanced':
            # 高级搜索
            title = data.get('title', '').strip()
            author = data.get('author', '').strip()
            keywords = data.get('keywords', '').strip()
            date_from = data.get('dateFrom')
            date_to = data.get('dateTo')
            
            if title:
                query = query.filter(Document.title.like(f'%{title}%'))
            if author:
                query = query.filter(Document.author.like(f'%{author}%'))
            if keywords:
                query = query.filter(Document.keywords.like(f'%{keywords}%'))
            if date_from:
                query = query.filter(Document.publish_date >= date_from)
            if date_to:
                query = query.filter(Document.publish_date <= date_to)
                
        elif search_type == 'query':
            # 检索式搜索
            search_query = data.get('query', '').strip()
            if search_query:
                # 这里需要实现检索式解析和查询逻辑
                pass
        
        # 执行分页查询
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        total_pages = pagination.pages
        documents = pagination.items
        
        # 使用to_dict方法格式化结果
        formatted_results = [doc.to_dict() for doc in documents]
        
        return jsonify({
            'documents': formatted_results,
            'total_pages': total_pages,
            'current_page': page
        })
        
    except Exception as e:
        print(f"搜索出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

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


