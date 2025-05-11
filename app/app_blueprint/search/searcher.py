from flask import Blueprint, render_template, request, jsonify
from Database.model import Work, SearchResult, Author, WorkAuthorship
from Database.config import db
from sqlalchemy import or_
from .search_utils import (
    record_search_session,
    record_search_results,
    # record_document_click,
    # update_dwell_time,
    # calculate_relevance_score,
    # update_search_result_score
)
from .basicSearch.search import search as basic_search
from .proSearch.search import search as advanced_search_engine
from .aiSearch.search import convert_to_structured_query, search as ai_search_engine
from .rank.rank import SORT_BY_RELEVANCE, SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED
import re
import sys
import os

# 添加项目根目录到sys.path，避免相对导入问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# 创建蓝图
searcher_bp = Blueprint('searcher', __name__,
                       template_folder='templates',
                       static_folder='static')

def highlight_text(text, keywords):
    if not text:
        return text
    
    if not keywords:
        return text
    
    # 如果keywords是字符串，转换为列表
    if isinstance(keywords, str):
        # 检查是否包含布尔操作符
        if any(op in keywords.upper() for op in [' AND ', ' OR ', ' NOT ']):
            # 提取布尔表达式中的关键词
            extracted = re.findall(r'\b\w+\b', keywords)
            keywords = [k for k in extracted if k.upper() not in ['AND', 'OR', 'NOT']]
        else:
            keywords = [keywords]
    
    # 处理多个关键词
    result = text
    for keyword in keywords:
        if keyword and len(keyword.strip()) > 0:
            pattern = re.compile(f'({re.escape(keyword.strip())})', re.IGNORECASE)
            result = pattern.sub(r'<span class="highlight">\1</span>', result)
    
    return result

@searcher_bp.route('/')
@searcher_bp.route('/index')
def index():
    """搜索首页"""
    return render_template('search/index.html')

@searcher_bp.route('/search_page')
def search_page():
    return render_template('search/index.html')

@searcher_bp.route('/search', methods=['POST'])
@searcher_bp.route('/api/basic/search', methods=['POST'])
def api_search():
    try:
        data = request.get_json()
        keyword = data.get('keyword', '')
        sort_method = data.get('sort_method', SORT_BY_RELEVANCE)
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)
        
        if not keyword:
            return jsonify({
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page
            })
        
        # 提取关键词，用于高亮
        keywords = []
        if ' AND ' in keyword.upper() or ' OR ' in keyword.upper() or ' NOT ' in keyword.upper():
            # 提取布尔表达式中的关键词
            extracted = re.findall(r'\b\w+\b', keyword)
            keywords = [k for k in extracted if k.upper() not in ['AND', 'OR', 'NOT']]
        else:
            keywords = [keyword]
        
        # 执行搜索
        work_ids = basic_search(query_text=keyword, use_db=True, sort_method=sort_method)
        total = len(work_ids)
        
        # 分页
        start = (page - 1) * per_page
        end = start + per_page
        paginated_ids = work_ids[start:end] if start < len(work_ids) else []
        
        # 获取搜索结果
        works = []
        for work_id in paginated_ids:
            work = Work.query.get(work_id)
            if work:
                # 获取作者
                authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
                authors = []
                for authorship in authorships:
                    if authorship.author_id:
                        author = Author.query.get(authorship.author_id)
                        if author:
                            authors.append(author.display_name)
                
                # 创建结果对象
                result = {
                    'id': work.openalex or work.id,  # 使用 openalex ID 或原始 ID
                    'title': highlight_text(work.title or work.display_name or '', keywords),
                    'authors': ', '.join(authors) if authors else '未知作者',
                    'year': work.publication_year,
                    'cited_by_count': work.cited_by_count or 0,
                    'abstract': work.abstract_inverted_index or ''
                }
                works.append(result)
        
        return jsonify({
            'results': works,
            'total': total,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        print(f"搜索出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@searcher_bp.route('/api/basic/advanced_search', methods=['POST'])
def api_advanced_search():
    try:
        data = request.get_json()
        title = data.get('title', '')
        author = data.get('author', '')
        keyword = data.get('keyword', '')
        institution = data.get('institution', '')
        date_from = data.get('date_from', '')
        date_to = data.get('date_to', '')
        sort_method = data.get('sort_method', SORT_BY_RELEVANCE)
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)
        
        # 构建高级搜索查询
        query_parts = []
        keywords_for_highlight = []
        
        if title:
            query_parts.append(f'title:"{title}"')
            keywords_for_highlight.append(title)
        if author:
            query_parts.append(f'author:"{author}"')
            keywords_for_highlight.append(author)
        if keyword:
            query_parts.append(f'keyword:"{keyword}"')
            keywords_for_highlight.append(keyword)
        if institution:
            query_parts.append(f'institution:"{institution}"')
            keywords_for_highlight.append(institution)
        
        # 添加时间范围
        if date_from or date_to:
            date_from = date_from or '1900-01-01'
            date_to = date_to or '2099-12-31'
            query_parts.append(f'time:"{date_from}~{date_to}"')
        
        # 组合查询
        advanced_query = ' AND '.join(query_parts)
        
        if not advanced_query:
            return jsonify({
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page
            })
        
        print(f"高级搜索查询: {advanced_query}")  # 添加日志
        
        # 执行高级搜索
        work_ids = advanced_search_engine(query_text=advanced_query, use_db=True, sort_method=sort_method)
        total = len(work_ids)
        
        print(f"搜索结果数量: {total}")  # 添加日志
        
        # 分页
        start = (page - 1) * per_page
        end = start + per_page
        paginated_ids = work_ids[start:end] if start < len(work_ids) else []
        
        # 获取搜索结果
        works = []
        for work_id in paginated_ids:
            work = Work.query.get(work_id)
            if work:
                # 获取作者
                authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
                authors = []
                for authorship in authorships:
                    if authorship.author_id:
                        author = Author.query.get(authorship.author_id)
                        if author:
                            authors.append(author.display_name)
                
                # 创建结果对象
                result = {
                    'id': work.openalex or work.id,  # 使用 openalex ID 或原始 ID
                    'title': highlight_text(work.title or work.display_name or '', keywords_for_highlight),
                    'authors': ', '.join(authors) if authors else '未知作者',
                    'year': work.publication_year,
                    'cited_by_count': work.cited_by_count or 0,
                    'abstract': work.abstract_inverted_index or ''
                }
                works.append(result)
        
        return jsonify({
            'results': works,
            'total': total,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        print(f"高级搜索出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@searcher_bp.route('/api/basic/ai_search', methods=['POST'])
def api_ai_search():
    try:
        data = request.get_json()
        nl_query = data.get('query', '')
        show_structured = data.get('show_structured', True)
        sort_method = data.get('sort_method', SORT_BY_RELEVANCE)
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)
        
        if not nl_query:
            return jsonify({
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page
            })
        
        try:
            # 使用AI转换为结构化查询
            structured_query = convert_to_structured_query(nl_query)
            
            # 提取关键词，用于高亮
            keywords_for_highlight = []
            if any(op in nl_query.upper() for op in [' AND ', ' OR ', ' NOT ']):
                # 提取布尔表达式中的关键词
                extracted = re.findall(r'\b\w+\b', nl_query)
                keywords_for_highlight = [k for k in extracted if k.upper() not in ['AND', 'OR', 'NOT']]
            else:
                keywords_for_highlight = [nl_query]
                
            # 如果是结构化查询，也提取其中的关键词
            if structured_query != nl_query:
                extracted = re.findall(r'(?:title:|author:|keyword:|time:|content:)([^:]+)(?:\s|$)', structured_query)
                if extracted:
                    for keyword in extracted:
                        if keyword.strip() and keyword.strip() not in keywords_for_highlight:
                            keywords_for_highlight.append(keyword.strip())
            
            # 直接使用AI搜索引擎，它会自动判断是基本查询还是高级查询
            work_ids = ai_search_engine(query_text=nl_query, use_db=True)
            
            # 搜索失败时返回空结果
            if work_ids is None:
                work_ids = []
        except Exception as search_error:
            print(f"AI搜索执行失败: {search_error}")
            # 出错时尝试使用基本搜索
            work_ids = basic_search(query_text=nl_query, use_db=True)
            keywords_for_highlight = [nl_query]
        
        total = len(work_ids)
        
        # 分页
        start = (page - 1) * per_page
        end = start + per_page
        paginated_ids = work_ids[start:end] if start < len(work_ids) else []
        
        # 获取搜索结果
        works = []
        for work_id in paginated_ids:
            work = Work.query.get(work_id)
            if work:
                # 获取作者
                authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
                authors = []
                for authorship in authorships:
                    if authorship.author_id:
                        author = Author.query.get(authorship.author_id)
                        if author:
                            authors.append(author.display_name)
                
                # 创建结果对象
                result = {
                    'id': work.openalex or work.id,  # 使用 openalex ID 或原始 ID
                    'title': highlight_text(work.title or work.display_name or '', keywords_for_highlight),
                    'authors': ', '.join(authors) if authors else '未知作者',
                    'year': work.publication_year,
                    'cited_by_count': work.cited_by_count or 0,
                    'abstract': work.abstract_inverted_index or ''
                }
                works.append(result)
        
        response = {
            'results': works,
            'total': total,
            'page': page,
            'per_page': per_page
        }
        
        if show_structured:
            try:
                response['structured_query'] = structured_query
            except:
                response['structured_query'] = nl_query  # 如果转换失败，使用原始查询
        
        return jsonify(response)
        
    except Exception as e:
        print(f"AI搜索路由出错: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': [],
            'total': 0,
            'page': 1,
            'per_page': 10
        }), 500
