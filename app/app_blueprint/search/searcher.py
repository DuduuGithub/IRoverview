from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from Database.model import Work, SearchResult, Author, WorkAuthorship, Topic, Concept, Institution, Source, WorkConcept, WorkTopic, WorkLocation, SearchSession
from Database.config import db
from sqlalchemy import or_
from .search_utils import (
    record_search_session,
    record_search_results,
    record_document_click,
    get_search_history,
    record_dwell_time,
    record_rerank_operation
)
from .basicSearch.search import search as basic_search
from .proSearch.search import search as advanced_search, sort_db_results
from .aiSearch.search import convert_to_structured_query, search as ai_search
from .rank.rank import SORT_BY_RELEVANCE, SORT_BY_TIME_DESC, SORT_BY_TIME_ASC, SORT_BY_COMBINED
import re
import sys
import os
import logging
from datetime import datetime
import math
import nltk
from nltk.corpus import stopwords
import json
import torch
from sort_ai.model_trainer import ModelTrainer
from transformers import AutoModel, AutoTokenizer

# 添加项目根目录到sys.path，避免相对导入问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 创建蓝图
searcher_bp = Blueprint('searcher', __name__,
                       template_folder='templates',
                       static_folder='static')

logger = logging.getLogger(__name__)

# 初始化重排序模型
rerank_model = None
try:
    rerank_model = ModelTrainer()
    logger.info("重排序模型加载成功")
except Exception as e:
    logger.error(f"重排序模型加载失败: {str(e)}")

# 初始化BERT模型用于文本嵌入
embedding_tokenizer = None
embedding_model = None

def init_embedding_model():
    """初始化用于文本嵌入的BERT模型"""
    global embedding_tokenizer, embedding_model
    try:
        if not rerank_model:
            logger.error("重排序模型未初始化，无法获取模型路径")
            return
            
        # 使用rerank_model的模型路径
        model_path = rerank_model.model_path
        logger.info(f"从路径加载文本嵌入模型: {model_path}")
        
        embedding_tokenizer = AutoTokenizer.from_pretrained(model_path)
        embedding_model = AutoModel.from_pretrained(model_path)
        
        if torch.cuda.is_available():
            embedding_model = embedding_model.cuda()
        embedding_model.eval()
        logger.info("文本嵌入模型加载成功")
    except Exception as e:
        logger.error(f"文本嵌入模型加载失败: {str(e)}")
        embedding_tokenizer = None
        embedding_model = None

"""
高亮文本的辅助函数，用于突出显示查询关键词
参数:
    - text: 要处理的原始文本
    - keywords: 要高亮的关键词，可以是字符串或列表
返回:
    - 处理后的包含HTML高亮标记的文本
"""
def highlight_text(text, keywords):
    if not text or not keywords:
        return text
    
    # 将字符串关键词转换为列表
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
            # 使用单词边界匹配，确保匹配完整单词而非子字符串
            pattern = re.compile(r'\b({0})\b'.format(re.escape(keyword.strip())), re.IGNORECASE)
            result = pattern.sub(r'<span class="highlight">\1</span>', result)
    
    return result

"""
从查询中提取关键词，用于高亮显示
参数:
    - query: 查询文本
返回:
    - 关键词列表
"""
def extract_keywords_for_highlight(query):
    if not query:
        return []
        
    # 检查是否是布尔查询
    if any(op in query.upper() for op in [' AND ', ' OR ', ' NOT ']):
        # 提取布尔表达式中的关键词
        extracted = re.findall(r'\b\w+\b', query)
        return [k for k in extracted if k.upper() not in ['AND', 'OR', 'NOT']]
    else:
        # 英文查询，使用单词边界分割
        # 确保NLTK资源已下载
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # 获取英文停用词
        stop_words = set(stopwords.words('english'))
        
        # 分割单词并过滤停用词
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 1]

"""
对结果进行分页处理
参数:
    - result_ids: 完整的结果ID列表
    - page: 当前页码
    - per_page: 每页条目数
返回:
    - 当前页的ID列表
"""
def paginate_results(result_ids, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page
    return result_ids[start:end] if start < len(result_ids) else []

"""
将倒排索引格式的摘要转换为可读文本
参数:
    - inverted_index: 倒排索引格式的摘要（JSON字符串或字典）
返回:
    - 可读的摘要文本
"""
def convert_abstract_to_text(inverted_index):
    if not inverted_index:
        return ""
    
    # 如果是字符串，尝试解析为JSON
    if isinstance(inverted_index, str):
        try:
            inverted_index = json.loads(inverted_index)
        except:
            return inverted_index  # 如果解析失败，直接返回原文本
    
    # 如果不是字典，无法处理
    if not isinstance(inverted_index, dict):
        return str(inverted_index)
    
    # 创建一个词-位置的映射
    word_positions = {}
    
    # 遍历倒排索引，为每个单词创建位置列表
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions[pos] = word
    
    # 按位置排序单词，重建文本
    sorted_positions = sorted(word_positions.keys())
    reconstructed_text = " ".join(word_positions[pos] for pos in sorted_positions)
    
    return reconstructed_text

"""
获取作品详情信息
参数:
    - work_ids: 作品ID列表
    - keywords: 用于高亮的关键词
返回:
    - 作品详情列表
"""
def get_work_details(work_ids, keywords=None):
    if not work_ids:
        return []
        
    from Database.model import Work, Author, WorkAuthorship
    
    works = []
    for work_id in work_ids:
        work = Work.query.get(work_id)
        if work:
            # 获取作者信息
            authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
            authors = []
            for authorship in authorships:
                if authorship.author_id:
                    author = Author.query.get(authorship.author_id)
                    if author:
                        authors.append(author.display_name)
            
            # 将倒排索引转换为可读文本
            abstract_text = convert_abstract_to_text(work.abstract_inverted_index)
            
            # 构建结果对象
            result = {
                'id': work.openalex or work.id,  # 使用 openalex ID 或原始 ID
                'title': highlight_text(work.title or work.display_name or '', keywords),
                'authors': ', '.join(authors) if authors else '未知作者',
                'year': work.publication_year,
                'cited_by_count': work.cited_by_count or 0,
                'abstract': highlight_text(abstract_text, keywords)
            }
            works.append(result)
    
    return works

# 路由定义
@searcher_bp.route('/')
@searcher_bp.route('/index')
def index():
    """搜索首页"""
    return render_template('search/index.html')

@searcher_bp.route('/basic')
@searcher_bp.route('/basic/index')
def basic_index():
    """基本搜索首页"""
    return render_template('search/index.html')

@searcher_bp.route('/search_page')
def search_page():
    """搜索页面"""
    return render_template('search/index.html')

@searcher_bp.route('/results')
def results_page():
    """搜索结果页面"""
    return render_template('search/results.html')

"""
基本搜索API
处理POST请求，执行基本搜索并返回分页结果
"""
@searcher_bp.route('/search', methods=['POST'])
@searcher_bp.route('/api/basic/search', methods=['POST'])
def api_search():
    try:
        # 获取请求数据
        data = request.get_json()
        keyword = data.get('keyword', '')
        sort_method = data.get('sort_method', SORT_BY_RELEVANCE)
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)
        count_only = data.get('count_only', False)  # 新增参数，仅用于统计
        
        # 空关键词检查
        if not keyword:
            return jsonify({
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page
            })
        
        # 提取关键词，用于高亮
        keywords = extract_keywords_for_highlight(keyword)
        
        # 获取日期范围（如果有）
        date_from = data.get('date_from', '')
        date_to = data.get('date_to', '')
        # 获取多选年份范围（如果有）
        year_ranges = data.get('year_ranges', [])
        
        # 执行搜索
        work_ids = basic_search(query_text=keyword, use_db=True, sort_method=sort_method)
        
        # 如果有日期筛选或年份范围筛选，应用筛选
        if date_from or date_to or year_ranges:
            from datetime import datetime
            
            try:
                # 解析日期范围
                if date_from:
                    from_date = datetime.strptime(date_from, '%Y-%m-%d')
                    from_year = from_date.year
                else:
                    from_year = 1900
                
                if date_to:
                    to_date = datetime.strptime(date_to, '%Y-%m-%d')
                    to_year = to_date.year
                else:
                    to_year = 2100
                
                # 处理多个年份范围筛选
                year_filter_active = False
                if year_ranges and len(year_ranges) > 0:
                    year_filter_active = True
                    # 生成所有可接受的年份列表
                    acceptable_years = set()
                    for year_range in year_ranges:
                        try:
                            start_year, end_year = map(int, year_range.split('-'))
                            for y in range(start_year, end_year + 1):
                                acceptable_years.add(y)
                        except:
                            print(f"解析年份范围出错: {year_range}")
                
                # 筛选结果
                filtered_ids = []
                for doc_id in work_ids:
                    work = Work.query.get(doc_id)
                    if work and work.publication_year:
                        year = work.publication_year
                        # 如果使用多选年份范围，检查年份是否在可接受范围内
                        if year_filter_active:
                            if year in acceptable_years:
                                filtered_ids.append(doc_id)
                        # 否则使用通用日期范围
                        elif from_year <= year <= to_year:
                            filtered_ids.append(doc_id)
                
                work_ids = filtered_ids
            except Exception as date_error:
                print(f"日期筛选出错: {date_error}")
                # 出错时不应用筛选
        
        total = len(work_ids)
        
        # 记录搜索会话
        session_id = record_search_session(keyword, total)
        if not session_id:
            logger.error("记录搜索会话失败")
            return jsonify({'error': '记录搜索会话失败'}), 500
        
        # 分页
        start = (page - 1) * per_page
        end = start + per_page
        paginated_ids = work_ids[start:end] if start < len(work_ids) else []
        
        # 获取搜索结果
        works = []
        raw_results = []  # 用于记录原始搜索结果
        for i, work_id in enumerate(paginated_ids):
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
                
                # 计算分页的起始位置
                start = (page - 1) * per_page
                
                # 将倒排索引摘要转换为可读文本
                abstract_text = convert_abstract_to_text(work.abstract_inverted_index)
                
                # 创建结果对象
                result = {
                    'id': work.id,  # 使用原始ID
                    'title': highlight_text(work.title or work.display_name or '', keywords),
                    'authors': ', '.join(authors) if authors else '未知作者',
                    'year': work.publication_year,
                    'cited_by_count': work.cited_by_count or 0,
                    'abstract': highlight_text(abstract_text, keywords),
                    'rank': start + i + 1,
                    'page': page,
                    'position': i + 1,
                    'url': f'/reader/document/{work.id}?session_id={session_id}'
                }
                works.append(result)
                
                # 创建原始结果对象用于记录
                raw_result = {
                    'id': work.id,
                    'relevance_score': 0.0,  # 初始相关性得分
                    'query_text': keyword
                }
                raw_results.append(raw_result)
        
        # 记录搜索结果
        if session_id and raw_results:
            try:
                record_search_results(
                    session_id=session_id,
                    results=raw_results,
                    page=page,
                    per_page=per_page
                )
            except Exception as e:
                logger.error(f"记录搜索结果失败: {str(e)}", exc_info=True)
        
        return jsonify({
            'results': works,
            'total': total,
            'page': page,
            'per_page': per_page,
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"搜索出错: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@searcher_bp.route('/api/basic/record_click', methods=['POST'])
def record_click():
    """记录文档点击"""
    logger.info("收到文档点击记录请求")
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        document_id = data.get('document_id')
        
        logger.info(f"请求数据: session_id={session_id}, document_id={document_id}")
        logger.info(f"请求头: {dict(request.headers)}")
        
        if not session_id or not document_id:
            logger.warning("缺少必要参数")
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 检查会话是否存在
        session = SearchSession.query.filter_by(session_id=session_id).first()
        if not session:
            logger.warning(f"会话不存在: session_id={session_id}")
            return jsonify({'error': '无效的会话ID'}), 400
            
        # 检查文档是否存在
        work = Work.query.get(document_id)
        if not work:
            logger.warning(f"文档不存在: document_id={document_id}")
            return jsonify({'error': '无效的文档ID'}), 400
            
        record_document_click(session_id, document_id)
        logger.info("点击记录处理完成")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"记录点击失败: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@searcher_bp.route('/api/basic/record_dwell_time', methods=['POST'])
def record_dwell():
    """记录文档停留时间"""
    logger.info("收到停留时间记录请求")
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        document_id = data.get('document_id')
        dwell_time = data.get('dwell_time')
        
        logger.info(f"请求数据: session_id={session_id}, document_id={document_id}, dwell_time={dwell_time}")
        logger.info(f"请求头: {dict(request.headers)}")
        
        if not all([session_id, document_id, dwell_time]):
            logger.warning("缺少必要参数")
            return jsonify({'error': '缺少必要参数'}), 400
            
        # 检查会话是否存在
        session = SearchSession.query.filter_by(session_id=session_id).first()
        if not session:
            logger.warning(f"会话不存在: session_id={session_id}")
            return jsonify({'error': '无效的会话ID'}), 400
            
        # 检查文档是否存在
        work = Work.query.get(document_id)
        if not work:
            logger.warning(f"文档不存在: document_id={document_id}")
            return jsonify({'error': '无效的文档ID'}), 400
            
        success = record_dwell_time(session_id, document_id, dwell_time)
        if success:
            logger.info("停留时间记录成功")
            return jsonify({'status': 'success'})
        else:
            logger.error("停留时间记录失败")
            return jsonify({'error': '记录停留时间失败'}), 500
    except Exception as e:
        logger.error(f"记录停留时间失败: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@searcher_bp.route('/api/basic/search-history/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """获取搜索历史"""
    try:
        history = get_search_history(session_id)
        if not history:
            return jsonify({'error': '未找到搜索历史'}), 404
            
        return jsonify(history)
        
    except Exception as e:
        print(f"获取搜索历史失败: {str(e)}")
        return jsonify({'error': '获取搜索历史失败'}), 500

"""
AI搜索API
处理POST请求，使用AI技术将自然语言查询转换为结构化查询，并返回结果
"""
@searcher_bp.route('/api/basic/ai_search', methods=['POST'])
@searcher_bp.route('/api/ai_search', methods=['POST'])
def api_ai_search():
    """
    处理AI检索请求，将自然语言转换为检索式
    """
    try:
        # 获取请求数据
        request_data = request.get_json()
        if not request_data or 'query' not in request_data:
            return jsonify({
                'status': 'error',
                'message': '缺少查询参数'
            }), 400
        
        nl_query = request_data['query']
        search_mode = request_data.get('search_mode', 'basic')
        print(f"收到AI检索请求: {nl_query}, 检索模式: {search_mode}")
        
        # 分析请求类型（新查询或修改现有查询）
        is_modification = "修改" in nl_query and ("检索式" in nl_query or "查询" in nl_query)
        original_query = None
        
        if is_modification:
            # 提取原始检索式
            try:
                pattern = r'[：:]\s*([^，。:,\n]+)'
                match = re.search(pattern, nl_query)
                if match:
                    original_query = match.group(1).strip()
                    print(f"提取到原始检索式: {original_query}")
            except Exception as e:
                print(f"提取原始检索式出错: {e}")
        
        # 针对修改请求和新查询使用不同的处理逻辑
        if is_modification and original_query:
            # 处理修改请求，分析修改意图
            modification_intent = analyze_modification_intent(nl_query, original_query)
            
            # 根据意图应用修改
            structured_query = apply_modification(original_query, modification_intent, search_mode)
            
            # 生成友好响应
            response = generate_modification_response(modification_intent, structured_query, search_mode)
        else:
            # 生成新的检索式
            try:
                structured_query = convert_to_structured_query(nl_query, search_mode)
                response = generate_ai_response(nl_query, structured_query, search_mode)
            except Exception as e:
                print(f"生成检索式出错: {e}")
                # 降级为简单处理
                structured_query = nl_query.replace("搜索", "").replace("查找", "").strip()
                response = "我尝试为您生成了一个简单的检索式，您可以直接使用或进行修改。"
        
        # 记录检索会话
        session_id = record_search_session(nl_query, total_results=0)
        
        print(f"AI检索响应: {response}, 结构化查询: {structured_query}")
        
        return jsonify({
            'status': 'success',
            'response': response,
            'query': structured_query,
            'session_id': session_id,
            'search_mode': search_mode
        })
    
    except Exception as e:
        print(f"AI检索API错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理请求时出错: {str(e)}'
        }), 500

"""
分析用户的修改意图
"""
def analyze_modification_intent(nl_query, original_query):
    # 初始化意图字典
    intent = {
        'time_filter': {
            'active': False,
            'range': None,
            'description': None
        },
        'author_filter': {
            'active': False,
            'authors': [],
            'description': None
        },
        'institution_filter': {
            'active': False,
            'institutions': [],
            'description': None
        },
        'title_filter': {
            'active': False,
            'terms': [],
            'description': None
        },
        'keyword_filter': {
            'active': False,
            'terms': [],
            'description': None
        },
        'logic_change': {
            'active': False,
            'operation': None,
            'description': None
        }
    }
    
    # 检测时间意图
    time_patterns = [
        (r'(近|最近|过去)(\d+)年', '最近几年'),
        (r'(\d{4})年(至|到|-)(\d{4})年', '特定年份范围'),
        (r'(\d{4})年(以来|之后|后)', '某年之后'),
        (r'(\d{4})年(以前|之前|前)', '某年之前')
    ]
    
    for pattern, description in time_patterns:
        match = re.search(pattern, nl_query)
        if match:
            intent['time_filter']['active'] = True
            intent['time_filter']['description'] = description
            
            # 提取年份范围
            from datetime import datetime
            current_year = datetime.now().year
            
            if '最近' in description or '近' in description or '过去' in description:
                years = int(match.group(2))
                intent['time_filter']['range'] = [current_year - years, current_year]
            elif '特定年份范围' in description:
                start_year = int(match.group(1))
                end_year = int(match.group(3))
                intent['time_filter']['range'] = [start_year, end_year]
            elif '某年之后' in description:
                year = int(match.group(1))
                intent['time_filter']['range'] = [year, current_year]
            elif '某年之前' in description:
                year = int(match.group(1))
                intent['time_filter']['range'] = [1900, year]
            
            break
    
    # 检测作者意图
    author_match = re.search(r'(作者|author)[：:]*\s*([^，。:,\n]+)', nl_query, re.IGNORECASE)
    if author_match:
        intent['author_filter']['active'] = True
        intent['author_filter']['authors'].append(author_match.group(2).strip())
        intent['author_filter']['description'] = "添加作者限定"
    
    # 检测机构意图
    institution_match = re.search(r'(机构|单位|学校|大学|institution)[：:]*\s*([^，。:,\n]+)', nl_query, re.IGNORECASE)
    if institution_match:
        intent['institution_filter']['active'] = True
        intent['institution_filter']['institutions'].append(institution_match.group(2).strip())
        intent['institution_filter']['description'] = "添加机构限定"
    
    # 检测逻辑操作变更
    if "改为或" in nl_query.lower() or "改为or" in nl_query.lower():
        intent['logic_change']['active'] = True
        intent['logic_change']['operation'] = "OR"
        intent['logic_change']['description'] = "将AND逻辑改为OR"
    elif "改为且" in nl_query.lower() or "改为and" in nl_query.lower():
        intent['logic_change']['active'] = True
        intent['logic_change']['operation'] = "AND"
        intent['logic_change']['description'] = "将OR逻辑改为AND"
    
    return intent

"""
根据意图修改查询
"""
def apply_modification(original_query, intent, search_mode):
    # 复制原始查询作为起点
    modified_query = original_query
    
    # 应用时间过滤
    if intent['time_filter']['active'] and intent['time_filter']['range']:
        time_range = intent['time_filter']['range']
        
        if search_mode == 'basic':
            # 基本模式下，保持查询不变，前端会在显示结果时提示用户关于时间过滤的信息
            # 这里不改变查询本身，但会在响应中包含时间过滤信息
            pass
        else:
            # 高级模式下，添加时间字段
            # 先检查查询中是否已有时间条件
            if "year:" in modified_query:
                # 替换现有的时间条件
                modified_query = re.sub(r'year:\[\d+ TO \d+\]', f"year:[{time_range[0]} TO {time_range[1]}]", modified_query)
            else:
                # 添加新的时间条件
                modified_query = f"{modified_query} AND year:[{time_range[0]} TO {time_range[1]}]"
    
    # 应用作者过滤
    if intent['author_filter']['active'] and intent['author_filter']['authors']:
        author = intent['author_filter']['authors'][0]  # 暂时只处理一个作者
        
        if search_mode == 'basic':
            # 基本模式下，简单添加作者名作为关键词
            if " AND " in modified_query or " OR " in modified_query:
                modified_query = f"({modified_query}) AND {author}"
            else:
                modified_query = f"{modified_query} AND {author}"
        else:
            # 高级模式下，使用author字段
            modified_query = f"{modified_query} AND author:\"{author}\""
    
    # 应用机构过滤
    if intent['institution_filter']['active'] and intent['institution_filter']['institutions']:
        institution = intent['institution_filter']['institutions'][0]  # 暂时只处理一个机构
        
        if search_mode == 'basic':
            # 基本模式下，简单添加机构名作为关键词
            if " AND " in modified_query or " OR " in modified_query:
                modified_query = f"({modified_query}) AND {institution}"
            else:
                modified_query = f"{modified_query} AND {institution}"
        else:
            # 高级模式下，使用institution字段
            modified_query = f"{modified_query} AND institution:\"{institution}\""
    
    # 应用逻辑操作变更
    if intent['logic_change']['active'] and intent['logic_change']['operation']:
        operation = intent['logic_change']['operation']
        
        # 简单替换所有的AND/OR操作符
        if operation == "AND" and " OR " in modified_query:
            modified_query = modified_query.replace(" OR ", " AND ")
        elif operation == "OR" and " AND " in modified_query:
            modified_query = modified_query.replace(" AND ", " OR ")
    
    return modified_query

"""
生成修改响应
"""
def generate_modification_response(intent, query, search_mode):
    response_parts = []
    
    # 添加时间过滤响应
    if intent['time_filter']['active']:
        if search_mode == 'basic' and intent['time_filter']['range']:
            time_range = intent['time_filter']['range']
            response_parts.append(f"我已记录您想要查找{time_range[0]}年到{time_range[1]}年的文献。在基本检索模式下，系统将在显示结果时自动过滤这个时间范围，而不在检索式中体现。")
        elif intent['time_filter']['range']:
            response_parts.append(f"我已将时间范围限定在{intent['time_filter']['range'][0]}年到{intent['time_filter']['range'][1]}年。")
    
    # 添加作者过滤响应
    if intent['author_filter']['active'] and intent['author_filter']['authors']:
        author = intent['author_filter']['authors'][0]
        response_parts.append(f"我已添加作者{author}的限定条件。")
    
    # 添加机构过滤响应
    if intent['institution_filter']['active'] and intent['institution_filter']['institutions']:
        institution = intent['institution_filter']['institutions'][0]
        response_parts.append(f"我已添加机构{institution}的限定条件。")
    
    # 添加逻辑变更响应
    if intent['logic_change']['active'] and intent['logic_change']['operation']:
        op_desc = "必须同时满足所有条件" if intent['logic_change']['operation'] == "AND" else "满足任一条件即可"
        response_parts.append(f"我已将检索逻辑修改为{op_desc}。")
    
    # 如果没有检测到任何意图，给出通用响应
    if not response_parts:
        return "我已根据您的要求修改了检索式。如果这不是您想要的效果，请更具体地描述您的需求。"
    
    # 组合所有响应部分
    combined_response = "，".join(response_parts)
    
    # 可能的提示说明
    if search_mode == 'basic' and any([intent['time_filter']['active'], 
                                       intent['author_filter']['active'], 
                                       intent['institution_filter']['active']]):
        combined_response += "。如果您需要更精确的字段检索，建议切换到高级检索模式。"
    
    return combined_response

"""
根据查询和检索式生成AI响应
"""
def generate_ai_response(nl_query, structured_query, search_mode):
    # 检查查询类型特征
    is_time_related = any(word in nl_query for word in ["近五年", "最近五年", "近3年", "近10年", "近十年", "2020年", "今年"])
    has_year_field = "year:" in structured_query or "year[" in structured_query
    is_advanced_format = ":" in structured_query
    
    # 从查询中提取可能的主题
    topic_match = re.search(r'关于([\w\s]+?)的', nl_query)
    topic = topic_match.group(1) if topic_match else None

    # 基于用户搜索意图生成友好响应
    if is_time_related:
        if search_mode == 'basic':
            return "我已生成检索式并注意到您关心时间范围。在基本检索模式下，系统将在处理结果时尊重您的时间需求，但检索式本身不会包含时间语法。如需使用更精确的时间过滤，可以切换到高级检索模式。"
        elif has_year_field:
            return "我已为您生成了包含时间限制的检索式，重点查找您指定年份范围的相关研究。您可以应用此检索式或要求我进行调整。"
    
    # 检测特定领域查询
    if "综述" in nl_query or "评论" in nl_query or "review" in nl_query.lower():
        return "我已生成查找综述/评论类文献的检索式。这类文献通常能够提供该领域的全面概括和研究现状。"
    
    if "对比" in nl_query or "比较" in nl_query or "versus" in nl_query.lower() or "vs" in nl_query.lower():
        return "我已生成针对比较研究的检索式。这将帮助您找到对比不同方法、技术或观点的文献。"
    
    if "应用" in nl_query and topic:
        return f"我已为您创建了关于{topic}应用领域的检索式。这将帮助您找到该技术在实际场景中的应用研究。"
    
    # 基于检索式格式给出不同响应    
    if is_advanced_format and search_mode == 'advanced':
        return "我已根据您的需求生成了结构化检索式，使用了字段限定以提高查询精度。您可以直接应用或要求我进行调整。"
    
    # 默认响应
    return "我已根据您的需求生成了检索式。您可以直接应用或告诉我需要哪些方面的调整。"

"""
API获取文档详情
"""
@searcher_bp.route('/api/document/<doc_id>', methods=['GET'])
def api_document_detail(doc_id):
    try:
        # 获取文档信息
        work = Work.query.get(doc_id)
        if not work:
            return jsonify({'error': '未找到文档'}), 404
            
        # 获取作者信息
        authorships = WorkAuthorship.query.filter_by(work_id=doc_id).all()
        authors = []
        for authorship in authorships:
            if authorship.author_id:
                author = Author.query.get(authorship.author_id)
                if author:
                    authors.append(author.display_name)
        
        # 获取关键词、主题等
        concepts = []
        concept_records = WorkConcept.query.filter_by(work_id=doc_id).all()
        for record in concept_records:
            concept = Concept.query.get(record.concept_id)
            if concept:
                concepts.append(concept.display_name)
                
        topics = []
        topic_records = WorkTopic.query.filter_by(work_id=doc_id).all()
        for record in topic_records:
            topic = Topic.query.get(record.topic_id)
            if topic:
                topics.append(topic.display_name)
        
        # 获取期刊/出版物信息
        venue_name = '未知期刊'
        try:
            # 使用WorkLocation关联表获取source_id
            work_location = WorkLocation.query.filter_by(work_id=doc_id, location_type='primary').first()
            if work_location and work_location.source_id:
                source = Source.query.get(work_location.source_id)
                if source:
                    venue_name = source.display_name or source.publisher or '未知期刊'
        except Exception as venue_error:
            print(f"获取期刊信息出错: {venue_error}")

        # 构建文档详细信息，确保所有引用的属性都存在
        try:
            # 将倒排索引摘要转换为可读文本
            abstract_text = convert_abstract_to_text(work.abstract_inverted_index)
            
            document_data = {
                'id': work.id,
                'title': work.title or work.display_name or 'Untitled',
                'authors': authors,
                'year': work.publication_year or 0,
                'venue': venue_name,
                'doi': work.doi or '',
                'abstract': abstract_text,
                'cited_by_count': work.cited_by_count or 0,
                'citations': work.cited_by_count or 0,
                'keywords': concepts + topics,
                'volume': work.volume or '',
                'issue': work.issue or '',
                'pages': f"{work.first_page or ''}-{work.last_page or ''}" if (work.first_page or work.last_page) else '',
                'references': [],
                'related': []
            }
            
            # 获取相关文档（这里简化，实际应根据需求实现更复杂的相关性计算）
            related_docs = []
            # 简单示例：获取同一作者的其他文章
            if authors:
                try:
                    # 查询作者ID
                    author_ids = [a.id for a in Author.query.filter(Author.display_name.in_(authors)).all()]
                    if author_ids:
                        other_works = WorkAuthorship.query.filter(
                            WorkAuthorship.author_id.in_(author_ids),
                            WorkAuthorship.work_id != doc_id
                        ).limit(5).all()
                        
                        for authorship in other_works:
                            other_work = Work.query.get(authorship.work_id)
                            if other_work:
                                other_authors = []
                                other_authorships = WorkAuthorship.query.filter_by(work_id=other_work.id).all()
                                for other_authorship in other_authorships:
                                    if other_authorship.author_id:
                                        author = Author.query.get(other_authorship.author_id)
                                        if author and author.display_name:
                                            other_authors.append(author.display_name)
                                
                                related_docs.append({
                                    'id': other_work.id,
                                    'title': other_work.title or other_work.display_name or 'Untitled',
                                    'authors': other_authors,
                                    'year': other_work.publication_year or 0,
                                    'cited_by_count': other_work.cited_by_count or 0
                                })
                except Exception as related_error:
                    print(f"获取相关文档出错: {related_error}")
            
            document_data['related'] = related_docs
            
            return jsonify(document_data)
        except Exception as format_error:
            print(f"格式化文档数据出错: {format_error}")
            return jsonify({'error': '处理文档数据时出错'}), 500
        
    except Exception as e:
        print(f"获取文档详情出错: {e}")
        return jsonify({'error': str(e)}), 500

"""
处理布尔表达式的工具函数
"""
def is_operator(token):
    """判断是否为运算符"""
    return token in ['AND', 'OR', 'NOT']

def get_operator_priority(operator):
    """获取运算符优先级"""
    priorities = {
        'NOT': 3,
        'AND': 2,
        'OR': 1,
        '(': 0
    }
    return priorities.get(operator, 0)

def infix_to_postfix(expression):
    """
    将中缀表达式转换为后缀表达式
    参数:
        - expression: 中缀表达式字符串
    返回:
        - 后缀表达式token列表
    """
    logger.info(f"开始将中缀表达式转换为后缀表达式: {expression}")
    
    # 分词，保留括号和运算符
    tokens = []
    current_term = []
    
    for char in expression:
        if char in '()':
            if current_term:
                tokens.append(''.join(current_term).strip())
                current_term = []
            tokens.append(char)
        elif char.isspace():
            if current_term:
                tokens.append(''.join(current_term).strip())
                current_term = []
        else:
            current_term.append(char)
    
    if current_term:
        tokens.append(''.join(current_term).strip())
    
    # 清理空token
    tokens = [t for t in tokens if t.strip()]
    
    # 处理运算符
    i = 0
    while i < len(tokens):
        if tokens[i].upper() in ['AND', 'OR', 'NOT']:
            tokens[i] = tokens[i].upper()
        i += 1
    
    logger.info(f"分词结果: {tokens}")
    
    # 转换为后缀表达式
    output = []
    operator_stack = []
    
    for token in tokens:
        if token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()  # 弹出左括号
        elif is_operator(token):
            while (operator_stack and operator_stack[-1] != '(' and 
                   get_operator_priority(operator_stack[-1]) >= get_operator_priority(token)):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        else:
            output.append(token)
    
    # 处理剩余的运算符
    while operator_stack:
        if operator_stack[-1] == '(':
            logger.warning("表达式中存在未匹配的左括号")
        output.append(operator_stack.pop())
    
    logger.info(f"转换后的后缀表达式: {output}")
    return output

def evaluate_postfix(postfix_tokens, field, all_work_ids):
    """
    计算后缀表达式
    参数:
        - postfix_tokens: 后缀表达式token列表
        - field: 查询字段
        - all_work_ids: 所有文档ID列表
    返回:
        - 计算结果（文档ID列表）
    """
    from .basicSearch.search import boolean_AND, boolean_OR, boolean_NOT
    
    logger.info(f"开始计算后缀表达式: {postfix_tokens}")
    stack = []
    
    for token in postfix_tokens:
        if not is_operator(token):
            # 处理搜索词
            result = process_single_field_query(field, token)
            stack.append(result)
            logger.info(f"处理检索词 '{token}' 得到结果数量: {len(result)}")
        else:
            # 处理运算符
            if token == 'NOT':
                if stack:
                    operand = stack.pop()
                    result = boolean_NOT(operand, all_work_ids)
                    stack.append(result)
                    logger.info(f"执行NOT运算，结果数量: {len(result)}")
            else:
                if len(stack) >= 2:
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    if token == 'AND':
                        result = boolean_AND(operand1, operand2)
                        logger.info(f"执行AND运算，结果数量: {len(result)}")
                    elif token == 'OR':
                        result = boolean_OR(operand1, operand2)
                        logger.info(f"执行OR运算，结果数量: {len(result)}")
                    stack.append(result)
    
    if stack:
        result = stack[0]
        logger.info(f"后缀表达式计算完成，最终结果数量: {len(result)}")
        return result
    
    logger.warning("后缀表达式计算失败，返回空列表")
    return []

def process_field_boolean_queries(field_queries):
    """
    处理高级检索的布尔表达式
    参数:
        - field_queries: 字段查询列表，每个元素包含field、logic和input
    返回:
        - 匹配的文档ID列表
    """
    from .basicSearch.search import boolean_AND, boolean_OR, boolean_NOT
    
    # 打印接收到的查询条件
    logger.info(f"开始处理布尔检索，接收到的查询条件: {field_queries}")
    
    # 如果没有查询条件，返回空列表
    if not field_queries:
        logger.warning("没有收到任何查询条件，返回空列表")
        return []
    
    # 获取所有文档ID作为初始集合
    all_work_ids = [w.id for w in Work.query.all()]
    if not all_work_ids:
        logger.warning("数据库中没有任何文档")
        return []
    
    result = None
    
    # 确保 field_queries 是列表类型
    if isinstance(field_queries, str):
        try:
            field_queries = json.loads(field_queries)
            logger.info(f"将字符串查询条件解析为JSON: {field_queries}")
        except:
            logger.error(f"解析查询条件失败: {field_queries}")
            return []
    
    if not isinstance(field_queries, list):
        field_queries = [field_queries]
        logger.info("将单个查询条件转换为列表")
    
    for query in field_queries:
        # 确保 query 是字典类型
        if isinstance(query, str):
            try:
                query = json.loads(query)
                logger.info(f"将字符串查询解析为JSON: {query}")
            except:
                logger.error(f"解析单个查询条件失败: {query}")
                continue
                
        field = query.get('field', '')
        logic = query.get('logic', 'AND')  # 与前一个查询条件的布尔关系
        input_text = query.get('input', '').strip()
        
        logger.info(f"处理查询条件: 字段={field}, 布尔关系={logic}, 输入={input_text}")
        
        if not input_text:
            logger.warning(f"字段 {field} 的输入为空，跳过处理")
            continue
            
        # 处理字段内的布尔表达式
        field_result = []
        
        # 检查输入是否包含布尔操作符
        has_boolean_ops = any(op in input_text.upper() for op in [' AND ', ' OR ', ' NOT ']) or '(' in input_text
        
        # 处理字段内的布尔表达式
        if has_boolean_ops:
            logger.info(f"检测到字段 {field} 内的布尔表达式: {input_text}")
            # 转换为后缀表达式并计算
            postfix_tokens = infix_to_postfix(input_text)
            field_result = evaluate_postfix(postfix_tokens, field, all_work_ids)
        else:
            # 处理简单查询
            logger.info(f"处理字段 {field} 的简单查询: {input_text}")
            field_result = process_single_field_query(field, input_text)
            logger.info(f"简单查询的结果数量: {len(field_result)}")
        
        # 根据逻辑运算符组合结果
        if result is None:
            result = field_result
            logger.info(f"第一个查询条件的结果数量: {len(result)}")
        else:
            if logic == 'AND':
                result = boolean_AND(result, field_result)
                logger.info(f"与前一个结果执行AND操作后的数量: {len(result)}")
            elif logic == 'OR':
                result = boolean_OR(result, field_result)
                logger.info(f"与前一个结果执行OR操作后的数量: {len(result)}")
            elif logic == 'NOT':
                not_result = boolean_NOT(field_result, all_work_ids)
                result = boolean_AND(result, not_result)
                logger.info(f"与前一个结果执行NOT操作后的数量: {len(result)}")
    
    logger.info(f"布尔检索最终结果数量: {len(result) if result is not None else 0}")
    return result if result is not None else []

def process_single_field_query(field, term):
    """
    处理单个字段的查询
    参数:
        - field: 查询字段
        - term: 查询词
    返回:
        - 匹配的文档ID列表
    """
    logger.info(f"开始处理单个字段查询: 字段={field}, 检索词={term}")
    
    if field == 'title':
        logger.info("执行标题字段查询")
        works = Work.query.filter(
            (Work.title.ilike(f'%{term}%')) | 
            (Work.display_name.ilike(f'%{term}%'))
        ).all()
        result = [w.id for w in works]
        logger.info(f"标题字段查询结果数量: {len(result)}")
        return result
        
    elif field == 'author':
        logger.info("执行作者字段查询")
        authors = Author.query.filter(Author.display_name.ilike(f'%{term}%')).all()
        author_ids = [a.id for a in authors]
        logger.info(f"找到匹配的作者数量: {len(author_ids)}")
        
        if author_ids:
            works = db.session.query(Work.id).select_from(Work).join(
                WorkAuthorship, Work.id == WorkAuthorship.work_id
            ).filter(
                WorkAuthorship.author_id.in_(author_ids)
            ).all()
            result = [w[0] for w in works]
            logger.info(f"作者字段查询结果数量: {len(result)}")
            return result
        logger.info("未找到匹配的作者")
        return []
        
    elif field == 'keyword':
        logger.info("执行关键词字段查询")
        # 匹配概念和主题
        concepts = Concept.query.filter(Concept.display_name.ilike(f'%{term}%')).all()
        concept_ids = [c.id for c in concepts]
        logger.info(f"找到匹配的概念数量: {len(concept_ids)}")
        
        topics = Topic.query.filter(Topic.display_name.ilike(f'%{term}%')).all()
        topic_ids = [t.id for t in topics]
        logger.info(f"找到匹配的主题数量: {len(topic_ids)}")
        
        # 查找相关作品
        concept_works = []
        if concept_ids:
            concept_works = db.session.query(Work.id).select_from(Work).join(
                WorkConcept, Work.id == WorkConcept.work_id
            ).filter(
                WorkConcept.concept_id.in_(concept_ids)
            ).all()
            concept_works = [w[0] for w in concept_works]
            logger.info(f"概念相关文献数量: {len(concept_works)}")
        
        topic_works = []
        if topic_ids:
            topic_works = db.session.query(Work.id).select_from(Work).join(
                WorkTopic, Work.id == WorkTopic.work_id
            ).filter(
                WorkTopic.topic_id.in_(topic_ids)
            ).all()
            topic_works = [w[0] for w in topic_works]
            logger.info(f"主题相关文献数量: {len(topic_works)}")
        
        result = list(set(concept_works + topic_works))
        logger.info(f"关键词字段查询最终结果数量: {len(result)}")
        return result
        
    elif field == 'abstract':
        logger.info("执行摘要字段查询")
        works = Work.query.filter(
            Work.abstract_inverted_index.cast(db.String).ilike(f'%{term}%')
        ).all()
        result = [w.id for w in works]
        logger.info(f"摘要字段查询结果数量: {len(result)}")
        return result
        
    elif field == 'institution':
        logger.info("执行机构字段查询")
        institutions = Institution.query.filter(
            Institution.display_name.ilike(f'%{term}%')
        ).all()
        institution_ids = [i.id for i in institutions]
        logger.info(f"找到匹配的机构数量: {len(institution_ids)}")
        
        if institution_ids:
            works = db.session.query(Work.id).select_from(Work).join(
                WorkAuthorship, Work.id == WorkAuthorship.work_id
            ).filter(
                WorkAuthorship.institution_id.in_(institution_ids)
            ).all()
            result = [w[0] for w in works]
            logger.info(f"机构字段查询结果数量: {len(result)}")
            return result
        logger.info("未找到匹配的机构")
        return []
        
    elif field == 'source':
        logger.info("执行来源字段查询")
        sources = Source.query.filter(Source.display_name.ilike(f'%{term}%')).all()
        source_ids = [s.id for s in sources]
        logger.info(f"找到匹配的来源数量: {len(source_ids)}")
        
        if source_ids:
            works = Work.query.filter(Work.source_id.in_(source_ids)).all()
            result = [w.id for w in works]
            logger.info(f"来源字段查询结果数量: {len(result)}")
            return result
        logger.info("未找到匹配的来源")
        return []
        
    elif field == 'country':
        logger.info("执行国家字段查询")
        institutions = Institution.query.filter(
            (Institution.country_code.ilike(f'%{term}%')) | 
            (Institution.country.ilike(f'%{term}%'))
        ).all()
        institution_ids = [i.id for i in institutions]
        logger.info(f"找到匹配的国家机构数量: {len(institution_ids)}")
        
        if institution_ids:
            works = db.session.query(Work.id).select_from(Work).join(
                WorkAuthorship, Work.id == WorkAuthorship.work_id
            ).filter(
                WorkAuthorship.institution_id.in_(institution_ids)
            ).all()
            result = [w[0] for w in works]
            logger.info(f"国家字段查询结果数量: {len(result)}")
            return result
        logger.info("未找到匹配的国家机构")
        return []
        
    elif field == 'topic':
        logger.info("执行主题字段查询")
        topics = Topic.query.filter(Topic.display_name.ilike(f'%{term}%')).all()
        topic_ids = [t.id for t in topics]
        logger.info(f"找到匹配的主题数量: {len(topic_ids)}")
        
        if topic_ids:
            works = db.session.query(Work.id).select_from(Work).join(
                WorkTopic, Work.id == WorkTopic.work_id
            ).filter(
                WorkTopic.topic_id.in_(topic_ids)
            ).all()
            result = [w[0] for w in works]
            logger.info(f"主题字段查询结果数量: {len(result)}")
            return result
        logger.info("未找到匹配的主题")
        return []
        
    elif field == 'concept':
        logger.info("执行概念字段查询")
        concepts = Concept.query.filter(Concept.display_name.ilike(f'%{term}%')).all()
        concept_ids = [c.id for c in concepts]
        logger.info(f"找到匹配的概念数量: {len(concept_ids)}")
        
        if concept_ids:
            works = db.session.query(Work.id).select_from(Work).join(
                WorkConcept, Work.id == WorkConcept.work_id
            ).filter(
                WorkConcept.concept_id.in_(concept_ids)
            ).all()
            result = [w[0] for w in works]
            logger.info(f"概念字段查询结果数量: {len(result)}")
            return result
        logger.info("未找到匹配的概念")
        return []
    
    logger.warning(f"未知的查询字段: {field}")
    return []

"""
新的高级检索API路由
处理POST请求，执行高级检索并返回结果
"""
@searcher_bp.route('/api/advanced_boolean_search', methods=['POST'])
def api_advanced_boolean_search():
    try:
        # 导入排序函数
        from .proSearch.search import sort_db_results
        
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': '请求数据无效'
            }), 400
        
        # 获取检索条件列表
        conditions = data.get('conditions', [])
        sort_method = data.get('sort_method', SORT_BY_RELEVANCE)
        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 10))
        
        if not conditions:
            return jsonify({
                'status': 'error',
                'message': '检索条件不能为空'
            }), 400
        
        # 记录搜索会话
        session_id = record_search_session(str(conditions))
        
        # 执行高级布尔检索
        result_ids = process_field_boolean_queries(conditions)
        
        if not result_ids:
            return jsonify({
                'status': 'success',
                'message': '未找到任何相关结果',
                'session_id': session_id,
                'results': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'pages': 0
            })
        
        # 对结果进行排序
        if sort_method:
            # 将conditions转换为parsed_query格式
            parsed_query = {
                "field_queries": {},
                "general_query": ""
            }
            for condition in conditions:
                field = condition.get('field', '')
                input_text = condition.get('input', '')
                if field and input_text:
                    parsed_query["field_queries"].setdefault(field, []).append(input_text)
                    if not parsed_query["general_query"]:
                        parsed_query["general_query"] = input_text
            
            result_ids = sort_db_results(result_ids, parsed_query, sort_method)
        
        # 分页处理
        total = len(result_ids)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_ids = result_ids[start_idx:end_idx]
        
        # 获取文档详细信息
        results = []
        for work_id in paginated_ids:
            work = Work.query.get(work_id)
            if work:
                # 获取作者信息
                authorships = WorkAuthorship.query.filter_by(work_id=work.id).all()
                authors = []
                for authorship in authorships:
                    if authorship.author_id:
                        author = Author.query.get(authorship.author_id)
                        if author:
                            authors.append(author.display_name)
                
                # 将倒排索引摘要转换为可读文本
                abstract_text = convert_abstract_to_text(work.abstract_inverted_index)
                
                # 创建结果对象
                result = {
                    'id': work.id,
                    'title': work.title or work.display_name or '',
                    'authors': ', '.join(authors) if authors else '未知作者',
                    'year': work.publication_year,
                    'cited_by_count': work.cited_by_count or 0,
                    'abstract': abstract_text,
                    'url': f'/reader/document/{work.id}?session_id={session_id}'
                }
                results.append(result)
        
        # 记录搜索结果
        if session_id:
            try:
                record_search_results(
                    session_id=session_id,
                    results=[{'id': r['id'], 'relevance_score': 0.0, 'query_text': str(conditions)} for r in results],
                    page=page,
                    per_page=per_page
                )
            except Exception as e:
                logger.error(f"记录搜索结果失败: {str(e)}", exc_info=True)
        
        return jsonify({
            'status': 'success',
            'message': f'找到 {total} 条相关结果',
            'session_id': session_id,
            'results': results,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': math.ceil(total / per_page)
        })
        
    except Exception as e:
        logger.error(f"高级布尔检索出错: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'检索处理出错: {str(e)}'
        }), 500
