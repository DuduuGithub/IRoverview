from flask import Blueprint, render_template, request, jsonify, current_app, session
import os
from pathlib import Path
import uuid
import json
from datetime import datetime
import threading
import logging
from werkzeug.utils import secure_filename
import sys
import concurrent.futures
import shutil
import glob
import traceback

# 导入用于处理倒排索引的函数
from .utils.create_vector import _parse_inverted_index

# 设置详细的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)
_log = logging.getLogger(__name__)

# 将IRoverview/Database添加到sys.path
database_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Database'))
sys.path.append(database_path)
_log.info(f"数据库路径设置: {database_path}")

# 导入RAG系统所需的各种处理器
from .src.pipeline import Pipeline, RunConfig
from .src.questions_processing import QuestionsProcessor
from .src.retrieval import VectorRetriever, HybridRetriever
from .src.pdf_parsing import PDFParser
from .src.text_splitter import TextSplitter
from .src.ingestion import VectorDBIngestor
from .src.document_processor import DocumentProcessor

# 导入新的意图处理模块
from .src.intent_classifier import IntentClassifier
from .src.intent_processor import IntentProcessor

# 创建Blueprint
rag_mcp_bp = Blueprint('rag_mcp', __name__, template_folder='templates')

# 定义全局变量但不初始化
DATA_DIR = None
UPLOAD_FOLDER = None
VECTOR_DB_DIR = None
DOCUMENTS_DIR = None
SUBSET_PATH = None
BM25_DB_DIR = None  # 添加BM25_DB_DIR变量

# 存储会话和处理任务的字典
sessions = {}
processing_tasks = {}

# 全局意图处理器
intent_processor = None

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'pdf'}

# 添加必要的导入
from Database.model import Work, Author, WorkAuthorship, WorkLocation, Source

def init_app(app):
    """初始化应用配置和目录"""
    global DATA_DIR, UPLOAD_FOLDER, VECTOR_DB_DIR, DOCUMENTS_DIR, SUBSET_PATH, BM25_DB_DIR, intent_processor
    
    # 初始化配置
    with app.app_context():
        DATA_DIR = Path(__file__).parent / "data"
        UPLOAD_FOLDER = DATA_DIR / 'pdf_uploads'
        VECTOR_DB_DIR = DATA_DIR / 'vector_dbs'
        DOCUMENTS_DIR = DATA_DIR / 'chunked_documents'
        SUBSET_PATH = DATA_DIR / 'subset.csv'
        BM25_DB_DIR = DATA_DIR / 'bm25_dbs'  # 添加BM25索引目录
        
        # 创建必要的目录
        for directory in [DATA_DIR, UPLOAD_FOLDER, VECTOR_DB_DIR, DOCUMENTS_DIR, BM25_DB_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化意图处理器
        intent_processor = IntentProcessor(
            data_dir=DATA_DIR,
            vector_db_dir=VECTOR_DB_DIR,
            documents_dir=DOCUMENTS_DIR,
            upload_folder=UPLOAD_FOLDER,
            subset_path=SUBSET_PATH if SUBSET_PATH and os.path.exists(SUBSET_PATH) else None,
            bm25_db_dir=BM25_DB_DIR
        )
            
    app.logger.info(f"RAG系统目录初始化完成，数据目录: {DATA_DIR}")
    app.logger.info(f"向量数据库目录: {VECTOR_DB_DIR}")
    app.logger.info(f"BM25索引目录: {BM25_DB_DIR}")
    app.logger.info(f"文档目录: {DOCUMENTS_DIR}")
    app.logger.info(f"上传目录: {UPLOAD_FOLDER}")
    app.logger.info(f"意图处理器已初始化")

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_or_create_session(create_new=False):
    """获取现有会话或创建新会话
    
    Args:
        create_new: 是否强制创建新会话
    """
    if create_new or 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        sessions[session['session_id']] = {
            'created_at': datetime.now(),
            'messages': [],
            'last_active': datetime.now(),
            'selected_docs': []  # 初始化预选文档为空列表
        }
        _log.info(f"创建新会话: {session['session_id']}")
    elif session['session_id'] not in sessions:
        sessions[session['session_id']] = {
            'created_at': datetime.now(),
            'messages': [],
            'last_active': datetime.now(),
            'selected_docs': []  # 初始化预选文档为空列表
        }
        _log.info(f"恢复丢失的会话: {session['session_id']}")
    
    # 如果是创建新会话，清除预选文档记录
    if create_new and session['session_id'] in sessions:
        # 清除会话中保存的预选文档
        sessions[session['session_id']]['selected_docs'] = []
        _log.info(f"新会话已清除预选文档记录")
        
        # 清除消息历史中的预选文档记录
        messages = sessions[session['session_id']].get('messages', [])
        for msg in messages:
            if 'selected_docs' in msg:
                msg['selected_docs'] = []
    
    return session['session_id']

@rag_mcp_bp.route('/')
def index():
    """渲染主页"""
    # global声明必须在变量使用前
    global DATA_DIR, UPLOAD_FOLDER, VECTOR_DB_DIR, DOCUMENTS_DIR, SUBSET_PATH, intent_processor
    
    # 确保目录已初始化
    if DATA_DIR is None:
        # 可能尚未初始化，使用默认值
        DATA_DIR = Path(__file__).parent / "data"
        UPLOAD_FOLDER = DATA_DIR / 'pdf_uploads'
        VECTOR_DB_DIR = DATA_DIR / 'vector_dbs'
        DOCUMENTS_DIR = DATA_DIR / 'chunked_documents'
        SUBSET_PATH = DATA_DIR / 'subset.csv'
        
        # 创建必要的目录
        for directory in [DATA_DIR, UPLOAD_FOLDER, VECTOR_DB_DIR, DOCUMENTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # 初始化意图处理器
        intent_processor = IntentProcessor(
            data_dir=DATA_DIR,
            vector_db_dir=VECTOR_DB_DIR,
            documents_dir=DOCUMENTS_DIR,
            upload_folder=UPLOAD_FOLDER,
            subset_path=SUBSET_PATH if os.path.exists(SUBSET_PATH) else None,
            bm25_db_dir=BM25_DB_DIR
        )
            
    get_or_create_session()
    return render_template('rag_mcp/index.html')

@rag_mcp_bp.route('/process', methods=['POST'])
def process_query():
    """处理用户查询并返回答案"""
    global intent_processor
    
    session_id = get_or_create_session()
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': '缺少查询参数'}), 400
    
    query = data['query']
    context_docs = data.get('context_docs', [])
    selected_docs = data.get('selected_docs', [])
    is_reference_selection = data.get('is_reference_selection', False)
    is_new_chat = data.get('is_new_chat', False)  # 添加新对话标志
    
    # 处理清除预选文档命令
    clear_commands = ["clear selection", "清除选择", "clear", "清除", "reset selection", "重置选择"]
    if any(cmd.lower() in query.lower() for cmd in clear_commands):
        _log.info(f"处理清除预选文档命令: '{query}'")
        
        # 清除会话中的预选文档
        if 'selected_docs' in sessions[session_id]:
            sessions[session_id]['selected_docs'] = []
        
        # 记录用户消息
        message_id = str(uuid.uuid4())
        sessions[session_id]['messages'].append({
            'id': message_id,
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat(),
            'selected_docs': []  # 清空选中的文档ID
        })
        
        # 返回确认信息
        confirmation_msg = "已清除所有预选文献。您现在可以进行新的检索查询或重新选择文献。"
        
        # 记录助手确认消息
        response_id = str(uuid.uuid4())
        sessions[session_id]['messages'].append({
            'id': response_id,
            'role': 'assistant',
            'content': confirmation_msg,
            'timestamp': datetime.now().isoformat()
        })
        
        _log.info("已清除预选文档")
        
        # 返回确认响应
        return jsonify({
            'message_id': message_id,
            'response_id': response_id,
            'response': confirmation_msg,
            'status': 'success'
        })
    
    # 处理参考文献选择命令
    if is_reference_selection and query.startswith('select'):
        _log.info(f"处理参考文献选择命令: '{query}'")
        
        # 记录用户消息
        message_id = str(uuid.uuid4())
        sessions[session_id]['messages'].append({
            'id': message_id,
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat(),
            'selected_docs': selected_docs  # 保存选中的文档ID
        })
        
        # 保存预选文档到会话中
        sessions[session_id]['selected_docs'] = selected_docs
        
        # 解析命令获取篇数
        doc_count = 0
        try:
            # 提取"共X篇"中的数字
            import re
            count_match = re.search(r'共(\d+)篇', query)
            if count_match:
                doc_count = int(count_match.group(1))
        except Exception as e:
            _log.error(f"解析参考文献数量出错: {str(e)}")
            
        # 返回确认信息
        confirmation_msg = f"已选择 {len(selected_docs)} 篇参考文献。请继续输入您的问题，我将基于这些文献为您解答。"
        
        # 记录助手确认消息
        response_id = str(uuid.uuid4())
        sessions[session_id]['messages'].append({
            'id': response_id,
            'role': 'assistant',
            'content': confirmation_msg,
            'timestamp': datetime.now().isoformat()
        })
        
        _log.info(f"已保存 {len(selected_docs)} 篇参考文献ID，等待用户提问")
        
        # 返回确认响应
        return jsonify({
            'message_id': message_id,
            'response_id': response_id,
            'response': confirmation_msg,
            'status': 'success'
        })
    
    # 添加长度检查，防止查询过长
    if len(query) > 1000:
        _log.warning(f"查询过长，长度为 {len(query)}，将被截断")
        original_length = len(query)
        query = query[:1000] + "..."
        
        # 创建一个提示性回答
        answer = f"您的查询过长（{original_length}个字符）已被截断至1000字符。请提供更简短、明确的查询以获得更好的结果。"
        
        # 返回直接响应
        return jsonify({
            'message_id': str(uuid.uuid4()),
            'response_id': str(uuid.uuid4()),
            'answer': answer,
            'response': answer,
            'search_results': [],
            'references': [],
            'result_type': 'error',
            'status': 'truncated_query'
        })
    
    # 记录用户消息
    message_id = str(uuid.uuid4())
    sessions[session_id]['messages'].append({
        'id': message_id,
        'role': 'user',
        'content': query,
        'timestamp': datetime.now().isoformat()
    })
    
    _log.info(f"处理用户查询 [session_id={session_id}]: '{query}'")
    
    # 使用LLM进行意图判断
    intent_result = determine_query_intent_with_llm(query, selected_docs, sessions[session_id].get('messages', []))
    is_search_query = intent_result['is_search_query']
    should_use_selected_docs = intent_result['should_use_selected_docs']
    
    
    # 如果LLM认为应该使用预选文档，确保selected_docs不为空
    if should_use_selected_docs and not selected_docs and not is_search_query:
        # 尝试从会话中获取预选文档
        if 'selected_docs' in sessions[session_id] and sessions[session_id]['selected_docs']:
            selected_docs = sessions[session_id]['selected_docs']
            _log.info(f"[LLM意图识别] 从会话中获取预选文档: {len(selected_docs)} 篇")
        # 如果没有找到，查找历史选择
        else:
            for msg in reversed(sessions[session_id].get('messages', [])):
                if msg.get('role') == 'user' and msg.get('content', '').startswith('select') and 'selected_docs' in msg:
                    selected_docs = msg.get('selected_docs', [])
                    _log.info(f"[LLM意图识别] 找到历史选择的参考文献: {len(selected_docs)} 篇")
                    break
    
    # 如果是检索类查询，清除预选文档
    if is_search_query:
        selected_docs = []
        context_docs = []
        
        # 清除会话中的预选文档
        if 'selected_docs' in sessions[session_id]:
            sessions[session_id]['selected_docs'] = []
    
    if selected_docs:
        context_docs = selected_docs
    else:
        _log.info("[LLM意图识别] 未使用预选文档，执行常规查询处理")
    
    _log.info(f"指定的上下文文档: {context_docs}")
    
    # 获取会话历史记录
    conversation_history = sessions[session_id]['messages'].copy()
    
    # 检查是否有正在处理的PDF
    has_pdf = False
    pdf_path = None
    for task_id, task in processing_tasks.items():
        if task.get('status') == 'completed' and task.get('session_id') == session_id:
            has_pdf = True
            pdf_path = task.get('file_path')
            break
    
    # 初始化格式化后的引用变量，确保在异常处理中也能使用
    formatted_references = []
    
    # 使用意图处理器处理查询
    try:
        if intent_processor is None:
            # 如果意图处理器未初始化，则初始化
            intent_processor = IntentProcessor(
                data_dir=DATA_DIR,
                vector_db_dir=VECTOR_DB_DIR,
                documents_dir=DOCUMENTS_DIR,
                upload_folder=UPLOAD_FOLDER,
                subset_path=SUBSET_PATH if os.path.exists(SUBSET_PATH) else None,
                bm25_db_dir=BM25_DB_DIR
            )
            _log.info("意图处理器初始化完成")
            
        # 设置更低的相似度阈值，以便获取更多结果
        similarity_threshold = 0.1
        _log.info(f"设置相似度阈值为 {similarity_threshold}，确保能获取更多潜在相关文献")
        
        # 增加参数标识是否使用预选文档
        result = intent_processor.process_query(
            query=query,
            conversation_history=conversation_history,
            selected_docs=context_docs,
            has_pdf=has_pdf,
            pdf_path=pdf_path,
            similarity_threshold=similarity_threshold,  # 降低相似度阈值以获取更多结果
            use_preselected_docs=bool(selected_docs)  # 添加标志，表明是否使用预选文档
        )
        
        # 记录AI回复
        response_id = str(uuid.uuid4())
        answer_content = result.get('answer', 'Unable to generate a response')
        formatted_references = []
        
        # 处理引用文档
        if 'references' in result and result['references']:
            for ref in result['references']:
                # 获取文档内容
                text = ref.get("text", "")
                
                # 处理摘要，确保有内容显示
                abstract = ref.get("abstract", "")
                if not abstract and text:
                    abstract = text[:300] + ("..." if len(text) > 300 else "")
                
                # 构建格式化的引用信息
                formatted_references.append({
                    "document_id": ref.get("document_id", ""),
                    "doc_id": ref.get("document_id", ""),  # 添加doc_id字段兼容前端
                    "id": ref.get("document_id", ""),      # 添加id字段兼容前端
                    "title": ref.get("title", "Untitled"),
                    "authors": ref.get("authors", []),
                    "year": ref.get("year", ""),
                    "source": ref.get("source", "Unknown Source"),
                    "text": text,
                    "abstract": abstract,
                    "snippet": abstract[:300] if abstract else (text[:300] if text else "内容不可用"),
                    "score": ref.get("score", 0)
                })
            _log.info(f"[引用处理] 处理了 {len(formatted_references)} 个引用文档")
            
            # 记录第一个引用的详细信息
            if formatted_references:
                first_ref = formatted_references[0]
                _log.info(f"[引用示例] 标题: {first_ref.get('title', '')}")
                _log.info(f"[引用示例] 内容长度: {len(first_ref.get('text', ''))}")
                _log.info(f"[引用示例] 摘要预览: {first_ref.get('abstract', '')[:100]}")
                _log.info(f"[引用示例] snippet长度: {len(first_ref.get('snippet', ''))}")
        
        # 如果没有生成回答但有检索结果，使用LLM生成回答
        if (not answer_content or answer_content == 'Unable to generate a response') and result.get('result_type') == 'search':
            # 如果有检索结果，使用LLM生成回答
            results = result.get('references', [])
            
            # 如果有结果，使用LLM生成回答
            if results:
                # 选择前5条结果用于生成回答
                summary_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:5]
                _log.info(f"[LLM生成] 为LLM摘要选择前 5 条结果，总数: {len(results)}")
                
                # 记录送给LLM的详细内容
                _log.info("="*50)
                _log.info("[LLM输入] 送给LLM的检索结果:")
                for i, item in enumerate(summary_results):
                    _log.info(f"[#{i+1}] 标题: {item.get('title', 'N/A')}")
                    _log.info(f"      作者: {', '.join(item.get('authors', []))}")
                    _log.info(f"      年份: {item.get('year', 'N/A')}")
                    _log.info(f"      来源: {item.get('source', 'N/A')}")
                    _log.info(f"      分数: {item.get('score', 0):.4f}")
                    text_preview = item.get('text', '')[:150] + '...' if len(item.get('text', '')) > 150 else item.get('text', '')
                    _log.info(f"      内容摘要: {text_preview}")
                    _log.info("-"*40)
                _log.info("="*50)
                
                # 构建LLM系统提示
                system_prompt = """You are an expert academic research assistant. 
Your task is to summarize search results in an informative and helpful way.
Focus on the main topics found in the papers, their key findings, and how they relate to the user's query.
Your summary should be comprehensive yet concise, highlighting the most relevant information.
Write in a professional academic style in Chinese.

Please format your response using Markdown for better readability:
- Use **bold** for important concepts
- Use headers (## and ###) for sections
- Use bullet points for lists of findings
- Use numbered lists for steps or ranked items
- Use `code blocks` for specific technical terms when appropriate
- Use > blockquotes for direct quotes from the papers

Structure your response with clear sections including an introduction, main findings, and a conclusion."""

                _log.info(f"[LLM生成] 系统提示: {system_prompt}")
                
                # 构建用户提示，包括查询和检索结果
                user_prompt = f"Query: {query}\n\nI found {len(results)} results with a similarity threshold of {0.1}. Here are the most relevant ones:\n\n"
                
                # 添加每个结果的详细信息
                for i, result in enumerate(summary_results):
                    title = result.get('title', 'Untitled')
                    authors = ', '.join(result.get('authors', ['Unknown']))
                    year = result.get('year', 'N/A')
                    source = result.get('source', 'Unknown source')
                    text = result.get('text', 'No content available')
                    score = result.get('score', 0)
                    
                    user_prompt += f"Result {i+1}:\n"
                    user_prompt += f"Title: {title}\n"
                    user_prompt += f"Authors: {authors}\n"
                    user_prompt += f"Year: {year}\n"
                    user_prompt += f"Source: {source}\n"
                    user_prompt += f"Similarity Score: {score:.4f}\n"
                    user_prompt += f"Text: {text}\n\n"
                
                user_prompt += "Please summarize these results in a helpful way, highlighting the most relevant information for the query. Respond in Chinese language as that's the user's preferred language. Make sure to use proper Markdown formatting."
                
                _log.info(f"[LLM生成] 用户提示: {user_prompt[:200]}... (已截断)")
                
                # 调用LLM生成回答
                try:
                    _log.info("[LLM生成] 正在调用GPT-4生成回答...")
                    from .src.api_requests import APIProcessor
                    api_processor = APIProcessor()
                    response = api_processor.send_message(
                        system_content=system_prompt,
                        human_content=user_prompt,
                        model="gpt-4o-mini-2024-07-18",
                        temperature=0.7
                    )
                    
                    # 提取回答
                    answer = response if isinstance(response, str) else json.dumps(response)
                    _log.info(f"[LLM生成] 生成回答 [session_id={session_id}]: '{answer[:200]}...'")
                    
                    # 记录模型使用和标记数量
                    model_info = api_processor.response_data if hasattr(api_processor, 'response_data') else {}
                    _log.info(f"[LLM生成] 使用模型: {model_info.get('model', '')}, 标记使用: {model_info}")
                    
                    formatted_references = []
                    for result in results:
                        formatted_references.append({
                            "document_id": result.get("document_id", ""),
                            "title": result.get("title", "Untitled"),
                            "authors": result.get("authors", []),
                            "year": result.get("year", ""),
                            "text": result.get("text", ""),
                            "score": result.get("score", 0)
                        })
                    
                    _log.info(f"[引用文档] 引用文档数量: {len(formatted_references)}")
                    
                    # 记录会话消息
                    message_id = str(uuid.uuid4())
                    sessions[session_id]['messages'].append({
                        'id': message_id,
                        'role': 'assistant',
                        'content': answer,
                        'timestamp': datetime.now().isoformat(),
                        'references': formatted_references
                    })
                    
                    # 返回结果
                    return jsonify({
                        'answer': answer,
                        'references': formatted_references,
                        'search_results': formatted_references,  # 同时添加search_results字段
                        'message_id': message_id
                    })
                    
                except Exception as e:
                    _log.error(f"[LLM生成] 生成回答时出错: {str(e)}", exc_info=True)
                    
                    return jsonify({
                        'error': f"生成回答时出错: {str(e)}",
                        'references': formatted_references if formatted_references else [],
                        'search_results': formatted_references if formatted_references else []  # 同时添加search_results字段
                    }), 500
            
            else:
                _log.warning("[检索结果] 未找到相关结果")
                # 没有结果时，返回友好提示
                answer_content = "未找到相关结果。请尝试修改您的查询或使用更一般的术语。"
        else:
            # 使用意图处理器生成的答案，不覆盖
            if not answer_content or answer_content == 'Unable to generate a response':
                # 如果answer_content为空或"Unable to generate a response"，则使用result中的answer
                answer_content = result.get('answer', '')
                
                # 检查答案是否有效
                if not answer_content:
                    _log.warning("意图处理器未返回有效答案")
                    answer_content = "对不起，无法为您的问题生成回答。请尝试重新表述您的问题。"
        
        sessions[session_id]['messages'].append({
            'id': response_id,
            'role': 'assistant',
            'content': answer_content,
            'references': result.get('references', []),
            'timestamp': datetime.now().isoformat()
        })
        
        # 记录AI回复的简短摘要
        answer_brief = answer_content[:100] + "..." if len(answer_content) > 100 else answer_content
        _log.info(f"生成回答 [session_id={session_id}]: '{answer_brief}'")
        _log.info(f"引用文档数量: {len(result.get('references', []))}")
        
        # 更新最后活动时间
        sessions[session_id]['last_active'] = datetime.now()
        
        return jsonify({
            'message_id': message_id,
            'response_id': response_id,
            'response': answer_content,  # 添加response字段以保持与前端兼容
            'answer': answer_content,
            'search_results': formatted_references,  # 使用格式化后的引用数据
            'references': formatted_references,  # 保持兼容性
            'reasoning': result.get('reasoning_summary', ''),
            'result_type': result.get('result_type', 'answer'),
            'status': result.get('status', 'success')
        })
        
    except Exception as e:
        _log.error(f"处理查询时出错: {str(e)}", exc_info=True)
        return jsonify({'error': f'处理查询时出错: {str(e)}'}), 500

def ensure_parsed_abstract(text_content):
    """确保文本被正确解析，无论是倒排索引还是普通文本
    
    Args:
        text_content: 文本内容，可能是倒排索引JSON或普通文本
        
    Returns:
        str: 解析后的可读文本
    """
    if not text_content:
        return ""
    
    # 如果已经是字符串但不是JSON格式，直接返回
    if isinstance(text_content, str) and not (text_content.strip().startswith('{') and text_content.strip().endswith('}')):
        return text_content[:300]  # 限制长度为300字符
    
    # 尝试解析JSON格式的倒排索引
    try:
        # 如果是字符串形式的JSON，先转为字典
        if isinstance(text_content, str):
            text_json = json.loads(text_content)
        else:
            text_json = text_content
            
        # 检查是否是字典格式
        if isinstance(text_json, dict):
            # 使用倒排索引解析函数
            parsed_text = _parse_inverted_index(text_json)
            _log.info(f"成功解析倒排索引: {parsed_text[:50]}...")
            return parsed_text[:300]
    except json.JSONDecodeError:
        _log.warning(f"JSON解析失败，文本内容: {text_content[:50]}...")
    except Exception as e:
        _log.error(f"解析倒排索引时出错: {str(e)}")
    
    # 如果解析失败或不是JSON，返回原始文本
    if isinstance(text_content, str):
        return text_content[:300]
    else:
        return str(text_content)[:300]

@rag_mcp_bp.route('/search', methods=['POST'])
def search():
    """执行文档检索，支持不同的检索方法"""
    global intent_processor
    
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({'error': '缺少搜索参数'}), 400
    
    query = data['query']
    limit = data.get('limit', 50)  # 默认50，但实际返回会受限于阈值过滤
    search_method = data.get('search_method', 'jina')  # 默认使用Jina重排
    
    # 如果选择了BM25方法，自动切换为向量检索
    if search_method == 'bm25':
        search_method = 'vector'
        _log.info("BM25检索已弃用，自动切换为向量检索")
    
    # 如果选择了hybrid方法，自动切换为向量+jina重排
    if search_method == 'hybrid':
        search_method = 'jina'
        _log.info("混合检索已弃用，自动切换为向量+Jina重排")
    
    _log.info(f"执行文档检索，查询: '{query}'，限制数量: {limit}，检索方法: {search_method}")
    
    try:
        if intent_processor is None:
            # 如果意图处理器未初始化，则初始化
            intent_processor = IntentProcessor(
                data_dir=DATA_DIR,
                vector_db_dir=VECTOR_DB_DIR,
                documents_dir=DOCUMENTS_DIR,
                upload_folder=UPLOAD_FOLDER,
                subset_path=SUBSET_PATH if os.path.exists(SUBSET_PATH) else None,
                bm25_db_dir=BM25_DB_DIR
            )
            _log.info("意图处理器初始化完成")
            
        # 提取关键词用于检索增强
        keywords = intent_processor.intent_classifier.extract_keywords(query)
        _log.info(f"提取的关键词: {keywords}")
        
        # 根据检索方法选择不同的检索器
        if search_method == 'vector':
            _log.info("使用向量检索器")
            retriever = VectorRetriever(
                vector_db_dir=VECTOR_DB_DIR,
                documents_dir=DOCUMENTS_DIR,
                reranking_strategy=None  # 不使用重排序
            )
            results = retriever.retrieve_by_query(
                query=query,
                top_n=limit,
                similarity_threshold=0.3  # 提高相似度阈值，只返回更相关的结果
            )
        else:  # 默认使用Jina重排
            _log.info("使用向量检索 + Jina重排")
            retriever = VectorRetriever(
                vector_db_dir=VECTOR_DB_DIR,
                documents_dir=DOCUMENTS_DIR,
                reranking_strategy="jina"  # 使用Jina重排序
            )
            results = retriever.retrieve_by_query(
                query=query,
                top_n=limit,
                llm_reranking_sample_size=limit * 2,  # 增加样本大小
                similarity_threshold=0.3  # 提高相似度阈值，只返回更相关的结果
            )
        
        _log.info(f"检索到 {len(results)} 条结果")
        
        # 处理结果格式
        formatted_results = []
        seen_docs = set()
        
        for item in results:
            # 跳过无效结果
            if not item:
                continue
                
            # 确保结果分数超过阈值，否则不添加到结果中
            if item.get('score', 0) < 0.3:  # 提高阈值，只显示相关度高的结果
                continue
                
            # 提取文档ID，优先使用document_id字段
            doc_id = item.get('document_id', '')
            if not doc_id:
                # 如果没有document_id，尝试其他可能的ID字段
                doc_id = item.get('id', '') or item.get('doc_id', '')
                if not doc_id:
                    # 如果仍然没有ID，生成一个临时ID
                    doc_id = f"doc_{len(formatted_results)}"
                # 将找到的ID设置到document_id字段
                item['document_id'] = doc_id
                
            if doc_id in seen_docs:
                continue
                
            seen_docs.add(doc_id)
            
            # 从数据库获取文档元数据
            _log.info(f"从数据库获取文档元数据，文档ID: {doc_id}")
            
            try:
                # 修改：正确处理文档ID查询
                # 首先尝试直接使用ID查询
                work = Work.query.get(doc_id)
                
                # 如果找不到，尝试解析文件名格式的ID
                if not work and isinstance(doc_id, str) and os.path.sep in doc_id:
                    # 可能是文件路径，尝试提取文件名作为ID
                    file_id = os.path.basename(doc_id)
                    if file_id:
                        work = Work.query.get(file_id)
                        _log.info(f"通过文件名 {file_id} 查找文档")
                
                # 如果仍然找不到，尝试模糊匹配
                if not work:
                    _log.warning(f"无法通过ID {doc_id} 直接找到文档，尝试模糊匹配")
                    # 尝试使用LIKE查询
                    work = Work.query.filter(Work.id.like(f"%{doc_id}%")).first()
                    
                if work:
                    _log.info(f"成功找到文档: {work.id}")
                
                # 处理文本内容
                text_content = item.get('text', '')
                
                # 获取标题
                title = item.get('title', '')
                if not title and work and hasattr(work, 'title'):
                    title = work.title
                
                if not title:
                    title = f'文档 {doc_id}'
                
                # 获取和处理摘要
                abstract_text = ""
                
                # 1. 从检索结果中获取abstract_inverted_index
                if 'abstract_inverted_index' in item and item['abstract_inverted_index']:
                    abstract_text = ensure_parsed_abstract(item['abstract_inverted_index'])
                    _log.info(f"从检索结果的abstract_inverted_index获取摘要: {abstract_text[:50]}...")
                
                # 2. 从work对象获取abstract_inverted_index
                elif work and hasattr(work, 'abstract_inverted_index') and work.abstract_inverted_index:
                    abstract_text = ensure_parsed_abstract(work.abstract_inverted_index)
                    _log.info(f"从数据库work对象的abstract_inverted_index获取摘要: {abstract_text[:50]}...")
                
                # 3. 从work对象获取abstract属性
                elif work and hasattr(work, 'abstract') and work.abstract:
                    abstract_text = work.abstract
                    _log.info(f"从数据库work对象的abstract属性获取摘要: {abstract_text[:50]}...")
                
                # 4. 最后尝试使用检索到的文本内容
                elif text_content:
                    # 如果文本内容看起来像倒排索引，尝试解析它
                    if isinstance(text_content, str) and text_content.strip().startswith('{') and text_content.strip().endswith('}'):
                        abstract_text = ensure_parsed_abstract(text_content)
                    else:
                        abstract_text = text_content
                    _log.info(f"使用检索结果文本作为摘要: {abstract_text[:50]}...")
                
                # 确保摘要不为空
                if not abstract_text:
                    abstract_text = "摘要不可用"
                
                # 为前端准备snippet字段，确保类型和长度合适
                snippet = abstract_text[:300] if abstract_text else "内容不可用"
                
                # 获取作者信息
                authors = []
                if work and hasattr(work, 'authors'):
                    authors = [author.name for author in work.authors]
                elif item.get('authors'):
                    authors = item.get('authors')
                
                # 获取年份
                year = ""
                if work and hasattr(work, 'year'):
                    year = work.year
                elif item.get('year'):
                    year = item.get('year')
                
                # 创建格式化结果
                formatted_result = {
                    'id': doc_id,
                    'document_id': doc_id,  # 添加document_id字段以与前端兼容
                    'doc_id': doc_id,      # 添加doc_id字段以与前端兼容
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'abstract': abstract_text,
                    'snippet': snippet,
                    'score': item.get('score', 0),
                    'text': text_content,  # 保留原始文本块
                    'relevance_score': item.get('score', 0)  # 添加relevance_score字段以与前端兼容
                }
                
                formatted_results.append(formatted_result)
                
            except Exception as e:
                _log.error(f"处理文档 {doc_id} 时出错: {str(e)}", exc_info=True)
                continue
        
        _log.info(f"返回 {len(formatted_results)} 条格式化结果")
        
        # 对结果按相似度分数排序
        formatted_results = sorted(formatted_results, key=lambda x: x.get('score', 0), reverse=True)
        
        return jsonify({
            'results': formatted_results,
            'query': query,
            'search_method': search_method,
            'status': 'success'
        })
        
    except Exception as e:
        _log.error(f"执行检索时出错: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'执行检索时出错: {str(e)}',
            'query': query,
            'status': 'error'
        }), 500

@rag_mcp_bp.route('/upload', methods=['POST'])
def upload_file():
    """上传文件处理
    
    Returns:
        上传处理结果（包含任务ID）
    """
    _log.info("接收文件上传请求")
    
    # 检查请求中是否有文件
    if 'file' not in request.files:
        _log.error("没有文件部分")
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    
    # 如果用户未选择文件，浏览器也会提交一个没有文件名的空文件部分
    if file.filename == '':
        _log.error("未选择文件")
        return jsonify({'error': '未选择文件'}), 400
    
    # 检查是否是PDF文件
    if file and allowed_file(file.filename):
        # 获取或创建会话ID
        session_id = get_or_create_session()
        
        # 检查请求中的额外参数
        args = request.form.to_dict()
        search_mode = args.get('search_mode', 'false').lower() == 'true'
        
        _log.info(f"PDF上传模式: {'搜索模式' if search_mode else '解析模式'}")
        
        # 创建安全的文件名
        filename = secure_filename(file.filename)
        # 添加UUID前缀避免文件名冲突
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # 确保上传目录存在
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        _log.info(f"文件已保存: {file_path}")
        
        # 创建任务记录
        task_id = str(uuid.uuid4())
        document_id = os.path.splitext(unique_filename)[0]
        
        processing_tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'file_path': file_path,
            'search_mode': search_mode,
            'document_id': document_id,
            'session_id': session_id,
            'created_at': datetime.now()
        }
        
        _log.info(f"创建PDF处理任务: {task_id}, 文档ID: {document_id}, 搜索模式: {search_mode}")
        
        # 启动一个单独的线程来处理PDF
        thread = threading.Thread(
            target=process_pdf_file,
            args=(file_path, document_id, task_id)
        )
        thread.daemon = True  # 设置为守护线程
        thread.start()
        
        return jsonify({
            'message': '文件已接收，处理中...',
            'task_id': task_id,
            'document_id': document_id
        })
    else:
        _log.error(f"不允许的文件类型: {file.filename}")
        return jsonify({'error': '只允许PDF文件'}), 400

def process_pdf_file(file_path, document_id, task_id):
    """处理PDF文件
    
    Args:
        file_path: PDF文件路径
        document_id: 文档ID
        task_id: 任务ID
    """
    try:
        _log.info(f"开始处理PDF文件: {file_path}")
        
        # 更新任务状态
        processing_tasks[task_id]['status'] = 'processing'
        processing_tasks[task_id]['progress'] = 10
        
        # 创建输出目录
        parsed_reports_dir = os.path.join(DATA_DIR, 'parsed_reports')
        os.makedirs(parsed_reports_dir, exist_ok=True)
        
        # 解析PDF
        _log.info(f"使用PDFParser解析文件: {file_path}")
        pdf_parser = PDFParser(output_dir=parsed_reports_dir)
        
        # 将字符串路径转换为Path对象
        from pathlib import Path
        pdf_path = Path(file_path)
        
        processing_tasks[task_id]['progress'] = 30
        
        # 解析PDF
        parse_result = pdf_parser.parse_and_export([pdf_path])
        
        _log.info(f"PDF解析结果: 成功处理 {parse_result[0]} 个文件，失败 {parse_result[1]} 个文件")
        
        processing_tasks[task_id]['progress'] = 50
        
        if parse_result[0] == 0:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = 'PDF解析失败'
            _log.error(f"PDF解析失败: {file_path}")
            return
        
        _log.info(f"PDF解析成功: {file_path}，成功处理 {parse_result[0]} 个文件，失败 {parse_result[1]} 个文件")
        
        # 查找解析后的JSON文件
        # 尝试找到与PDF文件名匹配的JSON文件
        pdf_filename = os.path.basename(file_path)
        pdf_filename_no_ext = os.path.splitext(pdf_filename)[0]
        
        # 查找文件名匹配的JSON文件
        json_files = glob.glob(os.path.join(parsed_reports_dir, "*.json"))
        
        matched_json = None
        if json_files:
            # 首先尝试精确匹配
            for json_file in json_files:
                if pdf_filename_no_ext in json_file:
                    matched_json = json_file
                    break
            
            # 如果没有找到精确匹配，使用最新创建的JSON文件
            if not matched_json:
                # 按文件创建时间排序
                json_files.sort(key=os.path.getctime, reverse=True)
                # 使用最新创建的文件
                matched_json = json_files[0]
                _log.info(f"未找到精确匹配的JSON文件，使用最新创建的: {matched_json}")
        
        if not matched_json:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = '未找到解析后的JSON文件'
            _log.error(f"未找到解析后的JSON文件，无法继续处理")
            return
        
        _log.info(f"找到对应的JSON文件: {matched_json}")
        
        # 更新任务进度
        processing_tasks[task_id]['progress'] = 60
        
        # 使用TextSplitter分割文本
        _log.info("分割PDF文本到文本块")
        chunked_documents_dir = os.path.join(DATA_DIR, 'chunked_documents')
        os.makedirs(chunked_documents_dir, exist_ok=True)
        
        # 计算分割后的文件路径（与源文件同名，但位于chunked_documents_dir目录）
        chunked_file = os.path.join(chunked_documents_dir, os.path.basename(matched_json))
        
        text_splitter = TextSplitter()
        # 使用正确的方法名split_single_report并处理其返回的布尔值
        success = text_splitter.split_single_report(
            Path(matched_json), 
            Path(chunked_documents_dir)
        )
        
        if not success:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = '文本分割失败'
            _log.error("文本分割失败")
            return
        
        _log.info("文本分割成功")
        
        # 确认分割后的文件存在
        if not os.path.exists(chunked_file):
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = '找不到分割后的文件'
            _log.error(f"找不到分割后的文件: {chunked_file}")
            return
            
        # 更新任务进度
        processing_tasks[task_id]['progress'] = 75
        
        # 如果是搜索模式，创建向量数据库
        search_results = []
        
        search_mode = processing_tasks[task_id].get('search_mode', False)
        if search_mode:
            _log.info("搜索模式: 开始为PDF创建向量索引")
            
            # 创建向量数据库
            _log.info("创建向量数据库")
            vector_db_dir = os.path.join(DATA_DIR, 'vector_dbs')
            os.makedirs(vector_db_dir, exist_ok=True)
            
            # 处理chunked_file来为PDF创建向量表示
            _log.info(f"处理分割后的JSON文件: {chunked_file}")
            ingestor = VectorDBIngestor()
            
            # 创建临时向量数据库
            temp_vector_db_dir = os.path.join(DATA_DIR, 'temp_vector_dbs')
            os.makedirs(temp_vector_db_dir, exist_ok=True)
            
            # 为PDF创建向量索引
            ingestor.process_single_report(Path(chunked_file), Path(temp_vector_db_dir))
            _log.info("向量化处理成功")
            
            # 更新进度
            processing_tasks[task_id]['progress'] = 80
            
            # 使用PDF进行相似文献检索
            try:
                # 初始化查询处理器
                intent_processor = IntentProcessor(
                    data_dir=DATA_DIR,
                    vector_db_dir=VECTOR_DB_DIR,
                    documents_dir=DOCUMENTS_DIR,
                    upload_folder=UPLOAD_FOLDER,
                    subset_path=SUBSET_PATH if os.path.exists(SUBSET_PATH) else None
                )
                
                _log.info(f"[PDF相似文献] 开始计算PDF与数据库文献的相似度，使用阈值0.1...")
                
                # 使用更低的相似度阈值(0.1)以获取更多结果
                result = intent_processor._process_pdf_similar_literature("", file_path, 0.1)
                
                if result and 'results' in result and result['results']:
                    search_results = result['results']
                    _log.info(f"[PDF相似文献] 找到 {len(search_results)} 条相似文献")
                    
                    # 记录前三条结果的详情
                    top_results = search_results[:min(3, len(search_results))]
                    for i, res in enumerate(top_results):
                        _log.info(f"[PDF相似文献] 结果 #{i+1}:")
                        _log.info(f"  - 标题: {res.get('title', 'N/A')}")
                        _log.info(f"  - 作者: {', '.join(res.get('authors', ['N/A']))}")
                        _log.info(f"  - 来源: {res.get('source', 'N/A')}")
                        _log.info(f"  - 年份: {res.get('year', 'N/A')}")
                        _log.info(f"  - 相似度: {res.get('score', 0):.6f}")
                else:
                    _log.warning(f"[PDF相似文献] 未找到相似文献。检查数据库内容和相似度阈值设置。")
            except Exception as e:
                _log.error(f"[PDF相似文献] 查找相似文献时出错: {str(e)}", exc_info=True)
        
        processing_tasks[task_id]['progress'] = 90
        
        # 清理处理过程中生成的临时文件
        # ... (可以添加清理代码)
        
        _log.info(f"PDF处理完成: {file_path}")
        
        # 更新任务状态为已完成
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['search_results'] = search_results
        processing_tasks[task_id]['completed_at'] = datetime.now()
        
    except Exception as e:
        _log.error(f"处理PDF文件时出错: {str(e)}", exc_info=True)
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)
        processing_tasks[task_id]['traceback'] = traceback.format_exc()

@rag_mcp_bp.route('/task/<task_id>', methods=['GET'])
def check_task(task_id):
    """检查处理任务的状态"""
    if task_id not in processing_tasks:
        return jsonify({'error': '找不到任务'}), 404
        
    task = processing_tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'file_path': task['file_path'],
        'document_id': task['document_id'],
        'progress': task['progress'],
        'summary': task.get('summary', ''),
        'error': task.get('error', ''),
        'search_results': task.get('search_results', [])
    })

@rag_mcp_bp.route('/history', methods=['GET'])
def get_history():
    """获取聊天历史"""
    session_id = get_or_create_session()
    
    if session_id not in sessions:
        return jsonify({'sessions': []}), 200
        
    # 获取此会话的消息
    messages = sessions[session_id].get('messages', [])
    
    # 按对话组织消息
    chats = []
    if messages:
        # 简单地取第一条消息作为标题
        first_user_msg = next((m for m in messages if m['role'] == 'user'), None)
        title = first_user_msg['content'][:30] + '...' if first_user_msg else '新对话'
        
        chats.append({
            'id': session_id,
            'title': title,
            'created_at': sessions[session_id]['created_at'].isoformat(),
            'last_active': sessions[session_id]['last_active'].isoformat(),
            'messages': messages
        })
    
    # 按时间分组
    today = []
    yesterday = []
    older = []
    
    today_date = datetime.now().date()
    for chat in chats:
        chat_date = datetime.fromisoformat(chat['last_active']).date()
        if chat_date == today_date:
            today.append(chat)
        elif (today_date - chat_date).days == 1:
            yesterday.append(chat)
        else:
            older.append(chat)
    
    return jsonify({
        'today': today,
        'yesterday': yesterday,
        'older': older
    })

@rag_mcp_bp.route('/tools', methods=['GET'])
def get_tools():
    """获取可用工具列表"""
    tools = [
        {
            'id': 'search',
            'name': '检索数据',
            'icon': '🔍',
            'description': '搜索资料库中的文献和文档'
        },
        {
            'id': 'chat',
            'name': '问答',
            'icon': '💬',
            'description': '与AI助手进行对话，获取信息和帮助'
        },
        {
            'id': 'pdf',
            'name': 'PDF解读',
            'icon': '📄',
            'description': '上传PDF文件，让AI帮您分析内容'
        }
    ]
    
    return jsonify({'tools': tools})

@rag_mcp_bp.route('/new_chat', methods=['POST'])
def create_new_chat():
    """创建新的聊天会话"""
    try:
        # 强制创建新会话
        session_id = get_or_create_session(create_new=True)
        
        # 返回新会话ID
        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'message': '新会话已创建'
        })
    except Exception as e:
        _log.error(f"创建新会话时出错: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'创建新会话时出错: {str(e)}',
            'status': 'error'
        }), 500

# 检查是否是检索类查询
def determine_query_intent_with_llm(query, selected_docs, conversation_history=None):
    """使用LLM判断查询意图
    
    Args:
        query: 用户查询
        selected_docs: 已选择的文档列表
        conversation_history: 对话历史
        
    Returns:
        dict: 包含意图判断结果的字典
        {
            'is_search_query': bool,  # 是否是检索类查询
            'should_use_selected_docs': bool,  # 是否应该使用预选文档
            'reasoning': str  # 推理过程
        }
    """
    _log.info(f"[LLM意图识别] 开始进行意图识别: '{query}'")
    
    # 构建系统提示
    system_prompt = """You are an intent classification system for an academic research assistant.
Your task is to determine whether a user's query is:
1. A search/retrieval request to find new documents (is_search_query=true)
2. A request to use already selected documents (should_use_selected_docs=true)

Important guidelines:
- If the user is asking to find, search or discover new papers/documents, classify as a search query
- If the user is referencing "these papers", "these documents", "selected papers" or similar phrases, they likely want to use the already selected documents
- If the user is asking for a comparison, analysis, literature review, or synthesis of information, they likely want to use the already selected documents
- Consider the conversation history and whether documents have been already selected

Return only a JSON object with three fields:
{
  "is_search_query": boolean,  // true if this is a search/retrieval query
  "should_use_selected_docs": boolean,  // true if we should use the already selected documents
  "reasoning": string  // brief explanation of your reasoning
}"""

    # 构建用户提示
    user_prompt = f"""Query: {query}

Selected documents: {"Yes - " + str(len(selected_docs)) + " documents" if selected_docs else "No documents selected"}

Recent conversation history:
"""
    
    if conversation_history:
        # 只取最近5条对话
        recent_messages = conversation_history[-5:]
        for msg in recent_messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if content:
                user_prompt += f"{role.capitalize()}: {content[:100]}..." if len(content) > 100 else f"{role.capitalize()}: {content}\n"
    else:
        user_prompt += "No conversation history available."
    
    user_prompt += "\nAnalyze the query and determine the intent. Return ONLY a JSON object with the specified fields."
    
    try:
        from .src.api_requests import APIProcessor
        api_processor = APIProcessor()
        response = api_processor.send_message(
            system_content=system_prompt,
            human_content=user_prompt,
            model="gpt-4o-mini-2024-07-18",  # 使用小型、快速的模型
            temperature=0.1
        )
        
        # 尝试解析JSON响应
        _log.info(f"[LLM意图识别] 原始响应: {response}")
        
        import json
        import re
        
        # 提取JSON部分（如果响应包含其他文本）
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            intent_result = json.loads(json_str)
        else:
            # 尝试直接解析整个响应
            intent_result = json.loads(response)
        
        _log.info(f"[LLM意图识别] 解析结果: {intent_result}")
        
        # 返回标准化的结果
        return {
            'is_search_query': bool(intent_result.get('is_search_query', False)),
            'should_use_selected_docs': bool(intent_result.get('should_use_selected_docs', False)),
            'reasoning': intent_result.get('reasoning', "No reasoning provided")
        }
    except Exception as e:
        _log.error(f"[LLM意图识别] 处理出错: {str(e)}")
        # 发生错误时使用传统方法回退
        search_keywords = [
            "papers about", "about", "search", "find", "retrieval", 
            "检索", "查找", "查询", "论文", "搜索", "give me", "tell me about",
            "papers", "文献", "文章"
        ]
        
        exclude_keywords = [
            "基于这些文献", "基于这", "基于上述", "基于选择", "based on", 
            "综述", "文献综述", "review", "literature review", "survey",
            "summary", "summarize", "总结", "汇总", 
            "对比", "分析", "比较", "这些文章", "这些文献", "这些论文", "这几篇", 
            "这7篇", "这些7篇", "这七篇", "这几篇", "analyze", "compare", "comparison",
            "these papers", "these documents", "these articles", "异同", "相同点", "不同点",
            "similarities", "differences", "区别", "共同", "特点"
        ]
        
        is_exclude_match = any(keyword.lower() in query.lower() for keyword in exclude_keywords)
        contains_reference_to_selected = "这" in query or "these" in query.lower() or "这些" in query
        is_search_query = (any(keyword.lower() in query.lower() for keyword in search_keywords) 
                          and not is_exclude_match 
                          and not contains_reference_to_selected)
        
        return {
            'is_search_query': is_search_query,
            'should_use_selected_docs': bool(selected_docs) and (is_exclude_match or contains_reference_to_selected),
            'reasoning': "使用传统关键词匹配方法回退，因为LLM意图识别失败"
        }
