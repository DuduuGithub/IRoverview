from flask_sqlalchemy import SQLAlchemy
import pymysql
import os

pymysql.install_as_MySQLdb() 
db = SQLAlchemy()

# 数据库配置
USERNAME = 'root'
PASSWORD = '123456'
HOST = '127.0.0.1'
PORT = '3306'
DATABASE = 'iroverview'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOST, PORT, DATABASE)

# SQLAlchemy配置
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True

# Flask配置
SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-for-literature-search'

# OpenAI配置
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')  # 请在环境变量中设置你的OpenAI API密钥

# KIMI API配置
MOONSHOT_API_KEY = 'sk-BfheznukbDFaEdnGe9ql2CtUuCfkMBWmrJViPkfj6hqwTNBO'
MOONSHOT_BASE_URL = 'https://api.moonshot.cn/v1'
MOONSHOT_MODEL = 'moonshot-v1-8k'
MOONSHOT_SYSTEM_PROMPT = '你是一个专业的古籍文书分析助手，由 Moonshot AI 提供支持。你擅长解读和分析古代文书，会用中文回答用户的问题。'

# 搜索配置
SEARCH_CONFIG = {
    'default_page_size': 10,
    'max_page_size': 50,
    'enable_model_rerank': True,  # 是否启用模型重排序
    'model_path': None,  # 使用默认模型路径
    'rerank_batch_size': 32  # 重排序批处理大小
}

# BERT模型配置
BERT_CONFIG = {
    'model_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'sort_ai', 'bert', 'bert-base-uncased'),
    'max_length': 512,
    'device': 'cuda'  # 或 'cpu'
}
