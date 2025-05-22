from flask_sqlalchemy import SQLAlchemy
import pymysql
import os

pymysql.install_as_MySQLdb() 
db = SQLAlchemy()

# 数据库配置
USERNAME = 'root'
PASSWORD = '0483'
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
