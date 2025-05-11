from flask import Blueprint

# 导入搜索蓝图
from .searcher import searcher_bp

# 创建搜索模块蓝图
search_bp = Blueprint('search', __name__)

# 注册子蓝图
search_bp.register_blueprint(searcher_bp, url_prefix='/searcher') 