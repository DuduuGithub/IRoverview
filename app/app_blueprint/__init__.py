import os
import importlib
from flask import Blueprint, Flask
from .reader import reader_bp
from .analysis import analysis_bp
from .search.basicSearch import search_bp

def register_blueprints(app: Flask):
    # 注册搜索蓝图
    app.register_blueprint(search_bp, url_prefix='/search')
    
    # 尝试导入旧的searcher蓝图，但不再作为必需组件
    try:
        from searcher import searcher_bp  # 直接从项目根目录导入，而不是从app包
        app.register_blueprint(searcher_bp, url_prefix='/searcher')
    except ImportError:
        # 如果导入失败，打印警告但不中断程序
        print("警告: 无法导入searcher蓝图，这不会影响应用的正常运行")
    
    app.register_blueprint(reader_bp, url_prefix='/reader')
    app.register_blueprint(analysis_bp, url_prefix='/analysis')
