import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, redirect, url_for, jsonify
from app_blueprint.search import searcher_bp
from app_blueprint.reader import reader_bp
from app_blueprint.thesaurus import thesaurus_bp
from app_blueprint.rag_mcp import rag_mcp_bp
from Database.model import *
from Database.config import db
import Database.config 
from sqlalchemy.sql import text
from sqlalchemy import inspect

def show_db_menu():
    while True:
        print("\n=== 请选择运行模式 ===")
        print("1. 正常运行")
        print("2. 重建数据库")
        choice = input("\n请选择 (1-2): ").strip()
        
        if choice == "1":
            return False
        elif choice == "2":
            confirm = input("警告：重建数据库将删除所有现有数据！确定要继续吗？(y/n): ").strip().lower()
            if confirm == 'y':
                return True
            else:
                continue
        else:
            print("无效的选择，请重试。")

def init_database(rebuild=False):
    print("\n正在初始化数据库...")
    with app.app_context():
        try:
            if rebuild:
                # 删除所有表
                db.drop_all()
                db.session.commit()
                
                # 创建所有表
                db.create_all()
                db.session.commit()
                print("数据库重建完成！")
            else:
                print("正常启动，保持现有数据库。")
        except Exception as e:
            print(f"数据库操作失败: {str(e)}")
            db.session.rollback()
            raise

def createApp(debug=False):
    app = Flask(__name__,
               static_folder='static',
               static_url_path='/static')
               
    # 加载配置
    app.config.from_object(Database.config)
    
    # 禁用SQL语句输出
    app.config['SQLALCHEMY_ECHO'] = False
    
    # 初始化数据库
    db.init_app(app)
    
    # 注册蓝图
    app.register_blueprint(searcher_bp, url_prefix='/search')
    app.register_blueprint(reader_bp, url_prefix='/reader')
    app.register_blueprint(thesaurus_bp, url_prefix='/word_clouds')
    app.register_blueprint(rag_mcp_bp, url_prefix='/rag_mcp')
    return app

app = createApp(debug=False)  # 设置为False来关闭SQL语句输出

# 根路由重定向到搜索首页
@app.route('/')
def index():
    return redirect(url_for('searcher.search_page'))  # 修改为searcher蓝图的search_page路由

if __name__ == '__main__':
    # 显示运行模式选择菜单
    rebuild_choice = show_db_menu() # 完成第一次插入数据后，可注释掉这一行
    
    # 初始化数据库
    init_database(rebuild=rebuild_choice) # 完成第一次插入数据后，可注释掉这一行
    
    app.run(debug=True)  # 这里的debug只控制Flask的调试模式，不影响SQL输出