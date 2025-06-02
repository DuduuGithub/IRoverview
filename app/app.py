import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, redirect, url_for, jsonify
from app_blueprint.search import searcher_bp
from app_blueprint.reader import reader_bp
from app_blueprint.thesaurus import thesaurus_bp
from Database.model import *
from Database.config import db
import Database.config 
from sqlalchemy.sql import text
from sqlalchemy import inspect

def init_database():
    print(0)
    with app.app_context():
        try:
            #检查数据库是否已存在
            inspector = inspect(db.engine)
            if not inspector.has_table('works'):
                # 删除所有表
                db.drop_all()
                db.session.commit()
                
                # 创建所有表
                db.create_all()
                db.session.commit()
                print("数据库初始化完成！")
            else:
                print("数据库已存在，跳过初始化。")
        except Exception as e:
            print(f"数据库初始化失败: {str(e)}")
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
    
    return app

app = createApp(debug=False)  # 设置为False来关闭SQL语句输出

# 根路由重定向到搜索首页
@app.route('/')
def index():
    return redirect(url_for('searcher.search_page'))  # 修改为searcher蓝图的search_page路由

if __name__ == '__main__':
    # 初始化数据库
    init_database()
    app.run(debug=True)  # 这里的debug只控制Flask的调试模式，不影响SQL输出