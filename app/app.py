import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from werkzeug.security import generate_password_hash
from flask import Flask, redirect, url_for, jsonify
from app_blueprint import register_blueprints
from Database.model import *
from utils import *
from Database.config import db
import Database.config 
from sqlalchemy.sql import text

def init_database():
    with app.app_context():
        try:
            # 删除所有表
            db.drop_all()
            db.session.commit()
            
            # 创建所有表
            db.create_all()
            db.session.commit()
            print("数据库初始化完成！")
            
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
    
    # 只在调试模式下显示SQL语句
    app.config['SQLALCHEMY_ECHO'] = debug
    
    # 初始化数据库
    db.init_app(app)
    
    # 注册蓝图
    register_blueprints(app=app)
    
    return app

app = createApp(debug=False)  # 设置为False来关闭SQL语句输出

# 根路由重定向到首页
@app.route('/')
def index():
    return redirect(url_for('search.index'))  # 修改为您的search蓝图索引路由


if __name__ == '__main__':
    # 初始化数据库
    init_database()
    app.run(debug=True)  # 这里的debug只控制Flask的调试模式，不影响SQL输出