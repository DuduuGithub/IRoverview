import sys
import os
# 将项目根目录添加到 sys.path
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
            # 先删除视图和表
            try:
                db.session.execute(text('DROP VIEW IF EXISTS DocumentDisplayView'))
                db.session.commit()
            except:
                pass
            
            db.drop_all()
            db.session.commit()
            
            # 创建所有表
            db.create_all()
            db.session.commit()
            print("表创建成功！")
            
            print("数据库初始化完成！")
            
        except Exception as e:
            print(f"数据库初始化失败: {str(e)}")
            db.session.rollback()
            raise

def createApp():
    app = Flask(__name__,
               static_folder='static',
               static_url_path='/static')
               
    
    # 加载配置
    app.config.from_object(Database.config)
    
    # 初始化数据库
    db.init_app(app)

    app.config['SQLALCHEMY_ECHO'] = True
    
    
    # 注册蓝图
    register_blueprints(app=app)
    

    return app

app = createApp()

# 根路由重定向到首页
@app.route('/')
def index():
    return redirect(url_for('searcher.index'))


if __name__ == '__main__':
    # 当数据库表结构发生变化时，需执行一下这个
    # init_database()
    app.run()