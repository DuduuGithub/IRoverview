from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    """初始化数据库"""
    db.init_app(app)
