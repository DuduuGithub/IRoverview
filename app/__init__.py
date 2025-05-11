from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from Database.config import db

login_manager = LoginManager()

def create_app(config_class=None):
    app = Flask(__name__)
    
    # 初始化扩展
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    return app 