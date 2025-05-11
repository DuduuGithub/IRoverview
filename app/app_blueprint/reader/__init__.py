from flask import Blueprint

reader_bp = Blueprint('reader', __name__,
                     template_folder='templates',
                     static_folder='static')

from . import routes 
from .routes import reader_bp