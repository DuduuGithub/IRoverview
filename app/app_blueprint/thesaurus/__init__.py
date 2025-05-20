from flask import Blueprint

thesaurus_bp = Blueprint('thesaurus', __name__,
                     template_folder='templates',
                     static_folder='static')

from . import routes 