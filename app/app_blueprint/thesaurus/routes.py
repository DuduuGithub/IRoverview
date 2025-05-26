from flask import Blueprint, render_template, request, jsonify
from . import thesaurus_bp

@thesaurus_bp.route('/')
def index():
    """渲染词云分析主页"""
    return render_template('word_clouds.html') 