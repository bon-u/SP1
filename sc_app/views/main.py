from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/',methods=['POST', 'GET'])
def index():
    return render_template('index.html'), 200