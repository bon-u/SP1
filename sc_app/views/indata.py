from flask import Blueprint, render_template, request
from . import pred

indata = Blueprint('indata', __name__)

@indata.route('/indata',methods=['POST', 'GET'])
def index():
    return render_template('index.html'), 200