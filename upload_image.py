import os
import sys
sys.path.append('./')
from argparse import ArgumentParser
from flask import Flask, render_template, request
from io import BytesIO
import urllib
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/test.png')
def index():
    # 「templates/index.html」のテンプレートを使う
    # 「message」という変数に"Hello"と代入した状態で、テンプレート内で使う
    return render_template('index.html')

if __name__ == '__main__':
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8000, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()
    port = int(os.environ.get('PORT', 55542))

    app.run(debug=options.debug, port=port, host='0.0.0.0')
