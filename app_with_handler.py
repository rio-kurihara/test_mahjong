# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.


import os
import sys
sys.path.append('./')
from argparse import ArgumentParser
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from PIL import Image
import linebot
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage,
    PostbackEvent, PostbackTemplateAction, ButtonsTemplate, TemplateSendMessage
)

import random
import tensorflow as tf
from mahjong_detection import detection_mahjong
from mahjong_detection.lib import point_calculater

app_name = "test-mahjong"
app = Flask(__name__, static_url_path="/static")

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)

if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token, timeout=10000)
handler = WebhookHandler(channel_secret)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

def _create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('make dir: {}'.format(dir_name))

DIR_INPUT = "static/input_images"
DIR_OUTPUT = "static/output_images"
_create_dir(DIR_INPUT)
_create_dir(DIR_OUTPUT)

# model build
ssd = None
graph = None

@handler.add(MessageEvent, message=TextMessage)
def message_image(event):
    # model buildを外に出すとアプリ起動に時間がかかりすぎるので、テキストメッセージが来たらmodel buildする
    global ssd
    global graph
    if ssd is None:
        ssd = detection_mahjong.build_model()
        graph = tf.get_default_graph()
    txt_msg = TextSendMessage(text='ok')
    line_bot_api.reply_message(event.reply_token, txt_msg)


@handler.add(MessageEvent, message=ImageMessage)
def message_image(event):
    # try:
        token = event.reply_token
        msg_id = event.message.id
        msg_content = line_bot_api.get_message_content(msg_id)
        # tmp_path = "{}/{}".format(DIR_INPUT, msg_id)
        tmp_path = "static/{}".format(msg_id)

        with open(tmp_path, "wb") as fw:
            for chunk in msg_content.iter_content():
                fw.write(chunk)

        with Image.open(tmp_path) as img:
            img_fmt = img.format
            # # return orginal image
            # os.rename(tmp_path, tmp_path + ".jpg")
            # print('*'*40, tmp_path)
            # url = "https://{}.herokuapp.com/{}.jpg".format(app_name, tmp_path)
            # img_msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            # line_bot_api.reply_message(event.reply_token, img_msg)

            # mahjong detector
            global graph
            # with graph.as_default():
            #     output_path, list_piname = detection_mahjong.main(img, DIR_OUTPUT, ssd)

        # print('*'*40, output_path)
        # print(os.path.exists(output_path))

        # return result image
        # url = "https://{}.herokuapp.com/{}".format(app_name, output_path)
        # print(url)
        # txt_msg = TextSendMessage(text='ok')
        # line_bot_api.reply_message(event.reply_token, txt_msg)

        # 点数計算して結果のテキストを返す
        win_pi = '2m'
        dora_pi = '8s'
        path_config = 'mahjong_detection/config_point_calculate.ini'
        print(os.path.exists(path_config))

        list_piname = ['1m', '2m', '3m', 'f', 'f', '3s', '4s', '5s', '5p', '5p', '5p', '2p', '3p', '4p']
        pc = point_calculater.PointCalculater(list_piname, win_pi, dora_pi, path_config)
        yaku, han, hu, parent_point, child_point = pc.main()
        result_txt = point_calculater.create_return_txt(yaku, han, hu, parent_point, child_point)

        txt_msg = TextSendMessage(text=result_txt)
        txt_msg_2 = TextSendMessage(text='え、リーのみ？')

        # img_msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
        line_bot_api.reply_message(event.reply_token, [txt_msg, txt_msg_2])

    # except:
    #     line_bot_api.reply_message(
    #         event.reply_token,
    #         TextSendMessage(text='Error'))


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8000, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()
    port = int(os.environ.get('PORT', 55400))
    app.run(debug=options.debug, port=port, host='0.0.0.0')
