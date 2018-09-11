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
from mahjong_detection import detection_mahjong

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

line_bot_api = LineBotApi(channel_access_token)
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

def load_file_from_s3():
    cmd = 'cd mahjong_detection/checkpoint\nwget https://s3-ap-northeast-1.amazonaws.com/test-mahjong/weights.25-0.05.hdf5'
    os.system(cmd)

def _create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print('make dir: {}'.format(dir_name))

DIR_INPUT = "static/input_images"
DIR_OUTPUT = "static/output_images"

@handler.add(MessageEvent, message=ImageMessage)
def message_image(event):
    # try:
        _create_dir(DIR_INPUT)
        _create_dir(DIR_OUTPUT)

        token = event.reply_token
        msg_id = event.message.id
        msg_content = line_bot_api.get_message_content(msg_id)
        tmp_path = "{}/{}".format(DIR_INPUT, msg_id)

        # load file
        if not os.path.exists('mahjong_detection/checkpoint/weights.25-0.05.hdf5'):
            load_file_from_s3()

        with open(tmp_path, "wb") as fw:
            for chunk in msg_content.iter_content():
                fw.write(chunk)

        with Image.open(tmp_path) as img:
            img_fmt = img.format
            # return orginal image
            os.rename(tmp_path, tmp_path + ".jpg")
            print('*'*40, tmp_path)
            url = "https://{}.herokuapp.com/{}.jpg".format(app_name, tmp_path))
            img_msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            line_bot_api.reply_message(event.reply_token, img_msg)

            # # mahjong detector
            # image_detected, list_result_label = detection_mahjong.main(img)
            # output_path = detection_mahjong.savefig(image_detected, DIR_OUTPUT)
            # print('*'*40, output_path)
            #
            # # return result image
            # url = "https://{}.herokuapp.com/{}.jpg".format(app_name, output_path)
            # img_msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            # # txt_msg = TextSendMessage(text='ok')
            # line_bot_api.reply_message(event.reply_token, img_msg)

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
