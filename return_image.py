import os
import sys
sys.path.append('./')
from argparse import ArgumentParser
from hellow_world import hello_world
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage
)

app_name = "test-mahjong"

os.makedirs("static/images")
app = Flask(__name__, static_url_path="/static")

…

@handler.add(MessageEvent, message=ImageMessage)
def message_image(event):
    try:
        token = event.reply_token
        msg_id = event.message.id
        msg_content = line_bot_api.get_message_content(msg_id)
        tmp_path = "static/images/{}".format(msg_id)

        with open(tmp_path, "wb") as fw:
            for chunk in msg_content.iter_content():
                fw.write(chunk)

        with Image.open(tmp_path) as img:
            img_fmt = img.format

            if img_fmt == "JPEG":
                os.rename(tmp_path, tmp_path + ".jpg")
                url = "https://{}.herokuapp.com/{}.jpg".format(app_name, tmp_path)

            img_msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
            line_bot_api.reply_message(event.reply_token, img_msg)

        sid = choose_source_id(event.source)
        line_bot_api.push_message(sid, TextSendMessage(text="message_id: {}, {}, {}".format(msg_id, img_fmt, str(exif_table))))
    except linebot.exceptions.LineBotApiError as e:
        # なんかエラー処理
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='Error'))
