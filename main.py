import os
import re
import gc
import time
import base64

import cv2
import numpy as np

from flask_socketio import SocketIO, emit
from flask import Flask, render_template, Response, request

import logging
from logging.handlers import RotatingFileHandler

from config import raw_frame_shape, video_ip
from utils import ImageCapture, base64_to_image, image_to_base64
from play_chess import get_predicted_result


# File hangler
def get_rotatingfilehandler():
    log_format_ = f"%(asctime)s - [%(levelname)s] - %(message)s"
    handler = RotatingFileHandler('log_file.log', maxBytes=2000000, backupCount=0)
    handler.setFormatter(logging.Formatter(log_format_))
    handler.setLevel(logging.DEBUG)
    return handler


def get_logger():
    logger = logging.getLogger('socketio')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_rotatingfilehandler())
    return logger


logger = get_logger()

app = Flask(__name__)
app.secret_key = "vatsal"
socketio = SocketIO(app, logger=logger, engineio_logger=False)

image_capture = ImageCapture(image_shape=raw_frame_shape)


@socketio.on('Captured_Image', namespace='/videofeed')
def process_images(captured_image_data):
    captured_image_data = re.sub(r'^data:image/.+;base64,', '', captured_image_data)

    # convert base64 to numpy image
    image = base64_to_image(captured_image_data)
    logger.debug(f'client {request.sid} request image size {image.shape}')

    # who_you_are = input("Are you White/Black [w/b] : ")
    who_you_are = 'w'

    # image = image_capture.get_frame(video_ip)
    # image = cv2.imread("Data/raw_images/65.png")

    # flag = 0 No chess board detected
    # flag = 1 chess board detected but move not
    # flag = 2 chess board detected and move predicted
    # flag = 3 chess board detected and game is over
    move_box_image, border_image, flag = get_predicted_result(image, who_you_are)
    logger.info(f"Image prediction : flag : {flag} for client {request.sid}")

    if flag in (2, 3):

        move_box_image = cv2.resize(move_box_image, image.shape[:2])
        border_image = cv2.resize(border_image, image.shape[:2])

        # convert numpy image to base64
        move_box_image_base64 = 'data:image/jpeg;base64,' + image_to_base64(move_box_image)
        border_image_base64 = 'data:image/jpeg;base64,' + image_to_base64(border_image)

        emit('Processed_Image', {'image_data': move_box_image_base64}, namespace='/videofeed')
        emit('Border_Image', {'image_data': border_image_base64}, namespace='/videofeed')


@socketio.on('connect', namespace='/videofeed')
def videofeed_connect():
    print(f'-> New client connected: {request.sid}')
    logger.debug(f'New client connected: {request.sid}')


@socketio.on('disconnect', namespace='/videofeed')
def videofeed_disconnect():
    print(f'-> client disconnected: {request.sid}')
    logger.debug(f'client disconnected: {request.sid}')


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":

    try:
        logger.info("App Started.")
        socketio.run(app, host="0.0.0.0", port=4812, debug=False,
                     certfile='SSL_Certificate/localhost.crt',
                     keyfile='SSL_Certificate/localhost.key'
                     )

    except Exception as e:
        print("Exception : ", e)
        logger.error(f'Exception: {e}')