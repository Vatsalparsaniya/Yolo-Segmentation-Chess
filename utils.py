import io
import os
import cv2
import json
import base64
import urllib.request
import numpy as np
from PIL import Image
from config import raw_frame_shape, raw_image_path, raw_annotated_path


class ImageCapture:

    def __init__(self, image_shape=None, method='ipcam'):
        self.image_shape = image_shape
        self.method = method

    def get_frame(self, data):
        if self.method == 'ipcam':

            # Use urllib to get the image from the IP camera
            # data = "http://192.168.0.106:4812/shot.jpg"
            image_request = urllib.request.urlopen(data)

            # Numpy to convert into a array
            image_numpy = np.array(bytearray(image_request.read()), dtype=np.uint8)

            # Finally decode the array to OpenCV usable format ;)
            image = cv2.imdecode(image_numpy, -1)

            if self.image_shape:
                image = cv2.resize(image, self.image_shape)

            return image

        if self.method == 'flask-api':
            return None


class SaveImage:

    def __init__(self, save_path):
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        self.save_path = save_path

    def save_image(self, image, image_name):
        cv2.imwrite(f"{self.save_path}/{image_name}.png", image)


def generate_missing_json():
    # creates a background json for the entire image if missing
    # this assumes you will never annotate a background class

    for im in os.listdir(raw_image_path):

        file_name = im.split('.')[0] + '.json'
        path = os.path.join(raw_annotated_path, file_name)

        if os.path.exists(path) is False:
            json_dict = {'shapes': [{"label": "background",
                                     "points": [[0, 0],
                                                [0, raw_frame_shape[1] - 1],
                                                [raw_frame_shape[0] - 1, raw_frame_shape[1] - 1],
                                                [raw_frame_shape[0] - 1, 0]]
                                     }]}

            with open(path, 'w') as handle:
                json.dump(json_dict, handle, indent=2)


def sort_files(path):
    return sorted(os.listdir(path), key=lambda x: int(x.split('.')[0]))


def base64_to_image(base64encoded_string):
    jpg_original = base64.b64decode(base64encoded_string)
    pil_image = Image.open(io.BytesIO(jpg_original))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def image_to_base64(image):
    retval, buffer_img = cv2.imencode('.jpeg', image)
    base64encoded_string = base64.b64encode(buffer_img).decode('utf-8')
    return base64encoded_string