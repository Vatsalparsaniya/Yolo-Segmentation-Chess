import os
import cv2
import urllib.request
from utils import ImageCapture, SaveImage
from config import raw_frame_shape, raw_image_path, video_ip

if __name__ == "__main__":

    image_capture = ImageCapture(image_shape=raw_frame_shape, method='ipcam')
    data_collection_save_image = SaveImage(raw_image_path)

    path, dirs, files = next(os.walk(raw_image_path))
    image_counter = len(files)

    while True:

        try:
            image = image_capture.get_frame(video_ip)

            cv2.imshow("Image Frame", image)

            if cv2.waitKey(25) & 0xFF == ord('s'):
                data_collection_save_image.save_image(image, image_counter)
                image_counter += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        except urllib.error.URLError as e:
            print('Caught this error: ' + str(e))
            break

    cv2.destroyAllWindows()
