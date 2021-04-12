import os
import cv2
import numpy as np
from models import Unet
from utils import sort_files
from config import model_name, frame_shape, image_path
from config import inference_input_image, inference_predicted_mask
from config import inference_border_image, inference_predicted_warped_image


class InferenceModel:

    def __init__(self, model, model_input_shape=(frame_shape[0], frame_shape[1], 3)):
        self.model = model
        self.model_input_shape = model_input_shape  # (height, width, channel)

    def image_prediction_mask(self, image):

        image_height, image_width = image.shape[0], image.shape[1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_reshaped = cv2.resize(image_rgb, (self.model_input_shape[0], self.model_input_shape[1]))
        image_batch = np.expand_dims(image_reshaped, axis=0)
        prediction = self.model.predict(image_batch)[0]

        chess_board_mask = prediction[:, :, 0]
        # background_mask = prediction[:, :, 1]

        chess_board_mask = np.array(chess_board_mask * 255, dtype=np.uint8)

        reshaped_chess_board_mask = cv2.resize(chess_board_mask, (image_height, image_width))
        return reshaped_chess_board_mask

    @staticmethod
    def get_mask_corner_points(mask, method="contour"):

        if method == "contour":

            _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if hierarchy is not None:

                cnt = max(contours, key=lambda x: cv2.contourArea(x))
                cnt_epsilon = 0.03 * cv2.arcLength(cnt, True)
                cnt_approx = cv2.approxPolyDP(cnt, cnt_epsilon, True)

                if len(cnt_approx) == 4:
                    return cnt_approx.reshape((4, 2))
                else:
                    return None
            else:
                return None

        if method == "goodFeaturesToTrack":

            corners = cv2.goodFeaturesToTrack(mask, maxCorners=4, qualityLevel=0.4, minDistance=100)

            if len(corners) == 4:
                return corners.reshape((4, 2))
            else:
                return None

    @staticmethod
    def get_order_points(pts):
        # first - top-left,
        # second - top-right
        # third - bottom-right
        # fourth - bottom-left

        rect = np.zeros((4, 2), dtype="float32")

        # top-left point will have the smallest sum
        # bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # top-right point will have the smallest difference
        # bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def point_transform(self, image, points):
        order_corner_points = self.get_order_points(points)

        height, width = image.shape[0], image.shape[1]

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(order_corner_points, dst)
        warped_image = cv2.warpPerspective(image, M, (width, height))

        return warped_image

    def mask_check(self, corner_points, image):

        if corner_points is not None:
            order_points = self.get_order_points(corner_points)
            image_height, image_width = image.shape[0], image.shape[1]
            chess_board_padding = 5

            x1, y1 = order_points[0]  # top-left,
            x2, y2 = order_points[1]  # top-right
            x3, y3 = order_points[2]  # bottom-right
            x4, y4 = order_points[3]  # bottom-left

            if (chess_board_padding < x1 < image_width - chess_board_padding) and \
                    (chess_board_padding < y1 < image_height - chess_board_padding) and \
                    (x1 < x2 < image_width - chess_board_padding) and \
                    (chess_board_padding < y2 < image_height - chess_board_padding) and \
                    (chess_board_padding < x3 < image_width - chess_board_padding) and \
                    (y2 < y3 < image_height - chess_board_padding) and \
                    (chess_board_padding < x4 < image_width - chess_board_padding) and \
                    (y1 < y4 < image_height - chess_board_padding):
                return True
            else:
                return False
        else:
            return False

    def get_predicted_warped_image(self, image):

        mask = self.image_prediction_mask(image)
        corner_points = self.get_mask_corner_points(mask)

        if self.mask_check(corner_points, image):
            warped_image = self.point_transform(image, corner_points)
            return warped_image
        else:
            return None

    @staticmethod
    def draw_border(img, corner_points):

        x1, y1 = corner_points[0]  # top-left,
        x2, y2 = corner_points[1]  # top-right
        x3, y3 = corner_points[2]  # bottom-right
        x4, y4 = corner_points[3]  # bottom-left

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # top-left -- top-right
        cv2.line(img, (x2, y2), (x3, y3), (255, 0, 0), 3)  # top-right -- bottom-right
        cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 3)  # bottom-right -- bottom-left
        cv2.line(img, (x4, y4), (x1, y1), (255, 0, 0), 3)  # bottom-left -- top-left

        cv2.circle(img, (x1, y1), 4, (0, 0, 255), -1)  # top-left,
        cv2.circle(img, (x3, y3), 4, (0, 0, 255), -1)  # top-right
        cv2.circle(img, (x4, y4), 4, (0, 0, 255), -1)  # bottom-right
        cv2.circle(img, (x2, y2), 4, (0, 0, 255), -1)  # bottom-left

        return img

    def get_marked_border_image(self, image):

        mask = self.image_prediction_mask(image)
        corner_points = self.get_mask_corner_points(mask)

        if self.mask_check(corner_points, image):
            border_image = self.draw_border(image, corner_points)
            return border_image
        else:
            # cv2.putText(image, "Border Not Detected", (10, 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 2)
            return image


if __name__ == "__main__":

    unet = Unet(model_name=model_name)
    model = unet.load_model(pretrained=True)
    inference_model = InferenceModel(model)

    image_paths = [os.path.join(inference_input_image, x) for x in sort_files(inference_input_image)]

    # for image_path in ['Data/inference_data/input_image/97.png']:
    for image_path in image_paths:
        file_name = os.path.basename(image_path)
        image = cv2.imread(image_path)

        warped_predicted_image = inference_model.get_predicted_warped_image(image)
        mask = inference_model.image_prediction_mask(image)
        border_image = inference_model.get_marked_border_image(image.copy())

        cv2.imshow("Image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("warped_predicted_image", warped_predicted_image)
        cv2.imshow("border_image", border_image)

        cv2.imwrite(os.path.join(inference_predicted_mask, file_name), mask)
        cv2.imwrite(os.path.join(inference_predicted_warped_image, file_name), warped_predicted_image)
        cv2.imwrite(os.path.join(inference_border_image, file_name), border_image)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
