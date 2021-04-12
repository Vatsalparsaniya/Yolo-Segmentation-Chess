import os
import cv2
import numpy as np
from config import *
from utils import sort_files
from more_itertools import run_length


class GenerateFen:
    def __init__(self, config_path, weights_path, class_fen_dict, classes_path, width, height):
        self.yolo_model = self.create_model(config_path, weights_path)
        self.output_layers = self.get_output_layers(self.yolo_model)
        self.class_fen_dict = class_fen_dict
        self.width = width
        self.height = height
        self.grid_contours = self.get_grid_contours()

        with open(classes_path, 'rt') as f:
            self.classes = f.read().strip('\n').split('\n')

    @staticmethod
    def create_model(config, weights):
        model = cv2.dnn.readNetFromDarknet(config, weights)
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CPU
        model.setPreferableBackend(backend)
        model.setPreferableTarget(target)
        return model

    @staticmethod
    def get_output_layers(model):
        layer_names = model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
        return output_layers

    def blob_from_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(image, 1 / 255., (self.height, self.width), [0, 0, 0], 1, crop=False)
        return blob

    def predict(self, blob):
        self.yolo_model.setInput(blob)
        outputs = self.yolo_model.forward(self.output_layers)
        return outputs

    def get_output_classes(self, outputs, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold):
        select_classes = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:

                scores = detection[5:]
                class_id = np.argmax(scores)
                class_name = self.classes[int(class_id)]
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    select_classes.append(class_name)
                    cx, cy, width, height = (
                            detection[0:4] * np.array([self.width, self.height, self.width, self.height])).astype("int")

                    x = int(cx - width / 2)
                    y = int(cy - height / 2)
                    boxes.append([x, y, int(width), int(height), cx, cy])
                    confidences.append(float(confidence))

        nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        if len(nms_indices) != 0:
            b_boxes = [boxes[ind] for ind in nms_indices.flatten()]
            s_classes = [select_classes[ind] for ind in nms_indices.flatten()]
            c_confidance = [confidences[ind] for ind in nms_indices.flatten()]
            return b_boxes, s_classes, c_confidance
        else:
            return None, None, None

    def get_grid_contours(self, n=8, m=8):

        M = self.height // n
        N = self.width // m

        grids_contours = []
        for y in range(0, self.height, M):
            for x in range(0, self.width, N):
                p1 = [x, y]
                p2 = [x + N, y]
                p3 = [x + N, y + M]
                p4 = [x, y + M]
                grids_contours.append([p1, p2, p3, p4])

        return np.array(grids_contours, dtype=np.int32)

    @staticmethod
    def convert_cell(value):
        if value == 'e':
            return None
        else:
            return value

    def convert_rank(self, rank):
        return ''.join(
            value * count if value else str(count)
            for value, count in run_length.encode(map(self.convert_cell, rank)))

    def fen_from_board(self, board, who_turn):
        return '/'.join(map(self.convert_rank, board)) + f' {who_turn} KQkq - 0 1'

    def get_fen(self, image, who_turn):

        blob = self.blob_from_image(image)
        outputs = self.predict(blob)
        boxes, select_classes, confidence_classes = self.get_output_classes(outputs)

        if boxes is not None and select_classes is not None:
            boxes_center_point = [(points[0] + points[2] // 2, points[1] + points[3] // 2) for points in boxes]
            board = np.full((8, 8), 'e').tolist()

            for grid_index, grid_box in enumerate(self.grid_contours):

                row = grid_index // 8
                col = grid_index % 8
                predicted_class = None
                # cv2.drawContours(image, [grid_box], 0, (0, 0, 255), 2)
                # cv2.imshow("image", image)

                for i, center_point in enumerate(boxes_center_point):

                    if cv2.pointPolygonTest(grid_box, center_point, False) > 0:
                        predicted_class = select_classes[i]
                        # print("predicted_class : ", predicted_class)
                        # print("center_point : ", center_point)
                        # print("grid_box", grid_box)

                        break

                if predicted_class is not None:
                    board[row][col] = self.class_fen_dict[predicted_class]
                else:
                    board[row][col] = 'e'

                # cv2.waitKey(0)

            fen_line = self.fen_from_board(board, who_turn)

            return fen_line

        else:
            return None


if __name__ == "__main__":

    fen_generator = GenerateFen(config_path, weights_path, class_fen_dict, classes_path, yolo_width, yolo_height)

    image_paths = [os.path.join(yolo_process_images_path, x) for x in sort_files(yolo_process_images_path)]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        fen_line = fen_generator.get_fen(image)
        print("fen_line : ", fen_line)
        cv2.imshow("Image", image)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
