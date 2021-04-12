import os
import json
import cv2
import numpy as np
from utils import sort_files
from config import inference_predicted_warped_image, yolo_visual_images_annotations
from config import inference_predicted_warped_image_annotation_yolo
from config import chess_piece_labels


class VisualYOLO:

    def __init__(self, image_path, yolo_annotation_path, image_save_path, chess_piece_labels):
        self.image_path = image_path
        self.annotation_path = yolo_annotation_path
        self.chess_piece_labels = chess_piece_labels
        self.image_save_path = image_save_path
        self.colors = np.random.randint(0, 255, size=(len(chess_piece_labels), 3))

    def view_yolo_annotations(self):

        image_paths = [os.path.join(self.image_path, x) for x in sort_files(self.image_path)]
        annot_paths = [os.path.join(self.annotation_path, x) for x in sort_files(self.annotation_path)]

        for i_path, a_path in zip(image_paths, annot_paths):

            image = cv2.imread(i_path)
            image_height, image_width = image.shape[0], image.shape[1]

            file_name = os.path.basename(a_path).split(".")[0]
            annotation_file = open(a_path, "r")

            for line in annotation_file.readlines():
                line = list(map(float, line.split()))

                label_class = int(line[0])
                x_center = line[1] * image_width
                y_center = line[2] * image_height
                box_width = line[3] * image_width
                box_height = line[4] * image_height

                top_left_corner = (int(x_center - box_width // 2), int(y_center - box_height // 2))
                bottom_right_corner = (int(x_center + box_width // 2), int(y_center + box_height // 2))
                color = (int(self.colors[label_class][0]),
                         int(self.colors[label_class][1]),
                         int(self.colors[label_class][2]))

                cv2.rectangle(image, top_left_corner, bottom_right_corner, color, 2)
                cv2.imshow("annotated image", image)
                cv2.imwrite(os.path.join(self.image_save_path, f"{file_name}.png"), image)

            cv2.waitKey(0)
            annotation_file.close()


if __name__ == "__main__":
    yolo_visual = VisualYOLO(inference_predicted_warped_image,
                             inference_predicted_warped_image_annotation_yolo,
                             yolo_visual_images_annotations,
                             chess_piece_labels)

    yolo_visual.view_yolo_annotations()
