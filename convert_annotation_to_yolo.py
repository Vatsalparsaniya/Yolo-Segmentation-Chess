import os
import json
import cv2
import argparse
from utils import sort_files
from config import inference_predicted_warped_image, inference_predicted_warped_image_annotation
from config import inference_predicted_warped_image_annotation_yolo
from config import chess_piece_labels


class ConvertToYOLO:

    def __init__(self, image_path, annotation_path, output_path, chess_piece_labels):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.output_path = output_path
        self.chess_piece_labels = chess_piece_labels

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

    def convert_to_yolo(self):

        image_paths = [os.path.join(self.image_path, x) for x in sort_files(self.image_path)]
        annot_paths = [os.path.join(self.annotation_path, x) for x in sort_files(self.annotation_path)]

        for i_path, a_path in zip(image_paths, annot_paths):
            with open(a_path, 'r') as annotation_file:
                annotation = json.load(annotation_file)

            file_name = os.path.basename(a_path).split(".")[0]
            image = cv2.imread(i_path)
            image_height, image_width = image.shape[0], image.shape[1]

            annotation_text_file = open(os.path.join(self.output_path, f"{file_name}.txt"), "w")

            annotated_shape_list = annotation['shapes']
            for label_dict in annotated_shape_list:

                if label_dict['shape_type'] == 'rectangle':
                    label = label_dict['label']
                    [x1, y1] = label_dict['points'][0]
                    [x2, y2] = label_dict['points'][1]

                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    box_height = y2 - y1
                    box_width = x2 - x1
                    label_class = self.chess_piece_labels.index(label)

                    annotation_text_file.write(
                        f"{label_class} {center_x / image_width} {center_y / image_height} {box_width / image_width} {box_height / image_height}\n")

            annotation_text_file.close()

    def create_train_txt_file(self):

        image_paths = [f"data/obj/{x}" for x in sort_files(self.image_path)]

        with open("yolo_config/train.txt", "w") as train_file:
            for i_path in image_paths:
                train_file.write(f"{i_path}\n")

            train_file.close()


if __name__ == "__main__":
    yolo_annotation_converter = ConvertToYOLO(inference_predicted_warped_image,
                                              inference_predicted_warped_image_annotation,
                                              inference_predicted_warped_image_annotation_yolo,
                                              chess_piece_labels)

    yolo_annotation_converter.convert_to_yolo()
    yolo_annotation_converter.create_train_txt_file()
