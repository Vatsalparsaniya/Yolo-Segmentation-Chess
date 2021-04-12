import os
import cv2
import sys
import numpy as np
import pandas as pd
from utils import sort_files
from config import *


def create_model(config, weights):
    model = cv2.dnn.readNetFromDarknet(config, weights)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU
    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    return model


def get_output_layers(model):
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    return output_layers


def blob_from_image(image, target_size):
    blob = cv2.dnn.blobFromImage(image, 1 / 255., target_size, [0, 0, 0], 1, crop=False)
    return blob


def predict(blob, model, output_layers):
    model.setInput(blob)
    outputs = model.forward(output_layers)
    return outputs


def get_output_classes(outputs, image_width, image_height, classes,
                       confidence_threshold=confidence_threshold, nms_threshold=nms_threshold):
    select_classes = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            class_name = classes[class_id]
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                select_classes.append(class_name)
                cx, cy, width, height = (
                        detection[0:4] * np.array([image_width, image_height, image_width, image_height])).astype(
                    "int")
                x = int(cx - width / 2)
                y = int(cy - height / 2)
                boxes.append([x, y, int(width), int(height), cx, cy])
                confidences.append(float(confidence))

    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    b_boxes = [boxes[ind] for ind in nms_indices.flatten()]
    s_classes = [select_classes[ind] for ind in nms_indices.flatten()]
    c_confidance = [confidences[ind] for ind in nms_indices.flatten()]
    return b_boxes, s_classes, c_confidance


def draw_boxes(image, boxes, select_classes, confidence_classes, classes):
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    overlay = image.copy()
    alpha = 0.4  # Transparency factor.

    for points, class_name, confidence in zip(boxes, select_classes, confidence_classes):
        color = [int(c) for c in colors[classes.index(class_name)]]
        cv2.rectangle(overlay, (points[0], points[1]),
                      (points[0] + points[2], points[1] + points[3]), color, -1)

        cv2.rectangle(overlay, (points[0], points[1]),
                      (points[0] + points[2], points[1] + points[3]), (255, 0, 0), 1)

        cv2.putText(image, f"{class_name}", (points[0], points[1] - 2),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)

        print(f"class : {class_name}, confidence : {confidence}")

    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image_new


def get_box_image(image_path, model, classes, width, height):
    original_image_BGR = cv2.imread(image_path)
    original_image_RGB = cv2.cvtColor(original_image_BGR, cv2.COLOR_BGR2RGB)

    output_layers = get_output_layers(model)

    blob = blob_from_image(original_image_RGB, (width, height))
    outputs = predict(blob, model, output_layers)

    boxes, select_classes, confidence_classes = get_output_classes(outputs, width, height, classes)

    box_image = original_image_BGR.copy()
    box_image = cv2.resize(box_image, (width, height))
    box_image = draw_boxes(box_image, boxes, select_classes, confidence_classes, classes)

    return box_image


if __name__ == "__main__":

    model = create_model(config_path, weights_path)

    with open(classes_path, 'rt') as f:
        classes = f.read().strip('\n').split('\n')

    image_paths = [os.path.join(yolo_process_images_path, x) for x in sort_files(yolo_process_images_path)]
    for image_path in image_paths:

        box_image = get_box_image(image_path, model, classes, yolo_width, yolo_height)
        cv2.imshow("image", box_image)
        file_name = os.path.basename(image_path)

        cv2.imwrite(f"{yolo_prediction_visual_images_path}/{file_name}", box_image)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
