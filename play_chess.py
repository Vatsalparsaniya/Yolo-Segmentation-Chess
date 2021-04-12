import gc
import cv2
import numpy as np
import chess
import chess.engine

import time
from utils import *
from config import *
from models import Unet

from inference_chess_board_model_unet import InferenceModel
from generate_fenline import GenerateFen

unet_model = Unet(model_name=model_name).load_model(pretrained=True)
inference_model = InferenceModel(unet_model)
fen_generator = GenerateFen(config_path, weights_path, class_fen_dict, classes_path, yolo_width, yolo_height)


def get_best_move(board, engine_path):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        result = engine.play(board, chess.engine.Limit(time=1.5))
        time.sleep(2)

        if result != "None":
            position1 = str(result.move)[0:2]
            position2 = str(result.move)[2:4]
            print(f"best move from {position1} to {position2}")
            engine.close()

            return position1, position2

        return None, None

    except Exception as e:

        print("Exception in get_best_move : ", e)
        engine.close()

        del engine
        gc.collect()

        return None, None


def get_grid_contours(height, width, n=8, m=8):
    M = height // n
    N = width // m

    grids_contours = []
    for y in range(0, height, M):
        for x in range(0, width, N):
            if y + M <= height and x + N <= width:
                p1 = [x, y]
                p2 = [x + N, y]
                p3 = [x + N, y + M]
                p4 = [x, y + M]
                grids_contours.append([p1, p2, p3, p4])

    return np.array(grids_contours, dtype=np.int32)


def draw_move_on_image(image, from_box, to_box):
    overlay = image.copy()

    box_to_id_mapping = {}
    index = 0
    for i in "87654321":
        for j in "abcdefgh":
            box_to_id_mapping[j + i] = index
            index += 1

    height, width, _ = image.shape
    boxes = get_grid_contours(height, width)
    print(boxes.shape)
    from_box_points = boxes[box_to_id_mapping[from_box]]
    to_box_points = boxes[box_to_id_mapping[to_box]]

    from_box_center_point = (from_box_points[0][0] + from_box_points[2][0]) // 2, \
                            (from_box_points[0][1] + from_box_points[2][1]) // 2

    to_box_center_point = (to_box_points[0][0] + to_box_points[2][0]) // 2, \
                          (to_box_points[0][1] + to_box_points[2][1]) // 2

    cv2.rectangle(overlay, (from_box_points[0][0], from_box_points[0][1]),
                  (from_box_points[2][0], from_box_points[2][1]), (0, 0, 255), -1)

    cv2.rectangle(overlay, (to_box_points[0][0], to_box_points[0][1]),
                  (to_box_points[2][0], to_box_points[2][1]), (0, 255, 255), -1)

    cv2.arrowedLine(overlay, from_box_center_point, to_box_center_point, (255, 0, 0), 3)

    cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

    return image


def get_predicted_result(image, who_you_are):
    warped_predicted_image = inference_model.get_predicted_warped_image(image)
    border_image = inference_model.get_marked_border_image(image.copy())

    if warped_predicted_image is not None:
        fen_line = fen_generator.get_fen(warped_predicted_image, who_turn=who_you_are.lower())

        if fen_line is not None:
            board = chess.Board(fen=fen_line)

            if not board.is_game_over():
                from_box, to_box = get_best_move(board, engine_path)
                if from_box is not None and to_box is not None:
                    move_box_image = draw_move_on_image(warped_predicted_image.copy(), from_box, to_box)
                    return move_box_image, border_image, 2
                else:
                    return warped_predicted_image, border_image, 1
            else:
                cv2.putText(warped_predicted_image, "Game Over", (10, 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
                return warped_predicted_image, border_image, 3
        else:
            return warped_predicted_image, border_image, 1
    else:
        return None, None, 0
