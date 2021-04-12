import os

# Data Collection Config

raw_frame_shape = (576, 576)  # Raw image frame size (height, width)
raw_image_path = 'Data/raw_images'  # Raw Image save path
raw_annotated_path = 'Data/raw_annotated'  # Raw annotation folder path
video_ip = 'http://192.168.0.104:4812/shot.jpg'  # Ip webcam path

# Class Information

labels = sorted(['chess_board'])  # class labels
hues = [30]  # class colour
hues_dict = {hue: label for label, hue in zip(labels, hues)}
n_classes = len(labels) + 1

# Model Config

model_name = "Unet"
logdir = "logs"
model_save_path = os.path.join('models', model_name + '.hdf5')
image_path = 'Data/reshape_images'
annotated_path = 'Data/reshape_annotated'
frame_shape = (256, 256)  # Model train frame shape
assert frame_shape[0] % 32 == 0 and frame_shape[1] % 32 == 0, "imshape should be multiples of 32."

# Inference model Config

inference_input_image = "Data/inference_data/input_image"
inference_predicted_mask = "Data/inference_data/predicted_mask"
inference_predicted_warped_image = "Data/inference_data/predicted_warped_image"
inference_border_image = "Data/visualize_data/inference_border_image"
inference_predicted_warped_image_annotation = "Data/inference_data/predicted_warped_image_annotation"
inference_predicted_warped_image_annotation_yolo = "Data/inference_data/predicted_warped_image_annotation_yolo"

if not os.path.exists(inference_predicted_mask):
    os.makedirs(inference_predicted_mask)

if not os.path.exists(inference_predicted_warped_image):
    os.makedirs(inference_predicted_warped_image)

if not os.path.exists(inference_border_image):
    os.makedirs(inference_border_image)

# YOLO model

chess_piece_labels = ['white_rook', 'white_knight', 'white_king', 'white_queen', 'white_bishop', 'white_pawn',
                      'black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_king', 'black_queen']

confidence_threshold = 0.6
nms_threshold = 0.4
yolo_width = 1 * 608
yolo_height = 1 * 608

# Yolo-3
# config_path = 'models/yolo3/yolov3_custom.cfg'
# weights_path = 'models/yolo3/yolov3_custom_last.weights'
# classes_path = 'models/yolo3/obj.names'

# Yolo-3-tiny
config_path = 'models/yolo3-tiny/yolov3-tiny_obj_custom.cfg'
weights_path = 'models/yolo3-tiny/yolov3-tiny_obj_custom_last.weights'
classes_path = 'models/yolo3-tiny/obj.names'

yolo_process_images_path = 'Data/inference_data/predicted_warped_image'
yolo_prediction_visual_images_path = "Data/visualize_data/yolo_visuals_predictions"
yolo_visual_images_annotations = "Data/visualize_data/yolo_visuals_annotations"

# FEN config
class_fen_dict = {'white_rook': "R", 'white_knight': "N", 'white_king': "K", 'white_queen': "Q",
                  'white_bishop': "B", 'white_pawn': "P", 'black_pawn': "p", 'black_rook': "r",
                  'black_knight': "n", 'black_bishop': "b", 'black_king': "k", 'black_queen': "q"}

# stockfish engine
engine_path = "models/stockfish/stockfish_10_x64.exe"
