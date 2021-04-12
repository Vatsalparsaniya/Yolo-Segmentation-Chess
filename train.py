import os
import tensorflow as tf
from data_generator import DataGenerator
from utils import generate_missing_json, sort_files
from config import model_name, raw_annotated_path, raw_image_path, image_path, annotated_path, labels, model_save_path
from models import Unet
from resize_data import Resizedata

import warnings
warnings.filterwarnings(action='ignore')

if len(os.listdir(raw_image_path)) != len(os.listdir(raw_annotated_path)):
    generate_missing_json()

rasize_data = Resizedata(raw_image_path, raw_annotated_path)
rasize_data.save_resized_file()

unet = Unet(model_name=model_name)
model = unet.load_model(pretrained=False)

image_paths = [os.path.join(image_path, x) for x in sort_files(image_path)]
annot_paths = [os.path.join(annotated_path, x) for x in sort_files(annotated_path)]

train_datagenerator = DataGenerator(image_paths=image_paths,
                                    annotation_paths=annot_paths,
                                    labels=labels,
                                    batch_size=5,
                                    augment=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path,
                                                monitor='dice',
                                                verbose=0,
                                                mode='max',
                                                save_weights_only=True)

model.fit(train_datagenerator,
          steps_per_epoch=len(train_datagenerator),
          epochs=500, verbose=1,
          callbacks=[checkpoint])
