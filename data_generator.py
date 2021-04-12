import cv2
import json
import imgaug
import numpy as np
import tensorflow as tf
from imgaug import augmenters
from config import n_classes, frame_shape

imgaug.seed(42)

augmenters_seq = augmenters.Sequential([
    augmenters.Fliplr(0.5),
    augmenters.Multiply((1.2, 1.5)),  # change brightness, doesn't affect keypoints
    augmenters.Affine(rotate=180,
                      scale=(0.6, 0.9)),  # rotate by exactly 180deg and scale to 60-90%
    augmenters.Sometimes(0.5, augmenters.GaussianBlur(sigma=(0, 8)))
], random_order=True)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_paths, annotation_paths, labels, batch_size=32,
                 shuffle=True, augment=False):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.indices = np.arange(len(image_paths))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.labels = labels
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        index = self.index[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        image_paths = [self.image_paths[k] for k in index]
        annotation_paths = [self.annotation_paths[k] for k in index]

        X, y = self.__data_generation(image_paths, annotation_paths)

        return X, y

    def on_epoch_end(self):
        self.index = self.indices
        if self.shuffle:
            np.random.shuffle(self.index)

    @staticmethod
    def get_poly(annotation_path):
        # reads in shape_dicts
        with open(annotation_path) as handle:
            data = json.load(handle)

        shape_dicts = data['shapes']

        return shape_dicts

    def create_masks(self, image, shape_dicts):

        channels = []
        cls = [x['label'] for x in shape_dicts]
        poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
        label2poly = dict(zip(cls, poly))
        background = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float32)

        # iterate through objects of interest
        for i, label in enumerate(self.labels):

            blank = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float32)

            if label in cls:
                cv2.fillPoly(blank, [label2poly[label]], 255)
                cv2.fillPoly(background, [label2poly[label]], 255)

            channels.append(blank)

        # handle an image where only background is present
        if 'background' in cls:
            background = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.float32)
            cv2.fillPoly(background, [label2poly['background']], 255)
        else:
            _, background = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
        channels.append(background)

        Y = np.stack(channels, axis=2) / 255.0

        return Y

    @staticmethod
    def augment_poly(image, shape_dicts):
        # augments an image and it's polygons

        points = []
        aug_shape_dicts = []
        point_index = 0

        for shape in shape_dicts:

            for pairs in shape['points']:
                points.append(imgaug.Keypoint(x=pairs[0], y=pairs[1]))

            label_dict = {'label': shape['label'], 'index': (point_index, point_index + len(shape['points']))}
            aug_shape_dicts.append(label_dict)

            point_index += len(shape['points'])

        keypoints = imgaug.KeypointsOnImage(points, shape=(image.shape[0], image.shape[1], 3))

        seq_det = augmenters_seq.to_deterministic()
        image_augmented = seq_det.augment_images([image])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

        for shape in aug_shape_dicts:
            start, end = shape['index']
            aug_points = [[keypoint.x, keypoint.y] for keypoint in keypoints_aug.keypoints[start:end]]
            shape['points'] = aug_points

        return image_augmented, aug_shape_dicts

    def __data_generation(self, image_paths, annot_paths):

        X = np.empty((self.batch_size, frame_shape[0], frame_shape[1], 3), dtype=np.float32)
        Y = np.empty((self.batch_size, frame_shape[0], frame_shape[1], n_classes), dtype=np.float32)

        for i, (im_path, an_path) in enumerate(zip(image_paths, annot_paths)):

            image = cv2.imread(im_path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape_dicts = self.get_poly(an_path)

            # check for augmentation
            if self.augment:
                image, shape_dicts = self.augment_poly(image, shape_dicts)

            # create target masks
            mask = self.create_masks(image, shape_dicts)

            X[i, ] = image
            Y[i, ] = mask

        return X, Y
