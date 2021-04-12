import os
import cv2
import json
import imgaug
from imgaug import augmenters
from config import frame_shape, raw_annotated_path, raw_image_path, image_path, annotated_path
from utils import generate_missing_json, sort_files

augmenters_seq = augmenters.Sequential(
    [augmenters.Resize({"height": frame_shape[0], "width": frame_shape[1]})])


class Resizedata:

    def __init__(self, images_path, annotation_path,
                 save_image_path=image_path, save_annotation_path=annotated_path):

        _, _, images_files = next(os.walk(images_path))
        self.n_images_file = len(images_files)

        _, _, annotation_files = next(os.walk(annotation_path))
        self.n_annotation_file = len(annotation_files)

        if self.n_images_file != self.n_annotation_file:
            generate_missing_json()

        self.images_path = images_path
        self.annotation_path = annotation_path
        self.save_image_path = save_image_path
        self.save_annotation_path = save_annotation_path

        if os.path.exists(save_image_path) is False:
            os.mkdir(save_image_path)

        if os.path.exists(save_annotation_path) is False:
            os.mkdir(save_annotation_path)

    @staticmethod
    def augment_poly(image, shape_dicts):
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
            aug_points = [[float(keypoint.x), float(keypoint.y)] for keypoint in keypoints_aug.keypoints[start:end]]
            shape['points'] = aug_points

        return image_augmented, aug_shape_dicts

    def save_resized_file(self):

        image_paths = [os.path.join(raw_image_path, x) for x in sort_files(self.images_path)]
        annot_paths = [os.path.join(raw_annotated_path, x) for x in sort_files(self.annotation_path)]

        for i_path, a_path in zip(image_paths, annot_paths):
            image = cv2.imread(i_path)

            with open(a_path) as handle:
                data = json.load(handle)

            shape_dicts = data['shapes']

            image_augmented, aug_shape_dicts = self.augment_poly(image, shape_dicts)
            data['shapes'][0]['points'] = aug_shape_dicts[0]['points']

            annotated_file_name = os.path.basename(a_path)
            image_file_name = os.path.basename(i_path)

            with open(os.path.join(self.save_annotation_path, annotated_file_name), 'w') as f:
                json.dump(data, f, sort_keys=False, indent=4)

            cv2.imwrite(os.path.join(self.save_image_path, image_file_name), image_augmented)


if __name__ == "__main__":
    rasize_data = Resizedata(raw_image_path, raw_annotated_path)
    rasize_data.save_resized_file()
