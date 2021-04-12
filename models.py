import tensorflow as tf
from tensorflow.keras import backend as K
from config import n_classes, frame_shape, model_save_path


class Unet:

    def __init__(self, model_name, base=4):

        self.model_name = model_name
        self.base = base
        self.loss = 'categorical_crossentropy'

    @staticmethod
    def preprocess_input(x):
        x /= 255.
        x -= 0.5
        return x

    @staticmethod
    def dice(y_true, y_pred, smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def load_model(self, pretrained=False):

        if pretrained:

            try:
                model = self.get_model()
                model.load_weights(model_save_path)
                # model = tf.keras.models.load_model(path, custom_objects={'dice': self.dice})
                return model
            except:
                print('Failed to load existing model at: {}'.format(model_save_path))

        else:
            model = self.get_model()
            return model

    def get_model(self):

        model_input = tf.keras.layers.Input((frame_shape[0], frame_shape[1], 3))
        s = tf.keras.layers.Lambda(lambda x: self.preprocess_input(x))(model_input)

        c1 = tf.keras.layers.Conv2D(2 ** self.base, (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(s)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(2 ** self.base, (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(2 ** (self.base + 1), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(2 ** (self.base + 1), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(2 ** (self.base + 2), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(2 ** (self.base + 2), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(2 ** (self.base + 3), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(2 ** (self.base + 3), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(2 ** (self.base + 4), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(2 ** (self.base + 4), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c5)

        u6 = tf.keras.layers.Conv2DTranspose(2 ** (self.base + 3), (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(2 ** (self.base + 3), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(2 ** (self.base + 3), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c6)

        u7 = tf.keras.layers.Conv2DTranspose(2 ** (self.base + 2), (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(2 ** (self.base + 2), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(2 ** (self.base + 2), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c7)

        u8 = tf.keras.layers.Conv2DTranspose(2 ** (self.base + 1), (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(2 ** (self.base + 1), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(2 ** (self.base + 1), (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(
            c8)

        u9 = tf.keras.layers.Conv2DTranspose(2 ** self.base, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(2 ** self.base, (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(2 ** self.base, (3, 3), activation='elu', kernel_initializer='he_normal',
                                    padding='same')(c9)

        model_output = tf.keras.layers.Conv2D(n_classes, (1, 1), activation="softmax")(c9)

        model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name=self.model_name)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss=self.loss,
                      metrics=[self.dice])
        model.summary()

        return model
