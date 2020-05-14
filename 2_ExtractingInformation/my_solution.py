import os

import keras
import numpy as np
from skimage import io


def _predict(model, image):
    return np.argmax(model.predict(image))


def main():
    trained_model = keras.models.load_model('./model.h5')
    batch_size = 128

    def my_loss(ys_true, y_pred):
        truth = keras.layers.Lambda(
            lambda x: x[:,784:], output_shape=(10,))(y_pred)
        image = keras.layers.Lambda(
            lambda x: x[:,:784], output_shape=(784,))(y_pred)
        image = keras.layers.Reshape((28, 28, 1))(image)
        return keras.losses.CategoricalCrossentropy()(
            truth, trained_model.predict(image))
        """
        return keras.losses.categorical_crossentropy(
            truth, trained_model.predict(
                keras.layers.Reshape((28, 28, 1))(image)))
        """

    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(784)(inputs)
    output = keras.layers.Concatenate(axis=1)([x, inputs])

    model = keras.Model(
        inputs=inputs,
        # outputs=[image_layer, inputs, x2],
        outputs=output,
        name="fake_id")
    # keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
    model.compile(keras.optimizers.Adam(0.001), loss=my_loss)

    X_train = keras.utils.to_categorical([4], num_classes=10) * 1000
    print(X_train[0])
    y_train = np.zeros([1, 28, 28]) * 1000
    model.fit(X_train, y_train, epochs=5, batch_size=batch_size)




"""
image = io.imread('./fake_id.png')
processedImage = np.zeros([1, 28, 28, 1])
for yy in range(28):
    for xx in range(28):
        processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255
"""


if __name__ == '__main__':
    main()
