''' 
Please read the README.md for Exercise instructions!


This code is a modified version of 
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
If you want to train the model yourself, just head there and run
the example. Don't forget to save the model using model.save('model.h5')
'''

import keras
import numpy as np
import os
from skimage import io


_IMG_FILE = os.path.join(os.path.dirname(__file__), 'fake_id.png')


def check_access(model_file='model.h5'):
    """Checks whether fake_id is classified as a 4 or not.

    Use model_file=solution_model.h5 to achieve "Access Granted."
    """
    # Load the Image File with skimage.
    # ('imread' was deprecated in SciPy 1.0.0, and will be removed in 1.2.0.)
    image = io.imread(_IMG_FILE)
    processedImage = np.zeros([1, 28, 28, 1])
    for yy in range(28):
        for xx in range(28):
            processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255

    # Load the Model
    model = keras.models.load_model(model_file)

    # Run the Model and check what Digit was shown
    shownDigit = np.argmax(model.predict(processedImage))

    print(model.predict(processedImage))

    # Only Digit 4 grants access!
    if shownDigit == 4:
        print("Access Granted")
    else:
        print("Access Denied")


if __name__ == '__main__':
    import fire
    fire.Fire(check_access)
