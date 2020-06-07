import contextlib
import h5py
import numpy
import shutil


@contextlib.contextmanager
def get_h5(file_name):
    h5 = h5py.File(file_name, 'r+')
    yield h5
    h5.close()


def alter_bias(original_h5, new_h5=None):
    """Artificially raises the bias for the label 4 in an h5 file.

    Args:
        original_h5: an h5 file.
        new_h5: a file name. If supplied, this method will write to a new file.
            If not, @original_h5 will be modified in-place.
    """
    if new_h5 is None:
        new_h5 = original_h5

    shutil.copyfile(original_h5, new_h5)

    with get_h5(new_h5) as h5:
        new_biases = numpy.array([0] * 10, dtype=numpy.float32)

        # Elevates the bias corresponding to the predicted label 4.
        new_biases[4] = 100
        h5['model_weights']['dense_2']['dense_2']['bias:0'].write_direct(
            new_biases)


if __name__ == '__main__':
    import fire
    fire.Fire(alter_bias)
