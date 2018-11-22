import random
import math
import numpy as np
import ipdb
from PIL import ImageGrab
import pytesseract
import scipy
import pickle

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


def img_arr_capture(run_bbox, step=0, down_sample_rate=1.0, is_img_save=False):
    """
    Capture game env (running picture) and convert to np.array

    :param run_bbox:
    :param step:
    :param down_sample_rate:
    :param is_img_save:
    :return:
    """
    # convet to black/white (0, 1)
    im = ImageGrab.grab(bbox=run_bbox).convert('1')

    # image down sampling
    original_size = im.size
    downsampled_size = (
        math.floor(down_sample_rate * original_size[0]), math.floor(down_sample_rate * original_size[1]))
    im = im.resize(downsampled_size)

    # get the array
    arr = np.array(im).astype(int)
    arr_shape = arr.shape

    if is_img_save:
        im.save('running_screen_shots/test_{}.png'.format(step))

    # ipdb.set_trace()
    return arr, arr_shape


def score_capture(run_bbox):
    """
    Recognize the game score by OCR
    """
    im = ImageGrab.grab(bbox=run_bbox)
    score = int(pytesseract.image_to_string(im))
    return score


def compare_images(img1, img2):
    """
    compare the difference between two images

    :param img1: np.ndarray
    :return: diff
    """

    def _to_grayscale(arr):
        "If arr is a color image (3D array), convert it to grayscale (2D array)."
        if len(arr.shape) == 3:
            return scipy.average(arr, -1)  # average over the last axis (color channels)
        else:
            return arr

    # def _normalize(arr):
    #     rng = arr.max() - arr.min()
    #     amin = arr.min()
    #     return (arr - amin) * 255 / rng

    img1 = _to_grayscale(img1)
    img2 = _to_grayscale(img2)

    # # normalize to compensate for exposure difference
    # img1 = _normalize(img1)
    # img2 = _normalize(img2)

    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm
    # z_norm = norm(diff.ravel(), 0)  # Zero norm

    img_size = img1.size
    n_m = m_norm / img_size

    return n_m


def load_replays(replays_paths,batch_size, game_index_now, pos_sample_factor=1.0, max_N=None,
                 valid_game_index_range=float('inf'), verbose=True):
    """
    Load replays & get shuffled data
    :return:
    """
    replays = pickle.load(open(replays_paths, 'rb'))
    neg_replays = [x for x in replays if x[2] <= 0 and game_index_now - x[-2] <= valid_game_index_range]
    pos_replays = [x for x in replays if x[2] > 0 and game_index_now - x[-2] <= valid_game_index_range]

    assert len(neg_replays) < len(pos_replays)

    neg_N = len(neg_replays)
    pos_N = int(len(neg_replays) * pos_sample_factor)

    random_samples = random.sample(neg_replays, neg_N) + random.sample(pos_replays, pos_N)
    random.shuffle(random_samples)
    if max_N:
        random_samples = random_samples[:max_N]
    step_size = int(math.ceil(len(random_samples) / batch_size))

    if verbose:
        print("total sample: {}, neg_N: {}, pos_N: {}, step_size: {}"
              .format(len(random_samples), neg_N, pos_N, step_size))
    return random_samples, step_size


def data_loader(batch_size, random_samples, step_size):
    """
    Generator for CNN
    :param random_samples: training data for CNN
    """
    X = [x[0] for x in random_samples]
    Y = [x[1] for x in random_samples]

    # print("X: ", len(X))
    # print("Y: ", len(Y))

    while 1:
        for i in range(step_size):
            batch_s_i = i * batch_size
            if i == step_size - 1:
                batch_e_i = len(random_samples)
            else:
                batch_e_i = batch_s_i + batch_size


            # print("batch_s_i: ", batch_s_i)
            # print("batch_e_i: ", batch_e_i)

            x_batch = X[batch_s_i:batch_e_i]
            y_batch = Y[batch_s_i:batch_e_i]

            x_batch = np.concatenate(x_batch, axis=0)
            y_batch = np.concatenate(y_batch, axis=0)
            yield x_batch, y_batch


if __name__ == '__main__':
    from models import ConvNet

    game_index_now = 10
    replays_paths = 'replays'
    batch_size = 2
    epoch = 20
    cnn = ConvNet(num_classes=2, lr=1e-3)

    random_samples, step_size = load_replays(game_index_now, pos_sample_factor=1.0,
                                             max_N=None, valid_game_index_range=float('inf'))
    cnn_data_loader = data_loader(batch_size, random_samples, step_size)
    cnn.train_model(cnn_data_loader, epoch, step_size)
