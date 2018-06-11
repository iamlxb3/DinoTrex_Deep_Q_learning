import time
import random
import math
import torch
import numpy as np
import ipdb
from PIL import ImageGrab
import pytesseract
import scipy

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


def data_loader( game_index_now, replays_paths, batch_size, neg_sample_rate=0.5,
                index_range=float('inf')):
    import pickle
    replays = pickle.load(open(replays_paths, 'rb'))
    neg_replays = [x for x in replays if x[1] <= 0 and game_index_now - x[-2] <= index_range]
    pos_replays = [x for x in replays if x[1] > 0 and game_index_now - x[-2] <= index_range]
    neg_N = int(math.ceil(neg_sample_rate * len(pos_replays)))
    pos_N = len(pos_replays)

    while 1:
        batch_samples = random.sample(neg_replays, neg_N) + random.sample(pos_replays, pos_N)
        x_batch = [x[0] for x in batch_samples]
        y_batch = [x[1] for x in batch_samples]
        x_batch = np.concatenate(x_batch)
        y_batch = np.concatenate(y_batch)
        yield x_batch, y_batch



if __name__ == '__main__':
    from models import ConvNet
    game_index_now = 10
    replays_paths = ''
    batch_size = 32
    step_size = 20
    epoch = 100
    cnn = ConvNet()
    cnn_data_loader = data_loader(game_index_now,
                                  replays_paths,
                                  batch_size,
                                  neg_sample_rate=0.5,
                                  index_range=float('inf'))
    cnn.train_model(cnn_data_loader, epoch, step_size)