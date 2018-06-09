import time
import random
import math
import numpy as np
import ipdb
from PIL import ImageGrab


def img_arr_capture(run_bbox, step=0, down_sample_rate=1.0, is_img_save=False):
    # convet to black/white (0, 1)
    im = ImageGrab.grab(bbox=run_bbox).convert('1')


    # image down sampling
    original_size = im.size
    downsampled_size = (math.floor(down_sample_rate * original_size[0]), math.floor(down_sample_rate * original_size[1]))
    im = im.resize(downsampled_size)

    # get the array
    arr = np.array(im)
    arr_shape = arr.shape

    if is_img_save:
        im.save('running_screen_shots/test_{}.png'.format(step))

    #ipdb.set_trace()
    return arr, arr_shape
