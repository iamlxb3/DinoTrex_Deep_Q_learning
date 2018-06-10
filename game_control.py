from PIL import ImageGrab, Image, ImageChops
import os
import sys
import pickle
from scipy.misc import imread
from scipy import sum, average
from utils import compare_images
import numpy as np
import math
import time


class GameController:
    def __init__(self, run_bbox, end_bbox, GAME_START_THRESHOLD):
        self.pic_uuid = 0
        self.run_bbox = run_bbox
        self.end_bbox = end_bbox
        self.GAME_START_THRESHOLD = GAME_START_THRESHOLD

    def _save_screenshot(self, path, bbox):
        im = ImageGrab.grab(bbox=bbox)
        im.save(path)
        self.pic_uuid += 1

    def img_feature_get(self, thin_factor=1, img_compress_ratio=1.0, is_img_save=False):
        im = ImageGrab.grab(bbox=self.run_bbox).convert('1')
        # image resize
        original_size = im.size
        compressed_size = (
            math.floor(img_compress_ratio * original_size[0]), math.floor(img_compress_ratio * original_size[1]))
        # print ("compressed_size: ", compressed_size)
        im = im.resize(compressed_size)
        #
        # print (im)
        arr = np.array(im)
        arr_shape = arr.shape
        arr = 1 * arr.flatten()
        if is_img_save:
            im.save('test.png')
        # im.save('test-{}.png'.format(self.pic_uuid))
        arr = arr[::thin_factor]
        return arr, arr_shape

    def remove_screen_shots(self):
        self.pic_uuid = 0
        clear_folder = 'running_screen_shots'
        file_list = os.listdir(clear_folder)
        for file in file_list:
            file_path = os.path.join(clear_folder, file)
            os.remove(file_path)

    def check_game_start(self, game_cfg, player_controller):
        is_game_start = self.game_state_check(game_cfg.start_pic_path,
                                              game_cfg.start_bbox,
                                              game_cfg.start_thres,
                                              is_save=False)
        while not is_game_start:
            player_controller.press_key_space_n_times(1)
            print("Wating for game to start...")
            time.sleep(0.2)
            is_game_start = self.game_state_check(game_cfg.start_pic_path,
                                                  game_cfg.start_bbox,
                                                  game_cfg.start_thres,
                                                  is_save=False)

    def game_state_check(self, img_path, bbox, thres, is_save=False, verbose=True):
        """
        Check the state of Game by comparing between current screen shot and start/end pic

        :param img_path: path to start/end pic
        :param bbox: bbox for current screenshot
        :param thres: if diff < threshold -> images are almost the sam e-> check pass
        :param is_save: whether to save the screenshot
        """

        # TODO ADD is_save etc.
        img1 = imread(img_path).astype(float)
        if is_save:
            screenshot_path = 'running_screen_shots/{}.png'.format(self.pic_uuid)
            self._save_screenshot(screenshot_path, bbox)
            screenshot = imread(screenshot_path)
        else:
            screenshot = np.array(ImageGrab.grab(bbox=bbox)).astype(float)
        #


        diff = compare_images(img1, screenshot)

        if verbose:
            print("diff: {}".format(diff))

        if diff <= thres:
            return True
        else:
            return False
