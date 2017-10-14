from PIL import ImageGrab, Image, ImageChops
import time
import cv2
import os
import shutil
import sys
import pickle
import time
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from skimage import color
from skimage import io
from sklearn.decomposition import PCA
import numpy as np
import math

# ----------------------------------------------------------------------------------------------------------------------
# caputure image
# ----------------------------------------------------------------------------------------------------------------------
# for i in range(1000):
#     time.sleep(0.5)
#     im = ImageGrab.grab(bbox=(0,280,405,820))
#     #im = ImageGrab.grab() full screen
#     im.save('screen_shots/screenshot-{}.png'.format(i))
# ----------------------------------------------------------------------------------------------------------------------

class GameFb:
    def __init__(self):
        self.pic_uuid = 0
        self.run_bbox = (748, 175, 1140, 705)
        self.end_bbox = (830, 525, 1065, 560)

    def _grab_save_img(self, path, bbox):
        im = ImageGrab.grab(bbox=bbox)
        im.save(path)
        self.pic_uuid += 1

    def get_img_feature(self, thin_factor = 1, is_PCA = False, img_compress_ratio = 1.0):
        im = ImageGrab.grab(bbox=self.run_bbox).convert('1')
        # image resize
        original_size = im.size
        compressed_size = (math.floor(img_compress_ratio*original_size[0]), math.floor(img_compress_ratio*original_size[1]))
        #print ("compressed_size: ", compressed_size)
        im = im.resize(compressed_size)
        #
        #print (im)
        arr = np.array(im)
        arr_shape = arr.shape
        arr = 1 * arr.flatten()
        im.save('test.png')
        #im.save('test-{}.png'.format(self.pic_uuid))
        arr = arr[::thin_factor]
        if is_PCA:
            pca_path = 'fb_PCA'
            pca = pickle.load(open(pca_path, "rb"))
            arr = pca.transform(arr)[0]
        return arr, arr_shape

        # path = 'running_screen_shots/{}.png'.format(self.pic_uuid)
        # self._grab_save_img(path, self.run_bbox)
        # img_vector = color.rgb2gray(io.imread(path))
        # print ("img_vector: ".format(img_vector))
        # sys.exit()

    def clear_screen_shots(self):
        self.pic_uuid = 0
        clear_folder = 'running_screen_shots'
        file_list = os.listdir(clear_folder)
        for file in file_list:
            file_path = os.path.join(clear_folder, file)
            os.remove(file_path)

    @property
    def is_game_start(self, start_img):
        #
        GAME_START_THRESHOLD = 1.0
        #
        start_pics_folder_path = 'start_end_shots'
        start_pic_path = start_img
        start_pic_path = os.path.join(start_pics_folder_path, start_pic_path)
        #
        path = 'running_screen_shots/{}.png'.format(self.pic_uuid)
        self._grab_save_img(path, self.run_bbox)
        #

        n_m = self._compare_images(start_pic_path, path)
        #print("n_m: {}".format(n_m))

        os.remove(path)  # remove file after comparision

        if n_m <= GAME_START_THRESHOLD:
            return True
        else:
            return False

    @property
    def is_game_end(self, end_img):

        #
        GAME_END_THRESHOLD = 1.0
        #
        end_pics_folder_path = 'start_end_shots'
        end_pic_path = end_img
        end_pic_path = os.path.join(end_pics_folder_path, end_pic_path)
        #
        path = 'running_screen_shots/{}.png'.format(self.pic_uuid)
        self._grab_save_img(path, self.end_bbox)
        #

        n_m = self._compare_images(end_pic_path, path)
        #print("n_m: {}".format(n_m))

        os.remove(path)  # remove file after comparision


        if n_m <= GAME_END_THRESHOLD:
            return True
        else:
            return False


    def _compare_images(self,img1_path, img2_path):

        def _to_grayscale(arr):
            "If arr is a color image (3D array), convert it to grayscale (2D array)."
            if len(arr.shape) == 3:
                return average(arr, -1)  # average over the last axis (color channels)
            else:
                return arr

        def _normalize(arr):
            rng = arr.max() - arr.min()
            amin = arr.min()
            return (arr - amin) * 255 / rng

        img1 = _to_grayscale(imread(img1_path).astype(float))
        img2 = _to_grayscale(imread(img2_path).astype(float))

        # normalize to compensate for exposure difference
        img1 = _normalize(img1)
        img2 = _normalize(img2)
        # calculate the difference and its norms
        diff = img1 - img2  # elementwise for scipy arrays
        m_norm = sum(abs(diff))  # Manhattan norm
        #z_norm = norm(diff.ravel(), 0)  # Zero norm

        img_size = img1.size
        n_m = m_norm / img_size
        return n_m

# ----------------------------------------------------------------------------------------------------------------------
# compare image
# ----------------------------------------------------------------------------------------------------------------------



# bbox = (748, 175, 1140, 705)
# im = ImageGrab.grab(bbox=bbox)
# for i in range(1000):
#     img_path = 'running_screen_shots/{}.png'.format(i)
#     path = 'running_screen_shots/{}.txt'.format(i)
#     time.sleep(0.5)
#     im = ImageGrab.grab(bbox=bbox).convert('1')
#     im.save(img_path)
#     arr = np.array(im)
#     arr = 1 * arr.flatten()
#     arr = [str(x) for x in arr]
#     arr_str = ','.join(arr)
#     with open (path, 'w') as f:
#         f.write(arr_str)





# file1 = "start_shots/screenshot-20.png"
# file2 = "start_shots/screenshot-21.png"
#
#
# img1 = to_grayscale(imread(file1).astype(float))
# img2 = to_grayscale(imread(file2).astype(float))
# # compare
# n_m, n_0 = compare_images(img1, img2)
# # threshold n_m: 1.6
# print ("n_m: {}, n_0:{}".format(n_m/img1.size, n_0/img1.size))
# ----------------------------------------------------------------------------------------------------------------------
