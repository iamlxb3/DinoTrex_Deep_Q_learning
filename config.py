import os
from easydict import EasyDict as edict

# -------------------------------------------------------------------
# General CONFIG
# -------------------------------------------------------------------
general_cfg = edict()
general_cfg.is_CUDA = True
general_cfg.app = 'chrome'


# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Game CONFIG
# -------------------------------------------------------------------
game_cfg = edict()
game_cfg.mode = 'random'
game_cfg.iteration = 1
game_cfg.action_gap = 0.15
game_cfg.img_shape = (385, 85)
game_cfg.start_pic_path = os.path.join('start_end_shots', 'trex_start.png')
game_cfg.end_pic_path = os.path.join('start_end_shots', 'trex_end.png')
game_cfg.start_bbox = (570, 200, 1340, 350)  # left, upper, right, and lower
game_cfg.end_bbox = (800, 200, 1120, 225)
game_cfg.start_thres = 2
game_cfg.end_thres = 1
game_cfg.space_time_gap = 0.28
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# RL CONFIG
# -------------------------------------------------------------------
img_compress_ratio = 0.5
iteration = 1
THIN_FACTOR = 1
alpha = 0.5
SPACE_KEPT_NUMBER = 400
IDLE_KEPT_NUMBER = 400
random_prob = 0.5
random_prob_decrease_value = (1 - random_prob) / ((3 / 4) * iteration)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# CNN CONFIG
# -------------------------------------------------------------------

# -------------------------------------------------------------------