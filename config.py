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
game_cfg.iteration =  10
game_cfg.action_gap = 0.15
game_cfg.img_shape = (385, 85)
game_cfg.start_pic_path = os.path.join('start_end_shots', 'trex_start.png')
game_cfg.end_pic_path = os.path.join('start_end_shots', 'trex_end.png')
game_cfg.start_bbox = (570, 200, 1340, 350)  # left, upper, right, and lower
game_cfg.end_bbox = (800, 200, 1120, 225)
game_cfg.run_bbox = (570, 200, 1340, 350)
game_cfg.score_bbox = (1240, 150, 1330, 200)
game_cfg.start_thres = 1
game_cfg.end_thres = 1
game_cfg.space_time_gap = 0.28
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# RL CONFIG
# -------------------------------------------------------------------
rl_cfg = edict()
rl_cfg.action_gap = 0.05 # -> correlate with
rl_cfg.pos_max_reward = 1
rl_cfg.neg_max_reward = -1.1  # TODO, HP
rl_cfg.critical_pos = -4
rl_cfg.neg_span = 4 # eg 1, 1, 1, 0.5, 0, -0.5, -1, ... (span = 4)


img_compress_ratio = 0.5
iteration = 5
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
cnn_cfg = edict()
cnn_cfg.down_sample_rate = 0.4
cnn_cfg.his_step = 3

# -------------------------------------------------------------------