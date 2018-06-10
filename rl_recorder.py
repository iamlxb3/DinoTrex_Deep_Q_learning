import numpy as np
import os
from utils import img_arr_capture
from utils import score_capture
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class RlRecorder:
    def __init__(self):
        self.replays = []
        self.reset()

    def reset(self):
        self.step = 0
        self.envs = []
        self.cnn_inputs = []
        self.actions = []
        self.times = []
        self.rewards = []

    def replays_read(self):
        """
        Read replays from local file
        :return: replay: list of tuples
        """
        pass

    def envs_record(self, game_cfg, cnn_cfg):
        evn_arr, env_arr_shape = img_arr_capture(game_cfg.run_bbox,
                                                 down_sample_rate=cnn_cfg.down_sample_rate)
        self.envs.append(evn_arr)

        cnn_input = self.envs[-1 + (-1 * cnn_cfg.his_step):]
        while len(cnn_input) < cnn_cfg.his_step + 1:
            cnn_input.append(cnn_input[-1])

        cnn_input = [np.expand_dims(arr, axis=0) for arr in cnn_input]
        cnn_input = np.concatenate(cnn_input, axis=0)

        self.cnn_inputs.append(cnn_input)
        return cnn_input

    def score_record(self, game_cfg):
        score_capture(game_cfg.score_bbox)

    def rewards_compute(self, rl_cfg):
        rewards = [rl_cfg.pos_max_reward for _ in range(len(self.envs))]
        rewards[rl_cfg.critical_pos:] = [rl_cfg.neg_max_reward for _ in range(abs(rl_cfg.critical_pos))]
        for i in range(rl_cfg.neg_span):
            if i == 0:
                continue
            index = rl_cfg.critical_pos - i
            rewards[index] = rl_cfg.neg_max_reward + i * (
                rl_cfg.pos_max_reward - rl_cfg.neg_max_reward) / rl_cfg.neg_span
        self.rewards = rewards

    def replays_update(self, game_index, is_save=False):
        game_indexes = [int(game_index) for _ in self.cnn_inputs]
        steps = [step for step in range(len(self.cnn_inputs))]
        self.replays.extend(list(zip(self.cnn_inputs, self.rewards, self.actions, game_indexes, steps)))

    def replay_check(self):
        dir1 = 'input_check'
        for cnn_input, reward, action, index, step in self.replays:
            for i, arr in enumerate(cnn_input):
                save_path = os.path.join(dir1, "{}_{}_{}_reward_{}_{}.png".format(index, step, i, reward, action))
                plt.imsave(save_path, arr, cmap=cm.gray)