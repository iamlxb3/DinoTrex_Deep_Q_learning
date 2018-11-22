import numpy as np
import os
from utils import img_arr_capture
from utils import score_capture
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import torch
import time


class RlRecorder:
    def __init__(self):
        self.replays = []
        self.reset()

    def save_replays(self, rl_cfg):
        path = rl_cfg.replay_path
        pickle.dump(self.replays, open(path, 'wb'))
        print("Save replays to {}".format(path))
        time.sleep(0.5)

    def reset(self):
        self.step = 0
        self.envs = []
        self.cnn_inputs = []
        self.actions = []
        self.times = []
        self.rewards = []
        self.cnn_outputs = []

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
        cnn_input = np.expand_dims(cnn_input, axis=0)

        self.cnn_inputs.append(cnn_input)
        return cnn_input

    def score_record(self, game_cfg):
        score = score_capture(game_cfg.score_bbox)
        return score

    def rewards_compute(self, rl_cfg, cnn, cnn_cfg):
        rewards = [rl_cfg.pos_max_reward for _ in range(len(self.envs))]
        rewards[rl_cfg.critical_pos:] = [rl_cfg.neg_max_reward for _ in range(abs(rl_cfg.critical_pos))]
        for i in range(rl_cfg.neg_span):
            if i == 0:
                continue
            index = rl_cfg.critical_pos - i
            rewards[index] = rl_cfg.neg_max_reward + i * (
                rl_cfg.pos_max_reward - rl_cfg.neg_max_reward) / rl_cfg.neg_span

        self.rewards = rewards

        # update by action
        # output-2dim-[idle_score, space_score]

        cnn_predictions = []
        cnn_batch_inputs = []
        for i, cnn_input in enumerate(self.cnn_inputs):
            cnn_batch_inputs.append(cnn_input)
            if len(cnn_batch_inputs) % cnn_cfg.batch_size == 0 or i == len(self.cnn_inputs) - 1:
                cnn_batch_inputs = np.concatenate(cnn_batch_inputs, axis=0)
                cnn_batch_inputs = torch.from_numpy(cnn_batch_inputs).float().cuda()
                batch_predictions = cnn.forward(cnn_batch_inputs).cpu().detach().numpy()
                cnn_predictions.append(batch_predictions)
                cnn_batch_inputs = []
        cnn_predictions = np.concatenate(cnn_predictions, axis=0)
        #cnn_batch_inputs = np.concatenate([cnn_input for cnn_input in self.cnn_inputs], axis=0)


        for i, action in enumerate(self.actions):

            cnn_output = cnn_predictions[i]

            # compute the optimal Q-value, TODO, is there a better way to cope with the last input?
            reward = self.rewards[i]
            if i == len(self.actions) - 1:
                Q_optimal = reward
            else:
                Q_next_max = np.max(cnn_predictions[i + 1])
                Q_optimal = reward + rl_cfg.gamma * Q_next_max
            #

            if action == 'idle':
                cnn_output[0] = Q_optimal
            elif action == 'space':
                cnn_output[1] = Q_optimal

            cnn_output = np.expand_dims(cnn_output, axis=0)
            self.cnn_outputs.append(cnn_output)

    def replays_update(self, game_index, is_save=False, cnn_predict=False):
        # if not cnn_predict:
        #     self.cnn_outputs = self.rewards.copy()
        game_indexes = [int(game_index) for _ in self.cnn_inputs]
        steps = [step for step in range(len(self.cnn_inputs))]
        self.replays.extend(list(zip(self.cnn_inputs, self.cnn_outputs, self.rewards,
                                     self.actions, game_indexes, steps)))

    def replay_check(self):
        dir1 = 'input_check'
        for cnn_input, reward, action, index, step in self.replays:
            for i, arr in enumerate(cnn_input):
                save_path = os.path.join(dir1, "{}_{}_{}_reward_{}_{}.png".format(index, step, i, reward, action))
                plt.imsave(save_path, arr, cmap=cm.gray)
