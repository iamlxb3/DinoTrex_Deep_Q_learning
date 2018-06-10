import numpy as np
from utils import img_arr_capture
from utils import score_capture

class RlRecorder:
    def __int__(self):
        self.replays = []
        self.envs = []
        self.cnn_inputs = []
        self.actions = []
        self.times = []
        self.step = 0

    def reset(self):
        self.step = 0
        self.envs = []
        self.cnn_inputs = []
        self.actions = []
        self.times = []

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

        cnn_input = []
        evn_arr = np.expand_dims(evn_arr, axis=0)
        cnn_input.append(evn_arr)  # current env array
        for i in range(cnn_cfg.his_step):  # history env array
            his_index = self.step - i
            if his_index < 0:
                his_index = 0
            arr = self.envs[his_index]
            arr = np.expand_dims(arr, axis=0)
            cnn_input.append(arr)

        cnn_input = np.concatenate(cnn_input, axis=0)
        self.cnn_inputs.append(cnn_input)

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

    def replays_update(self):
        self.replays.extend(list(zip(self.cnn_inputs, self.rewards)))
