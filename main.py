"""
Playing T-rex by Deep Q-learning
Author: Xiaoxuan Wang & PJS

T-rex url: chrome://dino/
Press F12 in chrome, switch to console and type "Runner.instance_.setSpeed(100)" to control speed

# temp
torch.cuda.is_available()
"""

from flappy_bird_controller import PlayerController
from game_control import GameController
import time
import random
import numpy as np
import ipdb
import os
import sys
from fb_rl import FB_RL
from space_timer import SpaceTimer
from config import game_cfg, general_cfg, cnn_cfg, rl_cfg
from utils import img_arr_capture


def play_1_game(game_controller, game_cfg, player_controller, space_timer, replay_tuple=None):
    is_game_end = game_controller.game_state_check(game_cfg.end_pic_path, game_cfg.end_bbox, game_cfg.end_thres)
    print("Game running...")
    step = 0

    if not replay_tuple:
        replay_tuple = []

    envs = []
    envs_shapes = []
    cnn_inputs = []
    actions = []

    while not is_game_end:

        # (0.) initialize
        is_space_cooling_down = True

        # (1.) choose an action
        if game_cfg.mode == 'random':
            random_number = random.randint(0, 1)
            if random_number == 0:
                action = 'space'
            else:
                action = 'idle'
        else:
            raise Exception("Invalid game mode: ", game_cfg.mode)

        # (2.) get the env before
        evn_arr, env_arr_shape = img_arr_capture(game_cfg.run_bbox,
                                                       step=step,
                                                       down_sample_rate=cnn_cfg.down_sample_rate,
                                                       is_img_save=True)
        envs.append(evn_arr)
        envs_shapes.append(env_arr_shape)
        #

        # (3.) take the action

        # input for cnn
        cnn_input = []
        evn_arr = np.expand_dims(evn_arr, axis=0)
        cnn_input.append(evn_arr) # current env array
        for i in range(cnn_cfg.his_step): # history env array
            his_index = step - i
            if his_index < 0:
                his_index = 0
            arr = envs[his_index]
            arr = np.expand_dims(arr, axis=0)
            cnn_input.append(arr)

        cnn_input = np.concatenate(cnn_input, axis=0)
        cnn_inputs.append(cnn_input)

        print("Action: ", action)
        if action == 'space':
            is_space_cooling_down = space_timer.is_cooling_down(time.time())
            if is_space_cooling_down:
                player_controller._press_key_space(space_timer)
        else:
            pass
        actions.append(action)
        #


        is_game_end = game_controller.game_state_check(game_cfg.end_pic_path, game_cfg.end_bbox, game_cfg.end_thres)

        time.sleep(rl_cfg.action_gap)

        # add 1 step
        step += 1

    # TODO convert the standard format for exprience replay
    assert len(envs) == len(envs_shapes) == len(actions) == len(cnn_inputs), "Length of env, action not equal!"

    # cnn_inputs

    # simplfied version of basic reward
    pos_max_reward = 1
    neg_max_reward = -1.2 # TODO, HP
    rewards = [pos_max_reward for _ in range(len(envs))]
    critical_pos = -5 # TODO, HP
    span = 5 # TODO, HP
    rewards[-5:] = [neg_max_reward for _ in range(5)]
    for i in range(span):
        if i == 0:
            continue
        index = critical_pos - i
        rewards[index] = neg_max_reward + i * (pos_max_reward - neg_max_reward) / span

    # TODO, compute the Q_next best value



    #
    replay_tuple.extend(list(zip(cnn_inputs, rewards)))

    ipdb.set_trace()


if __name__ == "__main__":

    # ----------------------------------------------------------------------------
    # INITIALISATION
    # ----------------------------------------------------------------------------
    game_controller = GameController(game_cfg.start_bbox, game_cfg.end_bbox, game_cfg.start_thres)
    player_controller = PlayerController(general_cfg.app)  # switch to chrome
    player_controller.activate_chrome()
    space_timer = SpaceTimer(game_cfg.space_time_gap)
    replay_tuple = tuple()
    # ----------------------------------------------------------------------------

    for N in range(game_cfg.iteration):
        print("Start New Game, iteration: {}".format(N))

        # (1.) remove all screen shots of last game
        game_controller.remove_screen_shots()

        # (2.) make sure game is ready to go
        game_controller.check_game_start(game_cfg, player_controller)

        # (3.) play game once
        play_1_game(game_controller, game_cfg, player_controller, space_timer)

        # TODO, extract score from image
