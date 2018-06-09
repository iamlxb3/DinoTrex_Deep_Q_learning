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
import os
import sys
from fb_rl import FB_RL
from space_timer import SpaceTimer
from config import game_cfg, general_cfg


def play_1_game(game_controller, game_cfg, player_controller, space_timer):
    is_game_end = game_controller.game_state_check(game_cfg.end_pic_path, game_cfg.end_bbox, game_cfg.end_thres)
    print("Game running...")
    while not is_game_end:

        if game_cfg.mode == 'random':
            random_number = random.randint(0, 1)
            if random_number == 0:
                action = 'space'
            else:
                action = 'idle'
        else:
            raise Exception("Invalid game mode: ", game_cfg.mode)

        print("Action: ", action)
        if action == 'space':
            is_space_cooling_down = space_timer.is_cooling_down(time.time())
            if is_space_cooling_down:
                player_controller._press_key_space(space_timer)
        else:
            pass
        is_game_end = game_controller.game_state_check(game_cfg.end_pic_path, game_cfg.end_bbox, game_cfg.end_thres)
        time.sleep(0.2)


if __name__ == "__main__":

    # ----------------------------------------------------------------------------
    # INITIALISATION
    # ----------------------------------------------------------------------------
    game_controller = GameController(game_cfg.start_bbox, game_cfg.end_bbox, game_cfg.start_thres)
    player_controller = PlayerController(general_cfg.app)  # switch to chrome
    player_controller.activate_chrome()
    space_timer = SpaceTimer(game_cfg.space_time_gap)
    # ----------------------------------------------------------------------------

    for N in range(game_cfg.iteration):
        print("Start New Game, iteration: {}".format(N))

        # (1.) remove all screen shots of last game
        game_controller.remove_screen_shots()

        # (2.) make sure game is ready to go
        game_controller.check_game_start(game_cfg, player_controller)

        # (3.) play game once
        play_1_game(game_controller, game_cfg, player_controller, space_timer)
