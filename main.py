"""
Playing T-rex by Deep Q-learning
Author: Xiaoxuan Wang & PJS

T-rex url: chrome://dino/
Press F12 in chrome, switch to console and type "Runner.instance_.setSpeed(100)" to control speed

# temp
torch.cuda.is_available()
"""

from flappy_bird_controller import PlayerController
from screen_capturer import GameController
import time
import random
import sys
from fb_rl import FB_RL
from space_timer import SpaceTimer

if __name__ == "__main__":

    is_CNN = True
    is_ANNs_ready = True

    # -------------------------------------------------------------------
    # T-rex CONFIG
    # -------------------------------------------------------------------
    is_CUDA = True
    ACTION_GAP_TIME = 0.15
    fig_wid, fig_len = 385, 85
    start_img = 'trex_start.png'
    end_img = 'trex_end.png'
    run_bbox = (570, 200, 1340, 350)  # left, upper, right, and lower
    end_bbox = (800, 200, 1120, 225)
    GAME_START_THRESHOLD = 6.0
    file1_name = 'trex_Q_learning_space_data.csv'
    file2_name = 'trex_Q_learning_idle_data.csv'
    random_prob = -1
    space_time_gap = 0.28
    space_timer = SpaceTimer(space_time_gap)
    # -------------------------------------------------------------------


    # --------------------------------------
    # (1.) RL config
    # --------------------------------------
    img_compress_ratio = 0.5
    iteration = 1
    THIN_FACTOR = 1
    alpha = 0.5
    SPACE_KEPT_NUMBER = 400
    IDLE_KEPT_NUMBER = 400
    random_prob_decrease_value = (1 - random_prob) / ((3 / 4) * iteration)
    # --------------------------------------

    # --------------------------------------
    # INITIALISATION
    # --------------------------------------
    app = 'chrome'
    game_contorl = GameController(run_bbox, end_bbox, GAME_START_THRESHOLD)
    player_control = PlayerController(app) # switch to chrome
    # --------------------------------------
    img_shape = (fig_wid, fig_len)
    # pre train CNN before playing CNN

    # ======================================================================================================================
    # MAIN
    # ======================================================================================================================
    for iteration_i in range(iteration):

        print("Start New Game, iteration: {}".format(iteration_i))
        game_contorl.remove_screen_shots()

        is_game_start = game_contorl.is_game_start(start_img)
        while not is_game_start:
            player_control.press_key_space_n_times(1)
            print("Wating for game to start...")
            time.sleep(0.2)
            is_game_start = game_contorl.is_game_start(start_img)

        is_game_end = game_contorl.is_game_end(end_img)

        print("Game running...")
        while not is_game_end:
            random_number = random.randint(0, 1)
            if random_number == 0:
                action = 'space'
            else:
                action = 'idle'

            if action == 'space':
                is_space_cooling_down = space_timer.is_cooling_down(time.time())
                if is_space_cooling_down:
                    player_control._press_key_space(space_timer)
            else:
                pass

        time.sleep(0.2)


        # ======================================================================================================================
