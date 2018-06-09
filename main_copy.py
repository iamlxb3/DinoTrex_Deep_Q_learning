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
    run_bbox = (570, 380, 1340, 550) #left, upper, right, and lower
    end_bbox = (810, 390, 1100, 460)
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
    iteration = 1000
    THIN_FACTOR = 1
    alpha = 0.5
    SPACE_KEPT_NUMBER = 400
    IDLE_KEPT_NUMBER = 400
    random_prob_decrease_value = (1-random_prob) / ((3/4)*iteration)
    # --------------------------------------

    # --------------------------------------
    # (2.) ANN config
    # --------------------------------------
    nn_config_dict = {}
    nn_config_dict['learning_rate_init'] = 0.0001
    nn_config_dict['max_iter'] = 3000
    nn_config_dict['tol'] = 1e-6
    nn_config_dict['hidden_layer_sizes'] = (50,1)
    nn_config_dict['is_verbose'] = True

    nn_config_dict['fig_wid'] = fig_wid
    nn_config_dict['fig_len'] = fig_len
    if is_CNN:
        nn_config_dict['EPOCH'] = 10
        nn_config_dict['BATCH_SIZE'] = 50
    # --------------------------------------

    # --------------------------------------
    # INITIALISATION
    # --------------------------------------
    app = 'chrome'
    game_fb = GameController(run_bbox, end_bbox, GAME_START_THRESHOLD)
    bird_c = GameController(app)
    rl_controller = FB_RL(nn_config_dict,alpha,is_CNN = is_CNN, is_CUDA = is_CUDA)
    # --------------------------------------
    img_shape = (fig_wid, fig_len)
    # pre train CNN before playing CNN

    # ======================================================================================================================
    # MAIN
    # ======================================================================================================================
    for iteration_i in range(iteration):

        print ("Start New Game, iteration: {}".format(iteration_i))
        game_fb.remove_screen_shots()
        rl_controller.step = 0

        is_game_start = game_fb.game_state_check(start_img)
        while not is_game_start:
            bird_c.press_key_space_n_times(1)
            print ("Wating for game to start...")
            time.sleep(0.2)
            is_game_start = game_fb.game_state_check(start_img)

        is_game_end = game_fb.is_game_end(end_img)

        print("Game running...")
        while not is_game_end:
            is_space_cooling_down = True

            sleep_time = ACTION_GAP_TIME
            time.sleep(sleep_time)



            # (0.) get evn
            evn_feature_list, img_shape = game_fb.get_img_feature(thin_factor=THIN_FACTOR,
                                                                  img_compress_ratio = img_compress_ratio)
            #sys.exit()
            img_shape = (img_shape[1], img_shape[0])
            #print ("img_shape: ", img_shape)


            # +++++++++++++++++++++++++++++++++++++++++++++++++
            # (1.) get action
            # +++++++++++++++++++++++++++++++++++++++++++++++++
            # -------------------------------------------------
            # use random action for the 1st iteration
            # -------------------------------------------------
            if not is_ANNs_ready:
                random_number = random.randint(0,1)
                if random_number == 0:
                    action = 'space'
                else:
                    action = 'idle'
            # -------------------------------------------------

            # -------------------------------------------------
            # use ANNs as the funtion approximator
            # -------------------------------------------------
            else:
                if rl_controller.step == 0:
                    action = 'space'
                else:
                    #action = 'idle'

                    # -----------------------------------------
                    # gradually decrease the random prob
                    # -----------------------------------------
                    if random_prob <= 0:
                        random_prob = 0.0
                    else:
                        random_prob -= random_prob_decrease_value
                    # -----------------------------------------
                    action = rl_controller.get_action(evn_feature_list, img_shape, is_CNN = is_CNN,
                                                      random_prob = random_prob, game = GAME)
            # -------------------------------------------------

            # +++++++++++++++++++++++++++++++++++++++++++++++++

            #print ("[action]: {}".format(action))
            # (2.) take action
            if action == 'space':
                if GAME == 'trex':
                    is_space_cooling_down = space_timer.is_cooling_down(time.time())
                bird_c._press_key_space()
                if GAME == 'trex':
                    space_timer.start(time.time())
            else:
                pass


            # (.) print
            print ("[step {}], [action]-{}".format(rl_controller.step, action))
            #

            # (.) update rl dicts
            if GAME == 'trex' and not is_space_cooling_down and action == 'space':
                is_game_end = game_fb.is_game_end(end_img)
            else:
                rl_controller.action_dict[rl_controller.step] = action

                if action == 'space':
                    reward = 0.9
                elif action == 'idle':
                    reward = 1

                rl_controller.reward_dict[rl_controller.step] = reward
                rl_controller.env_dict[rl_controller.step] = evn_feature_list
                rl_controller.step += 1
                is_game_end = game_fb.is_game_end(end_img)

        reward = -1

        # update RL
        rl_controller.reward_dict[rl_controller.step] = reward
        rl_controller.compute_reward()
        rl_controller.save_Q_learning_data(file1_name, file2_name,
                                           space_kept_number = SPACE_KEPT_NUMBER, idle_kept_number = IDLE_KEPT_NUMBER)
        print ("Training start... ")

        #print ("img_shape: ", img_shape)
        rl_controller.train_rl(img_shape, file1_name, file2_name, is_CNN = is_CNN)

        #
        print ("============================================")
        time.sleep(0.5)

    # ======================================================================================================================