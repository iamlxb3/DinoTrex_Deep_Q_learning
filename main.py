from flappy_bird_controller import FlappyBirdController
from screen_capturer import GameFb
import time
import random
import sys
import os
from fb_rl import FB_RL

#website = 'http://flappybird.io/'
is_CNN = False

# --------------------------------------
# (1.) RL config
# --------------------------------------
ACTION_GAP_TIME = 0.25
iteration = 1000
THIN_FACTOR = 1
alpha = 0.5
SPACE_KEPT_NUMBER = 200
IDLE_KEPT_NUMBER = 200
# --------------------------------------

# --------------------------------------
# (2.) ANN config
# --------------------------------------
nn_config_dict = {}
nn_config_dict['learning_rate_init'] = 0.0001
nn_config_dict['hidden_layer_sizes'] = (100,1)
nn_config_dict['is_verbose'] = True
# --------------------------------------

# --------------------------------------
# INITIALISATION
# --------------------------------------
app = 'chrome'
game_fb = GameFb()
bird_c = FlappyBirdController(app)
rl_controller = FB_RL(nn_config_dict,alpha)
# --------------------------------------


# ======================================================================================================================
# MAIN
# ======================================================================================================================
for iteration_i in range(iteration):


    print ("Start New Game, iteration: {}".format(iteration_i))
    game_fb.clear_screen_shots()
    rl_controller.step = 0


    is_game_start = game_fb.is_game_start
    while not is_game_start:
        bird_c.press_key_space_n_times(1)
        print ("Wating for game to start...")
        time.sleep(0.2)
        is_game_start = game_fb.is_game_start

    is_game_end = game_fb.is_game_end

    print("Game running...")
    while not is_game_end:
        sleep_time = ACTION_GAP_TIME
        time.sleep(sleep_time)

        reward = 1

        # (0.) get evn
        evn_feature_list, img_shape = game_fb.get_img_feature(thin_factor=THIN_FACTOR)


        # +++++++++++++++++++++++++++++++++++++++++++++++++
        # (1.) get action
        # +++++++++++++++++++++++++++++++++++++++++++++++++
        # -------------------------------------------------
        # use random action for the 1st iteration
        # -------------------------------------------------
        if iteration_i == 0:
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
                action = rl_controller.get_action(evn_feature_list, img_shape, is_CNN = is_CNN)
        # -------------------------------------------------

        # +++++++++++++++++++++++++++++++++++++++++++++++++




        print ("[action]: {}".format(action))
        # (2.) take action
        if action == 'space':
            bird_c._press_key_space()
        else:
            pass


        # (.) print
        print ("[step {}], [action]-{}".format(rl_controller.step, action))
        #

        # (.) update rl dicts
        rl_controller.action_dict[rl_controller.step] = action
        rl_controller.reward_dict[rl_controller.step] = reward
        rl_controller.env_dict[rl_controller.step] = evn_feature_list
        rl_controller.step += 1

        is_game_end = game_fb.is_game_end

    reward = -1

    # update RL
    rl_controller.reward_dict[rl_controller.step] = reward
    rl_controller.compute_reward()
    rl_controller.save_Q_learning_data(space_kept_number = SPACE_KEPT_NUMBER, idle_kept_number = IDLE_KEPT_NUMBER)
    print ("Training start... ")
    #
    img_shape = (img_shape[0], int(img_shape[1]/THIN_FACTOR)) # the compression of the data throws away column information
    #
    rl_controller.train_rl(img_shape, is_CNN = is_CNN)
    #
    print ("============================================")
    time.sleep(0.5)

# ======================================================================================================================




    # # detect whether game is still running
# while True:
#     # (1.) refresh the environment, set the refresh rate, update_env-dict
#     # (2.) choose the best action from RL
#     to_press_key = True
#     # (3.) do the action
#     if to_press_key:
#         bird_c.press_key_space()
#     # (4.) update action-dict, late-reward-dict
#
#
#
# # update Q learning