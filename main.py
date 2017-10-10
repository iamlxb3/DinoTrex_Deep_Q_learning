from flappy_bird_controller import FlappyBirdController
from screen_capturer import GameFb
import time
import random
import sys
from fb_rl import FB_RL

#website = 'http://flappybird.io/'

game_fb = GameFb()
app = 'chrome'
bird_c = FlappyBirdController(app)

nn_config_dict = {}
nn_config_dict['learning_rate_init'] = 0.1
nn_config_dict['hidden_layer_sizes'] = (20,1)

rl_controller = FB_RL(nn_config_dict)

iteration = 1000

# ======================================================================================================================
# MAIN
# ======================================================================================================================
for i in range(iteration):


    print ("Start New Game, iteration: {}".format(i))
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
        sleep_time = 0.25
        time.sleep(sleep_time)

        reward = 1

        # (0.) get evn
        evn_arr = game_fb.get_img_feature(thin_factor=4)

        # (1.) get action
        if rl_controller.step == 0:
            action = 'space'
        else:
            #action = 'idle'
            action = rl_controller.get_action(evn_arr)


        print ("[action]: {}".format(action))
        # (2.) take action
        if action == 'space':
            bird_c._press_key_space()
        else:
            pass


        # (.) update rl dicts
        rl_controller.action_dict[rl_controller.step] = action
        rl_controller.reward_dict[rl_controller.step] = reward
        rl_controller.env_dict[rl_controller.step] = evn_arr
        rl_controller.step += 1
        print ("rl_controller.step: ", rl_controller.step)

        is_game_end = game_fb.is_game_end

    reward = -1

    # update RL
    rl_controller.reward_dict[rl_controller.step] = reward
    rl_controller.compute_reward()
    print ("Training start... ")
    rl_controller.train_rl()
    #

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