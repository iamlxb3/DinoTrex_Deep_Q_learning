"""
Playing T-rex by Deep Q-learning
Author: Xiaoxuan Wang & PJS

T-rex url: chrome://dino/
Press F12 in chrome, switch to console and type "Runner.instance_.setSpeed(100)" to control speed

# temp
torch.cuda.is_available()
"""

from player_control import PlayerController
from game_control import GameController
from rl_recorder import RlRecorder
import time
import pandas as pd
from timer import Timer
from config import game_cfg, general_cfg, cnn_cfg, rl_cfg


def play_1_game(game_index, game_controller, game_cfg, player_controller, space_timer, rl_recorder):
    print("Game running...")
    rl_recorder.reset()

    is_game_end = False  # game_controller.game_state_check(game_cfg.end_pic_path, game_cfg.end_bbox, game_cfg.end_thres)
    while not is_game_end:
        time1 = time.time()

        # (1.) record envs
        cnn_input = rl_recorder.envs_record(game_cfg, cnn_cfg)

        # (2.) choose an action
        action = player_controller.action_choose(game_cfg, cnn_input=cnn_input)

        # (3.) take the action
        action = player_controller.action_take(action, space_timer)  # action taken
        rl_recorder.actions.append(action)  # record actions

        # (4.) check game end
        time.sleep(rl_cfg.action_gap)
        is_game_end = game_controller.game_state_check(game_cfg.end_pic_path,
                                                       game_cfg.end_bbox,
                                                       game_cfg.end_thres,
                                                       is_save=False)

        # (5.) update step & save time
        time2 = time.time()
        rl_recorder.step += 1
        rl_recorder.times.append(time2 - time1)

    # get score
    score = rl_recorder.score_record(game_cfg)

    # TODO convert the standard format for exprience replay
    assert len(rl_recorder.envs) == len(rl_recorder.actions) \
           == len(rl_recorder.cnn_inputs), "Length of env, action not equal!"

    # compute reward
    rl_recorder.rewards_compute(rl_cfg)
    # TODO, compute the Q_next best value

    # update replays
    rl_recorder.replays_update(game_index)


    #rl_recorder.replay_check()
    return score


def initialise():
    game_controller = GameController(game_cfg.start_bbox, game_cfg.end_bbox, game_cfg.start_thres)
    player_controller = PlayerController(general_cfg.app)
    rl_recorder = RlRecorder()
    # TODO, read replay from disk
    player_controller.activate_chrome()  # switch to chrome
    timer = Timer(game_cfg.space_time_gap)
    performances = {'iter': [], 'score': []}
    return game_controller, player_controller, rl_recorder, timer, performances,


def performances_save(save_path, performances):
    performances_df = pd.DataFrame(performances)
    performances_df.to_csv(save_path, index=False)
    print("save result df to {}".format(save_path))


if __name__ == "__main__":

    # (0.) initialise
    game_controller, player_controller, rl_recorder, timer, performances = initialise()

    for N in range(game_cfg.iteration):
        print("Start New Game, iteration: {}".format(N))

        # (1.) remove all screen shots of last game
        game_controller.remove_screen_shots()
        time.sleep(0.5)

        # (2.) make sure game is ready to go
        game_controller.check_game_start(game_cfg, player_controller)

        # (3.) play game once
        score = play_1_game(N, game_controller, game_cfg, player_controller, timer, rl_recorder)

        # (4.) record & save results
        performances['iter'].append(N)
        performances['score'].append(score)
        performances_save("performances.csv", performances)

        # (5.) load samples from exprience pool & train CNN
