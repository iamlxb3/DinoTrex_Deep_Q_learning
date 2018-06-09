import time
from screen_capturer import GameController


if __name__ == '__main__':
    run_bbox = (570, 200, 1340, 350)  # left, upper, right, and lower
    end_bbox = (800, 200, 1120, 225)
    game_contorl = GameController(run_bbox, end_bbox, 1)

    # for i in range(999):
    #     is_game_start = game_contorl.take_start_screen_shots()
    #     time.sleep(0.1)


    for i in range(5):
        is_game_start = game_contorl.take_end_screen_shots()
        time.sleep(0.4)