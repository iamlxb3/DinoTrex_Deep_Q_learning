import win32com.client as comctl
import time
import random

class PlayerController:
    def __init__(self, app):
        self.app = app
        self.wsh = comctl.Dispatch("WScript.Shell")

    def activate_chrome(self):
        self.wsh.AppActivate(self.app)

    def _press_key_space(self, space_timer):
        self.wsh.SendKeys("{ }")
        space_timer.t1 = time.time()

    def press_key_space_n_times(self, n):
        interval_t = 0.1
        for i in range(n):
            time.sleep(interval_t)
            self.wsh.SendKeys("{ }")

    def action_choose(self, game_cfg, cnn_input=None):
        if game_cfg.mode == 'random':
            random_number = random.randint(0, 1)
            if random_number == 0:
                action = 'space'
            else:
                action = 'idle'
        else:
            raise Exception("Invalid game mode: ", game_cfg.mode)
        return action

    def action_take(self, action, space_timer):
        print("Action: ", action)
        if action == 'space':
            is_space_cooling_down = space_timer.is_space_cooling_down(time.time())
            if is_space_cooling_down:
                self._press_key_space(space_timer)
            else:
                action = 'idle'
        else:
            pass

        return action