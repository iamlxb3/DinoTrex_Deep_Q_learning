import win32com.client as comctl
import time
import random



class FlappyBirdController:
    def __init__(self,app):
        self.app = app
        self.wsh = comctl.Dispatch("WScript.Shell")
        self.wsh.AppActivate(app)

        
    def game_initialisation(self):
        pass

    def take_action(self, action):
        if action == 'space':
            self._press_key_space

    def _press_key_space(self):
        self.wsh.SendKeys("{ }")

    def press_key_space_n_times(self, n):
        interval_t = 0.1
        for i in range(n):
            time.sleep(interval_t)
            self.wsh.SendKeys("{ }")