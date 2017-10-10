import win32com.client as comctl
import time
import random
wsh = comctl.Dispatch("WScript.Shell")

#import subprocess
#subprocess.call(['C:\Program Files (x86)\Tencent\WeChat\WeChat.exe'])


# Google Chrome window title
wsh.AppActivate("chrome")

for i in range(100):
    interval_t = random.random()
    time.sleep(interval_t)
    wsh.SendKeys("{ }")