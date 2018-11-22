import time
from game_control import GameController
#import pytesseract
from pytesseract  import image_to_string
import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

if __name__ == '__main__':
    run_bbox = (570, 200, 1340, 350)  # left, upper, right, and lower
    end_bbox = (800, 200, 1120, 225)
    score_bbox = (1240, 150, 1330, 200)

    # for i in range(999):
    #     is_game_start = game_contorl.take_start_screen_shots()
    #     time.sleep(0.1)


    # for i in range(5):
    #     is_game_start = game_contorl.take_end_screen_shots()
    #     time.sleep(0.4)

    # game_contorl = GameController(score_bbox, score_bbox, 1)
    # for i in range(5):
    #     is_game_start = game_contorl.take_end_screen_shots()
    #     time.sleep(0.4)

results = image_to_string(Image.open("running_screen_shots/6.png"))
print("results: ", int(results))