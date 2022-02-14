import time
import warnings

import numpy as np
import pyautogui

from src.solver import BruteForceSolver
from src import vision

def on_click(x, y, button, pressed):
    if not pressed: # Stop listener
        return False

corners = vision.get_corners_from_user()
print(f'Corners: {corners}')

print('Get the reset button...')
reset_position = vision.get_click_from_user()
print(f'Reset button: {reset_position}')

board = vision.Board(*corners)
board.recognize_from_screen()

delta_time = 0.2 # slow down on purpose to not break human record
pyautogui.PAUSE = 0.0
for igame in range(10):
    print('=' * 80)
    print(f'Game: {igame}')
    print('=' * 80)

    # reset game
    print('Resetting game...')
    pyautogui.moveTo(*reset_position, duration=delta_time)
    pyautogui.click()
    print('Reset done.')

    solver = BruteForceSolver(board.nrows, board.ncols)
    while True:
        # update board from screen
        time.sleep(0.01) # wait for screen to update before screenshot
        board.screenshot_board()
        # board.print_board(flush=True)
        # print('-' * 50)
        time.sleep(0.01) # wait for screen to update before screenshot

        if all(board.board.flatten() == solver.board.flatten()):
            print('Board is the same as the previous board!', flush=True)
            board.print_board(flush=True)

        solver.set_board(board.board)
        if solver.game_over:
            break

        # get suggestion from solver
        ix, iy = solver.suggest_click()
        px, py = board.get_cell_center(ix, iy)

        # execute suggestion
        pyautogui.moveTo(px, py, duration=delta_time)
        pyautogui.click()

print('Done')