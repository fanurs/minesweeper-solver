import itertools
import time

import numpy as np
import pyautogui

class Solver:
    status_char = {
        'unclicked': '.',
        'number_0': '0',
        'number_1': '1',
        'number_2': '2',
        'number_3': '3',
        'number_4': '4',
        'number_5': '5',
        'number_6': '6',
        'number_7': '7',
        'number_8': '8',
        'flag': 'F',
        'flag_incorrect': 'f',
        'mine_clicked': 'X',
        'mine_unclicked': 'x',
    }
    statuses = list(status_char.keys())

    def __init__(self, nrows, ncols, nmines=None):
        self.nrows = nrows
        self.ncols = ncols
        self.shape = (nrows, ncols)
        self.nmines = nmines
        self.board = np.array([
            [self.status_char['unclicked'] for ix in range(ncols)]
            for iy in range(nrows)
        ])
        self._get_all_unclicked_indices()
    
    def set_board(self, board):
        board = np.array(board)
        if board.shape != self.shape:
            raise ValueError('Board shape is not the same.')
        self.board = board.copy()
    
    def _get_all_unclicked_indices(self, board=None):
        if board is None:
            board = self.board
        self.unclicked_indices = np.argwhere(board == self.status_char['unclicked'])
        return self.unclicked_indices
    
    def suggest_click(self):
        return self._suggest_random_click()
    
    def _suggest_random_click(self):
        self._get_all_unclicked_indices()
        i = np.random.choice(range(len(self.unclicked_indices)))
        return tuple(self.unclicked_indices[i][::-1])
    
    def print_board(self, board=None):
        if board is None:
            board = self.board
        for row in board:
            print(' '.join(row))
    
    @property
    def game_over(self):
        if self.status_char['mine_clicked'] in self.board.flatten():
            return True
        self._get_all_unclicked_indices()
        if len(self.unclicked_indices) == 0:
            return True
        return False

class BruteForceSolver(Solver):
    def __init__(self, nrows, ncols, nmines=None):
        super().__init__(nrows, ncols, nmines)
        self.confident_board = self.board.copy()
    
    def update_confident_board(self):
        for ix, iy in itertools.product(range(self.ncols), range(self.nrows)):
            if self.confident_board[iy, ix] == self.status_char['flag']:
                continue
            if self.confident_board[iy, ix] == self.status_char['unclicked']:
                self.confident_board[iy, ix] = self.board[iy, ix]
    
    def suggest_click(self, n_iters=3):
        for i_iter in range(n_iters):
            self.update_confident_board()
            self._get_all_unclicked_indices(self.confident_board)
            for iy, ix in self.unclicked_indices:
                solution = self.solve_unclicked(ix, iy, board=self.confident_board)
                if solution == 'safe':
                    return ix, iy
                if solution == 'mine':
                    self.confident_board[iy, ix] = self.status_char['flag']
        return self._suggest_random_click()

    def _suggest_random_click(self):
        self._get_all_unclicked_indices(board=self.confident_board)
        i = np.random.choice(range(len(self.unclicked_indices)))
        return tuple(self.unclicked_indices[i][::-1])
    
    def get_nearby_indices(self, ix, iy, level=1):
        nearby_indices = np.array([
            xy_pair for xy_pair in itertools.product(
                range(ix - level, ix + level + 1),
                range(iy - level, iy + level + 1),
            ) if xy_pair != (ix, iy)
        ])
        return nearby_indices[
            (nearby_indices[:, 0] >= 0) & (nearby_indices[:, 0] < self.ncols) &
            (nearby_indices[:, 1] >= 0) & (nearby_indices[:, 1] < self.nrows)
        ]

    def get_nearby_subboard(self, ix, iy, board=None, level=1):
        if board is None:
            board = self.board
        nearby_indices = self.get_nearby_indices(ix, iy, level)
        return board[
            nearby_indices[:, 1].min():nearby_indices[:, 1].max() + 1,
            nearby_indices[:, 0].min():nearby_indices[:, 0].max() + 1,
        ]
    
    def solve_unclicked(self, ix, iy, board=None):
        if board is None:
            board = self.board

        if board[iy, ix] != self.status_char['unclicked']:
            raise ValueError(f'This cell is not unclicked. {ix} {iy} {board[iy, ix]}')

        result = 'unknown'
        nearby_indices = self.get_nearby_indices(ix, iy)
        for nearby_index in nearby_indices:
            nearby_cell = board[nearby_index[1], nearby_index[0]]
            if nearby_cell.isdigit():
                cell_num = int(nearby_cell)
                cells_around = np.array([
                    board[_xy[1], _xy[0]]
                    for _xy in self.get_nearby_indices(nearby_index[0], nearby_index[1])
                ])
                n_flags = np.sum(cells_around == self.status_char['flag'])
                n_unclickeds = np.sum(cells_around == self.status_char['unclicked'])
                if n_unclickeds == cell_num - n_flags:
                    result = 'mine'
                    break
                if cell_num - n_flags == 0:
                    result = 'safe'
                    break
        return result

if __name__ == '__main__':
    pass