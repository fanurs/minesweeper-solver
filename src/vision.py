import concurrent.futures
import glob
import itertools
import pathlib

import win32api
import win32gui

import cv2
import numpy as np
import pyautogui as gui
from pynput import mouse

class WindowsScreenPainter:
    device_context = win32gui.GetDC(0)
    win32_color = win32api.RGB(255, 0, 0) # default red

    @classmethod
    def set_color(cls, rgb):
        cls.win32_color = win32api.RGB(*rgb)

    @classmethod
    def draw_horizontal_line(cls, y, x1, x2, width=1):
        y -= width // 2 # center y according to width
        y = np.clip(y, 0, gui.size().height)

        for x, dy in itertools.product(range(x1, x2 + 1), range(width)):
            win32gui.SetPixel(cls.device_context, x, y + dy, cls.win32_color)
    
    @classmethod
    def draw_vertical_line(cls, x, y1, y2, width=1):
        x -= width // 2 # center x according to width
        x = np.clip(x, 0, gui.size().width)

        for y, dx in itertools.product(range(y1, y2 + 1), range(width)):
            win32gui.SetPixel(cls.device_context, x + dx, y, cls.win32_color)
    
    @classmethod
    def draw_circle(cls, x, y, radius, fill=True):
        x_low, x_upp = x - radius, x + radius
        y_low, y_upp = y - radius, y + radius

        for px, py in itertools.product(range(x_low, x_upp + 1), range(y_low, y_upp + 1)):
            if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                win32gui.SetPixel(cls.device_context, px, py, cls.win32_color)

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
    if not pressed: # Stop listener
        return False

def get_click_from_user():
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
        result = gui.position()
    return result

def get_corners_from_user():
    print('Click on the top left corner of the board...')
    top_left = get_click_from_user()
    
    print('Click on the bottom right corner of the board...')
    bottom_right = get_click_from_user()
    
    return top_left, bottom_right

class Board:
    label_char = {
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

    def __init__(self, top_left, bottom_right, redo_rgb_histogram=False):
        """
        We define the "region of vision" as a rectangular region of the screen
        bounded by the top left and bottom right corners.
        """
        self.set_board_corners(top_left, bottom_right)
        self.screenshot = None

        self.database = dict()
        self.rgb_histograms = dict()
        if redo_rgb_histogram:
            for img_path in glob.glob('./datasets/minesweeper_online/*.png'):
                img_path = pathlib.Path(img_path)
                label = img_path.stem

                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (30, 30))

                self.database[label] = img
                self.rgb_histograms[label] = self._get_rgb_histogram(img)
                print(self.rgb_histograms[label])
                np.savetxt(f'./datasets/minesweeper_online/{label}.txt', self.rgb_histograms[label])
        else:
            for txt_path in glob.glob('./datasets/minesweeper_online/*.txt'):
                label = pathlib.Path(txt_path).stem
                self.rgb_histograms[label] = np.loadtxt(str(txt_path))[:, None]

        self.board = None
    
    def set_board_corners(self, top_left, bottom_right):
        self.x_low, self.x_upp = top_left[0], bottom_right[0]
        self.y_low, self.y_upp = top_left[1], bottom_right[1]
        self.board_pixel_width = self.x_upp - self.x_low
        self.board_pixel_height = self.y_upp - self.y_low
    
    def take_screenshot(self):
        """Take a screenshot of the region of vision.
        """
        self.screenshot = gui.screenshot(
            region=(self.x_low, self.y_low, self.board_pixel_width, self.board_pixel_height),
        )
        self.screenshot = np.array(self.screenshot)
        return self.screenshot

    def recognize_from_screen(self):
        self.take_screenshot()

        # pre-process
        image = cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2GRAY)
        image = np.float32(image)

        # apply Harris corner detection
        image = cv2.cornerHarris(image, 5, 5, 0.1)

        # normalize and remove flat regions and edges
        image /= image.max()
        # image[image < 0.5] = -1
        image = cv2.dilate(image, None)

        # get pixels labeled as corners by Harris detection
        corner_pixels = np.argwhere(image > 0)

        for i, side in enumerate(['y', 'x']):
            # construct a 1D histogram of the projection of the corner pixels
            h, x = np.histogram(corner_pixels[:, i], bins=int(0.25 * image.shape[i]))
            x = 0.5 * (x[1:] + x[:-1]) # get the centers of the histogram bins
            h[h > 0] = 1 # now h is a binary mask; 1 is edge, 0 is not (cell region)
            h = h.astype(np.int8)

            # construct the derivative of h
            # h plot would look like some step function fluctuating between 0 and 1
            # 0 is flat cell region, 1 is edge (projected from corners)
            # hence edge to cell would have a negative slope;
            # cell to edge would have a positive slope
            dh = np.diff(h)
            x_dh = 0.5 * (x[1:] + x[:-1]) # get the centers of dh values
            edge_to_cell = np.argwhere(dh < 0).flatten()
            cell_to_edge = np.argwhere(dh > 0).flatten()
            cell_centers = 0.5 * (x_dh[edge_to_cell] + x_dh[cell_to_edge])

            # perform a linear fit on cell_centers
            n_cells = len(cell_centers)
            cell_enums = np.arange(n_cells)
            par = np.polyfit(cell_enums, cell_centers, 1)
            fitted_cell_centers = np.polyval(par, cell_enums)

            # get cell edges
            dx = fitted_cell_centers[1] - fitted_cell_centers[0]
            cell_edges = np.linspace(
                fitted_cell_centers[0] - 0.5 * dx,
                fitted_cell_centers[-1] + 0.5 * dx,
                n_cells + 1,
            )
            cell_edges = np.round(cell_edges).astype(np.int64)

            # save results
            if side == 'y':
                self.hlines = cell_edges.copy()
                self.nrows = len(self.hlines) - 1
            else: # side == 'x'
                self.vlines = cell_edges.copy()
                self.ncols = len(self.vlines) - 1

        # redine lower and upper bounds of x, y to tightly fit the board
        self.x_upp = self.vlines[-1] + self.x_low
        self.x_low = self.vlines[0] + self.x_low
        self.y_upp = self.hlines[-1] + self.y_low
        self.y_low = self.hlines[0] + self.y_low
        self.vlines -= self.vlines[0]
        self.hlines -= self.hlines[0]

    def draw_grid_lines_on_screen(self, rgb=(255, 0, 0)):
        WindowsScreenPainter.set_color(rgb)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for y in self.hlines:
                executor.submit(
                    WindowsScreenPainter.draw_horizontal_line,
                    y + self.y_low,
                    self.x_low, self.x_upp,
                    width=3
                )
            for x in self.vlines:
                executor.submit(WindowsScreenPainter.draw_vertical_line, x + self.x_low, self.y_low, self.y_upp, width=3)
    
    def get_cell_center(self, ix, iy, screen_coordinates=True):
        x = 0.5 * (self.vlines[ix] + self.vlines[ix + 1])
        y = 0.5 * (self.hlines[iy] + self.hlines[iy + 1])
        if screen_coordinates:
            x += self.x_low
            y += self.y_low
        return int(x), int(y)
    
    def draw_cell_center_on_screen(self, ix, iy, rgb=(0, 255, 128), radius=3):
        x, y = self.get_cell_center(ix, iy)
        WindowsScreenPainter.set_color(rgb)
        WindowsScreenPainter.draw_circle(x, y, radius=radius)

    def draw_cell_centers_on_screen(self, rgb=(0, 255, 128)):
        for iy in range(self.nrows):
            for ix in range(self.ncols):
                self.draw_cell_center_on_screen(ix, iy, rgb)
    
    def get_cell_screenshot(self, ix, iy, use_global=True, retake_screenshot=False):
        if use_global:
            if retake_screenshot:
                self.take_screenshot()
            x, y = self.get_cell_center(ix, iy, screen_coordinates=False)
            img = self.screenshot.copy()
            img = img[
                self.hlines[iy]:self.hlines[iy + 1],
                self.vlines[ix]:self.vlines[ix + 1],
            ]
        else:
            x, y = self.get_cell_center(ix, iy)
            img = gui.screenshot(
                region=(
                    self.vlines[ix], # left
                    self.hlines[iy], # top
                    self.vlines[ix + 1] - self.vlines[ix], # width
                    self.hlines[iy + 1] - self.hlines[iy], # height
                )
            )
            img = np.array(img)
        img = cv2.resize(img, (30, 30))
        return img

    def _get_rgb_histogram(self, rgb_img):
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        return cv2.calcHist([img], [0], None, [256], [0, 256])
    
    def _euclidean_dist2(self, x1, x2):
        return np.sum((x1 - x2)**2) / len(x1)
    
    def identify_cell(self, ix, iy):
        cell_screenshot = self.get_cell_screenshot(ix, iy)

        hist = self._get_rgb_histogram(cell_screenshot)

        lowest_penalty = np.inf
        best_label = None
        for label, hist_ref in self.rgb_histograms.items():
            penalty = self._euclidean_dist2(hist, hist_ref)
            if penalty < lowest_penalty:
                lowest_penalty = penalty
                best_label = label
        return best_label
    
    def __getitem__(self, key):
        return self.board[key]
    
    def screenshot_board(self):
        self.take_screenshot()
        if self.board is None:
            self.board = np.array([
                [self.label_char['unclicked'] for _ in range(self.ncols)]
                for _ in range(self.nrows)
            ])
        
        for iy, ix in itertools.product(range(self.nrows), range(self.ncols)):
            label = self.identify_cell(ix, iy)
            self.board[iy, ix] = self.label_char[label]
    
    def print_board(self, **kwargs):
        for row in self.board:
            print(' '.join(row), **kwargs)


if __name__ == '__main__':
    pass