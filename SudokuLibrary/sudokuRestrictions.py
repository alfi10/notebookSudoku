import numpy as np
from itertools import product


def _board2representation(board: np.ndarray):
    """
    Convierte un tablero de sudoku en una representación orientada a restricciones.
    """
    xy = list(product(range(9), range(9)))
    xyz = np.array([[x, y, int(x / 3) * 3 + int(y / 3)] for x, y, in xy])
    valids = []
    for coords in xyz:
        num = board[coords[0], coords[1]]
        if num != 0:
            num_array = np.zeros(9, dtype=bool)
            num_array[num - 1] = True
            valids.append(num_array)
        else:
            no_num = np.ones(9, dtype=bool)
            valids.append(no_num)
    valids = np.array(valids)
    return xyz, valids


def _representation2board(coords: np.ndarray, valids: np.ndarray):
    """
    Convierte la representación orientada a restricciones en un tablero de sudoku.
    La celda será un 0 si no se puede determinar su valor.
    """
    board = np.zeros((9, 9), dtype=int)
    for coord in coords:
        x, y, _ = coord
        if np.sum(valids[x, y]) == 1:
            board[x, y] = np.argmax(valids[x, y]) + 1
    return board


class Sudoku:
    def __init__(self, board: np.ndarray = np.zeros((9, 9), dtype=int)):
        self.board = board
        self.coords, self.valids = _board2representation(board)

    def basic_restrictions(self):
        """
        Restricciones básicas de un sudoku: filas, columnas y cuadrantes
        """
        for icoord in range(self.coords.shape[0]):
            if np.sum(self.valids[icoord]) > 1:
                # Skips if already determined
                # x, y, z = self.coords[icoord]
                collisions = []
                for dim in range(self.coords.shape[1]):
                    collisions.append(self.coords[self.coords[:, dim] == self.coords[icoord, dim]])
                collisions = np.array(collisions).reshape(-1, 3)
                # Update current coord valids
                for collision in collisions:
                    x = collision[0]
                    y = collision[1]
                    valid = self.valids[x*9 + y]
                    if np.sum(valid) == 1:
                        # Eliminate from possible values
                        inv_collision = np.logical_not(valid)
                        self.valids[icoord] = np.logical_and(self.valids[icoord], inv_collision)
                if np.sum(self.valids[icoord]) == 1:
                    # New name set
                    self.board[self.coords[icoord][0], self.coords[icoord][1]] = np.argmax(self.valids[icoord]) + 1
                    print(f"Coord {self.coords[icoord]} solo tiene una opción: {np.argmax(self.valids[icoord]) + 1}")
