'''
Inspirado en el código de py-sudoku
'''
import numpy as np


class Sudoku:
    def __init__(self, board: list[list[int]] = None, debug: bool = False):
        self.board = board
        self.debug = debug

        if board is None:
            self.board = self._generate_board()

    def __str__(self):
        string = str()
        for index_row, row in enumerate(self.board):
            if index_row % 3 == 0:
                string += ('-' * 31) + '\n'
            for index_cell, cell in enumerate(row):
                if index_cell % 3 == 0:
                    string += '|'
                string += ' {} '.format(cell)
            string += '|\n'
        string += ('-' * 31) + '\n'
        return string

    def _generate_board(self):
        if self.debug:
            # Generamos un sudoku estandar para la prueba
            return np.array(
                    [
                        [0, 0, 0, 4, 0, 0, 7, 0, 0],
                        [8, 0, 7, 0, 0, 6, 0, 0, 0],
                        [0, 6, 2, 0, 0, 0, 0, 1, 8],
                        [0, 0, 0, 3, 0, 1, 8, 9, 0],
                        [3, 0, 0, 6, 0, 0, 0, 4, 7],
                        [0, 2, 0, 0, 0, 8, 0, 0, 0],
                        [9, 0, 8, 0, 6, 3, 1, 5, 4],
                        [2, 4, 3, 0, 7, 5, 6, 8, 0],
                        [0, 1, 0, 0, 9, 4, 0, 0, 0]
                    ]
            )
        else:
            # Generamos un sudoku vacío. Pendiente de implementar de verdad
            return np.zeros((9, 9), dtype=int)

    def _is_valid(self, row_coord: int, col_coord: int, num: int) -> bool:
        # Comprobamos la fila y la columna
        if (np.any(self.board[row_coord] == num) or
                np.any(self.board[:, col_coord] == num)):
            return False

        # Comprobamos el cuadrante
        start_row = row_coord - row_coord % 3
        start_col = col_coord - col_coord % 3
        if np.any(self.board[start_row:start_row+3, start_col:start_col+3] == num):
            return False

        return True

    def fill_cell(self, row_coord: int, col_coord: int, num: int) -> bool:
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')
        if num < 1 or num > 9:
            raise ValueError('Invalid number')
        if self.board[row_coord][col_coord] != 0:
            raise ValueError('Cell ({}, {}) already filled'.format(row_coord, col_coord))

        # Rellenamos la celda si el valor es válido en este momento
        if self._is_valid(row_coord, col_coord, num):
            self.board[row_coord][col_coord] = num
            return True
        return False
