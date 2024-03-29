# Inspirado en el código de py-sudoku
import copy
from typing import Any
import numpy as np


class Sudoku:
    def __init__(self, board: np.ndarray[Any, np.dtype] = None, board_valids: np.ndarray = None, debug: bool = False):
        self.debug = debug
        self.board = self._generate_board() if board is None else board
        self.board_valids = self.calculate_board_valids() if board_valids is None else board_valids
        self.solution_path = []

    def __str__(self):
        string = str()
        solution_path = self.solution_path
        string += 'Estado Inicial\n' if not solution_path else 'Camino de solución: {}\n'.format(solution_path)
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
                ], dtype=int)
        else:
            # Generamos un sudoku vacío. Pendiente de implementar de verdad
            return np.zeros((9, 9), dtype=int)

    def get_cell(self, row_coord: int, col_coord: int) -> int:
        """
        Devuelve el número de la celda en las coordenadas dadas de la matriz del sudoku.
        :param row_coord: Coordenada de fila de la celda (Vertical)
        :param col_coord: Coordenada de columna de la celda (Horizontal)
        :return: El número almacenado en la celda de las coordenadas dadas.
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')

        return int(self.board[row_coord][col_coord])

    def _is_valid(self, row_coord: int, col_coord: int, num: int) -> bool:
        """
        Comprueba si un número es válido en una celda. Para ello, comprueba si el número ya está en la fila, columna o
        cuadrante.
        :param row_coord: Coordenada de fila (vertical)
        :param col_coord: Coordenada de columna (horizontal)
        :param num: Valor a comprobar si es válido
        :return: True si el número es válido, False en caso contrario
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')
        if num < 1 or num > 9:
            raise ValueError('Invalid number: {}'.format(num))

        # Comprobamos si la celda está rellena
        cell_number = self.get_cell(row_coord, col_coord)
        if cell_number != 0:
            """
            El número no supera la comprobación de validez, pero el estar implica que los superó cuando se puso.
            
            Si la celda tiene el número que estamos comprobando, devolvemos True. Hacemos esto para preservar la
            equivalencia de la matriz de validos con el numero rellenado. Así, cuando una celda tiene un numero, la
            matriz de validos tiene un True en la posición del número de la celda y el resto en False.
            """
            return True if cell_number == num else False
        # Comprobamos si el número está en la fila y en la columna
        num_in_row = np.any(self.board[row_coord] == num)
        num_in_col = np.any(self.board[:, col_coord] == num)
        # Comprobamos si el número está en el cuadrante
        start_row = row_coord - row_coord % 3
        start_col = col_coord - col_coord % 3
        num_in_cuadrante = np.any(self.board[start_row:start_row + 3, start_col:start_col + 3] == num)
        # Comprobación de validez
        if num_in_row or num_in_col or num_in_cuadrante:
            return False
        # Else es un número válido
        return True

    def _update_board_valids(self, row_coord: int, col_coord: int, num: int, erase: bool = False):
        """
        Actualiza los números válidos de la fila, columna y cuadrante correspondientes.
        Si erase, se añade el número a los válidos de filas, columnas y cuadrantes correspondientes.
        Si not erase, se quita directamente el número de los válidos de todas las filas, columnas y cuadrantes.

        :param row_coord: Coordenada de fila de la celda
        :param col_coord: Coordenada de columna de la celda
        :param num: Número a añadir o quitar de los válidos
        :param erase: Indica si se ha borrado el número de la celda. Si True, num es el borrado. Else, es el rellenado
        :return: None
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')
        if num < 0 or num > 9:
            raise ValueError('Invalid number: {}'.format(num))

        # Actualizamos los números válidos de la fila row_coord
        for col in self.board_valids[row_coord]:
            col[num - 1] = self._is_valid(row_coord, col_coord, num) if erase else erase  # Ver docstring
        # Actualizamos los números válidos de la columna col_coord
        for row in self.board_valids[:, col_coord]:
            row[num - 1] = self._is_valid(row_coord, col_coord, num) if erase else erase  # Ver docstring
        # Actualizamos los números válidos del cuadrante
        start_row = row_coord - row_coord % 3
        start_col = col_coord - col_coord % 3
        for row in range(start_row, start_row + 3):
            for col in range(start_col, start_col + 3):
                self.board_valids[row, col, num - 1] = (
                    self._is_valid(row, col, num)
                ) if erase else erase  # Ver docstring

    def fill_cell(self, row_coord: int, col_coord: int, num: int) -> bool:
        """
        Rellena una celda con un número si es válido en ese momento.
        :param row_coord: Coordenada de fila de la celda
        :param col_coord: Coordenada de columna de la celda
        :param num: Número a rellenar en la celda
        :return: True si se ha rellenado, False en caso contrario
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')
        if num < 1 or num > 9:
            raise ValueError('Invalid number: {}'.format(num))
        if self.get_cell(row_coord, col_coord) != 0:
            raise ValueError('Tried {} but Cell ({}, {}) already filled with {}'
                             .format(num, row_coord, col_coord, self.get_cell(row_coord, col_coord))
                             )

        # Rellenamos la celda si el valor es válido en este momento
        if self._is_valid(row_coord, col_coord, num):
            self.board[row_coord][col_coord] = num
            if hasattr(self, 'board_valids'):
                # Borramos el número añadido de los válidos correspondientes
                self._update_board_valids(row_coord, col_coord, num)
            return True
        return False

    def clear_cell(self, row_coord: int, col_coord: int):
        """
        Borra el número de una celda si no está vacía.
        :param row_coord: Coordenada de fila de la celda
        :param col_coord: Coordenada de columna de la celda
        :return: None
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')
        # Guardamos el número de la celda que vamos a vaciar
        num = self.get_cell(row_coord, col_coord)
        if num == 0:
            raise ValueError('Cell ({}, {}) already empty'.format(row_coord, col_coord))

        # Vaciamos la celda
        self.board[row_coord][col_coord] = 0
        if hasattr(self, 'board_valids'):
            # Restauramos el número borrado a los válidos correspondientes
            self._update_board_valids(row_coord, col_coord, num, erase=True)

    def get_cell_valids(self, row_coord: int, col_coord: int) -> np.ndarray:
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')

        # Calculamos los números válidos para la celda
        valids = np.zeros(9, dtype=bool)
        for num in range(9):
            valids[num] = self._is_valid(row_coord, col_coord, num + 1)
        return valids

    def calculate_board_valids(self) -> np.ndarray:
        valids = np.array([self.get_cell_valids(row, col) for row in range(9) for col in range(9)])
        return valids.reshape((9, 9, 9))

    def is_solved(self) -> bool:
        return np.all(self.board != 0)

    def get_successors(self, cost: int = None):
        successors = []
        valids_row_length = self.board_valids.shape[0]
        for irow in range(valids_row_length):
            valids_column_length = self.board_valids.shape[1]
            for icol in range(valids_column_length):
                # Si el board está vacío, calculamos los posibles números para rellenar la celda
                board_cell = self.get_cell(irow, icol)
                if board_cell == 0:
                    # La matriz board_valids es booleana. Numpy trata True como 1 y False como 0
                    possible_numbers = np.where(self.board_valids[irow][icol] == 1)[0] + 1
                    for num in possible_numbers:
                        sudoku = copy.deepcopy(self)
                        sudoku.fill_cell(irow, icol, num)
                        if cost is not None:  # Si cost no es None, devolvemos una tupla con el coste
                            # Coste de un nodo: el número de hijos que puede generar el padre más el coste acumulado
                            cost += possible_numbers.size
                            successors.append((sudoku, cost))
                        else:
                            successors.append(sudoku)
                    return successors
        raise Exception  # Nunca debería llegar aquí

    def heuristic(self):
        return np.sum(self.board_valids)
