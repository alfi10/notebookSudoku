# Inspirado en el código de py-sudoku
import copy
from typing import Any
import numpy as np
from itertools import product


class Sudoku:
    def __init__(self, board: np.ndarray[Any, np.dtype] = None, debug: bool = False):
        self.debug = debug
        self.board = self._generate_board() if board is None else board
        self.board_valids = self.calculate_board_valids()
        self.solution_path = []  # Lista de tuplas (row, col, num) con los pasos del camino hacia la solución

    def __str__(self):
        string = str()
        solution_path = self.solution_path
        string += 'Estado Inicial\n' if not solution_path else 'Camino de solución: {}\n'.format(solution_path)
        string += self.board_string()
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

    def board_string(self):
        string = str()
        raw_data = self._valids_to_board_data()
        for index_row, row in enumerate(raw_data):
            if index_row % 3 == 0:
                string += ('-' * 31) + '\n'
            for index_cell, cell in enumerate(row):
                if index_cell % 3 == 0:
                    string += '|'
                string += ' {} '.format(cell)
            string += '|\n'
        string += ('-' * 31) + '\n'
        return string

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
        if cell_number != 0:  # Si tiene número, es válido si es el mismo que el que se quiere comprobar
            return True if cell_number == num else False
        # Calculamos las coordenadas de la fila, columna y cuadrante cuyas valideces vamos a comprobar
        validity_coords = self.get_row_col_cuadrant_coords(row_coord, col_coord)
        for row, col in validity_coords:
            if self.get_cell(row, col) == num:
                return False
        # Else es un número válido
        return True

    def _valids_to_board_data(self):
        """
        Devuelve una representación del tablero a partir de la matriz de validos. Si una celda tiene un solo True, se
        rellena con el número correspondiente. Si tiene más de un True, se rellena con un 0. Si no tiene ningún True,
        se deja una x.
        :return: String con el tablero de sudoku
        """
        string = np.full((9, 9), 'x', dtype=str)
        for row in range(9):
            for col in range(9):
                if np.sum(self.board_valids[row][col]) == 1:
                    string[row][col] = str(np.where(self.board_valids[row][col] == 1)[0][0] + 1)
                elif np.sum(self.board_valids[row][col]) > 1:
                    string[row][col] = '0'
                # Else: deja 'x' que representa visualmente que es un estado erroneo
        return string

    def is_solved(self) -> bool:
        """
        Comprueba si el sudoku es válido mirando board_valids. Si hay alguna celda con más de 1 True, alguna celda solo
        Falses, alguna celda con True en un índice que sea True en otro índice de su misma fila, columna o cuadrante,
        devuelve False. En caso contrario, devuelve True.
        :return: True si el sudoku está resuelto, False en caso contrario
        """
        # Comprueba que cada celda tenga 1 True
        total_trues = np.sum(self.board_valids)
        size_rows = self.board_valids.shape[0]
        size_cols = self.board_valids.shape[1]
        required_trues = size_rows * size_cols
        if total_trues != required_trues:
            return False
        # Comprueba que cada columna tenga un True por número sumando los arrays de cada columna
        repeticiones_columnas = np.sum(self.board_valids, axis=0)
        if np.any(repeticiones_columnas != 1):
            return False
        # Comprueba que cada fila tenga un True por número sumando los arrays de cada fila
        repeticiones_filas = np.sum(self.board_valids, axis=1)
        if np.any(repeticiones_filas != 1):
            return False
        # Comprueba que cada cuadrante tenga un True por número sumando los arrays de cada cuadrante
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                cuadrante = self.board_valids[row:row + 3, col:col + 3]
                repeticiones_cuadrante = np.sum(cuadrante, axis=2)
                if np.any(repeticiones_cuadrante != 1):
                    return False
        return True

    def _update_board_valids(self, row_coord: int, col_coord: int, num: int):
        """
        Actualiza los números válidos de la fila, columna y cuadrante correspondientes. Se quita el número de los
        válidos de todas las filas, columnas y cuadrantes. No se hace si el número no es válido en la celda coordenadas.

        :param row_coord: Coordenada de fila de la celda
        :param col_coord: Coordenada de columna de la celda
        :param num: Número a añadir o quitar de los válidos
        :return: True si se ha actualizado, False en caso contrario
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates')
        if num < 1 or num > 9:
            raise ValueError('Invalid number: {}'.format(num))

        # Si el número no es válido en la celda, no se actualiza
        if not self._is_valid(row_coord, col_coord, num):
            return False
        # Es válido. Actualizamos la celda sujeto
        # Los números que no son num de la celda son no válidos
        self.board_valids[row_coord, col_coord, np.arange(9)] = False
        # El número num de la celda es válido
        self.board_valids[row_coord, col_coord, num - 1] = True

        # Calculamos las coordenadas de la fila, columna y cuadrante cuyas valideces vamos a actualizar
        change_coords = self.get_row_col_cuadrant_coords(row_coord, col_coord)
        # Actualizamos los números válidos de las coordenadas
        for row, col in change_coords:
            self.board_valids[row, col, num - 1] = False
        return True

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
            # Borramos el número añadido de los válidos correspondientes
            self._update_board_valids(row_coord, col_coord, num)
            # Agregamos el paso al camino hacia la solución
            self._update_solution_path(row_coord, col_coord, num)
            return True
        return False

    def get_cell_valids(self, row_coord: int, col_coord: int) -> np.ndarray:
        """
        Devuelve los números válidos para rellenar una celda.
        :param row_coord: Coordenada de fila de la celda
        :param col_coord: Coordenada de columna de la celda
        :return: np.ndarray de booleanos de 9 elementos. True si el número es válido, False en caso contrario
        """
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

    def get_successors(self, cost: int = None):
        """
        Devuelve una lista de sucesores del nodo actual. El nodo actual es el primero que, en orden izquierda-derecha y
        arriba-abajo, tiene una celda vacía.
        :param cost: Opcional. Coste acumulado del nodo actual
        :return: Lista de Sudoku sucesores o de tuplas (Sudoku, coste) sucesores si recibimos cost
        """
        successors = []
        # Obten el número de validos por celda
        valids_per_cell = np.sum(self.board_valids, axis=2)
        if np.any(valids_per_cell == 0):  # Si el sudoku tiene una celda sin válidos, no genera hijos
            return successors
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
                            cost += possible_numbers.size - 1
                            successors.append((sudoku, cost))
                        else:
                            successors.append(sudoku)
                    return successors
        raise Exception('Nunca debería llegar aquí. Debe ser que el Sudoku no tiene solución')

    def heuristic(self):
        # return np.sum(self.board_valids)
        return np.multiply(self.board_valids, self.board_valids).sum()

    def _update_solution_path(self, row_coord: int, col_coord: int, num: int):
        """
        Actualiza el camino hacia la solución con el paso actual si num es diferente de 0. Si num es 0, elimina el paso
        del camino.
        :param row_coord: Coordenada de fila del paso
        :param col_coord: Coordenada de columna del paso
        :param num: Número del paso dado. Si es 0, se elimina dado en las coordenadas dadas
        :return: None
        """
        # Comprobación de errores pre ejecución
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates: (X: {}, Y: {})'.format(row_coord, col_coord))
        if num < 0 or num > 9:
            raise ValueError('Invalid number: {}'.format(num))

        # Si num es 0, agregamos el paso a la solución
        if num != 0:
            self.solution_path.append((row_coord, col_coord, num))
        else:  # Si num es 0, eliminamos el paso de la solución
            ultimo_paso_coordenadas = self.solution_path[-1:][2:]
            # Si el paso a eliminar no es el último, lo buscamos
            if ultimo_paso_coordenadas != (row_coord, col_coord):
                # Buscamos el paso a eliminar
                indice = next(
                    (indice for indice, paso in enumerate(self.solution_path)
                     if paso[:2] == (row_coord, col_coord)
                     ), None
                )
                if indice is not None:
                    self.solution_path.pop(indice)
                else:
                    raise ValueError('No se encontró el paso a eliminar')
            else:  # El paso a eliminar es el último. Pop
                self.solution_path.pop()

    @staticmethod
    def get_row_col_cuadrant_coords(row_coord: int, col_coord: int):
        """
        Calcula las coordenadas correspondientes a la misma  fila, misma columna y mismo cuadrante de unas coordenadas
        dadas. Devuelve un conjunto con las coordenadas de las celdas menos las coordenadas dadas.
        :param row_coord: Coordenada de fila de la celda
        :param col_coord: Coordenada de columna de la celda
        :return: Tupla con las coordenadas de la fila, columna y cuadrante similares a las dadas
        """
        # Comprobación de errores
        if row_coord < 0 or row_coord > 8 or col_coord < 0 or col_coord > 8:
            raise ValueError('Invalid coordinates: (X: {}, Y: {})'.format(row_coord, col_coord))

        # Calculamos las coordenadas de la fila, columna y cuadrante
        row = set(zip([row_coord] * 9, range(9)))  # Coordenadas de la fila
        col = set(zip(range(9), [col_coord] * 9))  # Coordenadas de la columna
        # Coordenadas del cuadrante
        start_row = row_coord - row_coord % 3
        start_col = col_coord - col_coord % 3
        cuadrante = set(product(range(start_row, start_row + 3), range(start_col, start_col + 3)))
        # Unimos las coordenadas y quitamos la de la celda sujeto
        result = row.union(col).union(cuadrante)
        result.remove((row_coord, col_coord))
        return result
