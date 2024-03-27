import numpy as np
from .sudoku import Sudoku


class SudokuSolver:
    def __init__(self, sudoku: Sudoku):
        self.sudoku = sudoku
        self.operations_applied = np.array([], dtype=int)

    def _find_empty_cells(self):
        empty_cells = np.where(self.sudoku.board == 0)
        if len(empty_cells[0]) > 0:
            return np.array(list(zip(*empty_cells)))
        return None

    def solve_profundidad(self):
        sudoku = self.sudoku
        empty_cells = self._find_empty_cells()
        if empty_cells is None:
            return True
        else:
            cells_to_fill = len(empty_cells)

        solved = False
        i_empty_cell = 0
        while not solved:
            # Operar sobre el nodo que toca en este piso del arbol
            row_coord, col_coord = empty_cells[i_empty_cell]
            # Prueba cada número del 1 al 9
            first_number_test = sudoku.get_cell(row_coord, col_coord) + 1
            if first_number_test != 1:
                # Si la celda ya tiene un número, lo borramos porque hemos subido desde un nodo hijo cerrado
                sudoku.clear_cell(row_coord, col_coord)
            for num in range(first_number_test, 10):
                if sudoku.fill_cell(row_coord, col_coord, num):
                    i_empty_cell += 1
                    if i_empty_cell == cells_to_fill:
                        solved = True
                    break
            else:
                '''
                No se pudo rellenar la celda porque no hay más estados hijos.
                Sube al nodo padre
                '''
                i_empty_cell -= 1


        # # Recorre cada celda vacía
        # for row_coord, col_coord in empty_cells:
        #     try:
        #         # Prueba cada número del 1 al 9
        #         for num in range(1, 10):
        #             if self.sudoku.fill_cell(row_coord, col_coord, num):
        #                 if self.solve_profundidad():
        #                     return
        #             else:
        #                 print('False en Cell ({}, {}) intentando {}'.format(row_coord, col_coord, num))
        #         print('No fill en ({}, {})'.format(row_coord, col_coord))
        #     except ValueError as e:
        #         print(e)
