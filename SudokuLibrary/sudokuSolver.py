import copy
import numpy as np
from .sudoku import Sudoku


def show_sudoku(sudoku: Sudoku):
    print(sudoku)


class SudokuSolver:
    def __init__(self, sudoku: Sudoku):
        self.sudoku = copy.deepcopy(sudoku)
        self.operations_applied = np.array([], dtype=int)

    def _find_empty_cells(self):
        empty_cells = np.where(self.sudoku.board == 0)
        if len(empty_cells[0]) > 0:
            return np.array(list(zip(*empty_cells)))
        return None

    def protosolver(self):
        sudoku = copy.deepcopy(self.sudoku)
        empty_cells = self._find_empty_cells()
        if empty_cells is None:
            return sudoku
        else:
            cells_to_fill = len(empty_cells)

        solved = False
        i_empty_cell = 0
        while not solved:
            # Operar sobre el nodo que toca en este piso del arbol
            row_coord, col_coord = empty_cells[i_empty_cell]
            # Prueba cada número del 1 al 9
            next_number_test = sudoku.get_cell(row_coord, col_coord) + 1
            if next_number_test != 1:
                # Si la celda ya tiene un número, lo borramos porque hemos subido desde un nodo hijo cerrado
                sudoku.clear_cell(row_coord, col_coord)
            for num in range(next_number_test, 10):
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
        return sudoku

    def solve_profundidad(self):
        # Tema 3, diapositiva 42
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [sudoku]
        closed_nodes = []
        while open_nodes:
            current = open_nodes.pop()  # 1.
            closed_nodes.append(current)  # 2.
            if current.is_solved():
                return current  # 3.
            else:  # 4.
                succesors = current.get_successors()  # 4.1
                open_nodes.extend(succesors)  # 4.2
        raise Exception('No solution found')

    def solve_anchura(self):
        # Tema 3, diapositiva 25
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [sudoku]
        closed_nodes = []
        while open_nodes:
            current = open_nodes.pop(0)
            closed_nodes.append(current)
            if current.is_solved():
                return current
            else:
                succesors = current.get_successors()
                open_nodes.extend(succesors)
        raise Exception('No solution found')
