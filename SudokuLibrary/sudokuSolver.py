import bisect
import copy
import numpy as np
import time
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
            current = open_nodes.pop(0)  # 1.
            closed_nodes.append(current)  # 2.
            if current.is_solved():
                return current  # 3.
            else:
                succesors = current.get_successors()  # 4.
                open_nodes.extend(succesors)  # 5.
        raise Exception('No solution found')

    def solve_coste_uniforme(self):
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [(sudoku, 0)]  # (tablero, coste)
        closed_nodes = []
        while open_nodes:
            current, cost = open_nodes.pop(0)  # 1.
            closed_nodes.append((current, cost))  # 2.
            if current.is_solved():
                return current  # 3.
            else:
                succesors = current.get_successors(cost)  # 4.
                for successor in succesors:
                    candidate_sudoku, candidate_cost = successor
                    closed_successor_nodes = [node for node in closed_nodes if node[0] == candidate_sudoku]
                    candidate_cost_less_than_closed_costs = np.all(
                        [candidate_cost < node[1] for node in closed_successor_nodes]
                    )
                    if closed_successor_nodes and not candidate_cost_less_than_closed_costs:
                        continue  # 5.
                    # Else se insertará ordenadamente al final del bloque

                    iopen, open_successor_node = next(
                        (
                            (index, node) for index, node in enumerate(open_nodes)
                            if node[0] == candidate_sudoku
                        ), (None, None)
                    )
                    if open_successor_node is not None:
                        candidate_cost_less_than_open_cost = candidate_cost < open_successor_node[1]
                        if not candidate_cost_less_than_open_cost:
                            continue
                        open_nodes.pop(iopen)  # 6.
                        # Se inserta ordenadamente al final del bloque

                    bisect.insort(open_nodes, successor, key=lambda x: x[1])  # 7.
        raise Exception('No solution found')

    def solve_avara(self):
        # Tema 4, diapositiva 26
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [(sudoku, sudoku.heuristic())]
        closed_nodes = []
        while open_nodes:
            current, heuristic = open_nodes.pop(0)  # 1.
            closed_nodes.append((current, heuristic))  # 2.
            if current.is_solved():
                return current  # 3.
            else:
                succesors = current.get_successors()  # 4.
                for successor in succesors:
                    candidate_sudoku = successor
                    candidate_heuristic = candidate_sudoku.heuristic()
                    candidate_in_open = np.any([node[0] == candidate_sudoku for node in open_nodes])
                    candidate_in_closed = np.any([node[0] == candidate_sudoku for node in closed_nodes])
                    if not (candidate_in_open or candidate_in_closed):
                        bisect.insort(open_nodes, (candidate_sudoku, candidate_heuristic), key=lambda x: x[1])
                    # Else se descarta el nodo no insertándolo en la lista de nodos abiertos
        raise Exception('No solution found')

    def solve_a_estrella(self):
        # Tema 4, diapositiva 45
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [(sudoku, 0, 0)]  # (tablero, coste, heurística)
        closed_nodes = []
        while open_nodes:
            current, cost, heuristic = open_nodes.pop(0)  # 1.
            closed_nodes.append((current, cost, heuristic))  # 2.
            if current.is_solved():
                return current  # 3.
            else:
                succesors = current.get_successors(cost)  # 4.
                for successor in succesors:
                    candidate_sudoku, candidate_cost = successor
                    candidate_heuristic = candidate_sudoku.heuristic()

                    # Si coste de successor es menor que el coste de la lista de cerrados, insertar en abiertos
                    closed_successor_nodes = [node for node in closed_nodes if node[0] == candidate_sudoku]
                    candidate_cost_less_than_closed_costs = np.all(
                        [candidate_cost < node[1] for node in closed_successor_nodes]
                    )
                    if closed_successor_nodes and not candidate_cost_less_than_closed_costs:
                        continue  # 4.1
                    # Else se insertará ordenadamente al final del bloque

                    open_successor_node = next((node for node in open_nodes if node[0] == candidate_sudoku), None)
                    if open_successor_node is not None:
                        candidate_cost_less_than_open_cost = candidate_cost < open_successor_node[1]
                        if not candidate_cost_less_than_open_cost:
                            continue
                        open_nodes.remove(open_successor_node)  # 4.2
                        # Se inserta ordenadamente al final del bloque

                    bisect.insort(open_nodes,
                                  (candidate_sudoku, candidate_cost, candidate_heuristic),
                                  key=lambda x: x[1] + x[2]
                                  )  # 4.3
        raise Exception('No solution found')
