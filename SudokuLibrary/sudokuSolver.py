import bisect
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

    def solve_profundidad(self, measure=False):
        # Tema 3, diapositiva 42
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [sudoku]
        closed_nodes = []
        ciclos = 1
        while open_nodes:
            ciclos += 1
            current = open_nodes.pop()  # 1.
            closed_nodes.append(current)  # 2.
            if current.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
                return current  # 3.
            else:  # 4.
                succesors = current.get_successors()  # 4.1
                open_nodes.extend(succesors)  # 4.2
        raise Exception('No solution found')

    def solve_anchura(self, measure=False):
        # Tema 3, diapositiva 25
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [sudoku]
        closed_nodes = []
        ciclos = 1
        while open_nodes:
            ciclos += 1
            current = open_nodes.pop(0)  # 1.
            closed_nodes.append(current)  # 2.
            if current.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
                return current  # 3.
            else:
                succesors = current.get_successors()  # 4.
                open_nodes.extend(succesors)  # 5.
        raise Exception('No solution found')

    def solve_coste_uniforme(self, measure=False):
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [(sudoku, 0)]  # (tablero, coste)
        closed_nodes = []
        ciclos = 1
        while open_nodes:
            ciclos += 1
            current, cost = open_nodes.pop(0)  # 1.
            closed_nodes.append((current, cost))  # 2.
            if current.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
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

    def solve_avara(self, measure=False):
        # Tema 4, diapositiva 26
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [(sudoku, sudoku.heuristic())]
        closed_nodes = []
        ciclos = 1
        while open_nodes:
            ciclos += 1
            current, heuristic = open_nodes.pop(0)  # 1.
            closed_nodes.append((current, heuristic))  # 2.
            if current.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
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

    def solve_a_estrella(self, measure=False):
        # Tema 4, diapositiva 45
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [(sudoku, 0, 0)]  # (tablero, coste, heurística)
        closed_nodes = []
        ciclos = 1
        while open_nodes:
            ciclos += 1
            current, cost, heuristic = open_nodes.pop(0)  # 1.
            closed_nodes.append((current, cost, heuristic))  # 2.
            if current.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
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

    def solve_restricciones(self, measure=False):
        # Pseudocódigo de elaboración propia
        sudoku = copy.deepcopy(self.sudoku)
        solved = False
        ciclos = 1
        while not solved:
            ciclos += 1
            sudoku_before = copy.deepcopy(sudoku)
            self._restrictions(sudoku)
            if sudoku.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
                solved = True
            elif np.array_equal(sudoku_before.board, sudoku.board):
                print(f'Ciclos: {ciclos}')
                print(sudoku.board_string())
                raise Exception('No más avances posibles')

    def _restrictions(self, sudoku: Sudoku):
        # Última celda restante
        valids_cells = np.sum(sudoku.board_valids, axis=2)
        empty_cells = np.argwhere(valids_cells > 1)
        for cell in empty_cells:
            row, col = cell
            if row == 0 and col == 1:
                pass
            cell_valids = sudoku.board_valids[row, col]
            # Suma los válidos de cada número por fila, columna y cuadrante
            row_valids = sudoku.board_valids[row, :, :].sum(axis=0)
            col_valids = sudoku.board_valids[:, col, :].sum(axis=0)
            start_row = row - row % 3
            start_col = col - col % 3
            square_valids = sudoku.board_valids[start_row:start_row + 3, start_col:start_col + 3, :]
            square_valids = np.sum(np.sum(square_valids, axis=0), axis=0)
            # Multiplico las sumas de validos por los validos de la celda candidata
            cellxrow = np.multiply(row_valids, cell_valids)
            cellxcol = np.multiply(col_valids, cell_valids)
            cellxsqr = np.multiply(square_valids, cell_valids)
            # Resultado será 0, no era válido, o la suma. Si es 1, era el único válido del sector y cumple restricción
            for cellxsector in [cellxrow, cellxcol, cellxsqr]:
                if np.any(cellxsector == 1):
                    num = np.argwhere(cellxsector == 1)[0][0] + 1
                    sudoku.fill_cell(row, col, num)
                    break
        return sudoku
