import bisect
import copy
import numpy as np
from .sudoku import Sudoku
from .sudokuRestrictions import Sudoku as SudokuRestrictions


def show_sudoku(sudoku: Sudoku):
    print(sudoku)


class SudokuSolver:
    def __init__(self, sudoku: Sudoku):
        self.sudoku = copy.deepcopy(sudoku)

    def solve_profundidad(self, measure=False):
        # Tema 3, diapositiva 42
        sudoku = copy.deepcopy(self.sudoku)
        open_nodes = [sudoku]
        closed_nodes = []
        ciclos = 0
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
        ciclos = 0
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
        ciclos = 0
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
        ciclos = 0
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
        ciclos = 0
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
        sudoku_restricted = SudokuRestrictions(sudoku.board)
        solved = False
        ciclos = 0
        while not solved:
            pre_change = ciclos
            while sudoku_restricted.basic_restrictions():
                ciclos += 1
            while sudoku_restricted.obvious_pairs():
                ciclos += 1
            while sudoku_restricted.hidden_pairs():
                ciclos += 1
            if sudoku_restricted.is_solved():
                if measure:
                    print(f'Ciclos: {ciclos}')
                solved = True
            if pre_change == ciclos:
                print(ciclos, ' ciclos de restricciones aplicadas')
                print('Estado alcanzado:')
        return sudoku_restricted
