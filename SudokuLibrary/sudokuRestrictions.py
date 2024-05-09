import numpy as np
from itertools import product, combinations


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
    for icoord in range(coords.shape[0]):
        x, y, _ = coords[icoord]
        if np.sum(valids[icoord]) == 1:
            board[x, y] = np.argmax(valids[icoord]) + 1
    return board


class Sudoku:
    def __init__(self, board: np.ndarray = np.zeros((9, 9), dtype=int)):
        self.board = board
        self.coords, self.valids = _board2representation(board)

    def basic_restrictions(self):
        """
        Restricciones básicas de un sudoku: filas, columnas y cuadrantes
        Descubre números por última celda posible de cada número
        """
        change_ocurred = False
        num_coords = self.coords.shape[0]
        for icoord in range(num_coords):
            coord_eval = self.coords[icoord]
            valid_eval = self.valids[icoord]
            if np.sum(valid_eval) == 1:  # Anula el num en todas las colisiones
                icollisions = np.argwhere(np.any(self.coords == coord_eval, axis=1)).flatten()
                icollisions = [elem for elem in icollisions if not np.array_equal(elem, icoord)]
                remove_possible = np.logical_not(valid_eval)
                pre_change = self.valids.copy()
                pre_board = self.board.copy()
                self.valids[icollisions] = np.logical_and(self.valids[icollisions], remove_possible)
                if not change_ocurred:
                    change_ocurred = not np.array_equal(pre_change, self.valids)
        # Update board
        self.board = _representation2board(self.coords, self.valids)
        print(np.where(self.board != pre_board))
        print('asignado nums ', self.board[np.where(self.board != pre_board)])
        return change_ocurred

    def obvious_pairs(self):
        """
        Restricciones de pares obvios
        """
        change_ocurred = False
        # Se aplica por sectores: filas, columnas y cuadrantes
        num_sectors = self.coords.shape[1]
        for sector in range(num_sectors):
            for sector_num in range(9):
                sector_coords = self.coords[self.coords[:, sector] == sector_num]
                sector_valids = self.valids[self.coords[:, sector] == sector_num]
                # Suma de validos por candidato
                sum_valids_per_cand = np.sum(sector_valids, axis=1)
                sector_candidates = np.argwhere(sum_valids_per_cand == 2).flatten()
                pair_candidates = list(combinations(sector_candidates, 2))
                for pair in pair_candidates:
                    pair_valids = sector_valids[list(pair)]
                    if np.array_equal(pair_valids[0], pair_valids[1]):
                        # Seleccionar celdas menos las del par
                        condition = np.logical_not(  # Aquellos que no sean parte del par
                                    np.all(sector_valids == pair_valids[0], axis=1)
                        )
                        no_pair_sector_coords = sector_coords[condition]
                        no_pair_sector_valids = sector_valids[condition]
                        # Elimina los números del par de las celdas restantes multiplicando por la negación
                        no_pair_sector_valids = np.logical_and(no_pair_sector_valids, np.logical_not(pair_valids[0]))
                        pre_change = self.valids.copy()
                        # Actualiza las celdas
                        index_valids = np.where(np.all(self.coords[:, None, :] == no_pair_sector_coords, axis=2))
                        self.valids[index_valids[0]] = no_pair_sector_valids
                        if not np.array_equal(pre_change, self.valids):
                            print(f"Par encontrado en sector {sector} con valor {sector_num}")
                            print(f"Pareja: {pair}")
                            print(f"valids: {pair_valids[0]}")
                        if not change_ocurred:
                            change_ocurred = not np.array_equal(pre_change, self.valids)
        # Update board
        self.board = _representation2board(self.coords, self.valids)
        return change_ocurred

    def hidden_pairs(self):
        """
        Restricciones de pares ocultos
        """
        change_ocurred = False
        # Se aplica por sectores: filas, columnas y cuadrantes
        num_sectors = self.coords.shape[1]
        for sector in range(num_sectors):
            for sector_num in range(9):
                sector_coords = self.coords[self.coords[:, sector] == sector_num]
                sector_valids = self.valids[self.coords[:, sector] == sector_num]
                # Suma de validos por número [1-9]
                sum_valids_per_num = np.sum(sector_valids, axis=0)
                possible_hiddens = np.argwhere(sum_valids_per_num > 2).flatten()
                if len(possible_hiddens) > 1:
                    # Combinaciones de 2 números posibles ocultos
                    pair_candidates = list(combinations(possible_hiddens, 2))
                    for pair in pair_candidates:
                        # Seleccionamos todos los valids[pair]
                        pair_valids = sector_valids[:, list(pair)]
                        if np.all(np.sum(pair_valids, axis=0) == 2):
                            sector_coords_to_change = sector_coords[np.argwhere(np.all(pair_valids, axis=1)).flatten()]
                            change_coords = np.where(
                                np.all(
                                    np.any(
                                        self.coords[:, None] == sector_coords_to_change, axis=1), axis=1))[0]
                            new_valids = np.zeros(9, dtype=bool)
                            new_valids[list(pair)] = True
                            pre_change = self.valids.copy()
                            np.logical_and(self.valids[change_coords], new_valids)
                            if not change_ocurred:
                                change_ocurred = not np.array_equal(pre_change, self.valids)
        # Update board
        self.board = _representation2board(self.coords, self.valids)
        return change_ocurred
