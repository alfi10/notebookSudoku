from colorama import Fore, Style


# Imprimir el mejor movimiento
def imprimir_sudoku_con_resaltado(sudoku, mejor_movimiento):
    for i, fila in enumerate(sudoku):
        for j, valor in enumerate(fila):
            if mejor_movimiento and i == mejor_movimiento.x and j == mejor_movimiento.y:
                print(f"{Fore.RED}{valor}{Style.RESET_ALL}", end=" ")  # Resaltar en rojo el mejor movimiento
            else:
                print(valor, end=" ")
        print()


# Funcion de conteo de errores
def contar_errores(estado):
    tablero = estado.SUDOKU
    errores = 0
    for i in range(9):
        for j in range(9):
            num = tablero[i][j]
            if num != 0 and not es_movimiento_valido(tablero, i, j, num):
                errores += 1
    return errores


# Funcion de comprobacion de filas y columnas
def completar_filas_columnas_subgrupos(tablero):
    completados = 0
    for i in range(9):
        fila = set(tablero[i])
        columna = set(tablero[j][i] for j in range(9))
        if len(fila) == 9:
            completados += 1
        if len(columna) == 9:
            completados += 1
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            subgrupo = set()
            for x in range(3):
                for y in range(3):
                    subgrupo.add(tablero[i+x][j+y])
            if len(subgrupo) == 9:
                completados += 1
    return completados


# Funcion para evaluar el estado
def evaluar_estado(estado):
    puntuacion = 0
    tablero = estado.SUDOKU

    # Contar cuántos números se han colocado en el tablero
    for fila in tablero:
        for num in fila:
            if num != 0:
                puntuacion += 1

    # Verificar filas, columnas y subgrupos completados
    puntuacion += completar_filas_columnas_subgrupos(tablero) * 3

    # Verificar si el Sudoku está completo y asignar puntos adicionales
    if es_tablero_lleno(tablero) and es_tablero_valido(tablero):
        puntuacion += 5

    # Penalizar errores en la colocación de números
    penalizacion = contar_errores(estado)
    puntuacion -= penalizacion * 2

    return puntuacion


# Funcion para comprobar si es un movimiento valido
def es_movimiento_valido(tablero, fila, columna, num):
    # Verificar fila
    if num in tablero[fila]:
        return False

    # Verificar columna
    if num in [tablero[i][columna] for i in range(9)]:
        return False

    # Verificar subcuadro
    subcuadro_fila = fila // 3
    subcuadro_columna = columna // 3
    for i in range(subcuadro_fila * 3, (subcuadro_fila + 1) * 3):
        for j in range(subcuadro_columna * 3, (subcuadro_columna + 1) * 3):
            if tablero[i][j] == num:
                return False

    return True


# Funcion para generar hijos
def generar_hijos(estado):
    # Declaramos una lista de hijos vacia
    hijos = []
    # Recorremos el tablero en cada celda
    for i in range(9):
        for j in range(9):
            # Comprobamos que la celda elegida esté vacia
            if estado.SUDOKU[i][j] == 0:
                # Recorremos todos los numeros posibles [1,9]
                for num in range(1, 10):
                    # Comprobamos si es un movimiento valido
                    if es_movimiento_valido(estado.SUDOKU, i, j, num):
                        # Agregamos el hijo a la lista
                        nuevo_tablero = [fila[:] for fila in estado.SUDOKU]
                        nuevo_tablero[i][j] = num
                        nuevo_estado = Estado(nuevo_tablero, i, j, f'Poner en X={i} | Y={j} | num={num}',
                                              estado.profundidad + 1, estado)
                        hijos.append(nuevo_estado)
    # Agregamos el ultimo estado que es, no hacer nada (pasar turno)
    nuevo_estado = Estado(estado.SUDOKU, estado.x, estado.y, 'Pasar', estado.profundidad + 1, estado.padre)
    hijos.append(nuevo_estado)

    return hijos


# Funcion para comprobar si el tablero esta lleno
def es_tablero_lleno(tablero):
    # Recorremos las filas del tablero
    for fila in tablero:
        # Comprobamos que haya celdas vacias, (0)
        if 0 in fila:
            # Devuelve False (no esta lleno)
            return False
    return True


# Funcion MINIMAX
def minimax(estado, profundidad, jugador_maximizador):
    # Comprobamos que la profundidad es la establecida por nosotros o bien el tablero este lleno
    if profundidad == 0 or es_tablero_lleno(estado.SUDOKU):
        # Evaluamos el tablero, no devolvemos mejor movimiento pues debemos llevar el peso del estado a los padres
        return evaluar_estado(estado), None

    # Comprobamos si el estado es del jugador MAX o MIN
    if jugador_maximizador:
        # Inicializamos el menor de los valores en "-inf" el valor más bajo que puede ser
        mejor_valor = float('-inf')
        # Inicializamos el mejor movimiento en None
        mejor_movimiento = None
        # Recorremos todos los hijos posibles 1 a 1
        for hijo in generar_hijos(estado):
            # Por cada hijo volvemos a la funcion MINIMAX restando la profundidad e indicando que ahora se trata del jugador MIN
            valor, _ = minimax(hijo, profundidad - 1, False)
            # Si el valor del hijo es mayor que el ya existente
            if valor > mejor_valor:
                # Actualizamos los valores por los nuevos
                mejor_valor = valor
                mejor_movimiento = hijo
        # Devolvemos los mejores valores
        return mejor_valor, mejor_movimiento
    else:
        # Inicializamos el menor de los valores en "-inf" el valor más bajo que puede ser
        mejor_valor = float('inf')
        # Inicializamos el mejor movimiento en None
        mejor_movimiento = None
        # Recorremos todos los hijos posibles 1 a 1
        for hijo in generar_hijos(estado):
            # Por cada hijo volvemos a la funcion MINIMAX restando la profundidad e indicando que ahora se trata del jugador MAX
            valor, _ = minimax(hijo, profundidad - 1, True)
            # Si el valor del hijo es menor que el ya existente
            if valor < mejor_valor:
                # Actualizamos los valores por los nuevos
                mejor_valor = valor
                mejor_movimiento = hijo
        # Devolvemos los mejores valores
        return mejor_valor, mejor_movimiento


# Definimos una clase para almacenar un estado
class Estado:

    # Definimos un constructor para la clase
    # PARAMETROS:
    # SUDOKU => Tablero
    # x => Posicion horizontal de la casilla para introducir numeros
    # y => Posicion vertical de la casilla para introducir numeros
    # padre => "Estado" padre de ese "Estado" (puede estar vacio en el caso del nodo raiz)
    def __init__(self, SUDOKU, x, y, operador, profundidad, padre=None):
        self.padre = padre
        self.SUDOKU = SUDOKU
        self.x = x
        self.y = y
        self.hijos = []
        self.operador = operador
        self.coste = 0
        self.jugador = 0
        self.profundidad = profundidad

    # Funcion para agregar hijos al nodo
    def agregar_hijo(self, hijo):
        self.hijos.append(hijo)

    def agregar_coste(self, coste):
        self.coste = coste

    def agregar_heuristica(self, heuristica):
        self.heuristica = heuristica

    def asignar_jugador(self, jugador):
        self.jugador = jugador