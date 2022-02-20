import numpy as np
import metodos as met

# Constantes
itmax = 10000
epsilon = 10**(-15)

'''
A função a seguir implementa a fatoração QR, a qual transforma a matriz A 
em uma matriz ortogonal Q e uma matriz triangular R por meio de sucessivas 
transformações de Householder, sendo que A = Q.R. A função utiliza o fato 
de que, a cada iteração k do algoritmo, apenas a submatriz onde i>=k e 
j>=k é alterada.
'''
def QRFactorization(A):
    n = len(A)

    R = A.copy()
    Q = np.eye(n)

    for k in range(n-1):
        a = R[k:, k]
        delta = np.sign(a[0])
        e = np.zeros(n-k)
        e[0] += 1
        v = a + delta * np.linalg.norm(a) * e

        H = np.eye(n-k) - 2 * (np.outer(v, v) / np.dot(v, v))

        R[k:, k:] = np.dot(H, R[k:, k:])
        Q[0:, k:] = np.dot(Q[0:, k:], H)
    
    return (Q, R)

'''
A função a seguir verifica o critério do parada do método QR, recebendo a 
matriz Ak da iteração k do método e retornando um booleano: True para quando 
o critério está satisfeito e False caso contrário. O critério consiste em 
verificar se todos os coeficientes de Ak abaixo da diagonal principal são 
menores que um epsilon definido (valor pequeno e perto da precisão de máquina). 
Dessa forma, caso isso seja verdadeiro, Ak é quase triangular superior, no 
sentido de que os coeficientes abaixo da diagonal são pequenos.
'''
def QRMethodStopCriterionIsTrue(Ak):
    n = len(Ak)

    for i in range(n):
        for j in range(i):
            if np.abs(Ak[i][j]) > epsilon:
                return False
    
    return True

'''
A função a seguir implementa o método QR, o qual consiste em utilizar a fatoração 
QR para encontrar os autovalores e autovetores de uma dada matriz. A função recebe 
uma matriz A e retorna a matriz produto das iterações do método e obtida a partir 
de A (A_next), cuja diagonal principal contém uma boa aproximação dos autovalores 
de A; uma matriz que contém em suas colunas uma aproximação para os autovetores de 
A (V_next); e o número de interações até a convergência do método ou o alcance do 
número máximo de iterações definido itmax (k+1 - a adição de uma unidade é feita, 
pois a variável de iterações k possui valor inicial nulo).
'''
def QRMethod(A):
    A_0 = A.copy()
    V_previous = np.eye(len(A))

    for k in range(itmax):
        if k == 0:
            A_next = A_0
        else:
            A_next = np.dot(R_previous, Q_previous)

        Q_next, R_next = QRFactorization(A_next)

        V_next = np.dot(V_previous, Q_next)

        Q_previous = Q_next
        R_previous = R_next
        V_previous = V_next

        if QRMethodStopCriterionIsTrue(A_next):
            break

    return (A_next, V_next, k+1)

'''
A função a seguir serve para extrair os autovalores contidos na diagonal 
principal da matriz U, produto do método QR (denominada A_next na função 
que implementa o método QR).
'''
def extractEigValues(U):
    n = len(U)
    eigValues = np.zeros(n)

    for i in range(n):
        eigValues[i] = U[i][i]

    return eigValues

'''
A função a seguir serve para extrair os autovetores presentes nas 
colunas da matriz V, produto do método QR (denominada V_next na 
função que implementa o método QR).
'''
def extractNormalizedEigVectors(V):
    n = len(V)
    eigVectors = np.zeros((n, n))

    for i in range(n):
        # normalização dos autovetores
        eigVector = V[:, i] / np.linalg.norm(V[:, i])
        
        # o seguinte trecho de código torna positivo o primeiro valor dos 
        # autovetores ou o segundo valor, caso o primeiro seja nulo
        if (eigVector[0] < 0 or (eigVector[0] == 0 and eigVector[1] < 0)):
            eigVector *= -1
        
        eigVectors[:, i] = eigVector

    return eigVectors

'''
A função a seguir recebe os autovalores e respectivos autovetores de uma 
matriz e os retorna ordenados em ordem decrescente dos autovalores (o 
autovalor em eigValues[i] está associado ao autovetor eigVectors[:, i], 
tanto na entrada como na saída).
'''
def sortEigValuesAndVectors(eigValues, eigVectors):
    idx = eigValues.argsort()[::-1]   

    return(eigValues[idx], eigVectors[:,idx])

'''
A função a seguir recebe os autovalores de uma matriz e os retorna 
ordenados em ordem decrescente dos autovalores
'''
def sortEigValues(eigValues):
    eigValues = np.sort(eigValues)  

    return(eigValues)

'''
A função a seguir imprime os autovalores e respectivos autovetores 
recebidos como entrada.
'''
def printEigValuesAndVectors(eigValues, eigVectors):
    for i in range(len(eigValues)):
        print("\nAutovalor " + str(i+1) + ": " + str(eigValues[i]))
        print("Autovetor " + str(i+1) + " associado ao autovalor " + str(i+1) + ": "
         + str(eigVectors[:, i]))

'''
A função a seguir imprime os autovalores recebidos como entrada.
'''
def printEigValues(eigValues):
    for i in range(len(eigValues)):
        print("\nAutovalor " + str(i+1) + ": " + str(eigValues[i]))

'''
A função a seguir imprime a comparação entre os valores encontrados 
e os valores de referência dos autovalores e respectivos autovetores 
de uma matriz a partir da apresentação do erro entre eles.
'''
def printComparison(eigValuesRef, eigValues, eigVectorsRef, eigVectors):
    print("\n(+) Erros:")

    for i in range(len(eigValues)):
        eigValueError = met.getEigValueError(eigValues[i], eigValuesRef[i])
        eigVectorError = met.getEigVectorError(eigVectors[:, i], eigVectorsRef[:, i])

        print("\nErro entre o autovalor " + str(i+1) + " calculado e o de referência: "
         + str(eigValueError))
        print("Erro entre o autovetor " + str(i+1) + " calculado e o de referência: "
         + str(eigVectorError))
         
'''
A função a seguir imprime a comparação entre os valores encontrados 
e os valores de referência dos autovalores de uma matriz a partir 
da apresentação do erro entre eles.
'''
def printComparisonEigValues(eigValuesRef, eigValues):
    print("\n(+) Erros:")

    for i in range(len(eigValues)):
        eigValueError = met.getEigValueError(eigValues[i], eigValuesRef[i])

        print("\nErro entre o autovalor " + str(i+1) + " calculado e o de referência: "
         + str(eigValueError))

'''
A função a seguir imprime a comparação entre os valores encontrados 
para o autovalor dominante de uma matriz por meio de diferentes 
métodos (l1 e l2) a partir da apresentação do erro entre eles.
'''
def printComparisonDominantEigValue(l1, l2):
    eigValueError = met.getEigValueError(l1, l2)

    print("\n(+) Erro entre os autovalores dominates encontrados: " + str(eigValueError))

'''
A função a seguir imprime a matriz produto do método QR, que contém uma 
boa aproximação para os autovalores em sua diagonal principal, e o 
número de iterações do método QR
'''
def printAfinalAndNumberOfIterations(A_final, k):
    print("\nA_final = ")
    print(A_final)
    print("\nNúmero de iterações k = " + str(k))

# Programa principal

print("******************************************************")
print("Exercício 3.1")

# definição e apresentação da matriz A utilizada para o exercício 3.1
A = np.array([[6.0, -2.0, -1.0], [-2.0, 6.0, -1.0], [-1.0, -1.0, 5.0]])
print("\nMatriz A utilizada no exercício 3.1:")
print(A)

# apresentação dos valores de referência dos autovalores e autovetores de A
print("\n(+) Referência:")

eigValuesRef, eigVectorsRef = np.linalg.eig(A)
eigValuesRef, eigVectorsRef = sortEigValuesAndVectors(eigValuesRef, eigVectorsRef)

# o seguinte trecho de código torna positivo o primeiro valor dos autovetores 
# ou o segundo valor, caso o primeiro seja nulo
for i in range(len(eigVectorsRef)):
    if (eigVectorsRef[0][i] < 0 or (eigVectorsRef[0][i] == 0 and eigVectorsRef[1][i] < 0)): 
        eigVectorsRef[:, i] *= -1

printEigValuesAndVectors(eigValuesRef, eigVectorsRef)

# cálculo dos autovalores e autovetores de A por meio do método QR
print("\n(+) Calculado:")

A_final, V_final, k = QRMethod(A)

printAfinalAndNumberOfIterations(A_final, k)

eigValues = extractEigValues(A_final)
eigVectors = extractNormalizedEigVectors(V_final)

eigValues, eigVectors = sortEigValuesAndVectors(eigValues, eigVectors)

printEigValuesAndVectors(eigValues, eigVectors)

# comparação pela apresentação dos erros entre os valores calculados 
# e os de referência para os autovalores e autovetores de A
printComparison(eigValuesRef, eigValues, eigVectorsRef, eigVectors)

print("\n******************************************************")
print("Exercício 3.2")

# definição e apresentação da matriz A utilizada para o exercício 3.2
A = np.array([[1.0, 1.0], [-3.0, 1.0]])
print("\nMatriz A utilizada no exercício 3.2:")
print(A)

# apresentação dos valores de referência dos autovalores de A
print("\n(+) Referência:")

eigValuesRef = np.linalg.eigvals(A)
eigValuesRef = sortEigValues(eigValuesRef)

printEigValues(eigValuesRef)

# cálculo dos autovalores de A por meio do método QR
print("\n(+) Calculado:")

A_final, V_final, k = QRMethod(A)

printAfinalAndNumberOfIterations(A_final, k)

eigValues = extractEigValues(A_final)
eigValues = sortEigValues(eigValues)

printEigValues(eigValues)

# comparação pela apresentação dos erros entre os valores 
# calculados e os de referência para os autovalores de A
printComparisonEigValues(eigValuesRef, eigValues)

print("\n******************************************************")
print("Exercício 3.3")

# definição e apresentação da matriz A utilizada para o exercício 3.3
A = np.array([[3.0, -3.0], [0.33333, 5.0]])
print("\nMatriz A utilizada no exercício 3.3:")
print(A)

# apresentação dos valores de referência dos autovalores de A
print("\n(+) Referência:")

eigValuesRef = np.linalg.eigvals(A)
eigValuesRef = sortEigValues(eigValuesRef)

printEigValues(eigValuesRef)

# cálculo dos autovalores de A por meio do método QR
print("\n(+) Calculado:")

A_final, V_final, k = QRMethod(A)

printAfinalAndNumberOfIterations(A_final, k)

eigValues = extractEigValues(A_final)
eigValues = sortEigValues(eigValues)

printEigValues(eigValues)

# comparação pela apresentação dos erros entre os valores 
# calculados e os de referência para os autovalores de A
printComparisonEigValues(eigValuesRef, eigValues)

print("\n******************************************************")
print("Exercício 3.4")

# resolução do exercício para a matriz A1 do exercício 1.1
print("\n---> Matriz A1 do Exercício 1.1:")

# geração da mesma matriz a partir da utilização das mesmas funções 
# do exercício 1
B = met.generateRandomB(10, 10, 2021) # usando a mesma seed do exercício 1 para comparação
x0 = met.generateRandomX0(10, 2022)

A1 = met.createA1array(B)

# determinação do autovalor dominante de A1 pelo método das potências, 
# desenvolvido no exercício 1
print("\n(+) Método das Potências:")

results = met.powerMethod(A1, x0)
lambda1_A1_P = results[1]

print("\nAutovalor dominante de A1 obtido pelo método das potências: " + str(lambda1_A1_P))

# determinação do autovalor dominante de A1 pelo método QR
print("\n(+) Método QR:")

A1_final, V1_final, k1 = QRMethod(A1)

printAfinalAndNumberOfIterations(A1_final, k1)

eigValues1 = extractEigValues(A1_final)

lambda1_A1_QR = max(eigValues1)

print("\nAutovalor dominante de A1 obtido pelo método QR: " + str(lambda1_A1_QR))

printComparisonDominantEigValue(lambda1_A1_P, lambda1_A1_QR)

# resolução do exercício para a matriz A2 do exercício 1.2
print("\n---> Matriz A2 do Exercício 1.2:")

# geração da mesma matriz a partir da utilização das mesmas funções 
# do exercício 1
B = met.generateRandomB(5, 5, 2021) # usando a mesma seed do exercício 1 para comparação
x0 = met.generateRandomX0(5, 2022)
D = np.array([[1.5, 0, 0,   0,      0],
              [0,   1, 0,   0,      0], 
              [0,   0, 0.5, 0,      0],
              [0,   0, 0,   0.2,    0], 
              [0,   0, 0,   0,   0.05]])

A2 = met.createA2array(B, D)

# determinação do autovalor dominante de A2 pelo método das potências, 
# desenvolvido no exercício 1
print("\n(+) Método das Potências:")

results = met.powerMethod(A2, x0)
lambda1_A2_P = results[1]

print("\nAutovalor dominante de A2 obtido pelo método das potências: " + str(lambda1_A2_P))

# determinação do autovalor dominante de A1 pelo método QR
print("\n(+) Método QR:")

A2_final, V2_final, k2 = QRMethod(A2)

printAfinalAndNumberOfIterations(A2_final, k2)

eigValues2 = extractEigValues(A2_final)

lambda1_A2_QR = max(eigValues2)

print("\nAutovalor dominante de A2 obtido pelo método QR: " + str(lambda1_A2_QR))

printComparisonDominantEigValue(lambda1_A2_P, lambda1_A2_QR)