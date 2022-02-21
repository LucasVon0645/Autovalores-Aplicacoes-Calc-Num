import numpy as np

# Constantes para a função powerMethod
itmax = 70
epsilon = 10**(-15)

# Funções auxiliares

'''
O método a seguir é responsável por gerar uma matriz
de dimensões x por y com elementos aleatórios entre
0 e 1 por meio da semente 'seed'.
'''
def generateRandomB(x,y, seed):
    np.random.seed(seed)
    array = np.random.rand(x,y)
    return array

'''
O método a seguir é responsável por gerar um vetor
de dimensão x por 1 com elementos aleatórios entre
0 e 1 por meio da semente 'seed'.
'''
def generateRandomX0(x, seed):
    np.random.seed(seed)
    array = np.random.rand(x,1)
    return array

'''
O método a seguir é responsável por gerar uma matriz
de dimensões x por y com elementos aleatórios entre
0 e 1.
'''
def generateRandomArray(x, y):
    array = np.random.rand(x, y)
    return array

'''
A função a seguir retorna o indíce do maior autovalor de uma
lista que contém os autovalores de uma matriz.
'''
def findTheIndexOfHighestEigValue(array):
    array_list = array.tolist()
    max_value = max(array_list)
    return array_list.index(max_value)

'''
A função a seguir retorna o indíce do segundo maior autovalor de uma
lista que contém os autovalores de uma matriz.
'''
def findTheIndexOfSecondHighestEigValue(array):
    index_highest = findTheIndexOfHighestEigValue(array)
    array_list = array.tolist()
    array_list_copy = array_list.copy()
    array_list_copy.pop(index_highest)
    secondHighest = max(array_list_copy)
    return array_list.index(secondHighest)

'''
A rotina a seguir aplica o critério de Sassenfeld para uma matriz
A de ordem n. Ela retorna o parâmetro M do critério.
'''
def sassenfeldCriterion(A, n):
    sumFirstLine = 0
    for j in range(1,n):
        sumFirstLine = sumFirstLine + np.abs(A[0][j])
    beta_0 = (1/A[0][0])*(sumFirstLine)
    betaArray = [beta_0]
    for i in range(1,n):
        sumBeforeI = 0
        for j in range(i):
            beta_j = betaArray[j]
            sumBeforeI = sumBeforeI + np.abs(A[i][j])*beta_j
        sumAfterI = 0
        for j in range(i+1, n):
            sumAfterI = sumAfterI + np.abs(A[i][j])
        beta_i = (1/A[i][i])*(sumBeforeI + sumAfterI)
        betaArray.append(beta_i)
    return max(betaArray)

'''
O método a seguir imprime se o critério de Sassenfeld é satisfeito
ou não com base no valor do parâmetro M.
'''
def isSassenFeldCriterionSatisfied (M):
    if (M < 1):
        print("Criterio de Sassenfeld satisfeito: M < 1")
    else:
        print("Criterio de Sassenfeld nao satisfeito: M > 1")

'''
A função a seguir recebe dois vetores quaisquer u e v e retorna o módulo
da maior diferença entre os elementos correspondentes do vetores u e v.
'''
def greatestDifferenceBetweenArrays(u, v):
    n = len(u)
    difference = list(range(n))
    for i in range(n):
        difference[i] = np.abs(u[i] - v[i])
    return max(difference)

'''
A rotina a seguir é utilizada nos exs 1 e 2 para a impressão das condições
iniciais.
'''
def printInitialConditions(B, x0, n, labelB):
    print("- Condicoes iniciais")
    print(labelB+" ("+str(n)+"x"+str(n)+") = ")
    print(B)
    print("\n")
    print("x0 ("+str(n)+"x1) = ")
    print(x0)
    print("\n")

'''
O método a seguir serve para extrair da resposta retornada pelo
método 'np.linalg.eig' os seguintes elementos: o maior autovalor
de uma matriz, o respectivo autovetor associado e o segundo maior
autovalor.
'''
def getEigValuesAndVectors(reference, n):
    eigvalueArray = reference[0]
    indexHighestEigValue = findTheIndexOfHighestEigValue(eigvalueArray)
    indexSecondHighestEigValue = findTheIndexOfSecondHighestEigValue(
        eigvalueArray)
    highestEigValue = reference[0][indexHighestEigValue]
    secondHighestEigValue = reference[0][indexSecondHighestEigValue]
    eigVector = reference[1][:, indexHighestEigValue].reshape(n, 1)
    return (highestEigValue, secondHighestEigValue, eigVector)

'''
A função a seguir retorna o módulo da diferença entre
o autovalor calculado ('eigValue') e o valor de referência
('eigValueRef').
'''
def getEigValueError(eigValue, eigValueRef):
    return np.abs(eigValue - eigValueRef)

'''
A função a seguir retorna a norma do vetor diferença 
entre o autovetor calculado ('eigVector') e o de
refêrencia ('eigVectorRef').
'''
def getEigVectorError(eigVector, eigVectorRef):
    return np.linalg.norm(eigVector - eigVectorRef)

# A função a seguir está neste arquivo, pois é necessária para o exercício 3 e 4

'''
A função a seguir implementa o método da potências para a determinação
do maior autovalor de uma matriz A  e de seu respectivo autovetor 
associado. A função retorna uma tupla composta, nessa ordem, pelos 
seguintes elementos: autovetor, autovalor, quantidade de interações 
realizadas. O cálculo é feito a partir de um vetor inicial dado x0.
'''
def powerMethod(A, x0):
    x_previous = x0
    u = [[]]
    interactions = itmax
    for i in range(itmax):
        n = np.dot(A, x_previous)
        x_next = n/np.linalg.norm(n)
        u = np.dot(x_previous.T, n)/np.dot(x_previous.T, x_previous)
        if (np.linalg.norm(x_next - x_previous) < epsilon):
            interactions = i + 1
            break
        x_previous = x_next
    if (x_next[0][0] < 0 or (x_next[0][0] == 0 and x_next[1][0] < 0)):
            x_next *= -1
    return (x_next, u[0][0], interactions)

# As duas funções a seguir estão neste arquivo, pois são necessárias para o exercício 3

'''
A função a seguir cria a matriz A do exercício 1.1 a partir de uma
certa matriz B.
'''
def createA1array(B):
    return B + B.T

'''
A função a seguir cria a matriz A do exercício 1.2 a partir de uma
certa matriz B e de uma matriz diagonal D.
'''
def createA2array(B, D):
    return np.dot(B, np.dot(D, np.linalg.inv(B)))