import numpy as np

# funções auxiliares

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

def imprime(matrix):
    for i in range(len(matrix)):
        string = ""
        for j in range(len(matrix[i])):
            string = string + str(round(matrix[i][j], 3)) + " & "
        string = string + "\\"
        string = string.replace(".", ",")
        print(string)

def imprime2(matrix):
    string = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            string = string + str(round(matrix[i][j], 8)) + " & "
        string = string.replace(".", ",")
    print(string)