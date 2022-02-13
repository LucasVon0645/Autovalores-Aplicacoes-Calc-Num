import numpy as np

# funções auxiliares


def generateRandomArray(x, y):
    array = np.random.rand(x, y)
    return array


def findTheIndexOfHighestEigValue(array):
    array_list = array.tolist()
    max_value = max(array_list)
    return array_list.index(max_value)

def findTheIndexOfSecondHighestEigValue(array):
    index_highest = findTheIndexOfHighestEigValue(array)
    array_list = array.tolist()
    array_list_copy = array_list.copy()
    array_list_copy.pop(index_highest)
    secondHighest = max(array_list_copy)
    return array_list.index(secondHighest)

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

def isSassenFeldCriterionSatisfied (M):
    if (M < 1):
        print("Criterio de Sassenfeld satisfeito: M < 1")
    else:
        print("Criterio de Sassenfeld nao satisfeito: M > 1")

def greatestDifferenceBetweenArrays(u, v):
    n = len(u)
    difference = list(range(n))
    for i in range(n):
        difference[i] = np.abs(u[i] - v[i])
    return max(difference)

def printInitialConditions(B, x0, n, labelB):
    print("- Condicoes iniciais")
    print(labelB+" ("+str(n)+"x"+str(n)+") = ")
    print(B)
    print("\n")
    print("x0 ("+str(n)+"x1) = ")
    print(x0)
    print("\n")


def getEigValuesAndVectors(reference, n):
    eigvalueArray = reference[0]
    indexHighestEigValue = findTheIndexOfHighestEigValue(eigvalueArray)
    indexSecondHighestEigValue = findTheIndexOfSecondHighestEigValue(
        eigvalueArray)
    highestEigValue = reference[0][indexHighestEigValue]
    secondHighestEigValue = reference[0][indexSecondHighestEigValue]
    eigVector = reference[1][:, indexHighestEigValue].reshape(n, 1)
    return (highestEigValue, secondHighestEigValue, eigVector)


def getEigValueError(eigValue, eigValueRef):
    return np.abs(eigValue - eigValueRef)


def getEigVectorError(eigVector, eigVectorRef):
    if (eigVector[0][0]*eigVectorRef[0][0] < 0):
        return np.linalg.norm(eigVector + eigVectorRef)
    return np.linalg.norm(eigVector - eigVectorRef)
