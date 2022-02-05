import numpy as np
import metodos as met
import matplotlib.pyplot as plt

# Constantes
itmax = 70
epsilon = 10**(-15)


def powerMethod(A, x0):
    u_0 = np.dot(x0.T, np.dot(A, x0))/np.dot(x0.T, x0)
    x_previous = x0
    u = u_0
    interactions = itmax
    for i in range(itmax):
        n = np.dot(A, x_previous)
        x_next = n/np.linalg.norm(n)
        u = np.dot(x_next.T, np.dot(A, x_next))/np.dot(x_next.T, x_next)
        if (np.linalg.norm(x_next - x_previous) < epsilon):
            interactions = i + 1
            break
        x_previous = x_next
    return (x_next, u[0][0], interactions)

def powerMethodInteractions(A, x0, lambda1, lambda2):
    u_0 = np.dot(x0.T, np.dot(A, x0))/np.dot(x0.T, x0)
    x_previous = x0
    u = u_0
    resultEigenVector = []
    resultEigenValue = []
    assintoticErrorK = np.abs(lambda2/lambda1)
    squaredAssintoticErrorK = (lambda2/lambda1)**2
    assintoticErrorArray = [assintoticErrorK]
    squaredAssintoticErrorArray = [squaredAssintoticErrorK]
    interactions = itmax
    for i in range(itmax):
        if ( i != 0):
            assintoticErrorK = assintoticErrorK*(np.abs(lambda2/lambda1))
            squaredAssintoticErrorK = squaredAssintoticErrorK*((lambda2/lambda1)**2)
            assintoticErrorArray.append(assintoticErrorK)
            squaredAssintoticErrorArray.append(squaredAssintoticErrorK)
        n = np.dot(A, x_previous)
        x_next = n/np.linalg.norm(n)
        u = np.dot(x_next.T, np.dot(A, x_next))/np.dot(x_next.T, x_next)
        resultEigenVector.append(x_next)
        resultEigenValue.append(u[0][0])
        if (np.linalg.norm(x_next - x_previous) < epsilon):
            interactions = i + 1
            break
        x_previous = x_next
    return (resultEigenVector, resultEigenValue, assintoticErrorArray, squaredAssintoticErrorArray, interactions)


def createA1array(B):
    return B + B.T


def createA2array(B, D):
    return np.dot(B, np.dot(D, np.linalg.inv(B)))


def printInitialConditions(B, x0):
    print("- Condicoes iniciais")
    print("B (10x10) = ")
    print(B)
    print("\n")
    print("x0 (10x1) = ")
    print(x0)
    print("\n")


def printResults(result):
    print("- Resultados")
    print("numero de iteracoes: " + str(result[2]))
    print("autovalor dominante calculado (lambda1): " + str(result[1]))
    print("autovetor associado: ")
    print(result[0])
    print("\n")

def getEigValuesAndVectors(reference, n):
    eigvalueArray = reference[0]
    indexHighestEigValue = met.findTheIndexOfHighestEigValue(eigvalueArray)
    indexSecondHighestEigValue = met.findTheIndexOfSecondHighestEigValue(eigvalueArray)
    highestEigValue = reference[0][indexHighestEigValue]
    secondHighestEigValue = reference[0][indexSecondHighestEigValue]
    eigVector = reference[1][:, indexHighestEigValue].reshape(n, 1)
    return (highestEigValue, secondHighestEigValue, eigVector)

def getEigValueError (eigValue, eigValueRef):
    return np.abs(eigValue - eigValueRef)

def getEigVectorError(eigVector, eigVectorRef):
    if (eigVector[0][0]*eigVectorRef[0][0] < 0):
        return np.linalg.norm(eigVector + eigVectorRef)
    return np.linalg.norm(eigVector - eigVectorRef)

def printReference(highestEigValue, secondHighestEigValue, eigVector):
    print("- Valores de referencia")
    print("autovalor dominante (lambda1): " + str(highestEigValue))
    print("autovetor dominante associado")
    print(eigVector)
    print("segundo maior autovalor (lambda2): " + str(secondHighestEigValue))
    print("erro assintotico | lambda2/lambda1 | = " + str(np.abs(secondHighestEigValue/highestEigValue)))
    print("\n")

def printComparison (eigValue, eigValueRef, eigVector, eigVectorRef):
    eigValueError = getEigValueError(eigValue, eigValueRef)
    eigVectorError = getEigVectorError(eigVector, eigVectorRef)
    print("Erro entre o autovalor dominante calculado e o de referencia: " + str(eigValueError))
    print("\n")
    print("Erro entre o autovetor dominante calculado e o de referencia: " + str(eigVectorError))
    print("\n")

def createGraphic (A, x0, n, title, fileName):
    reference = np.linalg.eig(A)
    eigValuesAndVectors = getEigValuesAndVectors(reference, n) 
    eigValueRef = eigValuesAndVectors[0]
    lambda2Ref = eigValuesAndVectors[1]
    eigVectorRef = eigValuesAndVectors[2]
    results = powerMethodInteractions(A, x0, eigValueRef, lambda2Ref)
    eigVectorArray = results[0]
    eigValueArray = results[1]
    assintoticErrorArray = results[2]
    squaredAssintoticErrorArray = results[3]
    k = results[4]
    eigVectorErrorArray = list(map(lambda eigVector: getEigVectorError(eigVector, eigVectorRef), eigVectorArray))
    eigValueErrorArray = list(map(lambda eigValue: getEigValueError(eigValue, eigValueRef), eigValueArray))
    X = list(range(1, k+1))
    plt.plot(X, eigVectorErrorArray, label="Erro autovetor")
    plt.plot(X, eigValueErrorArray, label = "Erro autovalor")
    plt.plot(X, assintoticErrorArray, label = r'$| \frac{\lambda_1}{\lambda_2} |^k$')
    plt.plot(X, squaredAssintoticErrorArray, label=r'$| \frac{\lambda_1}{\lambda_2} |^{2k}$')
    plt.yscale("log")
    plt.xlabel('Iterações')
    plt.ylabel('Erro_L2')
    plt.title(title)
    plt.legend()
    plt.savefig("images/ex1/"+fileName+".png")
    plt. clf()

B = met.generateRandomArray(10, 10)
x0 = met.generateRandomArray(10, 1)

print("******************************************************")
print("Exercicio 1.1")
print("\n")
printInitialConditions(B, x0)
A = createA1array(B)
print("A (10x10) = ")
print(A)
print("\n")
results = powerMethod(A, x0)
printResults(results)
reference = np.linalg.eig(A)
eigValuesAndVectors = getEigValuesAndVectors(reference, 10) 
highestEigValue = eigValuesAndVectors[0]
secondHighestEigValue = eigValuesAndVectors[1]
eigVector = eigValuesAndVectors[2]
printReference(highestEigValue, secondHighestEigValue, eigVector)
printComparison(results[1], highestEigValue, results[0], eigVector)
createGraphic(A, x0, 10, "Exercício 1.1", "ex1_1")



B = met.generateRandomArray(5, 5)
x0 = met.generateRandomArray(5, 1)

print("******************************************************")
print("Exercicio 1.2")
print("\n")
printInitialConditions(B, x0)

print("-> Primeiro teste: lamda1 relativamente perto de lambda2")
D = np.array([[1.5, 0, 0, 0, 0], [0, 1, 0, 0, 0], [
             0, 0, 0.5, 0, 0], [0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0.05]])
A = createA2array(B, D)
print("D (5x5) = ")
print(D)
print("\n")
print("A (5x5) = ")
print(A)
print("\n")
results = powerMethod(A, x0)
printResults(results)
reference = np.linalg.eig(A)
eigValuesAndVectors = getEigValuesAndVectors(reference, 5) 
highestEigValue = eigValuesAndVectors[0]
secondHighestEigValue = eigValuesAndVectors[1]
eigVector = eigValuesAndVectors[2]
printReference(highestEigValue, secondHighestEigValue, eigVector)
printComparison(results[1], highestEigValue, results[0], eigVector)
createGraphic(A, x0, 5, "Exercício 1.2, $\lambda_{1} = 1.5$ e $\lambda_{2} = 1.0$", "ex1_2a")

print("-> Segundo teste: lamda1 relativamente distante de lambda2")
D = np.array([[5, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0.5, 0, 0], [
             0, 0, 0, 0.2, 0], [0, 0, 0, 0, 0.05]])
A = createA2array(B, D)
print("D (5x5) = ")
print(D)
print("\n")
print("A (5x5) = ")
print(A)
print("\n")
results = powerMethod(A, x0)
printResults(results)
reference = np.linalg.eig(A)
eigValuesAndVectors = getEigValuesAndVectors(reference, 5) 
highestEigValue = eigValuesAndVectors[0]
secondHighestEigValue = eigValuesAndVectors[1]
eigVector = eigValuesAndVectors[2]
printReference(highestEigValue, secondHighestEigValue, eigVector)
printComparison(results[1], highestEigValue, results[0], eigVector)
createGraphic(A, x0, 5, "Exercício 1.2, $\lambda_{1} = 5$ e $\lambda_{2} = 1$", "ex1_2b")
