import numpy as np
import metodos as met
import matplotlib.pyplot as plt

# Constantes
itmax = 70
epsilon = 10**(-15)
w = 1.25

''' 
Essa rotina recebe uma matriz quadrada A de ordem n e retorna 
True caso tal matriz satisfaça o critério das linhas. 
Caso contrário, retorna False.
'''
def linesCriterion(A, n):
    valid = True
    for i in range(n):
        lineSum = 0
        for j in range(n):
            if(j != i):
                lineSum = lineSum + np.abs(A[i][j])
        if (lineSum >= A[i][i]):
            valid = False
            break
    return valid

'''
Essa rotina implementa o método SOR (Sucessive Over-Relaxation) 
para a resolução de sistemas lineares da forma A.x = y, em que 
A é uma matriz quadrada de ordem n, enquanto x e y são vetores 
de dimensões nx1. w corresponde ao parâmetro w do método intera-
tivo. O critério de parada é limitado pelo critério de Sassenfeld
ou pelo número máximo de interações itmax. A função retorna a solução
x a partir de um valor inicial dado x0. 
'''
def solveEquationSystem(A, n, x0, y, w):
    M = met.sassenfeldCriterion(A, n)
    x_previous = x0
    x_next = list(range(n))
    for k in range(itmax):
        for i in range(n):
            x_previous_i = x_previous[i]
            sumBeforeI = 0
            for j in range(i):
                x_next_j = x_next[j]
                sumBeforeI = sumBeforeI + A[i][j]*x_next_j
            sumAfterI = 0
            for j in range(i+1, n):
                x_previous_j = x_previous[j]
                sumAfterI = sumAfterI + A[i][j]*x_previous_j
            residue = y[i] - sumBeforeI - sumAfterI
            x_next_i = (1 - w)*x_previous_i + (w/A[i][i])*residue
            x_next[i] = x_next_i
        greatestDifference = met.greatestDifferenceBetweenArrays(
            x_next, x_previous)
        if (greatestDifference <= (1-M)*epsilon/M):
            break
        x_previous = x_next.copy()
    return np.array(x_next)

'''
Esse método implementa o método da potência inversa para a determinação
do inverso do menor autovalor de uma matriz A de ordem n e de seu respectivo 
autovetor associado. A função retorna uma tupla composta, nessa ordem, 
pelos seguintes elementos: autovetor, autovalor, quantidade de interações 
realizadas. O cálculo é feito a partir de um vetor inicial dado x0.
'''
def inversePowerMethod(A, x0, n):
    x_previous = x0
    u = 0
    interactions = itmax
    random = met.generateRandomArray(n, 1)
    for i in range(itmax):
        c = solveEquationSystem(A, n, random, x_previous, w)
        x_next = c/np.linalg.norm(c)
        u = np.dot(x_previous.T, c)/np.dot(x_previous.T, x_previous)
        if (np.linalg.norm(x_next - x_previous) < epsilon):
            interactions = i + 1
            break
        x_previous = x_next
    if (x_next[0][0] < 0 or (x_next[0][0] == 0 and x_next[1][0] < 0)):
        x_next *= -1
    return (x_next, u[0][0], interactions)

'''
Essa rotina é semelhante à anterior ('inversePowerMethod'). No entanto,
ao invés de somente retornar as respostas finais, ela retorna listas
que possuem a sequência de resultados encontrados nas interações para os
valores do autovetor e do autovalor. O método também retorna os valores 
encontrados nas interações correspondentes a potência da razão entre os
autovalores de referência. A função é utilizada para a construção dos
gráficos. 'inverse_lambdan' e 'inverse_lambdan_1' são os valores de
referência  para os inversos do menor e segundo menor autovalor, 
respectivamente. 
'''
def inversePowerMethodInteractions(A, x0, inverse_lambdan, inverse_lambdan_1, n):
    x_previous = x0
    u = [[]]
    resultEigenVector = []
    resultEigenValue = []
    assintoticErrorK = np.abs(inverse_lambdan_1/inverse_lambdan)
    squaredAssintoticErrorK = (inverse_lambdan_1/inverse_lambdan)**2
    assintoticErrorArray = [assintoticErrorK]
    squaredAssintoticErrorArray = [squaredAssintoticErrorK]
    interactions = itmax
    random = met.generateRandomArray(n, 1)
    for i in range(itmax):
        if (i != 0):
            assintoticErrorK = assintoticErrorK * \
                (np.abs(inverse_lambdan_1/inverse_lambdan))
            squaredAssintoticErrorK = squaredAssintoticErrorK * \
                ((inverse_lambdan_1/inverse_lambdan)**2)
            assintoticErrorArray.append(assintoticErrorK)
            squaredAssintoticErrorArray.append(squaredAssintoticErrorK)
        c = solveEquationSystem(A, n, random, x_previous, w)
        x_next = c/np.linalg.norm(c)
        u = np.dot(x_previous.T, c)/np.dot(x_previous.T, x_previous)
        if (x_next[0][0] < 0 or (x_next[0][0] == 0 and x_next[1][0] < 0)):
            x_next *= -1
        resultEigenVector.append(x_next)
        resultEigenValue.append(u[0][0])
        if (np.linalg.norm(x_next - x_previous) < epsilon):
            interactions = i + 1
            break
        x_previous = x_next
    return (resultEigenVector, resultEigenValue, assintoticErrorArray, squaredAssintoticErrorArray, interactions)

'''
Essa função cria a matriz A de ordem n do exercício 2.1 a partir de uma
certa matriz B de mesma ordem.
'''
def createA1array(B, n):
    return B + B.T + n*np.identity(n)

'''
Essa função cria a matriz A de ordem n do exercício 2.2 a partir de uma
certa matriz B0, de uma matriz diagonal D e de um parâmetro p escolhido.
'''
def createA2array(B0, D, n, p):
    B = B0 + p*np.identity(n)
    return np.dot(B, np.dot(D, np.linalg.inv(B)))

'''
Esse método imprime os resultados obtidos a partir do método da potência inversa
implementado pela função 'inversePowerMethod'. Ela recebe uma tupla da forma
(autovetor encontrado, autovalor encontrado, n° de interações)
'''
def printResults(result):
    print("- Resultados")
    print("numero de iteracoes: " + str(result[2]))
    print("lambda_n^(-1): " + str(result[1]))
    print("autovetor de A^(-1) associado a lambda_n^(-1): ")
    print(result[0])
    print("\n")

'''
Esse método imprime os valores de referência obtidos a partir da biblioteca
numpy.
'''
def printReference(highestEigValue, secondHighestEigValue, eigVector):
    print("- Valores de referencia")
    print("lambda_n^(-1): " + str(highestEigValue))
    print("autovetor associado")
    print(eigVector)
    print(
        "segundo maior autovalor lambda_[n-1]^(-1): " + str(secondHighestEigValue))
    print("erro assintotico | lambda_[n-1]^(-1)/lambda_n^(-1) | = " +
          str(np.abs(secondHighestEigValue/highestEigValue)))
    print("\n")

'''
Essa função imprime dados que permitem a comparação dos valores encontrados
com os valores de referência.
'''
def printComparison(eigValue, eigValueRef, eigVector, eigVectorRef):
    eigValueError = met.getEigValueError(eigValue, eigValueRef)
    eigVectorError = met.getEigVectorError(eigVector, eigVectorRef)
    print("Erro entre o autovalor lambda_n^(-1) calculado e o de referencia: " +
          str(eigValueError))
    print("\n")
    print("Erro entre o autovetor de A^(-1) associado a lambda_n^(-1) calculado e o de referencia: " +
          str(eigVectorError))
    print("\n")

'''
Essa rotina é responsável por gerar os gráficos em escala logarítmica 
dos comportamentos dos erros dos autovalores e dos autovetores em função
do número de interações. Para isso, é aplicado o método 'inversePowerMethodInteractions'
e, em seguida, são calculados os erros para cada interação. Além disso, para efeitos
de comparação, o método também traz as retas correspondentes às potências da razão
entre os autovalores de referência.
'''
def createGraphic(A, x0, n, eigValuesAndVectorsInverseA, title, fileName):
    inverse_lambdanRef = eigValuesAndVectorsInverseA[0]
    inverse_lambdan_1Ref = eigValuesAndVectorsInverseA[1]
    eigVectorRef = eigValuesAndVectorsInverseA[2]
    if (eigVectorRef[0][0] < 0 or (eigVectorRef[0][0] == 0 and eigVectorRef[1][0] < 0)):
            eigVectorRef *= -1
    results = inversePowerMethodInteractions(
        A, x0, inverse_lambdanRef, inverse_lambdan_1Ref, n)
    eigVectorArray = results[0]
    eigValueArray = results[1]
    assintoticErrorArray = results[2]
    squaredAssintoticErrorArray = results[3]
    k = results[4]
    eigVectorErrorArray = list(map(lambda eigVector: met.getEigVectorError(
        eigVector, eigVectorRef), eigVectorArray))
    eigValueErrorArray = list(
        map(lambda eigValue: met.getEigValueError(eigValue, inverse_lambdanRef), eigValueArray))
    X = list(range(1, k+1))
    plt.plot(X, eigVectorErrorArray, label="Erro autovetor")
    plt.plot(X, eigValueErrorArray, label="Erro autovalor")
    plt.plot(X, assintoticErrorArray,
             label=r'$| \frac{(\lambda_{n-1})^{-1}}{(\lambda_n)^{-1}} |^k$')
    plt.plot(X, squaredAssintoticErrorArray,
             label=r'$| \frac{(\lambda_{n-1})^{-1}}{(\lambda_n)^{-1}} |^{2k}$')
    plt.yscale("log")
    plt.xlabel('Iterações')
    plt.ylabel('Erro_L2')
    plt.title(title)
    plt.legend()
    plt.savefig("images/ex2/"+fileName+".png")
    plt. clf()

# Programa principal

print("******************************************************")
print("Exercício 2.1")
print("\n")

# Geração das condições iniciais do ex 2.1
B = met.generateRandomB(10,10,2021)
x0 = met.generateRandomX0(10, 2022)

met.printInitialConditions(B, x0, 10, "B")
# Criação da matriz A do ex 2.1
A = createA1array(B, 10)
print("A (10x10) = ")
print(A)
print("\n")
print("\n")
# Aplicação do critério das linhas
print("Criterio das linhas para A: ", linesCriterion(A, 10))
print("\n")
# Aplicação do critério de Sassenfeld
M = met.sassenfeldCriterion(A, 10)
print("Criterio de Sassenfeld para A: M = ", M)
met.isSassenFeldCriterionSatisfied(M)
print("\n")
# Resultado pela resolução utilizando o método das potências inversas e SOR
results = inversePowerMethod(A, x0, 10)
printResults(results)
# Referências
reference = np.linalg.eig(np.linalg.inv(A))
eigValuesAndVectorsInverseA = met.getEigValuesAndVectors(reference, 10)
highestEigValueOfInverseA = eigValuesAndVectorsInverseA[0]
secondHighestEigValueOfInverseA = eigValuesAndVectorsInverseA[1]
eigVector = eigValuesAndVectorsInverseA[2]
# Escolha do sinal positivo para a primeira entrada do autovetor de referência
if (eigVector[0][0] < 0 or (eigVector[0][0] == 0 and eigVector[1][0] < 0)):
            eigVector *= -1
printReference(highestEigValueOfInverseA,
               secondHighestEigValueOfInverseA, eigVector)
# Comparação e gráficos
printComparison(results[1], highestEigValueOfInverseA, results[0], eigVector)
createGraphic(A, x0, 10, eigValuesAndVectorsInverseA, "Exercício 2.1", "ex2_1")

print("******************************************************")
print("Exercício 2.2")
print("\n")

# Geração das condições iniciais do ex 2.2
B0 = met.generateRandomB(5,5, 2021)
x0 = met.generateRandomX0(5,2022)
p = 10

met.printInitialConditions(B0, x0, 5, "B0")
print("p = 10")
print("\n")

# Exercício 2.2.a
print(
    "-> Primeiro teste: lambda_n^(-1) relativamente perto de lambda_[n-1]^(-1)")
D = np.array([[1.1, 0, 0,   0,      0],
              [0,   1, 0,   0,      0],
              [0,   0, 0.5, 0,      0],
              [0,   0, 0,   0.12,   0],
              [0,   0, 0,   0,    0.1]])
# Criação da matriz A do ex 2.2.a
A = createA2array(B0, D, 5, p)
print("D (5x5) = ")
print(D)
print("\n")
print("A (5x5) = ")
print(A)
print("\n")
# Aplicação do critério das linhas
print("Criterio das linhas para A: ", linesCriterion(A, 5))
print("\n")
# Aplicação do critério de Sassenfeld
M = met.sassenfeldCriterion(A, 5)
print("Criterio de Sassenfeld para A: M = ", M)
met.isSassenFeldCriterionSatisfied(M)
print("\n")
# Resultado pela resolução utilizando o método das potências inversas e SOR
results = inversePowerMethod(A, x0, 5)
printResults(results)
# Referências
reference = np.linalg.eig(np.linalg.inv(A))
eigValuesAndVectorsInverseA = met.getEigValuesAndVectors(reference, 5)
highestEigValueOfInverseA = eigValuesAndVectorsInverseA[0]
secondHighestEigValueInverseA = eigValuesAndVectorsInverseA[1]
eigVector = eigValuesAndVectorsInverseA[2]
# Escolha do sinal positivo para a primeira entrada do autovetor de referência
if (eigVector[0][0] < 0 or (eigVector[0][0] == 0 and eigVector[1][0] < 0)):
            eigVector *= -1
printReference(highestEigValueOfInverseA,
               secondHighestEigValueInverseA, eigVector)
# Comparação e gráficos
printComparison(results[1], highestEigValueOfInverseA, results[0], eigVector)
createGraphic(
    A, x0, 5, eigValuesAndVectorsInverseA, "Exercício 2.2, $\lambda_{n} = 0.1$ e $\lambda_{n-1} = 0.12$", "ex2_2a")

# Exercício 2.2.b (mesmas condições para B e x0)
print(
    "-> Segundo teste: lambda_n^(-1) relativamente distante de lambda_[n-1]^(-1)")
D = np.array([[5, 0, 0,   0,      0],
              [0, 1, 0,   0,      0],
              [0, 0, 0.5, 0,      0],
              [0, 0, 0,   0.4,    0],
              [0, 0, 0,   0,    0.1]])
# Criação da matriz A do ex 2.2.b
A = createA2array(B0, D, 5, 15)
print("D (5x5) = ")
print(D)
print("\n")
print("p = 15")
print("\n")
print("A (5x5) = ")
print(A)
print("\n")
# Aplicação do critério das linhas
print("Criterio das linhas para A: ", linesCriterion(A, 5))
print("\n")
# Aplicação do critério de Sassenfeld
M = met.sassenfeldCriterion(A, 5)
print("Criterio de Sassenfeld para A: M = ", M)
met.isSassenFeldCriterionSatisfied(M)
print("\n")
# Resultado pela resolução utilizando o método das potências inversas e SOR
results = inversePowerMethod(A, x0, 5)
printResults(results)
# Referências
reference = np.linalg.eig(np.linalg.inv(A))
eigValuesAndVectorsInverseA = met.getEigValuesAndVectors(reference, 5)
highestEigValueOfInverseA = eigValuesAndVectorsInverseA[0]
secondHighestEigValueInverseA = eigValuesAndVectorsInverseA[1]
eigVector = eigValuesAndVectorsInverseA[2]
# Escolha do sinal positivo para a primeira entrada do autovetor de referência
if (eigVector[0][0] < 0 or (eigVector[0][0] == 0 and eigVector[1][0] < 0)):
            eigVector *= -1
printReference(highestEigValueOfInverseA,
               secondHighestEigValueInverseA, eigVector)
# Comparação e gráficos
printComparison(results[1], highestEigValueOfInverseA, results[0], eigVector)
createGraphic(
    A, x0, 5, eigValuesAndVectorsInverseA, "Exercício 2.2, $\lambda_{n} = 0.1$ e $\lambda_{n-1} = 0.4$", "ex2_2b")
