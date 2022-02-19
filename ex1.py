import numpy as np
import metodos as met
import matplotlib.pyplot as plt

# Constantes
itmax = 70
epsilon = 10**(-15)

'''
Esse método implementa o método da potências para a determinação
do maior autovalor de uma matriz A  e de seu respectivo 
autovetor associado. A função retorna uma tupla composta, nessa ordem, 
pelos seguintes elementos: autovetor, autovalor, quantidade de interações 
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

'''
Essa rotina é semelhante à anterior ('powerMethod'). No entanto,
ao invés de somente retornar as respostas finais, ela retorna a lista
dos resultados das interações para o autovetor e o autovalor. O método
também retorna os valores  encontrados nas interações correspondentes à
potência da razão entre os autovalores de referência. A função é utilizada 
para a construção dos gráficos. 
'lambda1' e 'lambda2' são os valores de referência para o maior
e segundo maior autovalor, respectivamente. 
'''
def powerMethodInteractions(A, x0, lambda1, lambda2):
    x_previous = x0
    u = [[]]
    resultEigenVector = []
    resultEigenValue = []
    assintoticErrorK = np.abs(lambda2/lambda1)
    squaredAssintoticErrorK = (lambda2/lambda1)**2
    assintoticErrorArray = [assintoticErrorK]
    squaredAssintoticErrorArray = [squaredAssintoticErrorK]
    interactions = itmax
    for i in range(itmax):
        if (i != 0):
            assintoticErrorK = assintoticErrorK*(np.abs(lambda2/lambda1))
            squaredAssintoticErrorK = squaredAssintoticErrorK * \
                ((lambda2/lambda1)**2)
            assintoticErrorArray.append(assintoticErrorK)
            squaredAssintoticErrorArray.append(squaredAssintoticErrorK)
        n = np.dot(A, x_previous)
        x_next = n/np.linalg.norm(n)
        u = np.dot(x_previous.T, n)/np.dot(x_previous.T, x_previous)
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
Essa função cria a matriz A do exercício 1.1 a partir de uma
certa matriz B.
'''
def createA1array(B):
    return B + B.T

'''
Essa função cria a matriz A do exercício 1.2 a partir de uma
certa matriz B e de uma matriz diagonal D.
'''
def createA2array(B, D):
    return np.dot(B, np.dot(D, np.linalg.inv(B)))

'''
Esse método imprime os resultados obtidos a partir do método da potências
implementado pela função 'powerMethod'. Ela recebe uma tupla da forma
(autovetor encontrado, autovalor encontrado, n° de interações)
'''
def printResults(result):
    print("- Resultados")
    print("numero de iteracoes: " + str(result[2]))
    print("autovalor dominante calculado (lambda1): " + str(result[1]))
    print("autovetor associado: ")
    print(result[0])
    print("\n")

'''
Esse método imprime os valores de referência obtidos a partir da biblioteca
numpy.
'''
def printReference(highestEigValue, secondHighestEigValue, eigVector):
    print("- Valores de referencia")
    print("autovalor dominante (lambda1): " + str(highestEigValue))
    print("autovetor dominante associado")
    print(eigVector)
    print("segundo maior autovalor (lambda2): " + str(secondHighestEigValue))
    print("erro assintotico | lambda2/lambda1 | = " +
          str(np.abs(secondHighestEigValue/highestEigValue)))
    print("\n")

'''
Essa função imprime dados que permitem a comparação dos valores encontrados
com os valores de referência.
'''
def printComparison(eigValue, eigValueRef, eigVector, eigVectorRef):
    eigValueError = met.getEigValueError(eigValue, eigValueRef)
    eigVectorError = met.getEigVectorError(eigVector, eigVectorRef)
    print("Erro entre o autovalor dominante calculado e o de referencia: " +
          str(eigValueError))
    print("\n")
    print("Erro entre o autovetor dominante calculado e o de referencia: " +
          str(eigVectorError))
    print("\n")

'''
Essa rotina é responsável por gerar os gráficos em escala logarítmica 
dos comportamentos dos erros dos autovalores e dos autovetores em função
do número de interações. Para isso, é aplicado o método 'powerMethodInteractions'
e, em seguida, são calculados os erros para cada interação. Além disso, para efeitos
de comparação, o método também traz as retas correspondentes às potências da razão
entre os autovalores de referência.
'''
def createGraphic(A, x0, eigValuesAndVectors, title, fileName):
    eigValueRef = eigValuesAndVectors[0]
    lambda2Ref = eigValuesAndVectors[1]
    eigVectorRef = eigValuesAndVectors[2]
    if (eigVectorRef[0][0] < 0 or (eigVectorRef[0][0] == 0 and eigVectorRef[1][0] < 0)):
            eigVectorRef *= -1
    results = powerMethodInteractions(A, x0, eigValueRef, lambda2Ref)
    eigVectorArray = results[0]
    eigValueArray = results[1]
    assintoticErrorArray = results[2]
    squaredAssintoticErrorArray = results[3]
    k = results[4]
    eigVectorErrorArray = list(map(lambda eigVector: met.getEigVectorError(
        eigVector, eigVectorRef), eigVectorArray))
    eigValueErrorArray = list(
        map(lambda eigValue: met.getEigValueError(eigValue, eigValueRef), eigValueArray))
    X = list(range(1, k+1))
    plt.plot(X, eigVectorErrorArray, label="Erro autovetor")
    plt.plot(X, eigValueErrorArray, label="Erro autovalor")
    plt.plot(X, assintoticErrorArray,
             label=r'$| \frac{\lambda_2}{\lambda_1} |^k$')
    plt.plot(X, squaredAssintoticErrorArray,
             label=r'$| \frac{\lambda_2}{\lambda_1} |^{2k}$')
    plt.yscale("log")
    plt.xlabel('Iterações')
    plt.ylabel('Erro_L2')
    plt.title(title)
    plt.legend()
    plt.savefig("images/ex1/"+fileName+".png")
    plt. clf()




# Geração das condições iniciais do ex 1.1
B = met.generateRandomB(10,10,2021)
x0 = met.generateRandomX0(10, 2022)


print("******************************************************")
print("Exercicio 1.1")
print("\n")
met.printInitialConditions(B, x0, 10, "B")
# Criação da matriz A do ex 1.1
A = createA1array(B)
print("A (10x10) = ")
print(A)
print("\n")
# Resultado pela resolução utilizando o método das potências
results = powerMethod(A, x0)
printResults(results)
# Referências
reference = np.linalg.eig(A)
eigValuesAndVectors = met.getEigValuesAndVectors(reference, 10)
highestEigValue = eigValuesAndVectors[0]
secondHighestEigValue = eigValuesAndVectors[1]
eigVector = eigValuesAndVectors[2]
# Escolha do sinal positivo para a primeira entrada do autovetor de referência
if (eigVector[0][0] < 0 or (eigVector[0][0] == 0 and eigVector[1][0] < 0)):
            eigVector *= -1
printReference(highestEigValue, secondHighestEigValue, eigVector)
# Comparação e gráficos
printComparison(results[1], highestEigValue, results[0], eigVector)
createGraphic(A, x0, eigValuesAndVectors, "Exercício 1.1", "ex1_1")




# Geração das condições iniciais do ex 1.2
B = met.generateRandomB(5,5, 2021)
x0 =  met.generateRandomX0(5,2022)

print("******************************************************")
print("Exercicio 1.2")
print("\n")
met.printInitialConditions(B, x0, 5, "B")


# Exercício 1.2.a
print("-> Primeiro teste: lambda1 relativamente perto de lambda2")
D = np.array([[1.5, 0, 0,   0,      0],
              [0,   1, 0,   0,      0], 
              [0,   0, 0.5, 0,      0],
              [0,   0, 0,   0.2,    0], 
              [0,   0, 0,   0,   0.05]])
# Criação da matriz A do ex 1.2.a
A = createA2array(B, D)
print("D (5x5) = ")
print(D)
print("\n")
print("A (5x5) = ")
print(A)
print("\n")
# Resultado pela resolução utilizando o método das potências
results = powerMethod(A, x0)
printResults(results)
# Referências
reference = np.linalg.eig(A)
eigValuesAndVectors = met.getEigValuesAndVectors(reference, 5)
highestEigValue = eigValuesAndVectors[0]
secondHighestEigValue = eigValuesAndVectors[1]
eigVector = eigValuesAndVectors[2]
# Escolha do sinal positivo para a primeira entrada do autovetor de referência
if (eigVector[0][0] < 0 or (eigVector[0][0] == 0 and eigVector[1][0] < 0)):
            eigVector *= -1
printReference(highestEigValue, secondHighestEigValue, eigVector)
# Comparação e gráficos
printComparison(results[1], highestEigValue, results[0], eigVector)
createGraphic(
    A, x0, eigValuesAndVectors, "Exercício 1.2, $\lambda_{1} = 1.5$ e $\lambda_{2} = 1.0$", "ex1_2a")


# Exercício 1.2.b (mesmas condições para B e x0)
print("-> Segundo teste: lambda1 relativamente distante de lambda2")
D = np.array([[5, 0, 0,   0,      0],
              [0, 1, 0,   0,      0],
              [0, 0, 0.5, 0,      0], 
              [0, 0, 0,   0.2,    0], 
              [0, 0, 0,   0,   0.05]])
# Criação da matriz A do ex 1.2.b
A = createA2array(B, D)
print("D (5x5) = ")
print(D)
print("\n")
print("A (5x5) = ")
print(A)
print("\n")
# Resultado pela resolução utilizando o método das potências
results = powerMethod(A, x0)
printResults(results)
# Referências
reference = np.linalg.eig(A)
eigValuesAndVectors = met.getEigValuesAndVectors(reference, 5)
highestEigValue = eigValuesAndVectors[0]
secondHighestEigValue = eigValuesAndVectors[1]
eigVector = eigValuesAndVectors[2]
# Escolha do sinal positivo para a primeira entrada do autovetor de referência
if (eigVector[0][0] < 0 or (eigVector[0][0] == 0 and eigVector[1][0] < 0)):
            eigVector *= -1
printReference(highestEigValue, secondHighestEigValue, eigVector)
# Comparação e gráficos
printComparison(results[1], highestEigValue, results[0], eigVector)
createGraphic(
    A, x0, eigValuesAndVectors, "Exercício 1.2, $\lambda_{1} = 5$ e $\lambda_{2} = 1$", "ex1_2b")
