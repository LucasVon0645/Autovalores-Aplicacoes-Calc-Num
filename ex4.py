import numpy as np
import metodos as met
from ex1 import powerMethod

# Funções auxiliares

'''
A função a seguir recebe a matriz de arestas E e o número de vértices n 
de um grafo e gera sua matriz de adjacência A, sendo que, quando o 
vértice i possui conexão com o vértice j, os elementos A[i][j] e A[j][i] 
são iguais a 1 e, caso contrário, são iguais a 0 (Obs.: a diagonal 
principal de A é sempre nula, pois não se considera que um vértice está 
conectado a si mesmo).
'''
def constructAdjMatrix(E, n):
    A = np.zeros((n, n))

    for k in range(len(E)):
        i = E[k][0]
        j = E[k][1]

        A[i][j] = 1
        A[j][i] = 1

    return A

'''
A função a seguir recebe a matriz de adjacência de um grafo A e, 
a partir disso, calcula o grau médio dos vértices dmed e o grau 
máximo dos vértices dmax desse grafo, por meio da soma das linhas 
da matriz A e de funções de média (mean) e máximo (max) da 
biblioteca numpy.
'''
def calculateAverageAndMaximumDegree(A):
    S = A.sum(axis=1)

    dmed = np.mean(S)
    dmax = np.max(S)

    return (dmed, dmax)

# Programa principal

print("******************************************************")
print("Exercicio 4.1")

print("\nOs dois projetos de rede de metrô representados pelos grafos G1 e G2 são apresentados no relatório.")

print("\n******************************************************")
print("Exercicio 4.2")

# declaração e apresentação das matrizes de arestas para os grafos G1 e G2
E1 = np.array([[0 , 2 ],
               [2 , 5 ], 
               [5 , 9 ],
               [9 , 13], 
               [1 , 3 ],
               [3 , 5 ],
               [5 , 8 ],
               [8 , 12],
               [7 , 8 ],
               [8 , 9 ],
               [9 , 10],
               [10, 11],
               [4 , 6 ],
               [6 , 11],
               [11, 14],
               [14, 15]])

print("\nMatriz de arestas E1 para o grafo G1:")
print(E1)

E2 = np.array([[0 , 2 ],
               [2 , 6 ], 
               [6 , 9 ],
               [9 , 12], 
               [1 , 4 ],
               [4 , 7 ],
               [7 , 10],
               [10, 12],
               [3 , 7 ],
               [7 , 11],
               [11, 14],
               [14, 16],
               [5 , 8 ],
               [8 , 11],
               [11, 13],
               [13, 15]])

print("\nMatriz de arestas E2 para o grafo G2:")
print(E2)

# idetificação do número de vértices dos grafos G1 e G2
n1 = np.max(E1) + 1
n2 = np.max(E2) + 1

print("\nNúmero de vértices do grafo G1 associado à primeira rede de metrô: " + str(n1))
print("\nNúmero de vértices do grafo G2 associado à segunda rede de metrô: " + str(n2))

# construção e apresentação das matrizes de adjacência dos grafos G1 e G2
A1 = constructAdjMatrix(E1, 16)
A2 = constructAdjMatrix(E2, 17)

print("\nMatriz de adjacência A1 referente ao grafo G1: ")
print(A1)
print("\nMatriz de adjacência A2 referente ao grafo G2: ")
print(A2)

print("\n******************************************************")
print("Exercicio 4.3")

# utilização do método das potências (desenvolvido no exercício 1) para encontrar 
# os índices dos grafos G1 e G2, além dos autovetores associados a esses índices
x01 = met.generateRandomX0(len(A1), 42)
eigVector1, lambda1_G1, k1 = powerMethod(A1, x01)

x02 = met.generateRandomX0(len(A2), 42)
eigVector2, lambda1_G2, k2 = powerMethod(A2, x02)

# apresentação e comparação dos índices dos grafos G1 e G2
print("\nÍndice de G1: " + str(lambda1_G1))
print("\nÍndice de G2: " + str(lambda1_G2))

if lambda1_G1 > lambda1_G2:
    print("\nA rede do grafo G1 possui o maior índice.")
else:
    print("\nA rede do grafo G2 possui o maior índice.")

# cálculo dos graus médio e máximo dos vértices dos grafos G1 e G2 
# e comparação desses com os índices de cada grafo
dmed1, dmax1 = calculateAverageAndMaximumDegree(A1)

print("\nGrau médio dos vértices de G1: " + str(dmed1))
print("Grau máximo dos vértices de G1: " + str(dmax1))

if (lambda1_G1 >= dmed1 and lambda1_G1 <= dmax1):
    print("\nO valor do índice de G1, conforme esperado, é:")
    print("- maior do que o grau médio dos vértices de G1")
    print("- e menor do que o grau máximo dos vértices de G1")

dmed2, dmax2 = calculateAverageAndMaximumDegree(A2)

print("\nGrau médio dos vértices de G2: " + str(dmed2))
print("Grau máximo dos vértices de G2: " + str(dmax2))

if (lambda1_G2 >= dmed2 and lambda1_G2 <= dmax2):
    print("\nO valor do índice de G2, conforme esperado, é:")
    print("- maior do que o grau médio dos vértices de G2")
    print("- e menor do que o grau máximo dos vértices de G2")

print("\n******************************************************")
print("Exercicio 4.4")

# apresentação das aproximações numéricas dos autovetores 
# associados aos índices (autovalores) dos grafos G1 e G2
print("\nAutovetor associado ao índice (autovalor) de G1:")
print(eigVector1)

# determinação do vértice com maior centralidade de autivetor para os grafos G1 e G2
centralVertex1 = np.argmax(eigVector1)
print("\nVértice de maior centralidade de autovetor do grafo G1: v" + str(centralVertex1))

print("\nAutovetor associado ao índice (autovalor) de G2:")
print(eigVector2)

centralVertex2 = np.argmax(eigVector2)
print("\nVértice de maior centralidade de autovetor do grafo G2: v" + str(centralVertex2))