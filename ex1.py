import math
import numpy as np

# Constantes
itmax = 30
epsilon = 10**(-15)

# funções auxiliares
def generateRandomArray (x, y):
    return (100)*np.random.rand(x,y)

def calculateArrayX (A, x0):
    u_0 = np.dot(x0.T, np.dot(A, x0))/np.dot(x0.T, x0)
    x_anterior = x0
    u = u_0
    for i in range(itmax):
        x_prox = np.dot(A,x_anterior)/np.linalg.norm(np.dot(A,x_anterior))
        u = np.dot(x_prox.T, np.dot(A, x_prox))/np.dot(x_prox.T, x_prox)
        x_anterior = x_prox
    return (x_prox, u[0][0])

A = np.array([[-2,-4,2], [-2,1,2], [4,2,5]])
x0 = np.array([[1], [1], [1]])

print("A:\n")
print(A)
print("\n")
print("x0\n")
print(x0)
print("\n")
print("x calculado\n")
print(calculateArrayX(A, x0))
print("\n")
print("Valores reais\n")
print(np.linalg.eig(A))

