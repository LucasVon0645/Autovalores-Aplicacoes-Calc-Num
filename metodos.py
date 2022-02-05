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