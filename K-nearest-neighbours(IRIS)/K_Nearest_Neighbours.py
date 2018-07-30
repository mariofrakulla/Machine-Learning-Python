""" Application of K_Nearest_Neighbours in the Iris dataset """
import numpy as np
import pandas as pd
import math
import operator

data = pd.read_csv("iris.csv")
print(data.head())


def EuclideanDistance(X1, X2,length):
    distance = 0

    for ix in range(length):
        distance = distance + np.square(X1[ix] - X2[ix])
    return np.sqrt(distance)

def ManhattanDistance(X1, X2, length):
    distanceMH = 0

    for ix in range(length):
        distanceMH = distanceMH + np.abs(X1[ix] - X2[ix])
    return distanceMH

    """ Define our KNN model """
def KNN(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[1]
    """ This loop iterates over each element of the training set and computes the Euclidean
    Distance to the testInstance """
    for x in range(len(trainingSet)):
        dist = EuclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classVotes = {}

    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return (sortedVotes[0][0], neighbors)


testSet = [[1.2, 5.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
k = 3
result, neigh =  KNN(data, test, k)
print(result)
print(neigh)