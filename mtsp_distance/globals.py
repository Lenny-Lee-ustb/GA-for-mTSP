import random
import pandas as pd
import math
import numpy as np
'''
Contains all global variables specific to simulation
'''
# Defines range for coordinates when dustbins are randomly scattered
xMax = 1000
yMax = 1000
seedValue = 1
# numNodes = 51
numGenerations = 200
# size of population
populationSize = 100
mutationRate = 0.5
tournamentSize = 10
elitism = True
# number of trucks
numTrucks = 5

# read differnt dataset
# city_list =pd.read_csv("pr76.txt",sep=" ",header=0,names=['id','X','Y'])
# city_list =pd.read_csv("pr152.txt",sep=" ",header=0,names=['id','X','Y'])
city_list =pd.read_csv("pr226.txt",sep=" ",header=0,names=['id','X','Y'])
# city_list =pd.read_csv("mtsp51.txt",sep=" ",header=0,names=['id','X','Y'])
# city_list =pd.read_csv("mtsp100.txt",sep=" ",header=0,names=['id','X','Y'])
# city_list =pd.read_csv("mtsp150.txt",sep=" ",header=0,names=['id','X','Y'])
numNodes = len(city_list)
X = city_list['X'].tolist()
Y = city_list['Y'].tolist()

# dist_map = [[0]*numNodes]*numNodes
dist_map = np.zeros((numNodes,numNodes))
for i in range(numNodes):
    for j in range(numNodes):
        if i != j:
            dist_map[i][j] = math.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
        else:
            dist_map[i][j] = 100000

def random_range(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


# Randomly distribute number of dustbins to subroutes
# Maximum and minimum values are maintained to reach optimal result
def route_lengths():
    upper = (numNodes + numTrucks - 1)
    fa = upper/numTrucks*1.6 # max route length
    fb = upper/numTrucks*0.6 # min route length
    a = random_range(numTrucks, upper)
    while 1:
        if all( i < fa and i > fb  for i in a):
                break
        else:
                a = random_range(numTrucks, upper)
    return a
