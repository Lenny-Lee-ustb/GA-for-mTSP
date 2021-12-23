import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import numpy as np
import progressbar
import time

'''
Contains all global variables specific to simulation
'''
# Defines range for coordinates when dustbins are randomly scattered
xMax = 1000
yMax = 1000
seedValue = 1
# numNodes = 51
numGenerations = 180
# size of population
populationSize = 100
mutationRate = 0.5
tournamentSize = 10
elitism = True
# number of trucks
numTrucks = 5
test_time = 30


data_name = ["mtsp51", "mtsp100", "mtsp150", "pr76", "pr152", "pr226"]

for data in data_name:

    city_list = pd.read_csv("../" + data + ".txt", sep=" ",
                            header=0, names=['id', 'X', 'Y'])

    numNodes = len(city_list)
    X = city_list['X'].tolist()
    Y = city_list['Y'].tolist()

    df = pd.DataFrame(columns=['initial_distance',
                               'time_cost', 'global_min_dis'])
    print("Running ours method test on the data " + data)

    for k in range(test_time):
        tic = time.time()
        pbar = progressbar.ProgressBar()
        print(f"Running for the time {k+1}")

        # dist_map = [[0]*numNodes]*numNodes
        dist_map = np.zeros((numNodes, numNodes))
        for i in range(numNodes):
            for j in range(numNodes):
                if i != j:
                    dist_map[i][j] = math.sqrt(
                        (X[i] - X[j])**2 + (Y[i] - Y[j])**2)
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
            fa = upper/numTrucks*1.6  # max route length
            fb = upper/numTrucks*0.6  # min route length
            a = random_range(numTrucks, upper)
            while 1:
                if all(i < fa and i > fb for i in a):
                    break
                else:
                    a = random_range(numTrucks, upper)
            return a

        '''
        dustbin

        Represents nodes in the problem graph or network.
        Locatin coordinates can be passed while creating the object or they
        will be assigned random values.
        '''

        class Dustbin:
            # Good old constructor
            def __init__(self, x=None, y=None, ci=None):
                if x == None and y == None:
                    self.x = random.randint(0, xMax)
                    self.y = random.randint(0, yMax)
                else:
                    self.x = x
                    self.y = y
                    self.index = ci

            def getX(self):
                return self.x

            def getY(self):
                return self.y

            def getIndex(self):
                return self.index

            # Returns distance to the dustbin passed as argument
            def distanceTo(self, db):
                # old method
                # xDis = abs(self.getX() - db.getX())
                # yDis = abs(self.getY() - db.getY())
                # dis = math.sqrt((xDis*xDis) + (yDis*yDis))

                # new method from Zhuolun
                dis = dist_map[self.getIndex()][db.getIndex()]
                return dis

            # Gives string representation of the Object with coordinates

            def toString(self):
                s = '(' + str(self.getX()) + ',' + str(self.getY()) + ')'
                return s

            # Check if cordinates have been assigned or not
            # Dusbins with (-1, -1) as coordinates are created during creation on chromosome objects
            def checkNull(self):
                if self.x == -1:
                    return True
                else:
                    return False

        '''
        routemanager

        Holds all the dustbin objects and is used for
        creation of chromosomes by jumbling their sequence
        '''

        class RouteManager:
            destinationDustbins = []

            @classmethod
            def addDustbin(cls, db):
                cls.destinationDustbins.append(db)

            @classmethod
            def getDustbin(cls, index):
                return cls.destinationDustbins[index]

            @classmethod
            def numberOfDustbins(cls):
                return len(cls.destinationDustbins)

        '''
        route

        Represents the chromosomes in GA's population.
        The object is collection of individual routes taken by trucks.
        '''

        class Route:
            # Good old constructor
            def __init__(self, route=None):
                # 2D array which is collection of respective routes taken by trucks
                self.route = []
                # 1D array having routes in a series - used during crossover operation
                self.base = []
                # 1D array having route lengths
                self.routeLengths = route_lengths()

                for i in range(numTrucks):
                    self.route.append([])

                # fitness value and total distance of all routes
                self.fitness = 0
                self.distance = 0

                # creating empty route
                if route == None:
                    for i in range(RouteManager.numberOfDustbins()-1):
                        self.base.append(Dustbin(-1, -1))

                else:
                    self.route = route

            def generateIndividual(self, antpath):
                k = 0

                # ---Modify by ming---
                # Original version use a random method to create init generate:
                # ---
                # # put 1st member of RouteManager as it is (It represents the initial node) and shuffle the rest before adding
                # for dindex in range(1, RouteManager.numberOfDustbins()):
                #     self.base[dindex-1] = RouteManager.getDustbin(dindex)
                #     random.shuffle(self.base)
                # ---
                # Now we use ant colony algorithm to do this, every time we generate an individual,
                # we random choose one from ant colony.
                # The whole algorithm is in file ant.py

                # print(antpath)
                for i in range(numTrucks):
                    # add same first node for each route
                    self.route[i].append(RouteManager.getDustbin(0))
                    for j in range(self.routeLengths[i]-1):
                        Dustbin = RouteManager.getDustbin(int(antpath[k]))
                        # add shuffled values for rest
                        self.route[i].append(Dustbin)
                        k += 1
                # print(self.route)
            # Returns j'th dustbin in i'th route

            def getDustbin(self, i, j):
                return self.route[i][j]

            # Sets value of j'th dustbin in i'th route
            def setDustbin(self, i, j, db):
                self.route[i][j] = db
                #self.route.insert(index, db)
                self.fitness = 0
                self.distance = 0

            # Returns the fitness value of route
            def getFitness(self):
                if self.fitness == 0:
                    fitness = 1/self.getDistance()

                return fitness

            # Return total ditance covered in all subroutes
            def getDistance(self):
                if self.distance == 0:
                    routeDistance = 0

                    for i in range(numTrucks):
                        for j in range(self.routeLengths[i]):
                            fromDustbin = self.getDustbin(i, j)

                            if j+1 < self.routeLengths[i]:
                                destinationDustbin = self.getDustbin(i, j + 1)

                            else:
                                destinationDustbin = self.getDustbin(i, 0)

                            routeDistance += fromDustbin.distanceTo(
                                destinationDustbin)

                distance = routeDistance
                return routeDistance

            # Checks if the route contains a particular dustbin
            def containsDustbin(self, db):
                if db in self.base:  # base <-> route
                    return True
                else:
                    return False

            # Returns route in the form of a string
            def toString(self):
                geneString = '|'
                print(self.routeLengths)
                # for k in range(RouteManager.numberOfDustbins()-1):
                #    print (self.base[k].toString())
                for i in range(numTrucks):
                    for j in range(self.routeLengths[i]):
                        geneString += self.getDustbin(i, j).toString() + '|'
                    geneString += '\n'

                return geneString

        '''
        ant
        '''

        # 蚁群算法解决TSP

        class AntColony:
            def __init__(self):
                self.path_best = []
                self.distance_best = []

            def iterate(self, max_iter=20):
                MAX_iter = max_iter  # 最大迭代值
                AntCount = 100  # 蚂蚁数量
                # 信息素
                alpha = 1  # 信息素重要程度因子
                beta = 2  # 启发函数重要程度因子
                rho = 0.1  # 挥发速度

                Q = 1
                # 城市列表
                citys = []
                # 去掉第一个城市
                for i in range(1, len(X)):
                    citys.append([X[i], Y[i]])
                city_count = len(citys)

                # 预计算距离
                # Distance = np.zeros((city_count, city_count))
                # for i in range(city_count):
                #     for j in range(city_count):
                #         if i != j:
                #             Distance[i][j] = math.sqrt((citys[i][0] - citys[j][0]) ** 2 + (citys[i][1] - citys[j][1]) ** 2)
                #         else:
                #             Distance[i][j] = 100000
                Distance = dist_map[1:, 1:]

                # 初始信息素矩阵，全是为1组成的矩阵
                pheromonetable = np.ones((city_count, city_count))

                # 候选集列表,存放所有蚂蚁的路径
                candidate = np.zeros((AntCount, city_count)).astype(int)

                # path_best存放的是相应的，每次迭代后的最优路径，每次迭代只有一个值
                path_best = np.zeros((MAX_iter, city_count))

                # 存放每次迭代的最优距离
                distance_best = np.zeros(MAX_iter)

                # 倒数矩阵
                etable = 1.0 / Distance
                iter = 0  # 迭代初始值

                while iter < MAX_iter:
                    # first：蚂蚁初始点选择
                    # print(iter)
                    if AntCount <= city_count:
                        candidate[:, 0] = np.random.permutation(
                            range(city_count))[:AntCount]
                    else:
                        m = AntCount - city_count
                        n = 2
                        candidate[:city_count, 0] = np.random.permutation(range(city_count))[
                            :]
                        while m > city_count:
                            candidate[city_count*(n - 1):city_count*n,
                                      0] = np.random.permutation(range(city_count))[:]
                            m = m - city_count
                            n = n + 1
                        candidate[city_count*(n-1):AntCount,
                                  0] = np.random.permutation(range(city_count))[:m]
                    length = np.zeros(AntCount)  # 每次迭代的N个蚂蚁的距离值

                    # second：选择下一个城市选择
                    for i in range(AntCount):
                        # 移除已经访问的第一个元素
                        unvisit = list(range(city_count))  # 列表形式存储没有访问的城市编号
                        visit = candidate[i, 0]  # 当前所在点,第i个蚂蚁在第一个城市
                        unvisit.remove(visit)  # 在未访问的城市中移除当前开始的点
                        for j in range(1, city_count):  # 访问剩下的city_count个城市，city_count次访问
                            # 每次循环都更改当前没有访问的城市的转移概率矩阵1*30,1*29,1*28...
                            protrans = np.zeros(len(unvisit))
                            # 下一城市的概率函数
                            for k in range(len(unvisit)):
                                # 计算当前城市到剩余城市的（信息素浓度^alpha）*（城市适应度的倒数）^beta
                                # etable[visit][unvisit[k]],(alpha+1)是倒数分之一，pheromonetable[visit][unvisit[k]]是从本城市到k城市的信息素
                                protrans[k] = np.power(pheromonetable[visit][unvisit[k]], alpha) * np.power(
                                    etable[visit][unvisit[k]], (alpha + 1))

                            cumsumprobtrans = (
                                protrans / sum(protrans)).cumsum()
                            cumsumprobtrans -= np.random.rand()
                            # 求出离随机数产生最近的索引值
                            k = unvisit[list(cumsumprobtrans > 0).index(True)]
                            # 下一个访问城市的索引值
                            candidate[i, j] = k
                            unvisit.remove(k)
                            length[i] += Distance[visit][k]
                            visit = k  # 更改出发点，继续选择下一个到达点
                        # 最后一个城市和第一个城市的距离值也要加进去
                        length[i] += Distance[visit][candidate[i, 0]]

                    # 如果迭代次数为一次，那么无条件让初始值代替path_best,distance_best.
                    if iter == 0:
                        distance_best[iter] = length.min()
                        path_best[iter] = candidate[length.argmin()].copy()
                    else:
                        # 如果当前的解没有之前的解好，那么当前最优还是为之前的那个值；并且用前一个路径替换为当前的最优路径
                        if length.min() > distance_best[iter - 1]:
                            distance_best[iter] = distance_best[iter - 1]
                            path_best[iter] = path_best[iter - 1].copy()
                        else:  # 当前解比之前的要好，替换当前解和路径
                            distance_best[iter] = length.min()
                            path_best[iter] = candidate[length.argmin()].copy()

                    # 信息素的增加量矩阵
                    changepheromonetable = np.zeros((city_count, city_count))
                    for i in range(AntCount):
                        for j in range(city_count - 1):
                            changepheromonetable[candidate[i, j]
                                                 ][candidate[i][j + 1]] += Q / length[i]
                        # 最后一个城市和第一个城市的信息素增加量
                        changepheromonetable[candidate[i, j + 1]
                                             ][candidate[i, 0]] += Q / length[i]
                    # 信息素更新
                    pheromonetable = (1 - rho) * \
                        pheromonetable + changepheromonetable
                    iter += 1
                self.path_best = path_best
                self.distance_best = distance_best

        '''
        population

        Collection of routes (chrmosomes)
        '''

        class Population:
            routes = []
            # Good old contructor

            def __init__(self, populationSize, initialise):
                self.populationSize = populationSize
                if initialise:
                    # ---Modify by ming---
                    # 种群初始化时先通过蚁群算法生成tsp的解作为备选集，再随机从该集合中选择生成个体
                    ants = AntColony()
                    ants.iterate(20)  # 迭代次数多了以后会很慢
                    ant_path = ants.path_best
                    for i in range(populationSize):
                        newRoute = Route()  # Create empty route
                        choose = random.randint(1, len(ant_path))
                        newRoute.generateIndividual(
                            ant_path[choose - 1] + 1)  # Add route sequences
                        # Add route to the population
                        self.routes.append(newRoute)

            # Saves the route passed as argument at index
            def saveRoute(self, index, route):
                self.routes[index] = route

            # Returns route at index
            def getRoute(self, index):
                return self.routes[index]

            # Returns route with maximum fitness value
            def getFittest(self):
                fittest = self.routes[0]

                for i in range(1, self.populationSize):
                    if fittest.getFitness() <= self.getRoute(i).getFitness():
                        fittest = self.getRoute(i)

                return fittest

            def populationSize(self):
                return int(self.populationSize)

            # Equate current population values to that of pop
            def equals(self, pop):
                self.routes = pop.routes

        '''
        galogic

        The main helper class for Genetic Algorithm to perform
        crossover, mutation on populations to evolve them
        '''

        class GA:

            @classmethod
            # Evolve pop
            def evolvePopulation(cls, pop):

                newPopulation = Population(pop.populationSize, False)

                elitismOffset = 0
                # If fittest chromosome has to be passed directly to next generation
                if elitism:
                    newPopulation.saveRoute(0, pop.getFittest())
                    elitismOffset = 1

                # Performs tournament selection followed by crossover to generate child
                for i in range(elitismOffset, newPopulation.populationSize):
                    parent1 = cls.tournamentSelection(pop)
                    parent2 = cls.tournamentSelection(pop)
                    child = cls.crossover(parent1, parent2)
                    # Adds child to next generation
                    newPopulation.saveRoute(i, child)

                # Performs Mutation
                for i in range(elitismOffset, newPopulation.populationSize):
                    if random.randrange(100)/100 < mutationRate:
                        cls.mutate(newPopulation.getRoute(i))

                return newPopulation

            # Function to implement crossover operation
            @classmethod
            def crossover(cls, parent1, parent2):
                child = Route()
                # since size is (numNodes - 1) by default
                child.base.append(Dustbin(-1, -1))
                startPos = 0
                endPos = 0
                # while (startPos >= endPos):
                #     startPos = random.randint(1, numNodes-1)
                #     endPos = random.randint(1, numNodes-1)
                startPos = random.randint(1, numNodes-2)
                endPos = random.randint(startPos, numNodes-1)

                parent1.base = [parent1.route[0][0]]
                parent2.base = [parent2.route[0][0]]

                for i in range(numTrucks):
                    for j in range(1, parent1.routeLengths[i]):
                        parent1.base.append(parent1.route[i][j])

                for i in range(numTrucks):
                    for j in range(1, parent2.routeLengths[i]):
                        parent2.base.append(parent2.route[i][j])

                for i in range(1, numNodes):
                    if i > startPos and i < endPos:
                        child.base[i] = parent1.base[i]

                for i in range(numNodes):
                    if not(child.containsDustbin(parent2.base[i])):
                        for i1 in range(numNodes):
                            if child.base[i1].checkNull():
                                child.base[i1] = parent2.base[i]
                                break

                k = 0
                child.base.pop(0)
                for i in range(numTrucks):
                    # add same first node for each route
                    child.route[i].append(RouteManager.getDustbin(0))
                    for j in range(child.routeLengths[i]-1):
                        # add shuffled values for rest
                        child.route[i].append(child.base[k])
                        k += 1
                return child

            # Mutation opeeration
            @classmethod
            def mutate(cls, route):
                index1 = 0
                index2 = 0
                while index1 == index2:
                    index1 = random.randint(0, numTrucks - 1)
                    index2 = random.randint(0, numTrucks - 1)
                #print ('Indexes selected: ' + str(index1) + ',' + str(index2))

                # generate replacement range for 1
                route1startPos = 0
                route1lastPos = 0
                while route1startPos >= route1lastPos or route1startPos == 1:
                    route1startPos = random.randint(
                        1, route.routeLengths[index1] - 1)
                    route1lastPos = random.randint(
                        1, route.routeLengths[index1] - 1)

                # generate replacement range for 2
                route2startPos = 0
                route2lastPos = 0
                while route2startPos >= route2lastPos or route2startPos == 1:
                    route2startPos = random.randint(
                        1, route.routeLengths[index2] - 1)
                    route2lastPos = random.randint(
                        1, route.routeLengths[index2] - 1)

                #print ('startPos, lastPos: ' + str(route1startPos) + ',' + str(route1lastPos) + ',' + str(route2startPos) + ',' + str(route2lastPos))
                swap1 = []  # values from 1
                swap2 = []  # values from 2

                # if random.randrange(1) < mutationRate:
                # A change! reduce the random rate, former code has 100% mutate
                # pop all the values to be replaced
                for i in range(route1startPos, route1lastPos + 1):
                    swap1.append(route.route[index1].pop(route1startPos))

                for i in range(route2startPos, route2lastPos + 1):
                    swap2.append(route.route[index2].pop(route2startPos))

                del1 = (route1lastPos - route1startPos + 1)
                del2 = (route2lastPos - route2startPos + 1)

                # add to new location by pushing
                route.route[index1][route1startPos:route1startPos] = swap2
                route.route[index2][route2startPos:route2startPos] = swap1

                route.routeLengths[index1] = len(route.route[index1])
                route.routeLengths[index2] = len(route.route[index2])

            # Tournament Selection: choose a random set of chromosomes and find the fittest among them
            @classmethod
            def tournamentSelection(cls, pop):
                tournament = Population(tournamentSize, False)

                for i in range(tournamentSize):
                    randomInt = random.randint(0, pop.populationSize-1)
                    tournament.saveRoute(i, pop.getRoute(randomInt))

                fittest = tournament.getFittest()
                return fittest

        '''
        main
        '''

        # Add Dustbins
        for i in range(numNodes):
            RouteManager.addDustbin(Dustbin(X[i], Y[i], i))

        random.seed(seedValue)
        yaxis = []  # Fittest value (distance)
        xaxis = []  # Generation count

        pop = Population(populationSize, True)
        globalRoute = pop.getFittest()
        initial_distance = globalRoute.getDistance()

        # Start evolving
        for i in pbar(range(numGenerations)):
            pop = GA.evolvePopulation(pop)
            localRoute = pop.getFittest()
            if globalRoute.getDistance() > localRoute.getDistance():
                globalRoute = localRoute
            yaxis.append(localRoute.getDistance())
            xaxis.append(i)

        toc = time.time()
        df_row = df.shape[0]
        df.loc[df_row] = [initial_distance, round(
            toc-tic, 2), globalRoute.getDistance()]

        del pop
        del localRoute
        del Population
        del globalRoute
        del initial_distance
        del GA
        del Route
        del RouteManager
        del Dustbin
        del AntColony
        df.to_csv("ours_" + data + ".csv")
        print("Wrote output to the " + data + ".csv")
