import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import progressbar
import pandas as pd
import time

'''
globals.py

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
mutationRate = 0.25
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
    print("Running baseline method test on the data " + data)

    for k in range(test_time):
        tic = time.time()
        pbar = progressbar.ProgressBar()
        print(f"Running for the time {k+1}")

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
        dustbin.py

        Represents nodes in the problem graph or network.
        Locatin coordinates can be passed while creating the object or they
        will be assigned random values.
        '''

        class Dustbin:
            # Good old constructor
            def __init__(self, x=None, y=None):
                if x == None and y == None:
                    self.x = random.randint(0, xMax)
                    self.y = random.randint(0, yMax)
                else:
                    self.x = x
                    self.y = y

            def getX(self):
                return self.x

            def getY(self):
                return self.y

            # Returns distance to the dustbin passed as argument
            def distanceTo(self, db):
                xDis = abs(self.getX() - db.getX())
                yDis = abs(self.getY() - db.getY())
                dis = math.sqrt((xDis*xDis) + (yDis*yDis))
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
        routemanager.py


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
        route.py

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

            def generateIndividual(self):
                k = 0
                # put 1st member of RouteManager as it is (It represents the initial node) and shuffle the rest before adding
                for dindex in range(1, RouteManager.numberOfDustbins()):
                    self.base[dindex-1] = RouteManager.getDustbin(dindex)
                random.shuffle(self.base)

                for i in range(numTrucks):
                    # add same first node for each route
                    self.route[i].append(RouteManager.getDustbin(0))
                    for j in range(self.routeLengths[i]-1):
                        # add shuffled values for rest
                        self.route[i].append(self.base[k])
                        k += 1

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
        population.py

        Collection of routes (chrmosomes)
        '''

        class Population:
            routes = []
            # Good old contructor

            def __init__(self, populationSize, initialise):
                self.populationSize = populationSize
                if initialise:
                    for i in range(populationSize):
                        newRoute = Route()  # Create empty route
                        newRoute.generateIndividual()  # Add route sequences
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
        galogic.py

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
                while (startPos >= endPos):
                    startPos = random.randint(1, numNodes-1)
                    endPos = random.randint(1, numNodes-1)

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

                if random.randrange(1) < mutationRate:
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
            RouteManager.addDustbin(Dustbin(X[i], Y[i]))

        # random.seed(seedValue)
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
        del Population
        del globalRoute
        del initial_distance
        del GA
        del Route
        del RouteManager
        del Dustbin
        df.to_csv("baseline_" + data + ".csv")
        print("Wrote output to the " + data + ".csv")
