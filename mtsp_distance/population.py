'''
Collection of routes (chrmosomes)
'''
from route import *
from ant import *


class Population:
    routes = []
    # Good old contructor
    def __init__ (self, populationSize, initialise):
        self.populationSize = populationSize
        if initialise:
            # ---Modify by ming---
            # 种群初始化时先通过蚁群算法生成tsp的解作为备选集，再随机从该集合中选择生成个体
            ants = AntColony()
            ants.iterate(20)  # 迭代次数多了以后会很慢
            ant_path = ants.path_best
            for i in range(populationSize):
                newRoute = Route() # Create empty route
                choose = random.randint(1, len(ant_path))
                newRoute.generateIndividual(ant_path[choose - 1] + 1) # Add route sequences
                self.routes.append(newRoute) # Add route to the population

    # Saves the route passed as argument at index
    def saveRoute (self, index, route):
        self.routes[index] = route

    # Returns route at index
    def getRoute (self, index):
        return self.routes[index]

    # Returns route with maximum fitness value
    def getFittest (self):
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
